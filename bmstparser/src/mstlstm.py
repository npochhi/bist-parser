import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import *
from torch.autograd import Variable
from torch import optim
from utils import read_conll, write_conll
from operator import itemgetter
import utils, time, random, decoder
import numpy as np


def Parameter(shape, init=xavier_uniform):
    return Variable(init(torch.Tensor(*shape)), requires_grad=True)


class MSTParserLSTMModel(nn.Module):
    def __init__(self, vocab, pos, rels, w2i, options):
        super(MSTParserLSTMModel, self).__init__()
        random.seed(1)
        self.activations = {'tanh': F.tanh, 'sigmoid': F.sigmoid, 'relu': F.relu,
                            # Not yet supporting tanh3
                            # 'tanh3': (lambda x: nn.Tanh()(cwise_multiply(cwise_multiply(x, x), x)))
                            }
        self.activation = self.activations[options.activation]

        self.blstmFlag = options.blstmFlag
        self.labelsFlag = options.labelsFlag
        self.costaugFlag = options.costaugFlag
        self.bibiFlag = options.bibiFlag

        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims
        self.layers = options.lstm_layers
        self.wordsCount = vocab
        self.vocab = {word: ind + 3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind + 3 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels

        self.external_embedding, self.edim = None, 0
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding, 'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in
                                       external_embedding_fp}
            external_embedding_fp.close()

            self.edim = len(self.external_embedding.values()[0])
            self.noextrn = [0.0 for _ in xrange(self.edim)]
            self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            self.elookup = self.model.add_lookup_parameters((len(self.external_embedding) + 3, self.edim))
            for word, i in self.extrnd.iteritems():
                self.elookup.init_row(i, self.external_embedding[word])
            self.extrnd['*PAD*'] = 1
            self.extrnd['*INITIAL*'] = 2

            print 'Load external embedding. Vector dimensions', self.edim

        if self.bibiFlag:
            self.builders = [nn.LSTMCell(self.wdims + self.pdims + self.edim, self.ldims),
                             nn.LSTMCell(self.wdims + self.pdims + self.edim, self.ldims)]
            self.bbuilders = [nn.LSTMCell(1, self.ldims * 2, self.ldims),
                              nn.LSTMCell(1, self.ldims * 2, self.ldims)]
        elif self.layers > 0:
            assert self.layers == 1, 'Not yet support deep LSTM'
            self.builders = [
                nn.LSTMCell(self.layers, self.wdims + self.pdims + self.edim, self.ldims),
                nn.LSTMCell(self.layers, self.wdims + self.pdims + self.edim, self.ldims)]
        else:
            self.builders = [nn.RNNCell(self.wdims + self.pdims + self.edim, self.ldims),
                             nn.RNNCell(self.wdims + self.pdims + self.edim, self.ldims)]

        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units

        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2

        self.wlookup = nn.Embedding(len(vocab) + 3, self.wdims)
        self.plookup = nn.Embedding(len(pos) + 3, self.pdims)
        self.rlookup = nn.Embedding(len(rels), self.rdims)

        self.hidLayerFOH = Parameter((self.hidden_units, self.ldims * 2))
        self.hidLayerFOM = Parameter((self.hidden_units, self.ldims * 2))
        self.hidBias = Parameter((self.hidden_units))

        self.hid2Layer = Parameter((self.hidden2_units, self.hidden_units))
        self.hid2Bias = Parameter((self.hidden2_units))

        self.outLayer = Parameter(
            (1, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))

        if self.labelsFlag:
            self.rhidLayerFOH = Parameter((self.hidden_units, 2 * self.ldims))
            self.rhidLayerFOM = Parameter((self.hidden_units, 2 * self.ldims))
            self.rhidBias = Parameter((self.hidden_units))

            self.rhid2Layer = Parameter((self.hidden2_units, self.hidden_units))
            self.rhid2Bias = Parameter((self.hidden2_units))

            self.routLayer = Parameter(
                (len(self.irels), self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
            self.routBias = Parameter((len(self.irels)))

    def __getExpr(self, sentence, i, j, train):

        if sentence[i].headfov is None:
            sentence[i].headfov = torch.mm(self.hidLayerFOH, torch.cat([sentence[i].lstms[0], sentence[i].lstms[1]]))
        if sentence[j].modfov is None:
            sentence[j].modfov = torch.mm(self.hidLayerFOM, torch.cat([sentence[j].lstms[0], sentence[j].lstms[1]]))

        if self.hidden2_units > 0:
            output = torch.mm(
                self.outLayer,
                self.activation(
                    self.hid2Bias +
                    torch.mm(self.hid2Layer,
                             self.activation(sentence[i].headfov + sentence[j].modfov + self.hidBias))
                )
            )  # + self.outBias
        else:
            output = torch.mm(
                self.outLayer,
                self.activation(
                    sentence[i].headfov + sentence[j].modfov + self.hidBias)
            )  # + self.outBias

        return output

    def __evaluate(self, sentence, train):
        exprs = [[self.__getExpr(sentence, i, j, train) for j in xrange(len(sentence))] for i in xrange(len(sentence))]
        scores = np.array([[output for output in exprsRow] for exprsRow in exprs])

        return scores, exprs

    def __evaluateLabel(self, sentence, i, j):
        if sentence[i].rheadfov is None:
            sentence[i].rheadfov = torch.mm(self.rhidLayerFOH, torch.cat([sentence[i].lstms[0], sentence[i].lstms[1]]))
        if sentence[j].rmodfov is None:
            sentence[j].rmodfov = torch.mm(self.rhidLayerFOM, torch.cat([sentence[j].lstms[0], sentence[j].lstms[1]]))

        if self.hidden2_units > 0:
            output = torch.mm(
                self.routLayer,
                self.activation(
                    self.rhid2Bias +
                    torch.mm(
                        self.rhid2Layer,
                        self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias)
                    ) +
                    self.routBias)
            )
        else:
            output = torch.mm(
                self.routLayer.expr(),
                self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias)
            ) + self.routBias

        return output.value(), output

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.load(filename)

    def predict(self, sentence):
        for entry in sentence:
            wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0))] if self.wdims > 0 else None
            posvec = self.plookup[int(self.pos[entry.pos])] if self.pdims > 0 else None
            evec = self.elookup[int(self.extrnd.get(entry.form, self.extrnd.get(entry.norm,
                                                                                0)))] if self.external_embedding is not None else None
            entry.vec = torch.cat([wordvec, posvec, evec])

            entry.lstms = [entry.vec, entry.vec]
            entry.headfov = None
            entry.modfov = None

            entry.rheadfov = None
            entry.rmodfov = None

        if self.blstmFlag:
            lstm_forward = self.builders[0].initial_state()
            lstm_backward = self.builders[1].initial_state()

            for entry, rentry in zip(sentence, reversed(sentence)):
                lstm_forward = lstm_forward.add_input(entry.vec)
                lstm_backward = lstm_backward.add_input(rentry.vec)

                entry.lstms[1] = lstm_forward.output()
                rentry.lstms[0] = lstm_backward.output()

            if self.bibiFlag:
                for entry in sentence:
                    entry.vec = torch.cat(entry.lstms)

                blstm_forward = self.bbuilders[0].initial_state()
                blstm_backward = self.bbuilders[1].initial_state()

                for entry, rentry in zip(sentence, reversed(sentence)):
                    blstm_forward = blstm_forward.add_input(entry.vec)
                    blstm_backward = blstm_backward.add_input(rentry.vec)

                    entry.lstms[1] = blstm_forward.output()
                    rentry.lstms[0] = blstm_backward.output()

        scores, exprs = self.__evaluate(sentence, True)
        heads = decoder.parse_proj(scores)

        for entry, head in zip(sentence, heads):
            entry.pred_parent_id = head
            entry.pred_relation = '_'

        if self.labelsFlag:
            for modifier, head in enumerate(heads[1:]):
                scores, exprs = self.__evaluateLabel(sentence, head, modifier + 1)
                sentence[modifier + 1].pred_relation = self.irels[
                    max(enumerate(scores), key=itemgetter(1))[0]]

    def get_loss(self, sentence, errs, lerrs):

        for entry in sentence:
            c = float(self.wordsCount.get(entry.norm, 0))
            dropFlag = (random.random() < (c / (0.25 + c)))
            wordvec = self.wlookup[
                int(self.vocab.get(entry.norm, 0)) if dropFlag else 0] if self.wdims > 0 else None
            posvec = self.plookup[int(self.pos[entry.pos])] if self.pdims > 0 else None
            evec = None

            if self.external_embedding is not None:
                evec = self.elookup[self.extrnd.get(entry.form, self.extrnd.get(entry.norm, 0)) if (
                    dropFlag or (random.random() < 0.5)) else 0]
            entry.vec = torch.cat([wordvec, posvec, evec])

            entry.lstms = [entry.vec, entry.vec]
            entry.headfov = None
            entry.modfov = None

            entry.rheadfov = None
            entry.rmodfov = None

        if self.blstmFlag:
            lstm_forward = self.builders[0].initial_state()
            lstm_backward = self.builders[1].initial_state()

            for entry, rentry in zip(sentence, reversed(sentence)):
                lstm_forward = lstm_forward.add_input(entry.vec)
                lstm_backward = lstm_backward.add_input(rentry.vec)

                entry.lstms[1] = lstm_forward.output()
                rentry.lstms[0] = lstm_backward.output()

            if self.bibiFlag:
                for entry in sentence:
                    entry.vec = torch.cat(entry.lstms)

                blstm_forward = self.bbuilders[0].initial_state()
                blstm_backward = self.bbuilders[1].initial_state()

                for entry, rentry in zip(sentence, reversed(sentence)):
                    blstm_forward = blstm_forward.add_input(entry.vec)
                    blstm_backward = blstm_backward.add_input(rentry.vec)

                    entry.lstms[1] = blstm_forward.output()
                    rentry.lstms[0] = blstm_backward.output()

        scores, exprs = self.__evaluate(sentence, True)
        gold = [entry.parent_id for entry in sentence]
        heads = decoder.parse_proj(scores, gold if self.costaugFlag else None)

        if self.labelsFlag:
            for modifier, head in enumerate(gold[1:]):
                rscores, rexprs = self.__evaluateLabel(sentence, head, modifier + 1)
                goldLabelInd = self.rels[sentence[modifier + 1].relation]
                wrongLabelInd = \
                    max(((l, scr) for l, scr in enumerate(rscores) if l != goldLabelInd), key=itemgetter(1))[0]
                if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
                    lerrs.append(rexprs[wrongLabelInd] - rexprs[goldLabelInd])

        e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
        if e > 0:
            loss = [(exprs[h][i] - exprs[g][i]) for i, (h, g) in enumerate(zip(heads, gold)) if
                    h != g]
            errs.extend(loss)
        return e


class MSTParserLSTM:
    def __init__(self, vocab, pos, rels, w2i, options):
        self.model = MSTParserLSTMModel(vocab, pos, rels, w2i, options)

        self.trainer = {'sgd': optim.SGD, 'adam': optim.Adam}[options.opt](self.model.parameters(), **options)

    def predict(self, conll_path):
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP)):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                self.model.predict(conll_sentence)
                yield conll_sentence

    def train(self, conll_path):
        batch = 0
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        iSentence = 0
        start = time.time()
        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP))
            random.shuffle(shuffledData)
            errs = []
            lerrs = []
            for iSentence, sentence in enumerate(shuffledData):
                if iSentence % 100 == 0 and iSentence != 0:
                    print 'Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Errors:', (float(
                        eerrors)) / etotal, 'Time', time.time() - start
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                e = self.model.get_loss(conll_sentence, errs, lerrs)
                eloss += e
                mloss += e
                etotal += len(sentence)
                if iSentence % batch == 0 or len(errs) > 0 or len(lerrs) > 0:
                    if len(errs) > 0 or len(lerrs) > 0:
                        eerrs = torch.sum(errs + lerrs)  # * (1.0/(float(len(errs))))
                        eerrs.scalar_value()
                        eerrs.backward()
                        self.trainer.step()
                        errs = []
                        lerrs = []
        if len(errs) > 0:
            eerrs = (torch.sum(errs + lerrs))
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.step()
        print "Loss: ", mloss / iSentence

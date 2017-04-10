# A Pytorch implementation of the BIST Parsers (for graph based parser only)
The DyNet implementation by the author's of the original paper [Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations](https://www.transacl.org/ojs/index.php/tacl/article/viewFile/885/198) is [here](https://github.com/elikip/bist-parser) 

#### Required software

 * Python 2.7 interpreter
 * [Pytorch library](http://pytorch.org/)

#### Train a parsing model

The software requires having a `training.conll` and `development.conll` files formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat).

#### Train data a parsing model:

    python src/parser.py --outdir [results directory] --train training.conll --dev development.conll --epochs 30 --lstmdims 125 --lstmlayers 2 [--extrn extrn.vectors] --bibi-lstm

#### Parse data with your parsing model

The command for parsing a `test.conll` file formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat) with a previously trained model is:

    python src/parser.py --predict --outdir [results directory] --test test.conll [--extrn extrn.vectors] --model [trained model file] --params [param file generate during training]

The parser will store the resulting conll file in the out directory (`--outdir`).

import numpy as np
from dependency_decoding import chu_liu_edmonds


def parse_proj(scores):
    scores = scores.astype('double')
    heads, tree_score = chu_liu_edmonds(scores)
    return np.asanyarray(heads)

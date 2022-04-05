import numpy as np
from scipy import linalg
from collections import defaultdict


def sigmoid(z):
    # sigmoid function
    return 1.0 / (1.0 + np.exp(-z))


def get_idx(words, word2Ind):
    idx = []
    for word in words:
        idx = idx + [word2Ind[word]]
    return idx
    
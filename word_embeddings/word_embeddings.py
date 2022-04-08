import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter
from utils import sigmoid, get_batches, compute_pca, get_dict

import re                                                           #  Load the Regex-modul
with open('../data/word_embeddings/shakespeare.txt') as f:
    data = f.read()                                                 #  Read in the data
data = re.sub(r'[,!?;-]', '.',data)                                 #  Punktuations are replaced by .
data = nltk.word_tokenize(data)                                     #  Tokenize string to words
data = [ ch.lower() for ch in data if ch.isalpha() or ch == '.']    #  Lower case and drop non-alphabetical tokens
print("Number of tokens:", len(data),'\n', data[:15])               #  print data sample
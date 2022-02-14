# Importing packages and loading in the data set 
from utils import get_word_tag, preprocess  
import pandas as pd
from collections import defaultdict
import math
import numpy as np

# get training corpus

with open("../data/pos/WSJ_02-21.pos") as f:
    training_corpus=f.readlines()

# read the vocabulary data, split by each line of text, and save the list
with open("./data/hmm_vocab.txt", 'r') as f:
    voc_l = f.read().split('\n')

# vocab: dictionary that has the index of the corresponding words
vocab = {}

# Get the index of the corresponding words. 
for i, word in enumerate(sorted(voc_l)): 
    vocab[word] = i   



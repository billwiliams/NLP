import os 
import numpy as np
import pandas as pd
import random as rnd

import trax 

from trax.supervised import training
from trax import layers as tl

# set random seeds to make this notebook easier to replicate
rnd.seed(33)

data = pd.read_csv("../data/ner_dataset.csv", encoding = "ISO-8859-1") 
train_sents = open('../data/small/train/sentences.txt', 'r').readline()
train_labels = open('../data/small/train/labels.txt', 'r').readline()
print('SENTENCE:', train_sents)
print('SENTENCE LABEL:', train_labels)
print('ORIGINAL DATA:\n', data.head(5))
del(data, train_sents, train_labels)

vocab, tag_map = get_vocab('data/large/words.txt', 'data/large/tags.txt')
t_sentences, t_labels, t_size = get_params(vocab, tag_map, 'data/large/train/sentences.txt', 'data/large/train/labels.txt')
v_sentences, v_labels, v_size = get_params(vocab, tag_map, 'data/large/val/sentences.txt', 'data/large/val/labels.txt')
test_sentences, test_labels, test_size = get_params(vocab, tag_map, 'data/large/test/sentences.txt', 'data/large/test/labels.txt')
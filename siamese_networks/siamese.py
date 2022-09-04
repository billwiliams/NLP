import os
import nltk
import trax
from trax import layers as tl
from trax.supervised import training
from trax.fastmath import numpy as fastnp
import numpy as np
import pandas as pd
import random as rnd

import w4_unittest

nltk.download('punkt')

# set random seeds
rnd.seed(34)

data = pd.read_csv("data/questions.csv")
N=len(data)
print('Number of question pairs: ', N)
data.head()

N_train = 300000
N_test  = 10*1024
data_train = data[:N_train]
data_test  = data[N_train:N_train+N_test]
print("Train set:", len(data_train), "Test set:", len(data_test))
del(data) # remove to free memory
Q1_train_words = np.array(data_train['question1'][td_index])
Q2_train_words = np.array(data_train['question2'][td_index])

Q1_test_words = np.array(data_test['question1'])
Q2_test_words = np.array(data_test['question2'])
y_test  = np.array(data_test['is_duplicate'])
print('TRAINING QUESTIONS:\n')
print('Question 1: ', Q1_train_words[0])
print('Question 2: ', Q2_train_words[0], '\n')
print('Question 1: ', Q1_train_words[5])
print('Question 2: ', Q2_train_words[5], '\n')

print('TESTING QUESTIONS:\n')
print('Question 1: ', Q1_test_words[0])
print('Question 2: ', Q2_test_words[0], '\n')
print('is_duplicate =', y_test[0], '\n')

#create arrays
Q1_train = np.empty_like(Q1_train_words)
Q2_train = np.empty_like(Q2_train_words)

Q1_test = np.empty_like(Q1_test_words)
Q2_test = np.empty_like(Q2_test_words)

from collections import defaultdict

vocab = defaultdict(lambda: 0)
vocab['<PAD>'] = 1

for idx in range(len(Q1_train_words)):
        Q1_train[idx] = nltk.word_tokenize(Q1_train_words[idx])
            Q2_train[idx] = nltk.word_tokenize(Q2_train_words[idx])
                q = Q1_train[idx] + Q2_train[idx]
                    for word in q:
                                if word not in vocab:
                                                vocab[word] = len(vocab) + 1
                                                print('The length of the vocabulary is: ', len(vocab))

print(vocab['<PAD>'])
print(vocab['Astrology'])
print(vocab['Astronomy'])  #not in vocabulary, returns 0

for idx in range(len(Q1_test_words)): 
        Q1_test[idx] = nltk.word_tokenize(Q1_test_words[idx])
            Q2_test[idx] = nltk.word_tokenize(Q2_test_words[idx])

print('Train set has reduced to: ', len(Q1_train) ) 
print('Test set length: ', len(Q1_test) ) 
# Converting questions to array of integers
for i in range(len(Q1_train)):
        Q1_train[i] = [vocab[word] for word in Q1_train[i]]
            Q2_train[i] = [vocab[word] for word in Q2_train[i]]

                    
                    for i in range(len(Q1_test)):
                            Q1_test[i] = [vocab[word] for word in Q1_test[i]]
                                Q2_test[i] = [vocab[word] for word in Q2_test[i]]

# Splitting the data
cut_off = int(len(Q1_train)*.8)
train_Q1, train_Q2 = Q1_train[:cut_off], Q2_train[:cut_off]
val_Q1, val_Q2 = Q1_train[cut_off: ], Q2_train[cut_off:]
print('Number of duplicate questions: ', len(Q1_train))
print("The length of the training set is:  ", len(train_Q1))
print("The length of the validation set is: ", len(val_Q1))


# create data generator

batch_size = 2
res1, res2 = next(data_generator(train_Q1, train_Q2, batch_size))
print("First questions  : ",'\n', res1, '\n')
print("Second questions : ",'\n', res2)

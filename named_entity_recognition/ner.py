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

def data_generator(batch_size, x, y, pad, shuffle=False, verbose=False):
    '''
      Input: 
        batch_size - integer describing the batch size
        x - list containing sentences where words are represented as integers
        y - list containing tags associated with the sentences
        shuffle - Shuffle the data order
        pad - an integer representing a pad character
        verbose - Print information during runtime
      Output:
        a tuple containing 2 elements:
        X - np.ndarray of dim (batch_size, max_len) of padded sentences
        Y - np.ndarray of dim (batch_size, max_len) of tags associated with the sentences in X
    '''
    
    # count the number of lines in data_lines
    num_lines = len(x)
    
    # create an array with the indexes of data_lines that can be shuffled
    lines_index = [*range(num_lines)]
    
    # shuffle the indexes if shuffle is set to True
    if shuffle:
        rnd.shuffle(lines_index)
    
    index = 0 # tracks current location in x, y
    while True:
        buffer_x = [0] * batch_size # Temporal array to store the raw x data for this batch
        buffer_y = [0] * batch_size # Temporal array to store the raw y data for this batch
                
  
        
        # Copy into the temporal buffers the sentences in x[index] 
        # along with their corresponding labels y[index]
        # Find maximum length of sentences in x[index] for this batch. 
        # Reset the index if we reach the end of the data set, and shuffle the indexes if needed.
        max_len = 0 
        for i in range(batch_size):
             # if the index is greater than or equal to the number of lines in x
            if index >= num_lines:
                # then reset the index to 0
                index = 0
                # re-shuffle the indexes if shuffle is set to True
                if shuffle:
                    rnd.shuffle(lines_index)
            
            # The current position is obtained using `lines_index[index]`
            # Store the x value at the current position into the buffer_x
            buffer_x[i] =x[lines_index[index]]
            
            # Store the y value at the current position into the buffer_y
            buffer_y[i] = y[lines_index[index]]
            
            lenx = len(x[lines_index[index]]) #length of current x[]
            if lenx > max_len:
                max_len = lenx #max_len tracks longest x[]
            
            # increment index by one
            index += 1


        # create X,Y, NumPy arrays of size (batch_size, max_len) 'full' of pad value
        X = np.full((batch_size,max_len),pad)
        Y = np.full((batch_size,max_len),pad)

        # copy values from lists to NumPy arrays. Use the buffered values
        for i in range(batch_size):
            # get the example (sentence as a tensor)
            # in `buffer_x` at the `i` index
            x_i = buffer_x[i]
            
            # similarly, get the example's labels
            # in `buffer_y` at the `i` index
            y_i = buffer_y[i]
            
            # Walk through each word in x_i
            for j in range(len(x_i)):
                # store the word in x_i at position j into X
                X[i, j] = x_i[j]
                
                # store the label in y_i at position j into Y
                Y[i, j] = y_i[j]

        if verbose: print("index=", index)
        yield((X,Y))
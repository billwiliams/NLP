import math
import random
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')

nltk.data.path.append('../data/')

from .utils import preprocess_data,load_data,get_tokenized_data

data=load_data()

# Split into train and test sets
tokenized_data = get_tokenized_data(data)
random.seed(87)
random.shuffle(tokenized_data)

train_size = int(len(tokenized_data) * 0.8)
train_data = tokenized_data[0:train_size]
test_data = tokenized_data[train_size:]

minimum_freq = 2
train_data_processed, test_data_processed,\
 vocabulary = preprocess_data(train_data, test_data, minimum_freq)


def count_n_grams(data, n, start_token='<s>', end_token = '<e>'):
    """
    Count all n-grams in the data
    
    Args:
        data: List of lists of words
        n: number of words in a sequence
    
    Returns:
        A dictionary that maps a tuple of n-words to its frequency
    """
    
    # Initialize dictionary of n-grams and their counts
    n_grams = {}

    
    
    # Go through each sentence in the data
    for sentence in data: 
        
        # prepend start token n times, and  append the end token one time
        sentence = [start_token]*n +sentence+[end_token]
        
        # convert list to tuple
        # So that the sequence of words can be used as
        # a key in the dictionary
        sentence = tuple(sentence)
        
        
        # from index 0
        # to the last index where the end of the n-gram
        # is within the sentence.
        
        for i in range(len(sentence)-n +1): 

            # Get the n-gram from i to i+n
            n_gram = sentence[i:i+n]
            
            # check if the n-gram is in the dictionary
            if n_gram in n_grams: 
            
                # Increment the count for this n-gram
                n_grams[n_gram] += 1
            else:
                # Initialize this n-gram count to 1
                n_grams[n_gram] = 1
    
            
    return n_grams
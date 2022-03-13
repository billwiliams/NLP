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

def estimate_probability(word, previous_n_gram, 
                         n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """
    Estimate the probabilities of a next word using the n-gram counts with k-smoothing
    
    Args:
        word: next word
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary_size: number of words in the vocabulary
        k: positive constant, smoothing parameter
    
    Returns:
        A probability
    """
    # convert list to tuple to use it as a dictionary key
    previous_n_gram = tuple(previous_n_gram)
    
    
    # Set the denominator
    # If the previous n-gram exists in the dictionary of n-gram counts,
    # Get its count.  Otherwise set the count to zero
    # Use the dictionary that has counts for n-grams
    previous_n_gram_count = n_gram_counts.get(previous_n_gram,0)
            
    # Calculate the denominator using the count of the previous n gram
    # and apply k-smoothing
    denominator = n_gram_counts.get(previous_n_gram,0) + k * vocabulary_size

    # Define n plus 1 gram as the previous n-gram plus the current word as a tuple
    n_plus1_gram = previous_n_gram + (word,)
  
    # Set the count to the count in the dictionary,
    # otherwise 0 if not in the dictionary
    # use the dictionary that has counts for the n-gram plus current word    
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram,0)
            
    # Define the numerator use the count of the n-gram plus current word,
    # and apply smoothing
    numerator = n_plus1_gram_count +k
        
    # Calculate the probability as the numerator divided by denominator
    probability = numerator/denominator
    
  
    
    return probability

def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='<e>', unknown_token="<unk>",  k=1.0):
    """
    Estimate the probabilities of next words using the n-gram counts with k-smoothing
    
    Args:
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: positive constant, smoothing parameter
    
    Returns:
        A dictionary mapping from next words to the probability.
    """
    # convert list to tuple to use it as a dictionary key
    previous_n_gram = tuple(previous_n_gram)    
    
    # add <e> <unk> to the vocabulary
    # <s> is not needed since it should not appear as the next word
    vocabulary = vocabulary + [end_token, unknown_token]    
    vocabulary_size = len(vocabulary)    
    
    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram, 
                                           n_gram_counts, n_plus1_gram_counts, 
                                           vocabulary_size, k=k)
                
        probabilities[word] = probability

    return probabilities

def make_count_matrix(n_plus1_gram_counts, vocabulary):
    # add <e> <unk> to the vocabulary
    # <s> is omitted since it should not appear as the next word
    vocabulary = vocabulary + ["<e>", "<unk>"]
    
    # obtain unique n-grams
    n_grams = []
    for n_plus1_gram in n_plus1_gram_counts.keys():
        n_gram = n_plus1_gram[0:-1]        
        n_grams.append(n_gram)
    n_grams = list(set(n_grams))
    
    # mapping from n-gram to row
    row_index = {n_gram:i for i, n_gram in enumerate(n_grams)}    
    # mapping from next word to column
    col_index = {word:j for j, word in enumerate(vocabulary)}    
    
    nrow = len(n_grams)
    ncol = len(vocabulary)
    count_matrix = np.zeros((nrow, ncol))
    for n_plus1_gram, count in n_plus1_gram_counts.items():
        n_gram = n_plus1_gram[0:-1]
        word = n_plus1_gram[-1]
        if word not in vocabulary:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i, j] = count
    
    count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=vocabulary)
    return count_matrix
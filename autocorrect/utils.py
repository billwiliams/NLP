import numpy as np
import re
from collections import Counter

def process_file(file_name):
    """get the vocabulary by reading a file

    Args:
        file_name : path of the file to be read
    """
    vocab=[]

    with open(file_name,'r') as file:
        lines=file.read()
        for line in lines:
            line_lower_case=line.lower()
            line_words=re.findall('\w+', line_lower_case)
            for word in line_words.split():
                vocab.append(word.lower())
    
    return vocab

def get_count(vocab):
    """return word count for the words in the vocab

    Args:
        
        vocab : a set of words representing the corpus. 
    Output:
        word_count_dict: The wordcount dictionary where key is the word and value is its frequency.
    """
    word_count_dict=Counter(vocab)

    return word_count_dict

def get_probabilities(word_count):
    """ return a dictionary with the probably of word occurence in a corpus

    Args:
        
        word_count: The wordcount dictionary where key is the word and value is its frequency.
    Output:
        probs: A dictionary where keys are the words and the values are the probability that a word will occur. 
    """
    probs={}
    total=sum(word_count.values)

    for word,value in word_count.items():
        probs[word]=value/total
    
    return probs


    
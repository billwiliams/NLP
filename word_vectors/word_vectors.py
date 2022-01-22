import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# loading the data using using pandas
data = pd.read_csv('../data/capitals.txt', delimiter=' ')
data.columns = ['city1', 'country1', 'city2', 'country2']

# get the word embeddings
word_embeddings = pickle.load(open("../data/word_embeddings_subset.p", "rb"))
len(word_embeddings)  

def cosine_similarity(A,B):
    """ computes the cosine similarity of two vectors 

    Args:
        A : Vector A corresponding to the first word
        B : Vector corresponding to word B
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    """

    dot = np.dot(A,B)   
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)    
    cos = dot/(norma*normb)

    
    return cos
    

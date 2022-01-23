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
    
def get_country(city1, country1, city2, embeddings, cosine_similarity=cosine_similarity):
    """
    Input:
        city1: a string (the capital city of country1)
        country1: a string (the country of capital1)
        city2: a string (the capital city of country2)
        # CODE REVIEW COMMENT: Embedding incomplete code comment, should add "and values are their emmbeddings"
        embeddings: a dictionary where the keys are words and
    Output:
        countries: a dictionary with the most likely country and its similarity score
    """

    # store the city1, country 1, and city 2 in a set called group
    group = (city1,country1, city2)

    # get embeddings of city 1
    city1_emb = embeddings[city1]

    # get embedding of country 1
    country1_emb = embeddings[country1]

    # get embedding of city 2
    city2_emb = embeddings[city2]

    # get embedding of country 2 (it's a combination of the embeddings of country 1, city 1 and city 2)
    # Remember: King - Man + Woman = None
    vec = country1_emb-city1_emb +city2_emb

    # Initialize the similarity to -1 (it will be replaced by a similarities that are closer to +1)
    similarity = -1

    # initialize country to an empty string
    country = ''

    # loop through all words in the embeddings dictionary
    for word in embeddings.keys():

        # first check that the word is not already in the 'group'
        if word not in group:

            # get the word embedding
            word_emb = embeddings[word]

            # calculate cosine similarity between embedding of country 2 and the word in the embeddings dictionary
            cur_similarity = cosine_similarity(vec,word_emb)

            # if the cosine similarity is more similar than the previously best similarity...
            if cur_similarity > similarity:

                # update the similarity to the new, better similarity
                similarity = cur_similarity

                # store the country as a tuple, which contains the word and the similarity
                country = word,similarity


    return country
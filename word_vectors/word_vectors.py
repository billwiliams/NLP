import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# loading the data using using pandas
data = pd.read_csv('../data/capitals.txt', delimiter=' ')
data.columns = ['city1', 'country1', 'city2', 'country2']

# get the word embeddings
word_embeddings = pickle.load(open("../data/word_embeddings_subset.p", "rb"))
len(word_embeddings)  # there should be 243 words that will be used in this assignment

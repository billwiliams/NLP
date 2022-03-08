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
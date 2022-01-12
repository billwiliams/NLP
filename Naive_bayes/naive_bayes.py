import numpy as np
from LR.utils import process_tweet,build_features,prepare_data

# get the tweets data 
train_x,train_y,test_x,test_y= prepare_data()

# build a frequency dictionary using the training data
freqs=build_features(train_x,train_y)




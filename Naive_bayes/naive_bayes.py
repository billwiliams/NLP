import numpy as np
from LR.utils import process_tweet,build_features,prepare_data

# get the tweets data 
train_x,train_y,test_x,test_y= prepare_data()

# build a frequency dictionary using the training data
freqs=build_features(train_x,train_y)
print(freqs[0])

def naive_bayes(freqs,train_x,train_y):
    """Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels correponding to the tweets (0,1)
    Output:
        logprior: the log prior.
        loglikelihood: the log likelihood of you Naive bayes equation.
    """
    loglikelihood={}
    logprior=0
    
    # words in the training set 
    vocab=[word for word,label in freqs]

    # number of unique words in the training data
    V= len(set(vocab))

    # Calculating the total number of positive and negative words in the dictionary 
    N_pos=N_neg=0

    for (word,label) in freqs:
        pair=(word,label)
        if label==1:
            N_pos+=freqs[pair]
        else:
            N_neg=freqs[pair]
    # Number of Documents (D) in the training data
    D=len(train_y)

    # Number of positive and negative Documents
    D_pos=len([y for y in train_y if y==1])
    D_neg=D-D_pos # subtract positive documents from total documents

    # compute the logprior
    logprior=np.log(D_pos)-np.log(D_neg)

    # get the positive frequency of a word

    # Complute loglikelihood of each word
    for word in vocab:

        # get the positive and negative frequencies
        freq_pos=freqs[(word,1)]
        freq_neg=freqs[(word,0)]

        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos+1)/(N_pos+V)
        p_w_neg = (freq_neg+1)/(N_neg+V)

        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos/p_w_neg)
    
    return logprior,loglikelihood
    




        






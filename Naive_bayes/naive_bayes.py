import sys
sys.path.append('../LR/')

import numpy as np
from utils import process_tweet,build_features,prepare_data

# get the tweets data 
train_x,train_y,test_x,test_y= prepare_data()

# build a frequency dictionary using the training data
freqs=build_features(train_x,train_y)

def lookup(freqs,word,sentiment):
    """ returns the number of occurencies of the word,sentiment in the freqs dictionary

    Input:
        freqs: dictionary from (word, label) to how often the word appears
        word: specific processed word in the tweet
        sentiment: labels correponding to the word (0,1)
    Output:
        count: the number of occurencies of the word in the data if none it returns 0
    
    """

    return freqs.get((word,sentiment),0)

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
            N_neg+=freqs[pair]
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
        freq_pos=lookup(freqs,word,1)
        freq_neg=lookup(freqs,word,0)

        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos+1)/(N_pos+V)
        p_w_neg = (freq_neg+1)/(N_neg+V)

        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos/p_w_neg)
    
    return logprior,loglikelihood



def naive_bayes_predict(tweet,logprior,loglikelihood):
    """Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

    """
    # process the tweet
    processed_tweet=process_tweet(tweet)

    # initialize probability to zero
    p = 0.0

    # add the logprior
    p += logprior

    for word in processed_tweet:
        # add loglikelihood
        p+=loglikelihood.get(word,0)

    return p

#Training Naive Bayes classifier
logprior, loglikelihood = naive_bayes(freqs, train_x, train_y)
print(logprior)
print(len(loglikelihood))

#Testing Naive Bayes
my_tweet = 'She smiled.'
p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
print('The expected output is', p)

## checking performance on the test set

def predict_on_test_set(test_x,test_y,logprior,loglikelihood):
    """check performance on the test set

    Args:
        test_x : testing features
        test_y : testing labels
        logprior:the log prior.
        loglikelihood: the log likelihood of you Naive bayes equation.
    """
    y_hat=[]
    for tweet in test_x:
        pred=naive_bayes_predict(tweet,logprior,loglikelihood)
        if pred>0:
            y_hat.append(1.0)
        else:
            y_hat.append(0.0)
    
    accuracy=(sum(y_hat==test_y.flatten())/len(y_hat))*100
    print(accuracy)
    
    
    print(f"The accuracy of Naive Bayes on test set is {accuracy:.2f} \n")
    return accuracy
   
    
    
# Check prediction on the test set
predict_on_test_set(test_x,test_y,logprior,loglikelihood)

# Error Analysis
# for tweet,label in zip(test_x,test_y):
    
#     pred=naive_bayes_predict(tweet,logprior,loglikelihood)
#     # print misclassified tweets
#     if pred>0 and label==0.0:
#         print(f"{tweet} \n predicted 1 \n correct label {label}")
#     if pred<0 and label==1.0:
#         print(f"{tweet} \n predicted 0 \n correct label {label}")


# Checking for more positive words
# we can use the loglikelihood and set a  threshold or get the ration of the psotive and negative of the words

def get_pos_neg_ratio(freqs):
    """get the positive negative ratio of the words in the freqs dictionary

    Args:
        freqs: dictionary containing words and their postive negative count
    """
    pos_neg_ratio={}
    for word,label in freqs:

        # positive count
        positive_count=freqs.get((word,1),0)

        # negative count
        negative_count=freqs.get((word,0),0)

        # add the values to the dictionary

        # compute ratio and add it to dict
        ratio=(positive_count+1)/(negative_count+1)
        pos_neg_ratio[word]=[word,positive_count,negative_count,ratio]

    return pos_neg_ratio

def get_words_by_threshold(pos_neg_ratio,threshold=0.5):
    """Obtain words that meet a certain threshold based on the ratio

    Args:
        pos_neg_ratio : dcitionary containing the words and their rations
        threshold (float, optional): threshold. Defaults to 0.5.
    """
    for word in pos_neg_ratio:
        ratio=pos_neg_ratio[word][3]
        if threshold>1:
            if ratio>=threshold:
                print(f"{pos_neg_ratio[word][0]} ratio {pos_neg_ratio[word][3]}")
        else:
            if ratio<=threshold:
                print(f"{pos_neg_ratio[word][0]} ratio {pos_neg_ratio[word][3]}")

pos_neg_ratio=get_pos_neg_ratio(freqs)

# Print very negative words
get_words_by_threshold(pos_neg_ratio,0.05)  

# print very positive words
get_words_by_threshold(pos_neg_ratio,8.0)   

        
            
            
        
           












        






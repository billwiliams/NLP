import numpy as np 
import nltk
from nltk.corpus import twitter_samples
import re                                  # library for regular expression operations
import string                              # for string operations

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings
nltk.data.path.append('../data/')
# dowload twitter sentiment data
nltk.download('twitter_samples',download_dir="../data/")

# download stopwords
nltk.download('stopwords',download_dir="../data/")



def process_tweet(tweet):
    """
    Remove twitter handles, urls, stopwords
    Tokenize the string
    Perform stemming on the word

    """
    # remove handles form the tweet
    tweet2= re.sub(r'@([A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)','',tweet)
    # remove old style RT from tweet
    tweet2=re.sub(r'^RT[\s]+', '', tweet2)


    # remove hyperlinks from the tweet
    tweet2= re.sub(r'https?://[^\s\n\r]+','',tweet2)
    
    # remove hashtags from the tweet
    tweet2= re.sub(r'#','', tweet2)

    tweet_tokens = tokenize(tweet2)

    # Remove stopwords

    #Import the english stop words list from NLTK
    stopwords_english = stopwords.words('english') 

    # Remove stopwords
    tweet_clean = remove_stopwords(tweet_tokens, stopwords_english)

    # perform stemming

    tweet_stemmed = stem(tweet_clean)
    
    

    return tweet_stemmed

def stem(tweet):
    """"
    Return stems of the words in the tweet
    """
    # Stemming


    tweet_stemmed=[]

    stemmer=PorterStemmer()

    for word in tweet:
        stem_word=stemmer.stem(word)
        tweet_stemmed.append(stem_word)

    return tweet_stemmed

def remove_stopwords(tweet_tokens, stopwords_english):
    """
    Remove stopwords
    """
    tweet_clean=[]# clean tweet i.e. without stopwords

    for word in tweet_tokens:
        if word not in stopwords_english and word not in string.punctuation:
            tweet_clean.append(word)
    return tweet_clean

def tokenize(tweet):
    """
    Tokenize the tweet
    """
    # Tokenize the string and lowercase it
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    
    tweet_tokens=tokenizer.tokenize(tweet)
    
    return tweet_tokens

tweet=["This is is good @kim, @n, @123T_ #winning"]
print(process_tweet(tweet[0]))




def build_features(tweets,sentiments):
    """
    Build frequencies.
    Input:
        tweets: a list of tweets
        sentiments: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    
    """
    freqs={}
    for sentiment,tweet in zip(sentiments,tweets):

        for word in process_tweet(tweet):
            
            
            if (word,sentiment) in freqs:
                freqs[(word,sentiment)]+=1
            else:
                freqs[(word,sentiment)]=1
    return freqs
    
print(build_features(tweet,[1]))


def prepare_data():
    """
    prepares the training and testing data from the twitter samples data
    """
    # select the set of positive and negative tweets
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')
    # split the data into two pieces, one for training and one for testing (validation set) 
    test_pos = all_positive_tweets[4000:]
    train_pos = all_positive_tweets[:4000]
    test_neg = all_negative_tweets[4000:]
    train_neg = all_negative_tweets[:4000]

    train_x = train_pos + train_neg 
    test_x = test_pos + test_neg

    train_y=np.append(np.ones((len(train_pos),1)),np.zeros((len(train_neg),1)),axis=0)
    test_y=np.append(np.ones((len(test_pos),1)),np.zeros((len(test_neg),1)),axis=0)
    
    # check the sizes of the training and testing data
    print("training data shape is {}",train_y.shape)
    print("training data shape is {}",test_y.shape)
   

    return train_x,train_y,test_x,test_y





   
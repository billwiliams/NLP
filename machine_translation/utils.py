import numpy as np
import nltk
from nltk.corpus import twitter_samples
import re                                  # library for regular expression operations
import string                              # for string operations

from nltk.corpus import stopwords          # module for stop words that come with NLTK

from nltk.tokenize import TweetTokenizer   # module for tokenizing strings
nltk.data.path.append('../data/')
# dowload twitter sentiment data
nltk.download('twitter_samples',download_dir="../data/")

# download stopwords
nltk.download('stopwords',download_dir="../data/")


def load_dict(file):
    _dict={}
    with open(file) as f:
        for line in f:
            (key,val)=line.split()
            _dict[key]=val
    return _dict

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

def process_tweet(tweet):
    """
    Remove twitter handles, urls, stopwords
    Tokenize the string
    Perform stemming on the word

    """
    # remove handles form the tweet
    tweet2= re.sub(r'@\w+','',tweet)

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
    clean_tweet = remove_stopwords(tweet_tokens, stopwords_english)

    return clean_tweet

def remove_stopwords(tweet_tokens, stopwords_english):
    """
    Remove stopwords
    """
    clean_tweet=[]# clean tweet i.e. without stopwords

    for word in tweet_tokens:
        if word not in stopwords_english and word not in string.punctuation:
            clean_tweet.append(word)
    return clean_tweet

def tokenize(tweet):
    """
    Tokenize the tweet
    """
    # Tokenize the string and lowercase it
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    
    tweet_tokens=tokenizer.tokenize(tweet)
    
    
    return tweet_tokens
    
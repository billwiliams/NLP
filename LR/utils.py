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

    # Tokenize the string and lowercase it
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    
    tweet_tokens=tokenizer.tokenize(tweet2)

    # Remove stopwords

    #Import the english stop words list from NLTK
    stopwords_english = stopwords.words('english') 

    tweet_clean=[]# clean tweet i.e. without stopwords

    for word in tweet_tokens:
        if word not in stopwords_english and word not in string.punctuation:
            tweet_clean.append(word)

    

    # Stemming


    tweet_stemmed=[]

    stemmer=PorterStemmer()

    for word in tweet_clean:
        stem_word=stemmer.stem(word)
        tweet_stemmed.append(stem_word)
    
    

    return tweet_stemmed


print(process_tweet("This is is good @kim, @n, @123T_ #winning"))
def build_features():
    pass


   
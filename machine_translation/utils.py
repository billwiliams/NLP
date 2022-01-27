import numpy as np

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
    pass
    
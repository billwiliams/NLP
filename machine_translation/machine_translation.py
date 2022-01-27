import numpy as np
import pickle


# load both engilish and french subsets embeddings
en_embeddings_subset = pickle.load(open("../data/en_embeddings.p", "rb"))
fr_embeddings_subset = pickle.load(open("../data/fr_embeddings.p", "rb"))

def compute_loss(X, Y, R):
    '''
    Inputs: 
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
    Outputs:
        L: a matrix of dimension (m,n) - the value of the loss function for given X, Y and R.
    '''
   
    # m is the number of rows in X
    m = len(X)
        
    # diff is XR - Y    
    diff = np.matmul(X,R) -Y

    # diff_squared is the element-wise square of the difference    
    diff_squared = np.square(diff)

    # sum_diff_squared is the sum of the squared elements
    sum_diff_squared = np.sum(diff_squared)

    # loss i the sum_diff_squard divided by the number of examples (m)
    loss = sum_diff_squared/m
    
    return loss
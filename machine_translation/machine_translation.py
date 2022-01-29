import numpy as np
import pickle
from utils import load_dict,process_tweet,cosine_similarity


# load both engilish and french subsets embeddings
en_embeddings_subset = pickle.load(open("../data/en_embeddings.p", "rb"))
fr_embeddings_subset = pickle.load(open("../data/fr_embeddings.p", "rb"))

en_fr_train=load_dict("../data/en-fr.train.txt")
en_fr_test=load_dict("../data/en-fr.test.txt")

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


def get_matrices(en_fr, french_vecs, english_vecs):
    """
    Input:
        en_fr: English to French dictionary
        french_vecs: French words to their corresponding word embeddings.
        english_vecs: English words to their corresponding word embeddings.
    Output: 
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the French embeddings.
        R: the projection matrix that minimizes the F norm ||X R -Y||^2.
    """


    # X_l and Y_l are lists of the english and french word embeddings
    X_l = list()
    Y_l = list()
    en_fr_pairs=list()

    # get the english words (the keys in the dictionary) and store in a set()
    english_set = set(english_vecs.keys())
   

    # get the french words (keys in the dictionary) and store in a set()
    french_set = set(french_vecs.keys())

    # store the french words that are part of the english-french dictionary (these are the values of the dictionary)
    french_words = set(en_fr.values())

    # loop through all english, french word pairs in the english french dictionary
    for en_word, fr_word in en_fr.items():

        # check that the french word has an embedding and that the english word has an embedding
        if fr_word in french_set and en_word in english_set:

            # get the english embedding
            en_vec = english_vecs[en_word]

            # get the french embedding
            fr_vec = french_vecs[fr_word]

            # add the english embedding to the list
            X_l.append(en_vec)

            # add the french embedding to the list
            Y_l.append(fr_vec)
            en_fr_pairs.append((en_word,fr_word))

    # stack the vectors of X_l into a matrix X
    X = np.stack(X_l)

    # stack the vectors of Y_l into a matrix Y
    Y = np.stack(Y_l)

    return X, Y,en_fr_pairs

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

def compute_gradient(X, Y, R):
    '''
    Inputs: 
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
    Outputs:
        g: a scalar value - gradient of the loss function L for given X, Y and R.
    '''
   
    # m is the number of rows in X
    m = len(X)

    # gradient is X^T(XR - Y) * 2/m    
    gradient = np.matmul(X.T,((np.matmul(X,R)-Y))) *2/m
    
    
    return gradient

def align_embeddings(X, Y, train_steps=100, learning_rate=0.0003, verbose=True, compute_loss=compute_loss, compute_gradient=compute_gradient):
    '''
    Inputs:
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        train_steps: positive int - describes how many steps will gradient descent algorithm do.
        learning_rate: positive float - describes how big steps will  gradient descent algorithm do.
    Outputs:
        R: a matrix of dimension (n,n) - the projection matrix that minimizes the F norm ||X R -Y||^2
    '''
    np.random.seed(129)

    # the number of columns in X is the number of dimensions for a word vector (e.g. 300)
    # R is a square matrix with length equal to the number of dimensions in th  word embedding
    R = np.random.rand(X.shape[1], X.shape[1])

    for i in range(train_steps):
        if verbose and i % 25 == 0:
            print(f"loss at iteration {i} is: {compute_loss(X, Y, R):.4f}")
       
        # use the function that you defined to compute the gradient
        gradient = compute_gradient(X,Y,R)

        # update R by subtracting the learning rate times gradient
        
        R -= learning_rate*gradient
    
    return R

def nearest_neighbor(v, candidates, k=1, cosine_similarity=cosine_similarity):
    """
    Input:
      - v, the vector you are going find the nearest neighbor for
      - candidates: a set of vectors where we will find the neighbors
      - k: top k nearest neighbors to find
    Output:
      - k_idx: the indices of the top k closest vectors in sorted form
    """
    
    similarity_l = []

    # for each candidate vector...
    for row in candidates:
        # get the cosine similarity
        cos_similarity = cosine_similarity(v,row)

        # append the similarity to the list
        similarity_l.append(cos_similarity)

    # sort the similarity list and get the indices of the sorted list    
    sorted_ids = np.argsort(similarity_l)  
    
    # Reverse the order of the sorted_ids array
    sorted_ids = sorted_ids[::-1]
    
    # get the indices of the k most similar candidate vectors
    k_idx = sorted_ids[:k]
    
    return k_idx

def test_vocabulary(X, Y, R,en_fr_pairs, nearest_neighbor=nearest_neighbor):
    '''
    Input:
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the French embeddings.
        R: the transform matrix which translates word embeddings from
        English to French word vector space.
    Output:
        accuracy: for the English to French capitals
    '''

    # The prediction is X times R
    # ALTERNATIVE SOLUTION COMMENT:
    # pred = None
    pred = np.matmul(X,R)

    # initialize the number correct to zero
    num_correct = 0
    
    # initialize non correct pair predictions
    non_correct_pred=list()

    # loop through each row in pred (each transformed embedding)
    for i in range(len(pred)):
        # get the index of the nearest neighbor of pred at row 'i'; also pass in the candidates in Y
        pred_idx = nearest_neighbor(pred[i],Y)

        # if the index of the nearest neighbor equals the row of i... \
        if pred_idx == i:
            # increment the number correct by 1.
            num_correct += 1
        else:
            non_correct_pred.append(en_fr_pairs[i])

    # accuracy is the number correct divided by the number of rows in 'pred' (also number of rows in X)
    accuracy = num_correct/len(pred)


    return accuracy,non_correct_pred

# getting the training set:
X_train, Y_train,en_fr_train_pairs = get_matrices(
    en_fr_train, fr_embeddings_subset, en_embeddings_subset)

# finding R transformation matrix using align embeddings fucntion to minimize loss
R_train = align_embeddings(X_train, Y_train, train_steps=400, learning_rate=0.8)

X_val, Y_val,en_fr_test_pairs = get_matrices(en_fr_test, fr_embeddings_subset, en_embeddings_subset)

acc, wrong_preds = test_vocabulary(X_val, Y_val, R_train,en_fr_test_pairs)  # this might take a minute or two
print(f"accuracy on test set is {acc:.3f}")
print(f"some wrong predictions included following pairs{wrong_preds[:5]}")

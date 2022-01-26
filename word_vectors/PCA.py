import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn
import pickle

# Implementation of PCA algorithm for dimensionality reduction to enable 
# view the word vectors relationships from the document corpus



def pca(X, n_components=2):
    """
    Input:
        X: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output:
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """

    # mean center the data
    X_demeaned =(X- np.mean(X,axis=0))

    # calculate the covariance matrix
    covariance_matrix = np.cov(X_demeaned,rowvar=False)


    # calculate eigenvectors & eigenvalues of the covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix)
    

    # sort eigenvalue in increasing order (get the indices from the sort)
    idx_sorted = eigen_vals.argsort()
    
    
    # reverse the order so that it's from highest to lowest.
    idx_sorted_decreasing = idx_sorted[: : -1]

    

    

    # sort the eigen values by idx_sorted_decreasing
    eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]

    # sort eigenvectors using the idx_sorted_decreasing indices
    eigen_vecs_sorted = eigen_vecs[:,idx_sorted_decreasing]
    print(eigen_vecs_sorted.shape)

    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    eigen_vecs_subset = eigen_vecs_sorted[:,:n_components]
    
   
    

    # transform the data by multiplying the transpose of the eigenvectors with the transpose of the de-meaned data
    # Then take the transpose of that product.
    X_reduced =(np.dot(eigen_vecs_subset.T, X_demeaned.T)).T

    

    return X_reduced


def get_vectors(embeddings,words):
    """returns embedding vector of given words

    Args:
        words : a list of words
        embeddings : a dictionary contains words as keys and corresponding embeddings

    Returns:
        matrix: embedding vectors of given words
    """
    X=[]
    for i,word in enumerate(words):
        en_vec=embeddings.get(word,0)
        X.append(en_vec)
    return np.stack(X)

# Testing PCA
np.random.seed(1)
X = np.random.rand(3, 10)
X_reduced = pca(X, n_components=2)
print("original matrix was " + str(X.shape) + " and it became:")
print(X_reduced)


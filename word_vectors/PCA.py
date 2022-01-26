import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn

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

# Testing your function
np.random.seed(1)
X = np.random.rand(3, 10)
X_reduced = compute_pca(X, n_components=2)
print("Your original matrix was " + str(X.shape) + " and it became:")
print(X_reduced)

words = ['oil', 'gas', 'happy', 'sad', 'city', 'town',
         'village', 'country', 'continent', 'petroleum', 'joyful']

# given a list of words and the embeddings, it returns a matrix with all the embeddings
X = get_vectors(word_embeddings, words)

print('You have 11 words each of 300 dimensions thus X.shape is:', X.shape)
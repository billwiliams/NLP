import numpy as np
from utils import process_tweet,build_features,prepare_data


# get training data 

train_x,train_y,test_x,test_y=prepare_data()


freqs= build_features(train_x,train_y)


def extract_features(tweet,freqs,process_tweet=process_tweet):
    """
    Given a list of tweets, extract the features and store them in a matrix. 
    The first feature is the number of positive words in a tweet.
    The second feature is the number of negative words in a tweet.

    Input: 
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    """
    X=np.ones((1,3))

    #set bias term to 1
    X[0,0]=1
    
    # process the tweet
    words= process_tweet(tweet)

    for word in words:

        # increment positive count
        if (word,1) in freqs:
            X[0,1] += freqs[(word,1)]
        
        # increment negative count
        if (word,0) in freqs:
            X[0,2] += freqs[(word,0)]
    assert(X.shape == (1, 3))
    return X





def sigmoid(z):
    """
    compute sigmoid
    Input:
        z: float value
        
    Output:
        sigmoid of z
    """

    return 1/(1+np.exp(-z))

def gradient_descent(x,y,theta,alpha,num_iters):
    """perform gradient descent while updating weights 

    Args:
        x : matrix of m,n=1 features
        y : corresponding labels of training features x
        alpha: learning rate
        theta : weight vector of the dimension (n+1,1)
        num_iters (int): number of iterations to train model
    Output:
        J: final cost
        theta: final weight vector
    """
    # number of rows in matrix x
    m = x.shape[0]
    
    for i in range(0, num_iters):
        
        # get z, the dot product of x and theta
        z = np.dot(x,theta)
        
        
        # get the sigmoid of z
        h = sigmoid(z)
        
        # calculate the cost function
        J = (-1/m)*(np.dot(y.T,np.log(h))+np.dot((1-y).T , np.log(1-h)))

        # update the weights theta
        theta = theta -alpha/m *(np.dot(x.T,(h-y)))
        
    
    J = float(J)
    return J, theta
        

def predict(tweet, freqs,theta):
    """[summary]

    Args:
        tweet : a string 
        freqs  dictionary containing the word frequencey count
        theta : weights vector after training
    """
    x= extract_features(tweet,freqs)

    pred=sigmoid(np.dot(x,theta))

    return pred


def accuracy(y_hat,y):
    """Computes accuracy given the predictions and true labels

    Args:
        y_hat : the predictions from the model
        y : True labels from the data
    """

    return ((np.sum(y_hat==y))/len(y))*100




# ## Training the model
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

#training labels corresponding to X
Y = train_y

# Apply gradient descent

J, theta = gradient_descent(X, Y, np.zeros((3, 1)), 1e-9, 1500)


print(f"The cost after training is {J:.8f}.")

print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")


# Testing accuracy on the test set
y_hat=[] 

for i in range(len(test_x)):
    prediction=predict(test_x[i],freqs,theta)
    if prediction>0.5:
        y_hat.append(1.0)
    else:
        y_hat.append(0.0)

print(f"Accuracy on the test set is {accuracy(y_hat,test_y.flatten()):.2f}.")

#Error analysis for misclassified tweets

for x,y in zip(test_x,test_y):
    pred=predict(x,freqs,theta)
    if pred>0.5:
        pred=1.0
    else:
        pred=0.0

    if y!=pred:
        print("misclassified tweet\n")
        print(x+"\n")
        print(f"predicted label {pred}  correct label {y}")





# Twitter sentiment Analysis using logistic regression

## **Description**

implementing logistic regression for sentiment analysis on tweets. Given a tweet decide if it has a positive sentiment or a negative one. Below are the implemented:

* Extracting features for logistic regression given some text
* Logistic regression from scratch
* Apply logistic regression on a natural language processing task
* Testing using your logistic regression
* Error analysis

### Files

* LR- contains the logistic regression implementation, training and testing 
* utils- contains helper functions 
* tests- contains tests  

### Executing the code

Run LR from the command line `python LR.py` 


### Import some helper functions from the  utils.py file

* process_tweet: cleans the text, tokenizes it into separate words, removes stopwords, and converts words to stems.
* build_freqs: this counts how often a word in the 'corpus' (the entire set of tweets) was associated with a positive label '1' or a negative label '0', then builds the 'freqs' dictionary, where each key is the (word,label) tuple, and the value is the count of its frequency within the corpus of tweets.

### Prepare the data

* The `twitter_samples` contains subsets of five thousand positive_tweets, five thousand negative_tweets, and the full set of 10,000 tweets.  
    *Five thousand positive tweets and five thousand negative tweets are used

* Train test split: 20% will be in the test set, and 80% in the training set.

### Logistic regression: regression and a sigmoid

Logistic regression takes a regular linear regression, and applies a sigmoid to the output of the linear regression.

Regression:
$$z = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + ... \theta_N x_N$$
Note that the $\theta$ values are "weights".

Logistic regression
$$ h(z) = \frac{1}{1+\exp^{-z}}$$
$$z = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + ... \theta_N x_N$$
 'z' are the 'logits'.

### Cost function and Gradient

The cost function used for logistic regression is the average of the log loss across all training examples:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^m y^{(i)}\log (h(z(\theta)^{(i)})) + (1-y^{(i)})\log (1-h(z(\theta)^{(i)}))\tag{5} $$

* $m$ is the number of training examples
* $y^{(i)}$ is the actual label of training example 'i'.
* $h(z^{(i)})$ is the model's prediction for the training example 'i'.

The loss function for a single training example is
$$ Loss = -1 \times \left( y^{(i)}\log (h(z(\theta)^{(i)})) + (1-y^{(i)})\log (1-h(z(\theta)^{(i)})) \right)$$

* All the $h$ values are between 0 and 1, so the logs will be negative. That is the reason for the factor of -1 applied to the sum of the two loss terms.
* Note that when the model predicts 1 ($h(z(\theta)) = 1$) and the label 'y' is also 1, the loss for that training example is 0.
* Similarly, when the model predicts 0 ($h(z(\theta)) = 0$) and the actual label is also 0, the loss for that training example is 0.
* However, when the model prediction is close to 1 ($h(z(\theta)) = 0.9999$) and the label is 0, the second term of the log loss becomes a large negative number, which is then multiplied by the overall factor of -1 to convert it to a positive loss value. $-1 \times (1 - 0) \times log(1 - 0.9999) \approx 9.2$ The closer the model prediction gets to 1, the larger the loss.
* Likewise, if the model predicts close to 0 ($h(z) = 0.0001$) but the actual label is 1, the first term in the loss function becomes a large number: $-1 \times log(0.0001) \approx 9.2$.  The closer the prediction is to zero, the larger the loss.

#### Update the weights

To update weight vectors $\theta$, we apply gradient descent to iteratively improve the model's predictions.  
The gradient of the cost function $J$ with respect to one of the weights $\theta_j$ is:

$$\nabla_{\theta_j}J(\theta) = \frac{1}{m} \sum_{i=1}^m(h^{(i)}-y^{(i)})x^{(i)}_j \tag{5}$$

* 'i' is the index across all 'm' training examples.
* 'j' is the index of the weight $\theta_j$, so $x^{(i)}_j$ is the feature associated with weight $\theta_j$

* To update the weight $\theta_j$, we adjust it by subtracting a fraction of the gradient determined by $\alpha$:
$$\theta_j = \theta_j - \alpha \times \nabla_{\theta_j}J(\theta) $$
* The learning rate $\alpha$ is a value that we choose to control how big a single update will be.

## Extracting the features

* Given a list of tweets, we extract the features and store them in a matrix. We extract two features.
  * The first feature is the number of positive words in a tweet.
    * The second feature is the number of negative words in a tweet.
* Then train  logistic regression classifier on these features.
* Test the classifier on a validation set.

# Testing  logistic regression

To test  logistic regression function on some new input that the model has not seen before.

#### process

Predict whether a tweet is positive or negative.

* Given a tweet, process it, then extract the features.
* Apply the model's learned weights on the features to get the logits.
* Apply the sigmoid to the logits to get the prediction (a value between 0 and 1).

$$y_{pred} = sigmoid(\mathbf{x} \cdot \theta)$$

## Checking performance using the test set
After training the model using the training set above, we check how the model might perform on real, unseen data, by testing it against the test set.

#### process 

* Given the test data and the weights of the trained model, calculate the accuracy of your logistic regression model. 
* Make predictions on each tweet in the test set using `predict` function.
* If the prediction is > 0.5, set the model's classification 'y_hat' to 1, otherwise set the model's classification 'y_hat' to 0.
* A prediction is accurate when the y_hat equals the test_y.  Sum up all the instances when they are equal and divide by number of labels(m).

# Error Analysis

check what kind of tweets the model misclassifies.

examples

`@MarkBreech Not sure it would be good thing 4 my bottom daring 2 say 2 Miss B but Im gonna be so stubborn on mouth soaping ! #NotHavingit :p

predicted label 0.0  correct label [1.]

#### **misclassified tweet**

I'm playing Brain Dots : ) #BrainDots
http://t.co/UGQzOx0huu

predicted label 0.0  correct label [1.]

#### **misclassified tweet**

I'm playing Brain Dots : ) #BrainDots http://t.co/aOKldo3GMj http://t.co/xWCM9qyRG5

predicted label 0.0  correct label [1.]

#### **misclassified tweet**

I'm playing Brain Dots : ) #BrainDots http://t.co/R2JBO8iNww http://t.co/ow5BBwdEMY

predicted label 0.0  correct label [1.]

#### **misclassified tweet**

off to the park to get some sunlight : )

predicted label 0.0  correct label [1.]

#### **misclassified tweet**

@msarosh Uff Itna Miss karhy thy ap :p

predicted label 0.0  correct label [1.]

#### **misclassified tweet**

@phenomyoutube u probs had more fun with david than me : (

predicted label 1.0  correct label [0.]

#### **misclassified tweet**

pats jay : (

predicted label 1.0  correct label [0.]

#### **misclassified tweet**

Sr. Financial Analyst - Expedia, Inc.: (#Bellevue, WA) http://t.co/ktknMhvwCI #Finance #ExpediaJobs #Job #Jobs #Hiring

predicted label 1.0  correct label [0.]`

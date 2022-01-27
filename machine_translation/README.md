# Naive Machine Translation 
Translate English  into French words using word embeddings and vector space models. 


#### Look at the data

* en_embeddings_subset: the key is an English word, and the vaule is a
300 dimensional array, which is the embedding for that word.
```
'the': array([ 0.08007812,  0.10498047,  0.04980469,  0.0534668 , -0.06738281, ....
```

* fr_embeddings_subset: the key is a French word, and the vaule is a 300
dimensional array, which is the embedding for that word.
```
'la': array([-6.18250e-03, -9.43867e-04, -8.82648e-03,  3.24623e-02,...
```
#### Looking at the English French dictionary

* `en_fr_train` is a dictionary where the key is the English word and the value
is the French translation of that English word.
```
{'the': 'la',
 'and': 'et',
 'was': 'était',
 'for': 'pour',
```

* `en_fr_test` is similar to `en_fr_train`, but is a test set. 

##  Translation as linear transformation of embeddings

Given dictionaries of English and French word embeddings we create a transformation matrix `R`
* Given an English word embedding, $\mathbf{e}$,  multiply $\mathbf{eR}$ to get a new word embedding $\mathbf{f}$.
    * Both $\mathbf{e}$ and $\mathbf{f}$ are [row vectors](https://en.wikipedia.org/wiki/Row_and_column_vectors).
* Then compute the nearest neighbors to `f` in the french embeddings and recommend the word that is most similar to the transformed word embedding.

### Describing translation as the minimization problem

Find a matrix `R` that minimizes the following equation. 

$$\arg \min _{\mathbf{R}}\| \mathbf{X R} - \mathbf{Y}\|_{F}\tag{1} $$

### Frobenius norm

The Frobenius norm of a matrix $A$ (assuming it is of dimension $m,n$) is defined as the square root of the sum of the absolute squares of its elements:

$$\|\mathbf{A}\|_{F} \equiv \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n}\left|a_{i j}\right|^{2}}\tag{2}$$

### Actual loss function
The Frobenius norm loss:

$$\| \mathbf{XR} - \mathbf{Y}\|_{F}$$

is often replaced by it's squared value divided by $m$:

$$ \frac{1}{m} \|  \mathbf{X R} - \mathbf{Y} \|_{F}^{2}$$

where $m$ is the number of examples (rows in $\mathbf{X}$).

* The same R is found when using this loss function versus the original Frobenius norm.
* The reason for taking the square is that it's easier to compute the gradient of the squared Frobenius.
* The reason for dividing by $m$ is that we're more interested in the average loss per embedding than the  loss for the entire training set.
    * The loss for all training set increases with more words (training examples),
    so taking the average helps us to track the average loss regardless of the size of the training set.

### Implementation

####  1: Computing the loss
* The loss function will be squared Frobenoius norm of the difference between
matrix and its approximation, divided by the number of training examples $m$.
* Its formula is:
$$ L(X, Y, R)=\frac{1}{m}\sum_{i=1}^{m} \sum_{j=1}^{n}\left( a_{i j} \right)^{2}$$

where $a_{i j}$ is value in $i$th row and $j$th column of the matrix $\mathbf{XR}-\mathbf{Y}$.

###  2: Computing the gradient of loss in respect to transform matrix R

* Calculating the gradient of the loss with respect to transform matrix `R`.
* The gradient is a matrix that encodes how much a small change in `R`
affect the change in the loss function.
* The gradient gives  the direction in which we should decrease `R`
to minimize the loss.
* $m$ is the number of training examples (number of rows in $X$).
* The formula for the gradient of the loss function $𝐿(𝑋,𝑌,𝑅)$ is:

$$\frac{d}{dR}𝐿(𝑋,𝑌,𝑅)=\frac{d}{dR}\Big(\frac{1}{m}\| X R -Y\|_{F}^{2}\Big) = \frac{2}{m}X^{T} (X R - Y)$$

### 3: Finding the optimal R with gradient descent algorithm

#### Gradient descent

[Gradient descent](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html) is an iterative algorithm which is used in searching for the optimum of the function. 
* The gradient of the loss with respect to the matrix encodes how much a tiny change in some coordinate of that matrix affect the change of loss function.
* Gradient descent uses that information to iteratively change matrix `R` until  a point where the loss is minimized. 

Pseudocode:
1. Calculate gradient $g$ of the loss with respect to the matrix $R$.
2. Update $R$ with the formula:
$$R_{\text{new}}= R_{\text{old}}-\alpha g$$

Where $\alpha$ is the learning rate, which is a scalar.
#### Learning rate

* The learning rate or "step size" $\alpha$ is a coefficient which decides how much we want to change $R$ in each step.
* If we change $R$ too much, we could skip the optimum by taking too large of a step.
* If we make only small changes to $R$, we will need many steps to reach the optimum.
* Learning rate $\alpha$ is used to control those changes.
* Values of $\alpha$ are chosen depending on the problem
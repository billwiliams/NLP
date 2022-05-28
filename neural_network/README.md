# Sentiment with Deep Neural Networks
we explore sentiment analysis using deep neural networks. 

using  Logistic regression and Naive Bayes for sentiment analysis fails on examples like this:

<center> <span style='color:blue'> <b>This movie was almost good.</b> </span> </center>

The  models would have predicted a positive sentiment for that review. However, that sentence has a negative sentiment and indicates that the movie was not good. To solve those kinds of misclassifications, we use  deep neural networks to identify sentiment in text. 

Objectives:

- Understand how you can build/design a model using layers
- Train a model using a training loop
- Use a binary cross-entropy loss function
- Compute the accuracy of DNN model
- Predict using  own input

This model follows a similar structure to the ones  previously implemented i
- The only thing that changes is the model architecture, the inputs, and the outputs. 

We introduce  the Google library `trax` that we use for building and training models.


- Trax source code can be found on Github: [Trax](https://github.com/google/trax)
- The Trax code also uses the JAX library: [JAX](https://jax.readthedocs.io/en/latest/index.html)

##  Loading in the data

Import the data set.  
- Details of process_tweet function are available in utils.py file

## Building the vocabulary

Now we build the vocabulary.
- Map each word in each tweet to an integer (an "index"). 

- We assign an index to everyword by iterating over your training set.

The vocabulary will also include some special tokens
- `__PAD__`: padding
- `</e>`: end of line
- `__UNK__`: a token representing any word that is not in the vocabulary.

**Output**

Total words in vocab are 9088

- {'__PAD__': 0,
 '__</e>__': 1,
 '__UNK__': 2,
 'followfriday': 3,
 'top': 4,
 'engag': 5,
 'member': 6,
 'commun': 7,
 'week': 8,
 ':)': 9, }

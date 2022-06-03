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

##  Converting a tweet to a tensor

We convert each tweet to a tensor (a list of unique integer IDs representing the processed tweet).
- Note, the returned data type is a **regular Python `list()`**
    
- For words in the tweet that are not in the vocabulary,  we set them to the unique ID for the token `__UNK__`.

##### Example
Input a tweet:
```CPP
'@happypuppy, is Maria happy?'
```

The tweet_to_tensor  converts the tweet into a list of tokens (including only relevant words)
```CPP
['maria', 'happi']
```

Then it converts each word into its unique integer

```CPP
[2, 56]
```
- Notice that the word "maria" is not in the vocabulary, so it is assigned the unique integer associated with the `__UNK__` token, because it is considered "unknown."

## Creating a batch generator

Most of the time in Natural Language Processing, and AI in general  batches are used  when training our data sets. 
- If instead of training with batches of examples, we were to train a model with one example at a time, it would take a very long time to train the model. 
- we  build a data generator that takes in the positive/negative tweets and returns a batch of training examples. It returns the model inputs, the targets (positive or negative labels) and the weight for each target (ex: this allows us to can treat some examples as more important to get right than others, but commonly this will all be 1.0). 

We include it in a for loop

```CPP
for batch_inputs, batch_targets, batch_example_weights in data_generator:
    ...
```

We can also get a single batch like this:

```CPP
batch_inputs, batch_targets, batch_example_weights = next(data_generator)
```
The generator returns the next batch each time it's called. 
- This generator returns the data in a format (tensors) that you could directly use in your model.
- It returns a triplet: the inputs, targets, and loss weights:
    - Inputs is a tensor that contains the batch of tweets we put into the model.
    - Targets is the corresponding batch of labels that we train to generate.
    - Loss weights here are just 1s with same shape as targets. 

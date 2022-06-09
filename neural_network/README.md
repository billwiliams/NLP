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


# Defining classes

We write your own library of layers similar to the one used in Trax and also in Keras and PyTorch. 

Our framework is based on the following `Layer` class from utils.py.

```CPP
class Layer(object):
    """ Base class for layers.
    """
      
    # Constructor
    def __init__(self):
        # set weights to None
        self.weights = None

    # The forward propagation should be implemented
    # by subclasses of this Layer class
    def forward(self, x):
        raise NotImplementedError

    # This function initializes the weights
    # based on the input signature and random key,
    # This is implemented by subclasses of this Layer class
    def init_weights_and_state(self, input_signature, random_key):
        pass

    # This initializes and returns the weights, do not override.
    def init(self, input_signature, random_key):
        self.init_weights_and_state(input_signature, random_key)
        return self.weights
 
    # __call__ allows an object of this class
    # to be called like it's a function.
    def __call__(self, x):
        # When this layer object is called, 
        # it calls its forward propagation function
        return self.forward(x)


```
## Dense class 



Implementing the forward function of the Dense class. 
- The forward function multiplies the input to the layer (`x`) by the weight matrix (`W`)

$$\mathrm{forward}(\mathbf{x},\mathbf{W}) = \mathbf{xW} $$

- We use `numpy.dot` to perform the matrix multiplication.

Note that for more efficient code execution, we use the trax version of `math`, which includes a trax version of `numpy` and also `random`.

We implement the weight initializer `new_weights` function
- Weights are initialized with a random key.
- The second parameter is a tuple for the desired shape of the weights (num_rows, num_cols)
- The num of rows for weights should equal the number of columns in x, because for forward propagation, you will multiply x times weights.

Using `trax.fastmath.random.normal(key, shape, dtype=tf.float32)` to generate random values for the weight matrix. The key difference between this function
and the standard `numpy` randomness is the explicit use of random keys, which
need to be passed. 
- `key` is generated by calling `random.get_prng(seed=)` and passing in a number for the `seed`.
- `shape` is a tuple with the desired shape of the weight matrix.
    - The number of rows in the weight matrix should equal the number of columns in the variable `x`.  Since `x` may have 2 dimensions if it represents a single training example (row, col), or three dimensions (batch_size, row, col), we get the last dimension from the tuple that holds the dimensions of x.
    - The number of columns in the weight matrix is the number of units chosen for that dense layer.  Look at the `__init__` function to see which variable stores the number of units.
- `dtype` is the data type of the values in the generated matrix; we keep the default of `tf.float32`. 

setting the standard deviation of the random values to 0.1
- The values generated have a mean of 0 and standard deviation of 1.
- we Set the default standard deviation `stdev` to be 0.1 by multiplying the standard deviation to each of the values in the weight matrix.

## Model

Now we implement a classifier using neural networks.



For the model implementation, we use the Trax `layers` module, imported as `tl`.
 Trax layers are very similar to the ones implemented above,
but in addition to trainable weights  they also have a non-trainable state.
State is used in layers like batch normalization and for inference

A look at the code of the Trax Dense layer 
- [tl.Dense](https://github.com/google/trax/blob/master/trax/layers/core.py#L29): Trax Dense layer implementation

One other important layer that we use a lot is one that allows to execute one layer after another in sequence.
- [tl.Serial](https://github.com/google/trax/blob/master/trax/layers/combinators.py#L26): Combinator that applies layers serially.  
    - You can pass in the layers as arguments to `Serial`, separated by commas. 
    - For example: `tl.Serial(tl.Embeddings(...), tl.Mean(...), tl.Dense(...), tl.LogSoftmax(...))`

# Training

To train a model on a task, Trax defines an abstraction [`trax.supervised.training.TrainTask`](https://trax-ml.readthedocs.io/en/latest/trax.supervised.html#trax.supervised.training.TrainTask) which packages the train data, loss and optimizer (among other things) together into an object.

Similarly to evaluate a model, Trax defines an abstraction [`trax.supervised.training.EvalTask`](https://trax-ml.readthedocs.io/en/latest/trax.supervised.html#trax.supervised.training.EvalTask) which packages the eval data and metrics (among other things) into another object.

The final piece tying things together is the [`trax.supervised.training.Loop`](https://trax-ml.readthedocs.io/en/latest/trax.supervised.html#trax.supervised.training.Loop) abstraction that is a very simple and flexible way to put everything together and train the model, all the while evaluating it and saving checkpoints.
Using `Loop`  saves a lot of code compared to always writing the training loop by hand.

## Making a prediction

Now that we have trained a model, we can access it as `training_loop.model` object. We  use `training_loop.eval_model` 

#  Evaluation  


## Computing the accuracy on a batch

We write a function that evaluates the  model on the validation set and returns the accuracy. 
- `preds` contains the predictions.
    - Its dimensions are `(batch_size, output_dim)`.  `output_dim` is two in this case.  Column 0 contains the probability that the tweet belongs to class 0 (negative sentiment). Column 1 contains probability that it belongs to class 1 (positive sentiment).
    - If the probability in column 1 is greater than the probability in column 0, then interpret this as the model's prediction that the example has label 1 (positive sentiment).  
    - Otherwise, if the probabilities are equal or the probability in column 0 is higher, the model's prediction is 0 (negative sentiment).
- `y` contains the actual labels.
- `y_weights` contains the weights to give to predictions.

## Testing the  model on Validation Data

We  test the model's prediction accuracy on validation data. 

The program  takes in a data generator and the model. 
- The generator allows  to get batches of data. We can use it with a `for` loop:

```
for batch in iterator: 
   # do something with that batch
```

`batch` has dimensions `(batch size, 2)`. 
- Column 0 corresponds to the tweet as a tensor.
- Column 1 corresponds to its target (actual label, positive or negative sentiment).
- We feed the tweet into model and it will return the predictions for the batch. 
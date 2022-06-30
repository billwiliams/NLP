# Deep N-grams

Exploring Recurrent Neural Networks `RNN` by implementing a system to predict next set of characters using previous characters.
- We will be using the fundamentals of google's [trax](https://github.com/google/trax) package to implement the  deeplearning model. 

outcomes:-
- How to convert a line of text into a tensor
- Creating an iterator to feed data to the model
- Defining a GRU model using `trax`
- Training the model using `trax`
- Computing the accuracy of your model using the perplexity
- Predict using the model

### Overview

The task will be to predict the next set of characters using the previous characters. 

 - Many natural language tasks rely on using embeddings for predictions. 
- The model will convert each character to its embedding, run the embeddings through a Gated Recurrent Unit `GRU`, and run it through a linear layer to predict the next set of characters.

To predict the next character:
- We Use the softmax output and identify the word with the highest probability.
- The word with the highest probability is the prediction for the next word.

### Batch generator 

Most of the time in Natural Language Processing, and AI in general we use batches when training our data sets. We  build a data generator that takes in a text and returns a batch of text lines (lines are sentences).
- The generator converts text lines (sentences) into numpy arrays of integers padded by zeros so that all arrays have the same length, which is the length of the longest sentence in the entire data set.

The generator returns the data in a format that we can  directly use in the model when computing the feed-forward of the algorithm. This iterator returns a batch of lines and per token mask. The batch is a tuple of three parts: inputs, targets, mask. The inputs and targets are identical. The second column will be used to evaluate  predictions. Mask is 1 for non-padding tokens.

###  Repeating Batch generator 

The way the iterator is currently defined, it will keep providing batches forever.

Usually we want to cycle over the dataset multiple times during training (i.e. train for multiple *epochs*).

For small datasets we can use [`itertools.cycle`](https://docs.python.org/3.8/library/itertools.html#itertools.cycle) to achieve this easily.

#  Defining the GRU model

Now that we have the input and output tensors, we will go ahead and initialize the model. we will be implementing the `GRULM`, gated recurrent unit model. To implement this model, we will be using google's `trax` package. 
We will use the following packages when constructing the model: 


- `tl.Serial`: Combinator that applies layers serially (by function composition). [docs](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.combinators.Serial) / [source code](https://github.com/google/trax/blob/e65d51fe584b10c0fa0fccadc1e70b6330aac67e/trax/layers/combinators.py#L26)
    - we can pass in the layers as arguments to `Serial`, separated by commas. 
    - For example: `tl.Serial(tl.Embeddings(...), tl.Mean(...), tl.Dense(...), tl.LogSoftmax(...))`

___
- `tl.ShiftRight`: Allows the model to go right in the feed forward. [docs](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.attention.ShiftRight) / [source code](https://github.com/google/trax/blob/e65d51fe584b10c0fa0fccadc1e70b6330aac67e/trax/layers/attention.py#L560)
    - `ShiftRight(n_shifts=1, mode='train')` layer to shift the tensor to the right n_shift times
    
___

- `tl.Embedding`: Initializes the embedding. In this case it is the size of the vocabulary by the dimension of the model. [docs](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.core.Embedding) / [source code](https://github.com/google/trax/blob/e65d51fe584b10c0fa0fccadc1e70b6330aac67e/trax/layers/core.py#L130) 
    - `tl.Embedding(vocab_size, d_feature)`.
    - `vocab_size` is the number of unique words in the given vocabulary.
    - `d_feature` is the number of elements in the word embedding (some choices for a word embedding size range from 150 to 300, for example).
___

- `tl.GRU`: `Trax` GRU layer. [docs](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.rnn.GRU) / [source code](https://github.com/google/trax/blob/e65d51fe584b10c0fa0fccadc1e70b6330aac67e/trax/layers/rnn.py#L154)
    - `GRU(n_units)` Builds a traditional GRU of n_cells with dense internal transformations.
    - `GRU` paper: https://arxiv.org/abs/1412.3555
___

- `tl.Dense`: A dense layer. [docs](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.core.Dense) / [source code](https://github.com/google/trax/blob/e65d51fe584b10c0fa0fccadc1e70b6330aac67e/trax/layers/core.py#L34)
    - `tl.Dense(n_units)`: The parameter `n_units` is the number of units chosen for this dense layer.
___

- `tl.LogSoftmax`: Log of the output probabilities. [docs](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.core.LogSoftmax) / [source code](https://github.com/google/trax/blob/e65d51fe584b10c0fa0fccadc1e70b6330aac67e/trax/layers/core.py#L644)
    - Here, you don't need to set any parameters for `LogSoftMax()`.
___
#  Training

We define the cost function, the optimizer, and decide whether we will be training it on a `gpu` or `cpu`. We also have to feed in a built model. we use the `TrainTask` and `EvalTask` abstractions 

To train a model on a task, Trax defines an abstraction `trax.supervised.training.TrainTask` which packages the train data, loss and optimizer (among other things) together into an object.

Similarly to evaluate a model, Trax defines an abstraction `trax.supervised.training.EvalTask` which packages the eval data and metrics (among other things) into another object.

The final piece tying things together is the `trax.supervised.training.Loop` abstraction that is a very simple and flexible way to put everything together and train the model, all the while evaluating it and saving checkpoints.

An `epoch` is traditionally defined as one pass through the dataset.

Since the dataset is divided in `batches` we need several `steps` (gradient evaluations) in order to complete an `epoch`. So, one `epoch` corresponds to the number of examples in a `batch` times the number of `steps`. In short, in each `epoch` we go over all the dataset. 

The `max_length` variable defines the maximum length of lines to be used in training the data, lines longer than that length are discarded. 


###  Training the model

We write a function that takes in the model and trains it. To train the model we have to decide how many times we want to iterate over the entire data set. 

 Here is a list of things we do:

- We Create a `trax.supervised.trainer.TrainTask` object, this encapsulates the aspects of the dataset and the problem at hand:
    - labeled_data = the labeled data that we want to *train* on.
    - loss_fn = [tl.CrossEntropyLoss()](https://trax-ml.readthedocs.io/en/latest/trax.layers.html?highlight=CrossEntropyLoss#trax.layers.metrics.CrossEntropyLoss)
    - optimizer = [trax.optimizers.Adam()](https://trax-ml.readthedocs.io/en/latest/trax.optimizers.html?highlight=Adam#trax.optimizers.adam.Adam) with learning rate = 0.0005

- We Create a `trax.supervised.trainer.EvalTask` object, this encapsulates aspects of evaluating the model:
    - labeled_data = the labeled data that we want to *evaluate* on.
    - metrics = [tl.CrossEntropyLoss()](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.metrics.CrossEntropyLoss) and [tl.Accuracy()](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.metrics.Accuracy)
    - How frequently we want to evaluate and checkpoint the model.

- We Create a `trax.supervised.trainer.Loop` object, this encapsulates the following:
    - The previously created `TrainTask` and `EvalTask` objects.
    - the training model = [GRULM](#ex03)
    - optionally the evaluation model, if different from the training model. NOTE: in presence of Dropout etc we usually want the evaluation model to behave slightly differently than the training model.

# Evaluation  

### Evaluating using the deep nets

 To evaluate language models, we usually use perplexity which is a measure of how well a probability model predicts a sample. Note that perplexity is defined as: 

$$P(W) = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i| w_1,...,w_{n-1})}}$$

As an implementation hack, weou would usually take the log of that formula (to enable us to use the log probabilities we get as output of our `RNN`, convert exponents to products, and products into sums which makes computations less complicated and computationally more efficient). We should also take care of the padding, since we do not want to include the padding when calculating the perplexity (because we do not want to have a perplexity measure artificially good).


$$\log P(W) = {\log\left(\sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i| w_1,...,w_{n-1})}}\right)}$$$$ = \log\left(\left(\prod_{i=1}^{N} \frac{1}{P(w_i| w_1,...,w_{n-1})}\right)^{\frac{1}{N}}\right)$$
$$ = \log\left(\left({\prod_{i=1}^{N}{P(w_i| w_1,...,w_{n-1})}}\right)^{-\frac{1}{N}}\right)$$$$ = -\frac{1}{N}{\log\left({\prod_{i=1}^{N}{P(w_i| w_1,...,w_{n-1})}}\right)} $$$$ = -\frac{1}{N}{{\sum_{i=1}^{N}{\log P(w_i| w_1,...,w_{n-1})}}} $$

# Generating the language with the model

To use our own language model to generate new sentences  we need to make draws from a Gumble distribution.

The Gumbel Probability Density Function (PDF) is defined as: 

$$ f(z) = {1\over{\beta}}e^{(-z+e^{(-z)})} $$

where: $$ z = {(x - \mu)\over{\beta}}$$

The maximum value, which is what we choose as the prediction in the last step of a Recursive Neural Network `RNN` we are using for text generation, in a sample of a random variable following an exponential distribution approaches the Gumbel distribution when the sample increases asymptotically. For that reason, the Gumbel distribution is used to sample from a categorical distribution.

###  <span style="color:blue"> On statistical methods </span>

Using statistical method  will not give  results that are as good. The model will not be able to encode information seen previously in the data set and as a result, the perplexity will increase.  Furthermore, statistical ngram models take up too much space and memory. As a result, it will be inefficient and too slow. Conversely, with deepnets, We can get a better perplexity

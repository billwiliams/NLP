# Deep N-grams

Eexploring Recurrent Neural Networks `RNN` by implementing a system to predict next set of characters using previous characters.
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


T

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

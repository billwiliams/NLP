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

# Language Models: Auto-Complete

We build an auto-complete system.  Auto-complete systems examples are as below
- When you google something, you often have suggestions to help you complete your search. 
- When you are writing an email, you get suggestions telling you possible endings to your sentence.  

A key building block for an auto-complete system is a language model.
A language model assigns the probability to a sequence of words, in a way that more "likely" sequences receive higher scores.  For example, 
>"I have a pen" 
is expected to have a higher probability than 
>"I am a pen"
since the first one seems to be a more natural sentence in the real world.

We can take advantage of this probability calculation to develop an auto-complete system.  
Suppose the user typed 
>"I eat scrambled"
Then we can find a word `x`  such that "I eat scrambled x" receives the highest probability.  If x = "eggs", the sentence would be
>"I eat scrambled eggs"

While a variety of language models have been developed,  **N-grams**, is  a simple but powerful method for language modeling.
- N-grams are also used in machine translation and speech recognition. 


Here are the steps of taken:

1. Load and preprocess data
    - Load and tokenize data.
    - Split the sentences into train and test sets.
    - Replace words with a low frequency by an unknown marker `<unk>`.
1. Develop N-gram based language models
    - Compute the count of n-grams from a given data set.
    - Estimate the conditional probability of a next word with k-smoothing.
1. Evaluate the N-gram models by computing the perplexity score.
1. Use  model to suggest an upcoming word given your sentence. 

### Pre-process the data

Preprocessing data with the following steps:

1. Spliting data into sentences using "\n" as the delimiter.
1. Spliting each sentence into tokens. 
1. Assigning sentences into train or test sets.
1. Finding  tokens that appear at least N times in the training data.
1. Replacing  tokens that appear less than N times by `<unk>`

### Handling 'Out of Vocabulary' words

If the model is performing autocomplete, but encounters a word that it never saw during training, it won't have an input word to help it determine the next word to suggest. The model will not be able to predict the next word because there are no counts for the current word. 
- This 'new' word is called an 'unknown word', or <b>out of vocabulary (OOV)</b> words.
- The percentage of unknown words in the test set is called the <b> OOV </b> rate. 

To handle unknown words during prediction, we use a special token to represent all unknown words 'unk'. 
-




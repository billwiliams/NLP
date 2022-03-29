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

##  Developing n-gram based language models

We  develop the n-grams language model.
- Assuming the probability of the next word depends only on the previous n-gram.
- The previous n-gram is the series of the previous 'n' words.

The conditional probability for the word at position 't' in the sentence, given that the words preceding it are $w_{t-n}\cdots w_{t-2}, w_{t-1}$ is:

$$ P(w_t | w_{t-n}\dots w_{t-1} ) \tag{1}$$

We can estimate this probability  by counting the occurrences of these series of words in the training data.
- The probability can be estimated as a ratio, where
- The numerator is the number of times word 't' appears after words t-n through t-1 appear in the training data.
- The denominator is the number of times word t-n through t-1 appears in the training data.


$$ \hat{P}(w_t | w_{t-n} \dots w_{t-1}) = \frac{C(w_{t-n}\dots w_{t-1}, w_t)}{C(w_{t-n}\dots w_{t-1})} \tag{2} $$


- The function $C(\cdots)$ denotes the number of occurence of the given sequence. 
- $\hat{P}$ means the estimation of $P$. 
- Notice that denominator of the equation (2) is the number of occurence of the previous $n$ words, and the numerator is the same sequence followed by the word $w_t$.

Later, we can modify the equation (2) by adding k-smoothing, which avoids errors when any counts are zero.

The equation (2) tells us that to estimate probabilities based on n-grams, you need the counts of n-grams (for denominator) and (n+1)-grams (for numerator).


##  Perplexity

We generate the perplexity score to evaluate the model on the test set. 
- We also use back-off when needed. 
- Perplexity is used as an evaluation metric of the language model. 
- To calculate the perplexity score of the test set on an n-gram model, use: 

$$ PP(W) =\sqrt[N]{ \prod_{t=n+1}^N \frac{1}{P(w_t | w_{t-n} \cdots w_{t-1})} } \tag{4}$$

- where $N$ is the length of the sentence.
- $n$ is the number of words in the n-gram (e.g. 2 for a bigram).
- In math, the numbering starts at one and not zero.

In code, array indexing starts at zero, so the code will use ranges for $t$ according to this formula:

$$ PP(W) =\sqrt[N]{ \prod_{t=n}^{N-1} \frac{1}{P(w_t | w_{t-n} \cdots w_{t-1})} } \tag{4.1}$$

The higher the probabilities are, the lower the perplexity will be. 
- The more the n-grams tell us about the sentence, the lower the perplexity score will be. 


### Suggesting multiple words using n-grams of varying length

we suggest multiple words using n-grams of varying lengths (unigrams, bigrams, trigrams, 4-grams...6-grams).



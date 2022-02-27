# Parts-of-Speech Tagging (POS)

Implements the process of assigning a part-of-speech tag (Noun, Verb, Adjective...) to each word in an input text.

a name='0'></a>
##  Data Sources
We use two tagged data sets collected from the **Wall Street Journal (WSJ)**. 

[Here](http://relearn.be/2015/training-common-sense/sources/software/pattern-2.6-critical-fork/docs/html/mbsp-tags.html) is an example 'tag-set' or Part of Speech designation describing the two or three letter tag and their meaning. 
- One data set (**WSJ-2_21.pos**) is used for **training**.
- The other (**WSJ-24.pos**) for **testing**. 
- The tagged training data has been preprocessed to form a vocabulary (**hmm_vocab.txt**). 
- The words in the vocabulary are words from the training set that were used two or more times. 
- The vocabulary is augmented with a set of 'unknown word tokens', described below. 

The training is used to create the emission, transmission and tag counts. 

The test set (WSJ-24.pos) is read in to create `y`. 
- This contains both the test text and the true tag. 
- The test set has also been preprocessed to remove the tags to form **test_words.txt**. 
- This is read in and further processed to identify the end of sentences and handle words not in the vocabulary using functions provided in **utils_pos.py**. 
- This forms the list `prep`, the preprocessed text used to test our  POS taggers.

A POS tagger will necessarily encounter words that are not in its datasets. 
- To improve accuracy, these words are further analyzed during preprocessing to extract available hints as to their appropriate tag. 
- For example, the suffix 'ize' is a hint that the word is a verb, as in 'final-ize' or 'character-ize'. 
- A set of unknown-tokens, such as '--unk-verb--' or '--unk-noun--' will replace the unknown words in both the training and test corpus and will appear in the emission, transmission and tag data structures.



<a name='1.1'></a>
##  Training

 Finding  the words that are not ambiguous. 
- For example, the word `is` is a verb and it is not ambiguous. 
- In the `WSJ` corpus, $86$% of the token are unambiguous (meaning they have only one tag) 
- About $14\%$ are ambiguous (meaning that they have more than one tag)

<img src = "images/pos.png" style="width:400px;height:250px;"/>
. 

#### Transition counts
- The first dictionary is the `transition_counts` dictionary which computes the number of times each tag happened next to another tag. 

This dictionary will is used to compute: 
$$P(t_i |t_{i-1}) \tag{1}$$

This is the probability of a tag at position $i$ given the tag at position $i-1$.

In order to compute equation 1, we create a `transition_counts` dictionary where 
- The keys are `(prev_tag, tag)`
- The values are the number of times those two tags appeared in that order. 

#### Emission counts

The second dictionary we compute is the `emission_counts` dictionary. This dictionary is used to compute:

$$P(w_i|t_i)\tag{2}$$

we use it to compute the probability of a word given its tag. 

In order  to compute equation 2, we  will create an `emission_counts` dictionary where 
- The keys are `(tag, word)` 
- The values are the number of times that pair showed up in the  training set. 

#### Tag counts

The last dictionary we compute is the `tag_counts` dictionary. 
- The key is the tag 
- The value is the number of times each tag appeared.

<a name='3'></a>
# Part 3: Viterbi Algorithm and Dynamic Programming

We implement the Viterbi algorithm which makes use of dynamic programming. Specifically, We use the two matrices, `A` and `B` to compute the Viterbi algorithm.  

* **Initialization** - In this part we initialize the `best_paths` and `best_probabilities` matrices that we will be populating in `feed_forward`.
* **Feed forward** - At each step, we calculate the probability of each path happening and the best paths up to that point. 
* **Feed backward**: This allows us to find the best path with the highest probabilities. 

<a name='3.1'></a>
## Part 3.1:  Initialization 

We start by initializing two matrices of the same dimension. 

- best_probs: Each cell contains the probability of going from one POS tag to a word in the corpus.

- best_paths: A matrix that helps you trace through the best possible path in the corpus. 

We initializes the `best_probs` and the `best_paths` matrix. 

Both matrices will be initialized to zero except for column zero of `best_probs`.  
- Column zero of `best_probs` is initialized with the assumption that the first word of the corpus was preceded by a start token ("--s--"). 
- This allows us to reference the **A** matrix for the transition probability

vocab[corpus[0]] refers to the first word of the corpus (the word at position 0 of the corpus). 
-**vocab** is a dictionary that returns the unique integer that refers to that particular word.

Conceptually, it looks like this:
$\textrm{best_probs}[s_{idx}, i] = \mathbf{A}[s_{idx}, i] \times \mathbf{B}[i, corpus[0] ]$


In order to avoid multiplying and storing small values on the computer, we'll take the log of the product, which becomes the sum of two logs:

$best\_probs[i,0] = log(A[s_{idx}, i]) + log(B[i, vocab[corpus[0]]$

Also, to avoid taking the log of 0 (which is defined as negative infinity), the code itself will just set $best\_probs[i,0] = float('-inf')$ when $A[s_{idx}, i] == 0$

the implementation to initialize $best\_probs$ is as follows:

$ \textrm{if}\ A[s_{idx}, i] <> 0 : best\_probs[i,0] = log(A[s_{idx}, i]) + log(B[i, vocab[corpus[0]]])$

$ \textrm{if}\ A[s_{idx}, i] == 0 : best\_probs[i,0] = float('-inf')$

## Viterbi Forward

We implement  the `viterbi_forward` by  populating  `best_probs` and `best_paths` matrices.
- Walk forward through the corpus.
- For each word, compute a probability for each possible tag.

this  includes the path up to that (word,tag) combination. 
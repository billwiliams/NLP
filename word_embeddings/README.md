# Word Embeddings 

Exploring a classic way of generating word embeddings or representations.
- implementing   the continuous bag of words (CBOW) model. 

We:

- Train word vectors from scratch.
- create batches of data.
- Understand how backpropagation works.
- Plot and visualize  learned word vectors.

#  The Continuous bag of words model

For the following sentence: 
>**'I am happy because I am learning'**. 

- In continuous bag of words (CBOW) modeling, we try to predict the center word given a few context words (the words around the center word).
- For example, if you were to choose a context half-size of say $C = 2$, then you would try to predict the word **happy** given the context that includes 2 words before and 2 words after the center word:

> $C$ words before: [I, am] 

> $C$ words after: [because, I] 

- In other words:

$$context = [I,am, because, I]$$
$$target = happy$$


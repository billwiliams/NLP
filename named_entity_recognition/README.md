# Named Entity Recognition (NER)

named entity recognition (NER) is a subtask of information extraction that locates and classifies named entities in a text. The named entities could be organizations, persons, locations, times, etc.

#  Exploring the data

We use a dataset from Kaggle original data consists of four columns: the sentence number, the word, the part of speech of the word, and the tags.  A few tags you might expect to see are: 

* geo: geographical entity
* org: organization
* per: person 
* gpe: geopolitical entity
* tim: time indicator
* art: artifact
* eve: event
* nat: natural phenomenon
* O: filler word

## Data generator

A generator is a function that behaves like an iterator. It returns the next item in a pre-defined sequence. 

In many AI applications it is  useful to have a data generator. 

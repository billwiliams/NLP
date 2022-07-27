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

# Building the model

We implement the model that will be able to determining the tags of sentences 

Concretely, the  inputs will be sentences represented as tensors that are fed to a model with:

* An Embedding layer,
* A LSTM layer
* A Dense layer
* A log softmax layer

# Training the Model 

We  need to create the data generators for training and validation data. It is important to mask padding in the loss weights of the data, which can be done using the `id_to_mask` argument of [`trax.data.inputs.add_loss_weights`](https://trax-ml.readthedocs.io/en/latest/trax.data.html?highlight=add_loss_weights#trax.data.inputs.add_loss_weights).

# Computing Accuracy

We evaluate the model  in the test set. Previously, we have seen the accuracy on the training set and the validation (noted as eval) set. We now evaluate on  test set. To get a good evaluation, we need to create a mask to avoid counting the padding tokens when computing the accuracy. 

# Testing with on own sentence

we can test on our own sentence like below sample
- "Peter Navarro, the White House director of trade and manufacturing policy of U.S, said in an interview on Sunday morning that the White House was working to prepare for the possibility of a second wave of the coronavirus in the fall, though he said it wouldn’t necessarily come"

Sample output

- Peter B-per
- Navarro, I-per
- White B-org
- House I-org
- Sunday B-tim
- morning I-tim
- White B-org
- House I-org
- coronavirus B-tim
- fall, B-tim

**Expected Results**

```
Peter B-per
Navarro, I-per
White B-org
House I-org
Sunday B-tim
morning I-tim
White B-org
House I-org
coronavirus B-tim
fall, B-tim
```


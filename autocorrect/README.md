#  Autocorrect

Implementing an auto-correct system that is very effective and useful.

<a name='0'></a>

##  Overview
Tasks involved in the autocorrect system are as follows: 

- Get a word count given a corpus
- Get a word probability in the corpus 
- Manipulate strings 
- Filter strings 
- Implement Minimum edit distance to compare strings and to help find the optimal path for the edits. 

Such systems work as follows. 
- For example, if you type in the word **"I am lerningg"**, the autocorrect feature will try to predict that you meant **learning**
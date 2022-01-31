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


####  Edit Distance

We implement models that correct words that are 1 and 2 edit distances away. 
- Two words are n edit distance away from each other when  n edits are needed to change one word into another. 

An edit could consist of one of the following options: 

- Delete (remove a letter): ‘hat’ => ‘at, ha, ht’
- Switch (swap 2 adjacent letters): ‘eta’ => ‘eat, tea,...’
- Replace (change 1 letter to another): ‘jat’ => ‘hat, rat, cat, mat, ...’
- Insert (add a letter): ‘te’ => ‘the, ten, ate, ...’

Above four methods are used implement an Auto-correct. 
- To do so, we need to compute probabilities that a certain word is correct given an input. 

This auto-correct  was first created by [Peter Norvig](https://en.wikipedia.org/wiki/Peter_Norvig) in 2007. 
- [original article](https://norvig.com/spell-correct.html) 

The goal of the model is to compute the following probability:

$$P(c|w) = \frac{P(w|c)\times P(c)}{P(w)} \tag{Eqn-1}$$

The equation above is [Bayes Rule](https://en.wikipedia.org/wiki/Bayes%27_theorem). 
- Equation 1 says that the probability of a word being correct $P(c|w) $is equal to the probability of having a certain word $w$, given that it is correct $P(w|c)$, multiplied by the probability of being correct in general $P(C)$ divided by the probability of that word $w$ appearing $P(w)$ in general.


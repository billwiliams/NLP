import numpy as np
from utils import get_count,process_file,get_probabilities
from string_manipulations import StringManipulation


def get_corrections(word, probs, vocab, n=2, verbose = False):
    '''
    Input: 
        word: a user entered string to check for suggestions
        probs: a dictionary that maps each word to its probability in the corpus
        vocab: a set containing all the vocabulary
        n: number of possible word corrections you want returned in the dictionary
    Output: 
        n_best: a list of tuples with the most probable n corrected words and their probabilities.
    '''
    
    suggestions = []
    n_best = []
    
    
    # create suggestions as described above 
    vocab_edits=[]
    one_edits=[]
    two_edits=[]
    if word in vocab:
        vocab_edits.append(word)
    one_edits_=StringManipulation.edit_one_letter(word)
    for word in one_edits_:
        if word in vocab:
            one_edits.append(word)
    two_edits_=StringManipulation.edit_two_letters(word)
    for word in two_edits_:
        if word in vocab:
            two_edits.append(word)
    
        
                
        
        
    suggestions = vocab_edits or one_edits or two_edits
                    
    
    probabilities=[]
    for word in suggestions:
        probabilities.append(probs.get(word,0))
    
    # Get all your best words and return the most probable top n_suggested words as n_best
    idx=np.argsort(probabilities)[::-1]
    print(idx)
    
    n_best = [(suggestions[i],probabilities[i]) for i in idx]
    n_best=n_best[:n]
    
    
    
    if verbose: print("entered word = ", word, "\nsuggestions = ", suggestions)

    return n_best


vocab=process_file("../data/autocorrect/shakespeare.txt")
word_count=get_count(vocab)
probs=get_probabilities(word_count)
word=input("Enter a Word\n")

n_best=get_corrections(word,probs,vocab)

print("Suggestions are \n")
for i, word_prob in enumerate(n_best):
    print(f" word {i}: {word_prob[0]}, probability {word_prob[1]:.6f}")
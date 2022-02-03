

class StringManipulation:
    """ Perform string manipulations
    
    """
    def __init__(self,word) -> None:
        self.word=word
    
    def delete_letter(self,verbose=False):
        """deletes a letter from the word
        Output:
            delete_l: a list of all possible strings obtained by deleting 1 character from word
        """
        delete_l = []
        split_l = []
        
        ### START CODE HERE ###
        split_l=[(self.word[:i],self.word[i:]) for i in range(len(self.word)+1)]
        delete_l=[L + R[1:] for L, R in split_l if R]
        
        ### END CODE HERE ###

        if verbose: print(f"input word {self.word}, \nsplit_l = {split_l}, \ndelete_l = {delete_l}")

        return  delete_l



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
        
      
        split_l=[(self.word[:i],self.word[i:]) for i in range(len(self.word)+1)]
        delete_l=[L + R[1:] for L, R in split_l if R]
        
        

        if verbose: print(f"input word {self.word}, \nsplit_l = {split_l}, \ndelete_l = {delete_l}")

        return  delete_l

    def switch_letter(self, verbose=False):
        '''
    
        Output:
            switches: a list of all possible strings with one adjacent charater switched
        ''' 
        
        switch_l = []
        split_l = []
        
        
        split_l=[(self.word[:i],self.word[i:]) for i in range(len(self.word)+1)]
        switch_l=list(set([L+R[:i] + R[i+1] + R[i] + R[i+2:]for (L,R) in split_l for i in range(len(R)-1) if R]))
        
        
        
        if verbose: print(f"Input word = {self.word} \nsplit_l = {split_l} \nswitch_l = {switch_l}") 
        
        return switch_l
    
    def replace_letter(self, verbose=False):
        '''
        
        Output:
            replaces: a list of all possible strings where we replaced one letter from the original word. 
        ''' 
        
        letters = 'abcdefghijklmnopqrstuvwxyz'
        
        replace_l = []
        split_l = []
        
        
        split_l=[(self.word[:i],self.word[i:]) for i in range(len(self.word)+1)]
        for i in range(len(self.word)):
            for c in letters:
                new_word=self.word[:i]+c+self.word[i+1:]
                if new_word!=self.word:
                    replace_l.append(new_word)
    #     replace_l=[c+R for L, R in split_l for c in letters if  len(R)>1 and len(R)<len(word)]
        replace_set=set(replace_l)
        
        

        
        # turn the set back into a list and sort it, for easier viewing
        replace_l = sorted(list(replace_set))
        
        if verbose: print(f"Input word = {self.word} \nsplit_l = {split_l} \nreplace_l {replace_l}")   
        
        return replace_l
    
    def insert_letter(self, verbose=False):
        '''
        Output:
            inserts: a set of all possible strings with one new letter inserted at every offset
        ''' 
        letters = 'abcdefghijklmnopqrstuvwxyz'
        insert_l = []
        split_l = []
        
        
        split_l=[(self.word[:i],self.word[i:]) for i in range(len(self.word)+1)]
        for i in range(len(self.word)+1):
            for c in letters:
                new_word=self.word[:i]+c+self.word[i:]
                insert_l.append(new_word)
            
        
        if verbose: print(f"Input word {self.word} \nsplit_l = {split_l} \ninsert_l = {insert_l}")
        
        return insert_l

SM=StringManipulation('ate')
print(SM.delete_letter())
print(SM.switch_letter())
print(SM.replace_letter())

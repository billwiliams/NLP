import numpy as np

def get_vocab(file_name):
    """get the vocabulary by reading a file

    Args:
        file_name : path of the file to be read
    """
    vocab=[]

    with open(file_name,'r') as file:
        lines=file.read()
        for line in lines:
            line_lower_case=line.lower()
            for word in line_lower_case.split():
                vocab.append(word.lower())
    
    return vocab
    
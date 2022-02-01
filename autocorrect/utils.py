import numpy as np
import re

def process_file(file_name):
    """get the vocabulary by reading a file

    Args:
        file_name : path of the file to be read
    """
    vocab=[]

    with open(file_name,'r') as file:
        lines=file.read()
        for line in lines:
            line_lower_case=line.lower()
            line_words=re.findall('\w+', line_lower_case)
            for word in line_words.split():
                vocab.append(word.lower())
    
    return vocab
    

def load_data(file_name="./data/en_US.twitter.txt"):
    with open(file_name, "r") as f:
        data = f.read()
    print("Data type:", type(data))
    print("Number of letters:", len(data))
    print("First 300 letters of the data")
    print("-------")
    # display(data[0:300])
    print("-------")

    print("Last 300 letters of the data")
    print("-------")
    # display(data[-300:])
    print("-------")
    return data

def split_to_sentences(data):
    """
    Split data by linebreak "\n"
    
    Args:
        data: str
    
    Returns:
        A list of sentences
    """

    sentences = data.split("\n")
   
    
    # Additional clearning (This part is already implemented)
    # - Remove leading and trailing spaces from each sentence
    # - Drop sentences if they are empty strings.
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]
    
    return sentences    


def tokenize_sentences(sentences):
    """
    Tokenize sentences into tokens (words)
    
    Args:
        sentences: List of strings
    
    Returns:
        List of lists of tokens
    """
    
    # Initialize the list of lists of tokenized sentences
    tokenized_sentences = []
   
    
    # Go through each sentence
    for sentence in sentences: # complete this line
        
        # Convert to lowercase letters
        sentence = sentence.lower()
        
        # Convert into a list of words
        tokenized = nltk.word_tokenize(sentence)
        
        # append the list of words to the list of lists
        tokenized_sentences.append(tokenized)
    
    
    
    return tokenized_sentences


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

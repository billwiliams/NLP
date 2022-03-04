
def load_data():
    with open("./data/en_US.twitter.txt", "r") as f:
        data = f.read()
    print("Data type:", type(data))
    print("Number of letters:", len(data))
    print("First 300 letters of the data")
    print("-------")
    display(data[0:300])
    print("-------")

    print("Last 300 letters of the data")
    print("-------")
    display(data[-300:])
    print("-------")
    return data
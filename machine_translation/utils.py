
def load_dict(file):
    _dict={}
    with open(file) as f:
        for line in f:
            (key,val)=line.split()
            _dict[key]=val
    return _dict

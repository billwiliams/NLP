import os
import nltk
import trax
from trax import layers as tl
from trax.supervised import training
from trax.fastmath import numpy as fastnp
import numpy as np
import pandas as pd
import random as rnd

import w4_unittest

nltk.download('punkt')

# set random seeds
rnd.seed(34)

data = pd.read_csv("data/questions.csv")
N=len(data)
print('Number of question pairs: ', N)
data.head()

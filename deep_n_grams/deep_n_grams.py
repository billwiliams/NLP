import os
import shutil
import trax
import trax.fastmath.numpy as np
import pickle
import numpy
import random as rnd
from trax import fastmath
from trax import layers as tl

dirname = 'data/'
filename = 'shakespeare_data.txt'
lines = [] # storing all the lines in a variable. 

counter = 0

with open(os.path.join(dirname, filename)) as files:
    for line in files:        
        # remove leading and trailing whitespace
        pure_line = line.strip()

        # if pure_line is not the empty string,
        if pure_line:
            # append it to the list
            lines.append(pure_line)


# go through each line
for i, line in enumerate(lines):
    # convert to all lowercase
    lines[i] = line.lower()

print(f"Number of lines: {n_lines}")
print(f"Sample line at position 0 {lines[0]}")
print(f"Sample line at position 999 {lines[999]}")

eval_lines = lines[-1000:] # Create a holdout validation set
lines = lines[:-1000] # Leave the rest for training

print(f"Number of lines for training: {len(lines)}")
print(f"Number of lines for validation: {len(eval_lines)}")
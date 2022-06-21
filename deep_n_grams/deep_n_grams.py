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

def line_to_tensor(line, EOS_int=1):
    """Turns a line of text into a tensor

    Args:
        line (str): A single line of text.
        EOS_int (int, optional): End-of-sentence integer. Defaults to 1.

    Returns:
        list: a list of integers (unicode values) for the characters in the `line`.
    """
    
    # Initialize the tensor as an empty list
    tensor = []
    
    
    # for each character:
    for c in line:
        
        # convert to unicode int
        c_int = ord(c)
        
        # append the unicode integer to the tensor list
        tensor.append(c_int)
    
    # include the end-of-sentence integer
    tensor.append(EOS_int)
    
    

    return tensor


def data_generator(batch_size, max_length, data_lines, line_to_tensor=line_to_tensor, shuffle=True):
    """Generator function that yields batches of data

    Args:
        batch_size (int): number of examples (in this case, sentences) per batch.
        max_length (int): maximum length of the output tensor.
        NOTE: max_length includes the end-of-sentence character that will be added
                to the tensor.  
                Keep in mind that the length of the tensor is always 1 + the length
                of the original line of characters.
        data_lines (list): list of the sentences to group into batches.
        line_to_tensor (function, optional): function that converts line to tensor. Defaults to line_to_tensor.
        shuffle (bool, optional): True if the generator should generate random batches of data. Defaults to True.

    Yields:
        tuple: two copies of the batch (jax.interpreters.xla.DeviceArray) and mask (jax.interpreters.xla.DeviceArray).
        NOTE: jax.interpreters.xla.DeviceArray is trax's version of numpy.ndarray
    """
    # initialize the index that points to the current position in the lines index array
    index = 0
    
    # initialize the list that will contain the current batch
    cur_batch = []
    
    # count the number of lines in data_lines
    num_lines = len(data_lines)
    
    # create an array with the indexes of data_lines that can be shuffled
    lines_index = [*range(num_lines)]
    
    # shuffle line indexes if shuffle is set to True
    if shuffle:
        rnd.shuffle(lines_index)
    
    
    while True:
        
        # if the index is greater than or equal to the number of lines in data_lines
        if index>=num_lines:
            # then reset the index to 0
            index = 0
            # shuffle line indexes if shuffle is set to True
            if shuffle:
                rnd.shuffle(lines_index) 
                            
        # get a line at the `lines_index[index]` position in data_lines
        line = data_lines[lines_index[index]]
        
        # if the length of the line is less than max_length
        if len(line)<max_length:
            # append the line to the current batch
            cur_batch.append(line)
            
        # increment the index by one
        index += 1
        
        # if the current batch is now equal to the desired batch size
        if len(cur_batch)==batch_size:
            
            batch = []
            mask = []
            
            # go through each line (li) in cur_batch
            for li in cur_batch:
                # convert the line (li) to a tensor of integers
                tensor = line_to_tensor(li)
                
                # Create a list of zeros to represent the padding
                # so that the tensor plus padding will have length `max_length`
                pad = [0] * (max_length-len(tensor))
                
                # combine the tensor plus pad
                tensor_pad = tensor+pad
                
                # append the padded tensor to the batch
                batch.append(tensor_pad)

                # A mask for this tensor_pad is 1 whereever tensor_pad is not
                # 0 and 0 whereever tensor_pad is 0, i.e. if tensor_pad is
                # [1, 2, 3, 0, 0, 0] then example_mask should be
                # [1, 1, 1, 0, 0, 0]
                example_mask = [1  if i!=0 else 0 for i in tensor_pad]
                mask.append(example_mask) # @ KEEPTHIS
               
            # convert the batch (data type list) to a numpy array
            batch_np_arr = np.array(batch)
            mask_np_arr = np.array(mask)
            
            
            
            # Yield two copies of the batch and mask.
            yield batch_np_arr, batch_np_arr, mask_np_arr
            
            # reset the current batch to an empty list
            cur_batch = []


# import itertools

# infinite_data_generator = itertools.cycle(
#     data_generator(batch_size=2, max_length=10, data_lines=tmp_lines))

def GRULM(vocab_size=256, d_model=512, n_layers=2, mode='train'):
    """Returns a GRU language model.

    Args:
        vocab_size (int, optional): Size of the vocabulary. Defaults to 256.
        d_model (int, optional): Depth of embedding (n_units in the GRU cell). Defaults to 512.
        n_layers (int, optional): Number of GRU layers. Defaults to 2.
        mode (str, optional): 'train', 'eval' or 'predict', predict mode is for fast inference. Defaults to "train".

    Returns:
        trax.layers.combinators.Serial: A GRU language model as a layer that maps from a tensor of tokens to activations over a vocab set.
    """
    
    model = tl.Serial( 
      tl.ShiftRight(mode=mode), # Stack the ShiftRight layer
      tl.Embedding(vocab_size, d_model), # Stack the embedding layer
      [tl.GRU(d_model) for i in range(n_layers)], # Stack GRU layers of d_model units keeping n_layer parameter in mind (use list comprehension syntax)
      tl.Dense(vocab_size), # Dense layer
      tl.LogSoftmax() # Log Softmax
    ) 
   
    return model

from trax.supervised import training


def train_model(model, data_generator, lines, eval_lines, batch_size=32, max_length=64, n_steps=1, output_dir='model/'): 
    """Function that trains the model

    Args:
        model (trax.layers.combinators.Serial): GRU model.
        data_generator (function): Data generator function.
        batch_size (int, optional): Number of lines per batch. Defaults to 32.
        max_length (int, optional): Maximum length allowed for a line to be processed. Defaults to 64.
        lines (list): List of lines to use for training. Defaults to lines.
        eval_lines (list): List of lines to use for evaluation. Defaults to eval_lines.
        n_steps (int, optional): Number of steps to train. Defaults to 1.
        output_dir (str, optional): Relative path of directory to save model. Defaults to "model/".

    Returns:
        trax.supervised.training.Loop: Training loop for the model.
    """
    
    
    bare_train_generator = data_generator(batch_size=batch_size, max_length=max_length, data_lines=lines,shuffle=True)
    infinite_train_generator = itertools.cycle(data_generator(batch_size=batch_size, max_length=max_length, data_lines=lines,shuffle=True))
    
    bare_eval_generator = data_generator(batch_size=batch_size, max_length=max_length, data_lines=eval_lines,shuffle=True)
    infinite_eval_generator = itertools.cycle(data_generator(batch_size=batch_size, max_length=max_length, data_lines=eval_lines,shuffle=True))
    
    train_task = training.TrainTask( 
        labeled_data=infinite_train_generator, # Use infinite train data generator
        loss_layer= tl.CrossEntropyLoss(),   # Don't forget to instantiate this object
        optimizer=trax.optimizers.Adam(learning_rate=0.0005)      # Don't forget to add the learning rate parameter TO 0.0005
    ) 
    
    eval_task = training.EvalTask( 
        labeled_data=infinite_eval_generator,    # Use infinite eval data generator
        metrics=[tl.CrossEntropyLoss(),  tl.Accuracy()], # Don't forget to instantiate these objects
        n_eval_batches=3  # For better evaluation accuracy in reasonable time 
    ) 
    
    training_loop = training.Loop(model, 
                                  train_task, 
                                  eval_tasks=[eval_task], 
                                  output_dir=output_dir) 

    training_loop.run(n_steps=n_steps)
    
    
    
    # We return this because it contains a handle to the model, which has the weights etc.
    return training_loop

# Train the model 1 step and keep the `trax.supervised.training.Loop` object.
output_dir = './model/'

try:
    shutil.rmtree(output_dir)
except OSError as e:
    pass

training_loop = train_model(GRULM(), data_generator, lines=lines, eval_lines=eval_lines)




def test_model(preds, target):
    """Function to test the model.

    Args:
        preds (jax.interpreters.xla.DeviceArray): Predictions of a list of batches of tensors corresponding to lines of text.
        target (jax.interpreters.xla.DeviceArray): Actual list of batches of tensors corresponding to lines of text.

    Returns:
        float: log_perplexity of the model.
    """


    log_p = np.sum(tl.one_hot(target,preds.shape[-1]) * preds, axis= -1) # HINT: tl.one_hot() should replace one of the Nones

    non_pad = 1.0 - np.equal(target, 0)          # You should check if the target equals 0
    log_p = log_p * non_pad                             # Get rid of the padding    
    
    log_ppx = np.sum(log_p, axis=1) / np.sum(non_pad, axis=1) # Remember to set the axis properly when summing up
    log_ppx = np.mean(log_ppx) # Compute the mean of the previous expression
    
    
  
    
    return -log_ppx   

# Testing 
model = GRULM()
model.init_from_file('model.pkl.gz')
batch = next(data_generator(batch_size, max_length, lines, shuffle=False))
preds = model(batch[0])
log_ppx = test_model(preds, batch[1])
print('The log perplexity and perplexity of your model are respectively', log_ppx, np.exp(log_ppx))


def gumbel_sample(log_probs, temperature=1.0):
    """Gumbel sampling from a categorical distribution."""
    u = numpy.random.uniform(low=1e-6, high=1.0 - 1e-6, size=log_probs.shape)
    g = -np.log(-np.log(u))
    return np.argmax(log_probs + g * temperature, axis=-1)

def predict(num_chars, prefix):
    inp = [ord(c) for c in prefix]
    result = [c for c in prefix]
    max_len = len(prefix) + num_chars
    for _ in range(num_chars):
        cur_inp = np.array(inp + [0] * (max_len - len(inp)))
        outp = model(cur_inp[None, :])  # Add batch dim.
        next_char = gumbel_sample(outp[0, len(inp)])
        inp += [int(next_char)]
       
        if inp[-1] == 1:
            break  # EOS
        result.append(chr(int(next_char)))
    
    return "".join(result)

print(predict(32, ""))
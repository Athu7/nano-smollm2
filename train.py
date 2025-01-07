import torch
from pathlib import Path
import numpy as np

# dataloader for causal language modelling
class SmolLoaderLM:
    
    def __init__(self, tokens_file:str, lens_file:str, block_size:int, batch_size:int, pad_index:int = 0, ignore_token = -100):

        self.tokens_file = tokens_file
        if not Path(tokens_file).exists():
            raise FileNotFoundError(f"{tokens_file} not found")
         
        self.lens_file = lens_file
        if not Path(lens_file).exists():
            raise FileNotFoundError(f"{lens_file} not found")

        self.block_size = block_size
        self.bs = batch_size
        self.start = 0
        self.end = 1
        self.epoch = 0
        self.iter = 0
        self.pad_index = pad_index
        self.ignore_token = -100

    def next(self): 
        # load the memmaps of tokens and lens
        tokens = np.memmap(self.tokens_file, dtype=np.uint32, mode='r') # memmap of all the tokens in our dataset
        lens = np.memmap(self.lens_file, dtype=np.uint32, mode='r') # memmap of the lengths of each example in our dataset

        try:
            _ = lens[self.end] # try to fetch the end index
        except IndexError:
            #TODO: Skipping the last batch for now
            self.start = 0
            self.end = 1
            self.epoch += 1

        cbs = 0 #to store the elements added to the batch 
        batch= np.zeros(shape=(self.bs, self.block_size)) # temp array to store the batch
        while cbs < self.bs:
            tok_end = np.sum(lens[0:self.end]) # tokens end index
            tok_start = np.sum(lens[0:self.start]) # tokens start index
            curr_len = tok_end - tok_start
            if curr_len >= self.block_size: # means we have got a packed example for out batch
                if self.start != self.end - 1: 
                    ex = tokens[tok_start:np.sum(lens[0:self.end-1])] # get the packed tokens leaving the last which exceeded the block size
                    diff = self.block_size - ex.shape[0]  # get the amount of pad tokens to insert
                    pad = np.full(diff, self.pad_index) # create the pad array
                    ex = np.concatenate((ex, pad)) # get the packed padded example 
                    batch[cbs] = ex # append to the batch 
                    cbs += 1 
                    self.start = self.end - 1

                else:  # single example is greater than the block size
                    ex = tokens[np.sum(lens[0:self.start]):np.sum(lens[0:self.end])]
                    ex = ex[0:self.block_size] # trim the example to block size
                    batch[cbs] = ex # append to the batch
                    cbs += 1 # increase the batchsize count
                    self.start += 1
                    self.end += 1
            else:
                self.end += 1
        self.iter += 1
        inputs = batch[:, :-1].copy()
        targets = batch[:, 1:].copy()

        for ind,row in enumerate(targets):
            pad_inds = np.where(row == self.pad_index)[0]   
            targets[ind][pad_inds] = self.ignore_token #replace all pad tokens with ignore token

        return torch.tensor(inputs).long(), torch.tensor(targets).long() 
import torch
import inspect
import time
from pathlib import Path
import numpy as np
from model import SmolLM2

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
                try:
                    _ = lens[self.end] # try to fetch the end index
                except IndexError:
                    #TODO: Skipping the last batch for now
                    self.start = 0
                    self.end = 1
                    self.epoch += 1


            # print(self.start, self.end, curr_len)
        self.iter += 1
        inputs = batch[:, :-1].copy()
        targets = batch[:, 1:].copy()

        for ind,row in enumerate(targets):
            pad_inds = np.where(row == self.pad_index)[0]   
            targets[ind][pad_inds] = self.ignore_token #replace all pad tokens with ignore token

        return torch.tensor(inputs).long(), torch.tensor(targets).long() 
    

block_size = 1024
batch_size = 4
learning_rate = 3e-4
weight_decay = 1e-1
seed = 0

loader = SmolLoaderLM(tokens_file = "dataset_tokens.bin", lens_file = "dataset_lens.bin", batch_size = batch_size, block_size = block_size)
inps, targets = loader.next()

# set the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    torch.device("cpu") 

# set the seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None

model = SmolLM2()
model.load_state_dict(torch.load("model.pt"), strict = False)
model = model.to(device)
# model = torch.compile(model)


def get_optimizer(model, weight_decay, learning_rate, device):
    param_dict = {pn : p for pn, p in model.named_parameters()}
    param_dict = {pn : p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay' : weight_decay},
        {'params': nodecay_params, 'weight_decay' : 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)

    print(f"num_decay_params: {num_decay_params}")
    print(f"num_nodecay_params: {num_nodecay_params}")

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device.type == "cuda"

    print(f"using fused adam: {use_fused}")
    optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = (0.9, 0.95), eps = 1e-8, fused = use_fused)

    return optimizer
    
optimizer = get_optimizer(model = model, weight_decay=weight_decay, learning_rate=learning_rate, device = device)

for i in range(94):
    t1 = time.perf_counter()
    x,y = loader.next()
    x,y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x,y)
    loss.backward()
    
    optimizer.step()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    dt = (t2 - t1) * 1000
    tokens_per_sec = (loader.bs * loader.block_size) / (t2-t1)
    print(f"step {i}, loss: {loss.item():.4f}, dt: {dt:.2f}, tok/sec: {tokens_per_sec:.2f}")


import torch
from dataclasses import dataclass

@dataclass
class SmolLM2Config:
    n_embd: int = 576
    n_hidden: int = 1536
    bias: bool = False
    block_size:int = 8192 
    n_layer: int = 30
    n_head:int = 9
    n_kv_heads:int = 3
    norm_eps: float = 1e-05
    dtype: torch.dtype = torch.bfloat16
    rope_theta:int = 100000
    vocab_size: int = 49152    
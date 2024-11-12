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

# Minimal rope implementation
def pre_compute_rope(config:SmolLM2Config):
    head_dim = config.n_embd // config.n_head
    positions = torch.arange(config.block_size) # (1, T)
    thetas = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos_thetas = torch.outer(positions, thetas)
    pos_thetas = torch.concatenate([pos_thetas, pos_thetas], dim = -1 )
    cos = torch.cos(pos_thetas)
    sin = torch.sin(pos_thetas)
    return cos, sin
def apply_rope(x:torch.Tensor, cos:torch.Tensor, sin:torch.Tensor):
    B,n,T,h = x.shape
    x1 = x[..., :h//2]
    x2 = x[..., h//2:]
    rotated = torch.cat((-x2, x1), dim = -1)
    roped = (x * cos[:T, :]) + (rotated * sin[:T, :])
    return roped.to(dtype=x.dtype)
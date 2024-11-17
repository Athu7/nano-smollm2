import torch
import torch.nn as nn
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
    return cos.to(dtype = config.dtype), sin.to(dtype = config.dtype)
def apply_rope(x:torch.Tensor, cos:torch.Tensor, sin:torch.Tensor):
    B,n,T,h = x.shape
    x1 = x[..., :h//2]
    x2 = x[..., h//2:]
    rotated = torch.cat((-x2, x1), dim = -1)
    roped = (x * cos[:T, :]) + (rotated * sin[:T, :])
    return roped.to(dtype=x.dtype)


class MHA(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config
        self.group_size = config.n_head // config.n_kv_heads
        self.head_dim = config.n_embd // config.n_head
        self.q_proj = nn.Linear(self.config.n_embd, self.config.n_embd, bias = False, dtype = config.dtype)
        self.k_proj = nn.Linear(self.config.n_embd, self.config.n_kv_heads * self.head_dim, bias = False, dtype = config.dtype)
        self.v_proj = nn.Linear(self.config.n_embd, self.config.n_kv_heads * self.head_dim, bias = False, dtype = config.dtype)
        self.o_proj = nn.Linear(self.config.n_embd, self.config.n_embd, bias = False, dtype = config.dtype)
        self.register_buffer("mask", torch.tril(torch.ones(1,1, self.config.block_size, self.config.block_size, dtype= config.dtype)))
        cos, sin = pre_compute_rope(config = config)
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)
    
    def forward(self, x:torch.Tensor):
        B,T,C = x.shape
        n_head = self.config.n_head
        head_dim = self.config.n_embd // n_head
        kv_head = self.config.n_kv_heads

        # calculate q,k,v
        q = self.q_proj(x).view(B,T,n_head, head_dim).transpose(1,2) # (B, n_head, T, head_dim)
        k = self.k_proj(x).view(B,T,kv_head, head_dim).transpose(1,2) # (B, n_kv_heads, T, head_dim)
        v = self.v_proj(x).view(B,T,kv_head, head_dim).transpose(1,2) # (B, n_kv_heads, T, head_dim)
        
        # rotate q and k 
        q = apply_rope(q, sin = self.sin, cos = self.cos) 
        k = apply_rope(k, sin = self.sin, cos = self.cos) 

        # repeat the k and v groups 
        k = k.repeat_interleave(self.group_size, dim = 1) # (B, n_kv_heads, T, head_dim) -> (B, n_head, T, head_dim)
        v = v.repeat_interleave(self.group_size, dim = 1) # (B, n_kv_heads, T, head_dim) -> (B, n_head, T, head_dim)

        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p= 0, is_causal=True)
        # attn_scores = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(k.size(-1)))
        # attn_scores = torch.masked_fill(attn_scores, self.mask[:,:, :T,:T]== 0, float("-inf"))
        # attn_scores = torch.nn.functional.softmax(attn_scores, dim = -1)
        # attn_out = attn_scores @ v
        attn_out = attn_out.transpose(1,2).contiguous().view(B,T,C)
        attn_out = self.o_proj(attn_out)
        return attn_out

class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.n_embd, config.n_hidden, bias = config.bias, dtype = config.dtype)
        self.down_proj = nn.Linear(config.n_hidden, config.n_embd, bias = config.bias, dtype = config.dtype)
        self.gate_proj = nn.Linear(config.n_embd, config.n_hidden, bias = config.bias, dtype = config.dtype)

        # TODO: find out why our implementation doesn't work
        # self.act_fn = SiLU()
        self.act_fn = torch.nn.SiLU() 
    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class RMSNorm(nn.Module):
    def __init__(self, config: SmolLM2Config):

        super().__init__()
        self.embd_dim = config.n_embd
        self.eps = config.norm_eps # (1)
        self.weight = nn.Parameter(torch.ones(self.embd_dim, dtype=config.dtype)) # (C)

    def forward(self, x):
        means = x.pow(2).mean(dim=-1, keepdim=True) # (B, T, 1)
        x_normed = x * torch.rsqrt(means + self.eps) # (B, T, C) / root((B, T, 1) + (1)) -> (B, T, C)
        return (x_normed * self.weight).to(dtype=x.dtype) # (B, T, C) * (C) -> (B, T, C) 
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mha = MHA(config = config)
        self.rms1 = RMSNorm(config = config)
        self.ffn = FFN(config = config)
        self.rms2 = RMSNorm(config = config)
    
    def forward(self, x:torch.Tensor):
        x = x + self.mha(self.rms1(x))
        x = x + self.ffn(self.rms2(x))
        return x
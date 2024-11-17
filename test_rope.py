import torch
from model import SmolLM2Config
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as apply_rotary_pos_emb_llama

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

@torch.inference_mode()
def test_rope_smollm2():
    config = SmolLM2Config()
    head_dim = config.n_embd // config.n_head 
    rope_theta = config.rope_theta
    rot_emb = LlamaRotaryEmbedding(head_dim, scaling_factor=None, base=rope_theta)
    batch_size, seq_len = 1, config.block_size
    qk_tensor = torch.randn(batch_size, seq_len, head_dim)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    theirs_cos, theirs_sin = rot_emb(qk_tensor, position_ids)

    # our rope
    my_cos, my_sin = pre_compute_rope(config)
    torch.testing.assert_close(theirs_cos.squeeze(0), my_cos)
    torch.testing.assert_close(theirs_sin.squeeze(0), my_sin)
    num_heads = config.n_head

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)

    my_q_rot = apply_rope(queries, my_cos, my_sin)
    my_k_rot = apply_rope(keys, my_cos, my_sin)
    theirs_q_rot, theirs_k_rot = apply_rotary_pos_emb_llama(queries, keys, theirs_cos, theirs_sin)

    torch.testing.assert_close(theirs_q_rot, my_q_rot)
    torch.testing.assert_close(theirs_k_rot, my_k_rot)
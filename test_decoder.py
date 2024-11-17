import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SmolLM2Config, FFN, MHA, RMSNorm
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig


@torch.inference_mode()
def test_decoder():
    torch.manual_seed(123)

    # initialize configs
    config = SmolLM2Config()
    config_hf = LlamaConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    torch.set_default_dtype(config.dtype) # so that hf modules are also use the dt
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

    block = Block(config = config)
    config_hf._attn_implementation = "sdpa"
    block_hf = LlamaDecoderLayer(config = config_hf, layer_idx= None)
    model_data = torch.load("model.pt")
    model_data.keys()
    layer = 2
    dic = {
        "mha.q_proj.weight": model_data.get(f"model.layers.{layer}.self_attn.q_proj.weight"),
        "mha.k_proj.weight": model_data.get(f"model.layers.{layer}.self_attn.k_proj.weight"),
        "mha.v_proj.weight": model_data.get(f"model.layers.{layer}.self_attn.v_proj.weight"),
        "mha.o_proj.weight": model_data.get(f"model.layers.{layer}.self_attn.o_proj.weight"),
        "mha.mask": block.mha.mask,
        "mha.sin": block.mha.sin,
        "mha.cos": block.mha.cos,
        "ffn.gate_proj.weight": model_data.get(f"model.layers.{layer}.mlp.gate_proj.weight"),
        "ffn.up_proj.weight": model_data.get(f"model.layers.{layer}.mlp.up_proj.weight"),
        "ffn.down_proj.weight": model_data.get(f"model.layers.{layer}.mlp.down_proj.weight"),
        "rms1.weight": model_data.get(f"model.layers.{layer}.input_layernorm.weight"),
        "rms2.weight": model_data.get(f"model.layers.{layer}.post_attention_layernorm.weight"),
    }

    hf_dic = {
        "self_attn.q_proj.weight": model_data.get(f"model.layers.{layer}.self_attn.q_proj.weight"),
        "self_attn.k_proj.weight": model_data.get(f"model.layers.{layer}.self_attn.k_proj.weight"),
        "self_attn.v_proj.weight": model_data.get(f"model.layers.{layer}.self_attn.v_proj.weight"),
        "self_attn.o_proj.weight": model_data.get(f"model.layers.{layer}.self_attn.o_proj.weight"),
        "mlp.gate_proj.weight": model_data.get(f"model.layers.{layer}.mlp.gate_proj.weight"),
        "mlp.up_proj.weight": model_data.get(f"model.layers.{layer}.mlp.up_proj.weight"),
        "mlp.down_proj.weight": model_data.get(f"model.layers.{layer}.mlp.down_proj.weight"),
        "input_layernorm.weight": model_data.get(f"model.layers.{layer}.input_layernorm.weight"),
        "post_attention_layernorm.weight": model_data.get(f"model.layers.{layer}.post_attention_layernorm.weight"),
    } 
    
    block_hf.load_state_dict(hf_dic)
    block.load_state_dict(dic, strict = False)

    # test our implementation
    torch.manual_seed(123)
    B,T,C = 1, 200, config.n_embd
    inp = torch.randn(B, T, C)
    out = block(inp.to(dtype = config.dtype))

    position_ids = torch.arange(T, dtype = config.dtype).unsqueeze(0)
    out_hf = block_hf(hidden_states = inp.to(dtype = config.dtype), position_ids = position_ids)[0]

    torch.testing.assert_close(out_hf, out)
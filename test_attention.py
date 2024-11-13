import torch
import torch.nn as nn
from model import SmolLM2Config, pre_compute_rope, apply_rope, MHA
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, LlamaConfig

@torch.inference_mode()
def test_attention_smollm2():
    torch.manual_seed(123)

    # initialize configs
    config = SmolLM2Config()
    smollm2_config_hf = LlamaConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    torch.set_default_dtype(config.dtype) # so that hf modules are also use the dtype that we are using

    # initialize our attention
    attn = MHA(config=config)
    
    # initialize hf attention
    attn_hf = LlamaSdpaAttention(config = smollm2_config_hf)

    # load same weights in both implementations
    model_data = torch.load("model.pt")
    hf_dic = {
        "q_proj.weight": model_data.get("model.layers.0.self_attn.q_proj.weight"),
        "k_proj.weight": model_data.get("model.layers.0.self_attn.k_proj.weight"),
        "v_proj.weight": model_data.get("model.layers.0.self_attn.v_proj.weight"),
        "o_proj.weight": model_data.get("model.layers.0.self_attn.o_proj.weight"),
    }

    dic = {
        "q_proj.weight": model_data.get("model.layers.0.self_attn.q_proj.weight"),
        "k_proj.weight": model_data.get("model.layers.0.self_attn.k_proj.weight"),
        "v_proj.weight": model_data.get("model.layers.0.self_attn.v_proj.weight"),
        "o_proj.weight": model_data.get("model.layers.0.self_attn.o_proj.weight"),
        "mask": attn.mask,
        "sin": attn.sin,
        "cos": attn.cos
    }

    attn_hf.load_state_dict(hf_dic)
    attn.load_state_dict(dic)
    
    # test our implementation
    B,T,C = 1, config.block_size, config.n_embd
    inp = torch.randn(B, T, C)
    position_ids = torch.arange(T).unsqueeze(0)
    attn_out_hf = attn_hf(inp, position_ids = position_ids)[0]
    attn_out = attn(inp.to(dtype = config.dtype))
    torch.testing.assert_close(attn_out_hf, attn_out)
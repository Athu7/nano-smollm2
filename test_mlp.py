import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SmolLM2Config, FFN
from transformers.models.llama.modeling_llama import LlamaMLP, ACT2FN, LlamaConfig


@torch.inference_mode()
def test_mlp():
    torch.manual_seed(123)

    # initialize configs
    config = SmolLM2Config()
    config_hf = LlamaConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    torch.set_default_dtype(config.dtype) # so that hf modules are also use the dtype that we are using

    # ------------------ testing activation function implementations--------------
    class SiLU(nn.Module): 
        def __init__(self):
            super().__init__()    
        def forward(self, x):
            return x * torch.sigmoid(x) 


    silu = SiLU() # our implementation
    silu_torch = torch.nn.SiLU() # torch implementation
    silu_hf = ACT2FN["silu"] # hf implementation

    # dummy input
    B,T,C = 2, config.block_size, config.n_embd
    x = torch.randn(B,T,C, dtype=config.dtype)

    # get the output from all implementations
    x_out = silu(x)
    x_out_torch = silu_torch(x)
    x_out_hf = silu_hf(x)

    # check that the outputs are the same
    torch.testing.assert_close(x_out, x_out_torch)
    torch.testing.assert_close(x_out, x_out_hf)

    # ------------------ testing mlp----------------------------------------------
    ffn_hf = LlamaMLP(config = config_hf)
    ffn = FFN(config = config)
    model_data = torch.load("model.pt")

    # load same weights in both implementations
    dic = {
    "gate_proj.weight": model_data.get("model.layers.0.mlp.gate_proj.weight"),
    "up_proj.weight": model_data.get("model.layers.0.mlp.up_proj.weight"),
    "down_proj.weight": model_data.get("model.layers.0.mlp.down_proj.weight"),
    }

    ffn.load_state_dict(dic)
    ffn_hf.load_state_dict(dic)

    x_out = ffn(x)
    x_out_hf = ffn_hf(x)

    torch.testing.assert_close(x_out, x_out_hf)
import torch
from model import SmolLM2Config, RMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm

@torch.inference_mode()
def test_rms():    
    torch.manual_seed(123)

    # initialize configs
    config = SmolLM2Config()
    torch.set_default_dtype(config.dtype) # so that hf modules are also use the dtype that we are using

    rms_hf = LlamaRMSNorm(eps = config.norm_eps, hidden_size=config.n_embd)
    rms = RMSNorm(config=config)
    model_data = torch.load("model.pt")
    # load same weights in both implementations
    dic = {
        "weight": model_data.get("model.layers.0.input_layernorm.weight"),
    }

    rms.load_state_dict(dic)
    rms_hf.load_state_dict(dic)

    B,T,C = 2, config.block_size, config.n_embd
    x = torch.randn(B,T,C, dtype=config.dtype)

    x_out = rms(x)
    x_out_hf = rms_hf(x)

    # TODO: RMSNorm is not exact
    torch.testing.assert_close(x_out, x_out_hf, rtol=0.01, atol = 0.01)
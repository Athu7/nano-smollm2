import torch
from model import SmolLM2Config, SmolLM2
from transformers import AutoModelForCausalLM

@torch.inference_mode()
def test_smollm2():
    torch.manual_seed(123)

    # initialize configs
    config = SmolLM2Config()
    torch.set_default_dtype(config.dtype) # so that hf modules are also use the dtype that we are using

    # initialize our model 
    model = SmolLM2(config = config)

    # initialize hf model
    checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model_hf = AutoModelForCausalLM.from_pretrained(checkpoint)

    # load same weights in both implementations
    model_data = torch.load("model.pt")
    weight_names = model_data.keys()

    # in case of weight tying copy the embedding weights to lm_head
    if "model.lm_head.weight" not in weight_names:
        model_data["model.lm_head.weight"] = model_data.get("model.embed_tokens.weight")

    # rename the wieght names to match hf_names so we can use load_state_dict directly 
    replace_key = lambda x: x.replace("model.", "")
    dic_keys = [i for i in model_data.keys()]
    for i in dic_keys:
        model_data[replace_key(i)] = model_data.pop(i)
    
    model.load_state_dict(model_data, strict = False)
    
    # test our implementation
    B,T = 1, 30 
    inp = torch.randint(0, config.vocab_size, (B, T))
    logits_hf = model_hf(inp).logits
    logits = model(inp)
    torch.testing.assert_close(logits_hf, logits)
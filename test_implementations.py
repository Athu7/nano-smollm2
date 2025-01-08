import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SmolLM2Config, FFN, SmolLM2, RMSNorm, Block, pre_compute_rope, apply_rope, MHA
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    ACT2FN,
    LlamaConfig,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaSdpaAttention,
    LlamaDecoderLayer
)
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb as apply_rotary_pos_emb_llama,
)

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

@torch.inference_mode()
def test_decoder():
    torch.manual_seed(123)

    # initialize configs
    config = SmolLM2Config()
    config_hf = LlamaConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    torch.set_default_dtype(config.dtype) # so that hf modules are also use the dt

    block = Block(config = config)
    config_hf._attn_implementation = "sdpa" # as we are using torch's sdpa in our implementation as well
    block_hf = LlamaDecoderLayer(config = config_hf, layer_idx= None)
    model_data = torch.load("model.pt")
    model_data.keys()
    layer = 2
    dic = {
        "self_attn.q_proj.weight": model_data.get(f"model.layers.{layer}.self_attn.q_proj.weight"),
        "self_attn.k_proj.weight": model_data.get(f"model.layers.{layer}.self_attn.k_proj.weight"),
        "self_attn.v_proj.weight": model_data.get(f"model.layers.{layer}.self_attn.v_proj.weight"),
        "self_attn.o_proj.weight": model_data.get(f"model.layers.{layer}.self_attn.o_proj.weight"),
        "self_attn.mask": block.self_attn.mask,
        "self_attn.sin": block.self_attn.sin,
        "self_attn.cos": block.self_attn.cos,
        "mlp.gate_proj.weight": model_data.get(f"model.layers.{layer}.mlp.gate_proj.weight"),
        "mlp.up_proj.weight": model_data.get(f"model.layers.{layer}.mlp.up_proj.weight"),
        "mlp.down_proj.weight": model_data.get(f"model.layers.{layer}.mlp.down_proj.weight"),
        "input_layernorm.weight": model_data.get(f"model.layers.{layer}.input_layernorm.weight"),
        "post_attention_layernorm.weight": model_data.get(f"model.layers.{layer}.post_attention_layernorm.weight"),
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


def test_tokenizer():
    from tokenizer import Tokenizer
    from transformers import GPT2Tokenizer

    special_token_file = {
        "<|endoftext|>": 0,
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "<repo_name>": 3,
        "<reponame>": 4,
        "<file_sep>": 5,
        "<filename>": 6,
        "<gh_stars>": 7,
        "<issue_start>": 8,
        "<issue_comment>": 9,
        "<issue_closed>": 10,
        "<jupyter_start>": 11,
        "<jupyter_text>": 12,
        "<jupyter_code>": 13,
        "<jupyter_output>": 14,
        "<jupyter_script>": 15,
        "<empty_output>": 16,
    }
    tok = Tokenizer(
        vocab_file="vocab.json",
        merges_file="merges.txt",
        special_tokens_file=special_token_file,
    )
    tok_hf = GPT2Tokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

    with open("taylorswift.txt", encoding="utf-8") as f:
        text = f.read()

    tokens_hf = tok_hf.encode(text)
    tokens = tok.encode(text)

    assert tokens_hf == tokens

    tokens_decoded_hf = tok_hf.decode(tokens)
    tokens_decoded = tok.decode(tokens)

    assert tokens_decoded_hf == tokens_decoded
    
    
    
def test_cross_entropy_loss():
    import torch

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        torch.device("cpu") 
    torch.manual_seed(0)

    B,T,V= 2,6,49152
    inps = torch.randint(V, (B, T)) 
    inputs = inps[:, :-1].contiguous().to(device) # inputs are orig tokens shifted left
    logits = torch.randn(B*(T-1), V) # model(inputs) | (B, T-1 ) -> (B, T-1, V) -> (B*(T-1), V) we flatten the batch
    targets = inps[:, 1:].contiguous().to(device)  # (B, T-1 ) | targets are orig tokens shifted right
    padded_targets = torch.where(torch.tril(targets) == 0, -100, targets).to(device)

    def cross_entropy_loss(logits, targets, ignore_index = -100):
        # logits are flattened across the batch so the shape is (B*T, V)
        targets = targets.view(-1, 1).contiguous() # (B,T) -> (B*T, 1) | flatten the batch dim to match logits 
        non_pad_inds = torch.where(targets != -100)[0]
        targets = targets[non_pad_inds]
        logits = torch.index_select(logits, 0, non_pad_inds)
        neg_log_liklihood = -torch.log_softmax(logits, dim = -1)  # (B*T, 1) -> (B*T, 1) |  negative log liklihood of all the possible tokens
        vals = torch.gather(neg_log_liklihood, 1, targets).squeeze(-1) # (B*T, 1) | negative log lilihood corresponding to targets
        loss = torch.mean(vals) # (B*T, 1) -> (1) | mean over the batch and time
        return loss

    # padded input test
    loss = torch.nn.functional.cross_entropy(logits.to(device), padded_targets.view(-1).to(device), ignore_index=-100, reduction= "mean")
    our_loss = cross_entropy_loss(logits.to(device), padded_targets.to(device), ignore_index = -100)
    torch.testing.assert_close(loss, our_loss)

    # unpadded input test
    loss = torch.nn.functional.cross_entropy(logits.to(device), targets.view(-1).to(device), ignore_index=-100, reduction= "mean")
    our_loss = cross_entropy_loss(logits.to(device), targets.to(device), ignore_index = -100)
    torch.testing.assert_close(loss, our_loss)
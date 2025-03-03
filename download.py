from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# download from hf
repo_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
# repo_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
model = hf_hub_download(repo_id=repo_id, filename="model.safetensors", local_dir=".")

# convert safe tensors to pt and save
safetensor_file = "model.safetensors"
model_data = load_file(safetensor_file)
torch.save(model_data, "model.pt")

# modify the weight names to match our model and save the pt file
model_data = torch.load("model.pt")
replace_key = lambda x: x.replace("model.", "")
dic_keys = [i for i in model_data.keys()]
for i in dic_keys:
    model_data[replace_key(i)] = model_data.pop(i)
torch.save(model_data, "model.pt")

# download the tokenizer files
tokenizer_vocab = hf_hub_download(repo_id=repo_id, filename="vocab.json", local_dir=".")
tokenizer_merges = hf_hub_download( repo_id=repo_id, filename="merges.txt", local_dir=".")

# download a text file Andrej Karpathy uses in the minbpe repo to test the tokenizer
res = requests.get( "https://raw.githubusercontent.com/karpathy/minbpe/refs/heads/master/tests/taylorswift.txt")
text = res.content.decode("utf-8")
Path("taylorswift.txt").write_text(text)
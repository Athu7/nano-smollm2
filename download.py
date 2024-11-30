from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# download from hf
repo_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = hf_hub_download(repo_id=repo_id, filename="model.safetensors", local_dir=".")

# convert safe tensors to pt
safetensor_file = "model.safetensors"
model_data = load_file(safetensor_file)

# save the pt file
torch.save(model_data, "model.pt")

# download the tokenizer files
tokenizer_vocab = hf_hub_download(repo_id=repo_id, filename="vocab.json", local_dir=".")
tokenizer_merges = hf_hub_download( repo_id=repo_id, filename="merges.txt", local_dir=".")

# download a text file Andrej Karpathy uses in the minbpe repo to test the tokenizer
res = requests.get( "https://raw.githubusercontent.com/karpathy/minbpe/refs/heads/master/tests/taylorswift.txt")
text = res.content.decode("utf-8")
Path("taylorswift.txt").write_text(text)
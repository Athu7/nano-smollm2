import torch
from safetensors.torch import load_file
import argparse
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", "-n", type = str, required= True, default="HuggingFaceTB/SmolLM2-135M-Instruct")
parser.add_argument("--file-name", "-f", type = str, required= True, default="model.safetensors")
args = parser.parse_args()

# download from hf
model = hf_hub_download(repo_id=args.model_name, filename = args.file_name, local_dir=".")

# convert safe tensors to pt
safetensor_file = "model.safetensors"
model_data = load_file(safetensor_file)
# save the pt file
torch.save(model_data, "model.pt")
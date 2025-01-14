import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Union
import requests
from huggingface_hub import hf_hub_download, list_repo_files

from tokenizer import Tokenizer


class DataPipeline:
    def __init__(self, repo_id: str = "HuggingFaceTB/smoltalk"):
        self.data_folder_path = Path("data") 
        self.data_folder_path.mkdir(exist_ok=True, parents=True) 
        self.repo_id = repo_id
        self.org, self.dataset_name = repo_id.split("/")

    def download(self, folders: list[str] = ["data/everyday-conversations"]):
        all_downloaded_paths = []
        if self.repo_id in ["HuggingFaceTB/smoltalk"]:
            files = list_repo_files(self.repo_id, repo_type = "dataset")
            folder_files = [f for f in files for folder_name in folders if f.startswith(folder_name) ]
            train_files = [i for i in folder_files if "train" in i]
            test_files = [i for i in folder_files if "test" in i]
            train_file_urls = [f"https://huggingface.co/datasets/{self.org}/{self.dataset_name}/resolve/main/{i}" for i in train_files]
            _ = [f"https://huggingface.co/datasets/{self.org}/{self.dataset_name}/resolve/main/{i}" for i in test_files]

            for ind, url in enumerate(train_file_urls):
                data = requests.get(url)
                data_bytes = data.content
                dataset_folder_path = self.data_folder_path / Path(f"{self.dataset_name}")
                dataset_folder_path.mkdir(exist_ok=True, parents= True)
                data_path = dataset_folder_path / Path(f"{ind:02}.parquet")
                all_downloaded_paths.append(str(data_path))
                data_path.write_bytes(data_bytes) 
        else:
            pass

        return all_downloaded_paths
        
    def process(self, all_downloaded_paths:list[str]):
        all_json_paths = []
        all_downloaded_paths = [Path(i) for i in all_downloaded_paths]
        if self.repo_id in ["HuggingFaceTB/smoltalk"]:
            for file_path in all_downloaded_paths: 
                if file_path.is_file() and file_path.suffix == ".parquet" and file_path.exists():
                    df = pd.read_parquet(file_path)
                    json_data = df.to_json(orient='records', lines=False)
                    dic = json.loads(json_data)
                    messages = [i["messages"] for i in dic]
                    data_json_path = file_path.parent / Path(f"{file_path.stem}.json")
                    data_json_path.write_text(json.dumps(messages))
                    all_json_paths.append(data_json_path)
                else:
                    pass
        return all_json_paths 

    @classmethod
    def tokenize(cls, tokenizer:Tokenizer, conversation: list[dict]):
        ids = tokenizer.encode(conversation = conversation)
        return {"ids": ids, "len" : len(ids)}

    @classmethod
    def get_tokenized_dataset(cls, file_path:Union[str,Path], tokenizer:Tokenizer):
        if isinstance(file_path, str):
            file_path = Path()
        file_content = file_path.read_text()
        dataset = json.loads(file_content)
        tokenized_dataset = []
        dataset_size = 0
        for ind, i in enumerate(dataset):
            ids = cls.tokenize(tokenizer, i)
            tokenized_dataset.append(ids)
            dataset_size += ids["len"]

        return tokenized_dataset, dataset_size

    @classmethod
    def get_binaries(cls, folder_paths:list[str], tokenizer:Tokenizer = Tokenizer(vocab_file= "vocab.json", merges_file="merges.txt", special_tokens_file="special_tokens.json")):
        entire_dataset = []
        entires_dataset_size = 0
        folder_paths = [Path(i) for i in folder_paths]
        for folder in folder_paths:
            for file in folder.iterdir():
                if file.is_file() and file.suffix == ".json":
                    tokenized_dataset, dataset_size = cls.get_tokenized_dataset(file, tokenizer)
                    entire_dataset.extend(tokenized_dataset)
                    entires_dataset_size += dataset_size
        dataset_tokens_arr = np.memmap("dataset_tokens.bin", dtype=np.uint32 , mode='w+', shape=(entires_dataset_size,))
        dataset_lens_arr = np.memmap("dataset_lens.bin", dtype = np.uint32, mode = "w+", shape = (len(entire_dataset,)))
        # dataset_tensors = { "dataset" : torch.zeros(dataset_size).long(), "offsets" : torch.zeros(len(tokenized_dataset)).long() }
        offset = 0
        for ind,i in enumerate(entire_dataset):
            leng = i["len"]
            ids = i["ids"]
            # ten = torch.tensor(ids, dtype = torch.long)
            arr = np.array(ids, dtype=np.uint32)
            dataset_tokens_arr[offset: offset + leng] = arr
            dataset_lens_arr[ind] = leng 
            # # dataset_tensors["dataset"][offset : offset+leng] = ten
            # # dataset_tensors["offsets"][ind] = offset
            offset = offset + leng
        dataset_tokens_arr.flush()
        dataset_lens_arr.flush()
        
if __name__ == "__main__":
    pipe = DataPipeline()
    paths = pipe.download()
    json_paths = pipe.process(paths)
    binaries = DataPipeline.get_binaries(folder_paths = ["data/smoltalk"])



            
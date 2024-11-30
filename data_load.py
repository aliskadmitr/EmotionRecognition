import os
from datasets import load_dataset

class DataLoader:
    @staticmethod
    def load_dataset_from_path(path: str):
        return load_dataset("imagefolder", data_dir=path)
import os
import pickle
import torch
import json
from torch.utils.data import Dataset, TensorDataset
from .preprocess import PreprocessData

class TokenDataset(Dataset):
    def __init__(self, dataset):
        self.examples = dataset

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])

class DataHelper(object):
    """docstring for DataHelper"""
    def __init__(self, data_dir, tokenizer, load_dataset=True):
        super(DataHelper, self).__init__()
        preprocesser = PreprocessData(data_dir, tokenizer)
        self.PAD = preprocesser.PAD
        self.SEP = preprocesser.SEP
        self.END = preprocesser.END

        if load_dataset:
            with open(preprocesser.token_path, 'rb') as handle:
                dataset = pickle.load(handle)

            self.trainset = TokenDataset(dataset['train'])
            self.devset= TokenDataset(dataset['dev'])
            self.testset= TokenDataset(dataset['test'])


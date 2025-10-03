import json
import os
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

class TDataset(Dataset):
    def __init__(self, path, specaug=False):
        # read the data points
        with open(path) as f:
            self.data = json.load(f)  
        
        # keys are dataset index (FBank_ID)
        self.data_idx = list(self.data.keys())
        
        # SpecAugment, boolean value, wether or not using augmentation
        self.specaug = specaug
        
    def __len__(self):  # number of the points
        return len(self.data)

    def __getitem__(self, index):  # get one data point from FBank file
        data = self.data[self.data_idx[index]]  # using data point index to extract data
        
        data_path = data["fbank"]  # path of generated FBank file

        fbank = torch.load(data_path)
        fbank_mean = torch.mean(fbank, dim=0, keepdims=True)
        fbank_std = torch.std(fbank, dim=0, keepdims=True)
        fbank = (fbank - fbank_mean) / fbank_std  # standardlized fbank feature

        phn = data["phn"]  # phn

        duration = data["duration"]  # duration

        return fbank, phn, duration

def collate_wrapper(batch):
    fbank = pad_sequence([i[0] for i in batch])  # features of the whole batch (Xs)
    lens = torch.tensor([len(i[0]) for i in batch], dtype=torch.long)  # feature length
    phn = [i[1] for i in batch]  # phns of the batch
    duration = [i[2] for i in batch] # durations of the batch

    return fbank, lens, phn, duration

def get_dataloader(path, bs, shuffle, specaug=False):
    dataset = TDataset(path, specaug)
    return DataLoader(
        dataset, 
        batch_size=bs, 
        shuffle=shuffle,
        collate_fn=collate_wrapper, 
        pin_memory=True
    )


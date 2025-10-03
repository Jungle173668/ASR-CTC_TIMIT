from dataloader import get_dataloader
import torch
from collections import Counter
from datetime import datetime
from trainer import train
import models
from decoder import decode
import numpy as np
import argparse
import random
from collections import namedtuple
import models


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# arguments
args = {
    "seed": 123,
    "train_json": "fbank/train_fbank.json",
    "dev_json": "fbank/dev_fbank.json",
    "test_json": "fbank/test_fbank.json",
    "batch_size": 4,
    "num_layers": 2,
    "fbank_dims": 23,
    "model_dims": 128,
    "concat": 1,
    "lr": 0.5,
    "vocab": {
        "_": 0,
        "d": 1,
        "ay": 2,
        "y": 3,
        "iy": 4,
        "uh": 5,
        "th": 6,
        "er": 7,
        "ng": 8,
        "v": 9,
        "b": 10,
        "aa": 11,
        "ch": 12,
        "jh": 13,
        "t": 14,
        "dh": 15,
        "k": 16,
        "g": 17,
        "n": 18,
        "l": 19,
        "oy": 20,
        "z": 21,
        "s": 22,
        "p": 23,
        "ah": 24,
        "sh": 25,
        "m": 26,
        "sil": 27,
        "dx": 28,
        "eh": 29,
        "ae": 30,
        "ih": 31,
        "aw": 32,
        "r": 33,
        "hh": 34,
        "ow": 35,
        "w": 36,
        "ey": 37,
        "uw": 38,
        "f": 39
    },
    "report_interval": 50,
    "num_epochs": 20,
    "max_norm": 1.0,
    # "device": "cuda:0"
    "device":'cpu'
}

args = namedtuple('x', args)(**args)

# define model
model = models.BiLSTM(
    args.num_layers, args.fbank_dims * args.concat, args.model_dims, len(args.vocab),0.5)
num_params = sum(p.numel() for p in model.parameters())

print('Total number of model parameters is {}'.format(num_params))

# load model
model_path = 'checkpoints/20241215_182647/model_20'
print('Loading model from {}'.format(model_path))

# load model to cpu
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model.eval()
model.to(device)

# evaluate the model
results = decode(model, args, args.test_json)
print("SUB: {:.2f}%, DEL: {:.2f}%, INS: {:.2f}%, COR: {:.2f}%, PER: {:.2f}%".format(*results))

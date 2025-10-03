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

parser = argparse.ArgumentParser(description = 'Running MLMI2 experiments')

# set arguments for training and decoding. 
parser.add_argument('--seed', type=int, default=123)
# parser.add_argument('--train_json', type=str, default="json/train_fbank.json")
# parser.add_argument('--dev_json', type=str, default="json/dev_fbank.json")
# parser.add_argument('--test_json', type=str, default="json/test_fbank.json")

parser.add_argument('--train_json', type=str, default="fbank/train_fbank.json")  # train fbank metadata path
parser.add_argument('--dev_json', type=str, default="fbank/dev_fbank.json")  # dev fbank metadata path
parser.add_argument('--test_json', type=str, default="fbank/test_fbank.json")  # test fbank metadata path

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=3, help="number of rnn layers")
parser.add_argument('--fbank_dims', type=int, default=23, help="filterbank dimension")
parser.add_argument('--model_dims', type=int, default=128, help="model size for rnn layers")
parser.add_argument('--concat', type=int, default=1, help="concatenating frames")

parser.add_argument('--lr', type=float, default=0.5, help="learning rate")
parser.add_argument('--vocab', type=str, default="vocab_39.txt", help="vocabulary file path")
parser.add_argument('--report_interval', type=int, default=50, help="report interval during training")
parser.add_argument('--num_epochs', type=int, default=20)

parser.add_argument('--max_norm', type=int, default=3.0)

args = parser.parse_args()

# set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# read vocabulary file and create mapping(from phonemes to number) dictionary
vocab = {}
with open(args.vocab) as f:
    for id, text in enumerate(f):
        vocab[text.strip()] = id

# choose device
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

print(args)
args.device = device
args.vocab = vocab

# dropout probabilities
probs = [0.1, 0.3, 0.7, 0.9]

# gradient clipping max norms
max_norms = [1.0, 3.0, 5.0]

# learning rate for SGD
# lrs = [0.5, 0.1, 0.05, 0.01]
# lrs = [1.0, 0.75]

# learning rate for Adam
# lrs = [1e-3, 5*1e-4, 1e-4, 1e-5]
# lrs = [5*1e-3]

args.max_norm = max_norms[0]

# define model
model = models.BiLSTM(args.num_layers, args.fbank_dims * args.concat, args.model_dims, len(args.vocab),0.5)

if torch.__version__ == "2.1.0":
    model = torch.compile(model)

num_params = sum(p.numel() for p in model.parameters())
print('Total number of model parameters is {}'.format(num_params))

# model training
start = datetime.now()
model.to(args.device)
model_path = train(model, args)  # train the model
end = datetime.now()

duration = (end - start).total_seconds()  # training time

print('Training finished in {:.2f} minutes.'.format(duration / 60))
print('Model saved to {}'.format(model_path))


# # evaluate model
# print('Loading model from {}'.format(model_path))
# model.load_state_dict(torch.load(model_path))
# model.eval()
# model.to(device)

# results = decode(model, args, args.test_json)
# print("SUB: {:.2f}%, DEL: {:.2f}%, INS: {:.2f}%, COR: {:.2f}%, PER: {:.2f}%".format(*results))

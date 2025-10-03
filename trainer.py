from datetime import datetime
from pathlib import Path

import torch
from torch.nn import CTCLoss
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import log_softmax
from torch.optim import SGD
from torch.optim import Adam
from decoder import decode
from utils import concat_inputs
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloader import get_dataloader

import matplotlib.pyplot as plt

import os
import json

import numpy as np

# 3.1 Training 
def plot_loss(avg_train_losses, avg_dev_losses, picpath):
    # tensor to float
    avg_train_losses = [v if isinstance(v, float) else v.cpu().item() for v in avg_train_losses]
    avg_dev_losses = [v.cpu().item() if isinstance(v, torch.Tensor) else v for v in avg_dev_losses]

    # avg_train_losses = avg_train_losses.cpu().numpy()
    # avg_dev_losses = avg_dev_losses.cpu().numpy()

    # x-axis
    epochs = range(1, len(avg_train_losses) + 1)  # [1, 2, 3, ...]

    plt.figure(figsize=(8, 6))  # size
    plt.plot(epochs, avg_train_losses, label='Train Loss', marker='o', linestyle='-',color='#9F1529')
    plt.plot(epochs, avg_dev_losses, label='Dev Loss', marker='s', linestyle='--')

    # title & label
    plt.title('Loss vs Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    # Set x-axis ticks to be integers
    plt.xticks(range(1, len(avg_train_losses) + 1))

    # grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # legend
    plt.legend(fontsize=12)

    # graph
    # plt.show()
    plt.savefig(picpath)


def plot_PERs(PER_dev, PER_test):
    PER_dev = [v if isinstance(v, float) else v.cpu().item() for v in PER_dev]
    PER_test = [v.cpu().item() if isinstance(v, torch.Tensor) else v for v in PER_test]

    # x-axis
    epochs = range(1, len(PER_dev) + 1)  # [1, 2, 3, ...]

    plt.figure(figsize=(8, 6))  # size
    plt.plot(epochs, PER_dev, label='dev/validation PERs', marker='o', linestyle='-',color='#9F1529')
    plt.plot(epochs, PER_test, label='test PERs', marker='s', linestyle='--')

    # title & label
    plt.title('PERs vs Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('PERs', fontsize=12)

    # Set x-axis ticks to be integers
    plt.xticks(range(1, len(PER_test) + 1))

    # grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # legend
    plt.legend(fontsize=12)

    # graph
    # plt.show()
    plt.savefig('PERs_plot.png')


def plot_lr(lrs):

    plt.figure()
    epochs = np.arange(1, len(lrs) + 1)
    
    plt.plot(epochs, lrs, label='SGD Learning rate', color='#9F1529')

    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('SGD Learning Rate')

    plt.legend()
    plt.savefig('optimizer_lr_plot.png')


def train(model, args):
    torch.manual_seed(args.seed)

    # generate dataloaders
    train_loader = get_dataloader(args.train_json, args.batch_size, True)
    dev_loader = get_dataloader(args.dev_json, args.batch_size, False)

    # loss function
    criterion = CTCLoss(zero_infinity=True)
    
    # optimiser - SGD
    optimiser = SGD(model.parameters(), lr=args.lr)
    # Add scheduler (factor - a half)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=1, verbose=True)

    # optimiser - Adam
    # optimiser  = Adam(model.parameters(), lr=args.lr)

    def train_one_epoch(epoch):
        running_loss = 0.
        last_loss = 0.

        for idx, data in enumerate(train_loader):
            inputs, in_lens, trans, _ = data
            inputs = inputs.to(args.device)
            in_lens = in_lens.to(args.device)
            inputs, in_lens = concat_inputs(inputs, in_lens, factor=args.concat)
            targets = [torch.tensor(list(map(lambda x: args.vocab[x], target.split())),
                                    dtype=torch.long)
                       for target in trans]
            out_lens = torch.tensor(
                [len(target) for target in targets], dtype=torch.long)
            targets = pad_sequence(targets, batch_first=True)
            targets = targets.to(args.device)

            optimiser.zero_grad()
            outputs = log_softmax(model(inputs), dim=-1)
            loss = criterion(outputs, targets, in_lens, out_lens)
            loss.backward()
            '''
            3.1.1 Gradient clipping
            '''
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)

            optimiser.step()

            running_loss += loss.item()
            if idx % args.report_interval + 1 == args.report_interval:
                last_loss = running_loss / args.report_interval
                print('  batch {} loss: {}'.format(idx + 1, last_loss))
                tb_x = epoch * len(train_loader) + idx + 1
                running_loss = 0.
        return last_loss

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    Path('checkpoints/{}'.format(timestamp)).mkdir(parents=True, exist_ok=True)
    best_per = 200
    best_epoch = 0

    avg_train_losses, avg_dev_losses = [], []
    PER_dev, PER_test = [], []
    lrs = []
    
    for epoch in range(args.num_epochs):
        print('EPOCH {}:'.format(epoch + 1))
        model.train(True)
        avg_train_loss = train_one_epoch(epoch)

        model.train(False)
        
        # Evaluate dev loss
        running_dev_loss = 0.
        for idx, data in enumerate(dev_loader):
            inputs, in_lens, trans, _ = data
            inputs = inputs.to(args.device)
            in_lens = in_lens.to(args.device)
            inputs, in_lens = concat_inputs(inputs, in_lens, factor=args.concat)
            targets = [torch.tensor(list(map(lambda x: args.vocab[x], target.split())),
                                    dtype=torch.long)
                       for target in trans]
            out_lens = torch.tensor(
                [len(target) for target in targets], dtype=torch.long)
            targets = pad_sequence(targets, batch_first=True)
            targets = targets.to(args.device)
            outputs = log_softmax(model(inputs), dim=-1)
            dev_loss = criterion(outputs, targets, in_lens, out_lens)
            running_dev_loss += dev_loss

        avg_dev_loss = running_dev_loss / len(dev_loader)
        
        '''
        3.1.2 Optimiser Scheduler
        '''
        # Pass the dev loss to scheduler
        scheduler.step(avg_dev_loss)
        lrs.append(optimiser.param_groups[0]['lr'])

        dev_decode = decode(model, args, args.dev_json)

        '''
        ## 3.1 Training PER
        '''
        # test_decode = decode(model, args, args.test_json)

        # PER_dev.append(dev_decode[4])
        # PER_test.append(test_decode[4])


        '''
         # # 3.1 Training 
        '''
        avg_train_losses.append(avg_train_loss)
        avg_dev_losses.append(avg_dev_loss)

        print('LOSS train {:.5f} valid {:.5f}, valid PER {:.2f}%'.format(
            avg_train_loss, avg_dev_loss, dev_decode[4])
            )
        

        if dev_decode[4] < best_per:
            best_per = dev_decode[4]
            best_epoch = epoch + 1
            

    # end of training
    # save the model
    model_path = 'checkpoints/{}/model_{}'.format(timestamp, best_epoch)
    torch.save(model.state_dict(), model_path)
    
    # test results
    results = decode(model, args, args.test_json)

    # num of params
    num_params = sum(p.numel() for p in model.parameters())

    # save the results
    file_path = 'checkpoints/{}/results_{}'.format(timestamp, best_epoch)
    with open(file_path, 'a') as f:
        # write final loss
        f.write('LOSS train {:.5f} valid {:.5f}, valid PER {:.2f}%\n'.format(
            avg_train_loss, avg_dev_loss, dev_decode[4]
        ))
        f.write('\n')
       
        # write test results
        f.write("SUB: {:.2f}%, DEL: {:.2f}%, INS: {:.2f}%, COR: {:.2f}%, PER: {:.2f}%".format(*results))
        f.write('\n')

        f.write('Total number of model parameters is {}'.format(num_params))
        f.write('\n')
        # write arguments
        # f.write(str(args))
        json.dump(vars(args),f,indent=4)
        f.write('\n')

    
    '''
    3.1 Training - plot the losses
    '''
    picpath = 'checkpoints/{}/pic_{}.png'.format(timestamp, best_epoch)
    plot_loss(avg_train_losses, avg_dev_losses, picpath)

    '''
    3.1 Training - plot the PERs
    '''
    # plot_PERs(PER_dev, PER_test)

    '''
    3.1.2 Optimizer Learning rate changing
    '''
    # plot_lr(lrs)
            
    return model_path

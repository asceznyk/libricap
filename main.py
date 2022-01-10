import argparse

import numpy as np

import torch
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import torchaudio

from dataset import *
from model import *
from trainer import Trainer

def main(args):
    torch.manual_seed(7)

    train_dataset = torchaudio.datasets.LIBRISPEECH("/kaggle/working", url="train-clean-100", download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH("/kaggle/working", url="test-clean", download=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {} 
    train_loader = DataLoader(
                dataset=train_dataset, 
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=lambda x: data_preprocess(x, 'train'),
                **kwargs)
    test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=lambda x: data_preprocess(x, 'valid'),
                **kwargs) 

    model = SpeechRecognizer()

    trainer = Trainer(model, train_loader, test_loader, args)
    trainer.fit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--learning_rate', type=int, default=5e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--ckpt_path', default='./state.best',  type=str, help='path of the model to be saved')

    options = parser.parse_args()
    main(options)


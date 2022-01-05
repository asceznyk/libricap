import argparse

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import torchaudio

from dataset import *
from model import *

def main(args):
    torch.manual_seed(7)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset = torchaudio.datasets.LIBRISPEECH("/kaggle/working", url="train-clean-100", download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH("/kaggle/working", url="test-clean", download=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=lambda x: data_preprocess(x, 'train'),
                              **kwargs)

    test_loader = DataLoader(dataset=test_dataset, 
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=lambda x: data_preprocess(x, 'valid'),
                             **kwargs)

    print(next(iter(train_loader)), next(iter(test_loader)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--learning_rate', type=int, default=5e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--save_model_path', default='./state.best',  type=str, help='path of the model to be saved')

    options = parser.parse_args()
    main(options)


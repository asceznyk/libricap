import argparse

import numpy as np

import torch
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

import torchaudio

from dataset import *
from model import *
from trainer import Trainer

def main(args):
    torch.manual_seed(7)

    specs = torch.randn(1, 1, 128, 10)
    labels = torch.Tensor([[1, 28, 3, 2, 6, 8, 9, 22, 27, 8, 0]])
    ils = [10]
    lls = [10]

    #[specs, labels, ils, lls] = torch.load('/kaggle/input/speechsample/train_samp.pt') 
    loader = DataLoader(TensorDataset(specs, labels, torch.Tensor(ils).to(torch.int32), torch.Tensor(lls).to(torch.int32)), 1)

    hparams = {
        'n_res_layers': 3,
        'n_gru_layers': 2,
        'n_class': 29,
        'n_feats': 128,
        'gru_dim': 512 
    }

    model = SpeechRecognizer(**hparams)

    trainer = Trainer(model, loader, None, args)
    trainer.fit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--ckpt_path', default='./state.best',  type=str, help='path of the model to be saved')
       
    options = parser.parse_args()
    main(options)


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

    train_dataset = torch.load('/kaggle/input/speechsample/train_samples.pt') 
    loader = DataLoader(
        dataset=train_dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=lambda x: data_preprocess(x, 'train'))

    hparams = {
        'n_res_layers': 3,
        'n_gru_layers': 5,
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


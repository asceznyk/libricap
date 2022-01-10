import argparse

import numpy as np

import torch
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import torchaudio

from dataset import *
from model import *

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, args): 
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        self.model = model

        self.train_loader = DataLoader(
                dataset=train_dataset, 
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=lambda x: data_preprocess(x, 'train'),
                **kwargs)
        self.test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=lambda x: data_preprocess(x, 'valid'),
                **kwargs) 

        self.args = args
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
 
        self.criterion = nn.CTCLoss(blank=28).to(self.device)
        self.optimizer = optim.AdamW(model.parameters(), args.learning_rate)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.learning_rate, 
            steps_per_epoch=int(len(self.train_loader)),
            epochs=args.epochs,
            anneal_strategy='linear'
        )

        if self.use_cuda:
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)

    def save_checkpoint(self, best_loss):
        print(f"saving.. {self.args.ckpt_path}!")
        torch.save({
            'best_loss':best_loss,
            'model':self.model.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'optimizer':self.optimizer.state_dict(),
        }, self.args.ckpt_path)

    def load_checkpoint(self):
        print(f"loading from checkpoint.. ")
        ckpt = torch.load(args.ckpt_path)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        return ckpt['best_loss']

    def fit(self):
        model, optimizer, scheduler, args = self.model, self.optimizer, self.scheduler, self.args 

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            loader = self.train_loader if is_train else self.test_loader

            avg_loss = 0
            pbar = tqdm(enumerate(loader), total=len(loader)) 
            for i, (spectrograms, labels, input_lengths, label_lengths) in pbar:
                spectrograms = spectrograms.to(self.device)
                labels = labels.to(self.device)
                
                if is_train:
                    optimizer.zero_grad()

                with torch.set_grad_enabled(is_train):
                    outputs = model(spectrograms)
                    outputs = F.log_softmax(outputs, dim=2)
                    loss = criterion(outputs.transpose(0, 1), labels, input_lengths, label_lengths)
                    avg_loss += loss.item() / len(loader)

                if is_train:
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                pbar.set_description(f"epoch {e+1}, iter {i}: {split}, current loss: {loss.item():.3f}, avg loss: {avg_loss}, lr: {args.learning_rate:e}")

            return avg_loss

        best_loss = self.load_checkpoint()
        self.tokens = 0

        for e in range(args.max_epochs):
            train_loss = run_epoch('train')
            test_loss = run_epoch('valid') if self.test_dataset is not None else train_loss

        good_model = self.test_dataset is None or test_loss < best_loss
        if self.args.ckpt_path is not None and good_model:
            best_loss = test_loss
            self.save_checkpoint(best_loss)

def main(args):
    torch.manual_seed(7)

    train_dataset = torchaudio.datasets.LIBRISPEECH("/kaggle/working", url="train-clean-100", download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH("/kaggle/working", url="test-clean", download=True)

    model = SpeechRecognizer()

    trainer = Trainer(model, train_dataset, test_dataset, args)
    trainer.fit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--learning_rate', type=int, default=5e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--ckpt_path', default='./state.best',  type=str, help='path of the model to be saved')

    options = parser.parse_args()
    main(options)


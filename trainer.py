import os

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

class Trainer:
    def __init__(self, model, train_loader, test_loader, args): 
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
        self.criterion = nn.CTCLoss(blank=28).to(self.device)
        self.optimizer = optim.AdamW(model.parameters(), args.learning_rate)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=args.learning_rate, 
            steps_per_epoch=int(len(self.train_loader)),
            epochs=args.epochs,
            anneal_strategy='linear'
        )

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
        ckpt = torch.load(self.args.ckpt_path)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        return ckpt['best_loss']

    def fit(self):
        model, optimizer, scheduler, criterion, args = self.model, self.optimizer, self.scheduler, self.criterion, self.args 

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

                pbar.set_description(f"epoch: {e+1}, iter {i}, {split}, current loss: {loss.item():.3f}, avg loss: {avg_loss:.2f}, lr: {args.learning_rate:e}")

            return avg_loss

        best_loss = self.load_checkpoint() if os.path.exists(args.ckpt_path) else float('inf') 
        self.tokens = 0

        for e in range(args.epochs):
            train_loss = run_epoch('train')
            test_loss = run_epoch('valid') if self.test_loader is not None else train_loss

        good_model = self.test_loader is None or test_loss < best_loss
        if self.args.ckpt_path is not None and good_model:
            best_loss = test_loss
            self.save_checkpoint(best_loss)


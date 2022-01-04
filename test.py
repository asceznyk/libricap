import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import torchaudio

from dataset import *

train_dataset = torchaudio.datasets.LIBRISPEECH("/kaggle/working", url="train-clean-100", download=True)
test_dataset = torchaudio.datasets.LIBRISPEECH("/kaggle/working", url="test-clean", download=True)

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=8,
                          shuffle=True,
                          collate_fn=lambda x: data_preprocess(x, 'train'))

print(next(iter(train_loader)))

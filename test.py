import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import torchaudio

train_dataset = torchaudio.datasets.LIBRISPEECH("./", url="train-clean-100", download=True)
test_dataset = torchaudio.datasets.LIBRISPEECH("./", url="test-clean", download=True)

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=8,
                          shuffle=True,
                          collate_fn=lambda x: data_preprocess(x, 'train'))

print(next(train_loader))

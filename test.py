import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import torchaudio

from model import *

torch.manual_seed(7)

x = torch.randn(8, 1, 128, 1000)
n_feats = 128 // 2
gru_dim = 512
cnn = nn.Conv2d(1, 32, 3, stride=2, padding=3//2)
rescnn = ResCNN(32, 32, kernel=3, stride=1, dropout=0.5, n_feats=n_feats)
fc = nn.Linear(32*n_feats, gru_dim)
bigru = BiGRU(gru_dim, 256, 0.5, batch_first=True)

y = cnn(x)
y = rescnn(y)
size = y.size()
print(size)
y = y.view(size[0], size[1] * size[2], size[3])
y = fc(y.transpose(1,2))
y = bigru(y)

print(x, x.size())
print(y, y.size())



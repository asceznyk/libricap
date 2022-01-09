import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import torchaudio

from model import *

torch.manual_seed(7)

x = torch.randn(8, 1, 128, 1024)
cnn = nn.Conv2d(1, 32, 3, stride=1, padding=3//2)
rescnn = ResCNN(32, 32, 3, 1, 0.5, 128)
bigru = BiGRU(128, 256, 0.5, batch_first=True)

y = cnn(x)
y = rescnn(y)
size = y.size()
y = y.view(size[0], size[1] * size[3], size[2])
y = bigru(y)

print(x, x.size())
print(y, y.size())



import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import torchaudio

from model import *

torch.manual_seed(7)

x = torch.randn(8, 1, 128, 1024)
cnn = nn.Conv2d(1, 32, 3, stride=2, padding=3//2)
rescnn = ResCNN(32, 32, 3, 2, 0.5, 128)
y = cnn(x)
y = rescnn(y) 

print(x)
print(y)


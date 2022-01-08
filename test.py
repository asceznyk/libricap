import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import torchaudio

from model import *

torch.manual_seed(7)

x = torch.randn(8, 1, 128, 1024)
fln = FeatureLayerNorm(128)
y = fln(x) 

print(x)
print(y)


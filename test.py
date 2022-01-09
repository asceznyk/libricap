import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import torchaudio

from model import *

torch.manual_seed(7)

n_feats = 128
gru_dim = 512
x = torch.randn(8, 1, n_feats, 1000)
speech_model = SpeechRecognizer(3, 5, 29, n_feats, gru_dim, 0.1, 2)
y = speech_model(x)

print(x, x.size())
print(y, y.size())



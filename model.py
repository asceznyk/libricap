import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureLayerNorm(nn.Module):
    def __init__(self, n_feats):
        super(FeatureLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(2,3) #.contiguous() ## x.shape (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2,3) #.contiguous() ## x.shape (batch, channel, feature, time)

class ResCNN(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, dropout, n_feats):
        super(ResCNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_c, out_c, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_c, out_c, kernel, stride, padding=kernel//2)
        self.layer_norm1 = FeatureLayerNorm(n_feats)
        self.layer_norm2 = FeatureLayerNorm(n_feats)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        res = x

        print(res)
        x = self.layer_norm1(x) 
        print(res)

        x = self.dropout1(F.gelu(x))
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = self.dropout2(F.gelu(x))
        x = self.cnn2(x)

        return x + res





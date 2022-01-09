import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureLayerNorm(nn.Module):
    def __init__(self, n_feats):
        super(FeatureLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(2,3).contiguous() ## x.shape (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2,3).contiguous() ## x.shape (batch, channel, feature, time)

class ResCNN(nn.Module):
    def __init__(self, inp_c, out_c, kernel, stride, dropout, n_feats):
        super(ResCNN, self).__init__()
        self.cnn1 = nn.Conv2d(inp_c, out_c, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_c, out_c, kernel, stride, padding=kernel//2)
        self.layer_norm1 = FeatureLayerNorm(n_feats)
        self.layer_norm2 = FeatureLayerNorm(n_feats)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.layer_norm1(x) 
        x = self.dropout1(F.gelu(x))
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = self.dropout2(F.gelu(x))
        x = self.cnn2(x)
        return x + res ## x.shape (batch, channel, feature, time)

class BiGRU(nn.Module):
    def __init__(self, gru_dim, hidden_size, dropout, batch_first):
        super(BiGRU, self).__init__()
        self.bigru = nn.GRU(gru_dim, hidden_size, 
                          num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(gru_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.bigru(x)
        return self.dropout(x)



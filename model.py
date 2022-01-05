import numpy as np

import torch
import torch.nn as nn

class FeatureLayerNorm(nn.Module):
    def __init__(self, n_feats):
        super(FeatureLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(2,3).contiguous() ## x.shape (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2,3).contiguous() ## x.shape (batch, channel, feature, time)



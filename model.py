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

class SpeechRecognizer(nn.Module):
    def __init__(self, n_res_layers, n_gru_layers, n_class, n_feats, gru_dim, dropout, stride=2):
        super(SpeechRecognizer, self).__init__()
        #n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)
        self.rescnns = nn.Sequential(*[ResCNN(32, 32, kernel=3, stride=1, 
                                              dropout=dropout, n_feats=n_feats) 
                                      for _ in range(n_res_layers)])
        self.fc = nn.Linear(n_feats*32, gru_dim)
        self.bigrus = nn.Sequential(*[BiGRU(gru_dim if i==0 else gru_dim*2, gru_dim, 
                                            dropout=dropout, batch_first=i==0) 
                                     for i in range(n_gru_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(gru_dim*2, gru_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x) 
        x = self.rescnns(x) ## (batch, channel, feature, time)
        xsize = x.size()
        x = x.view(xsize[0], xsize[1] * xsize[2], xsize[3]) ## (batch, feature, time)
        x = self.fc(x.transpose(1,2)) ## (batch, time, feature)
        x = self.bigrus(x) ## (batch, time, 2*feature)
        return self.classifier(x) ##(batch, time, classes)



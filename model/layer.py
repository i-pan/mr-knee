"""
Different NN Layers.
"""
import torch
from torch import nn

class FCLayer(nn.Module):
    def __init__(self, in_features, num_classes, dropout=None): 
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else: 
            self.dropout = None

    def forward(self, x): 
        if self.dropout is not None: 
            return self.fc(self.dropout(x)) 
        else:
            return self.fc(x)

class RegressionFCLayer(nn.Module): 
    def __init__(self, in_features, num_classes, dropout=None): 
        super(RegressionFCLayer, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else: 
            self.dropout = None

    def forward(self, x): 
        if self.dropout is not None: 
            return torch.sigmoid(self.fc(self.dropout(x))) 
        else:
            return torch.sigmoid(self.fc(x))


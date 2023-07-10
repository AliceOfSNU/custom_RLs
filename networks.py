import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from helpers import *

device = torch.device("cpu")

class SharedNetwork(nn.Module):
    def __init__(self, state_dims:int):
        super(SharedNetwork, self).__init__()
        self.state_dims = state_dims
        self.hidden_dims = [64, 64, ]
        dims = [state_dims,] + self.hidden_dims
        self.gate = F.tanh
        self.layers = nn.ModuleList([layer_init(nn.Linear(in_feature, out_feature), 1e-3) for (in_feature, out_feature) in zip(dims[:-1], dims[1:])])
        self.out_dims = dims[-1]


    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x
    

class ActorNetwork(nn.Module):
    def __init__(self, feature_dims:int, action_dims = int):
        super(ActorNetwork, self).__init__()
        self.feature_dims = feature_dims
        self.action_dims = action_dims
        self.layers = nn.ModuleList([layer_init(nn.Linear(feature_dims, 32), 1e-3),  layer_init(nn.Linear(32, action_dims), 1e-3)])
        self.gate = F.tanh

    
    def forward(self, x):
        #for layer in self.layers:
        #    x = layer(x)
        x = self.gate(self.layers[0](x))
        return self.layers[1](x)
    
class CriticNetwork(nn.Module):
    def __init__(self, feature_dims:int):
        super(CriticNetwork, self).__init__()
        self.feature_dims = feature_dims
        self.value_dims = 1
        self.layers = nn.ModuleList([layer_init(nn.Linear(feature_dims, 32), 1e-3),  layer_init(nn.Linear(32, 1), 1e-3)])
        self.gate = F.tanh
    
    def forward(self, x):
        #for layer in self.layers:
        #    x = layer(x)

        x = self.gate(self.layers[0](x))
        return self.layers[1](x)
        return x



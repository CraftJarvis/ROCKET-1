from typing import Dict
import torch
import torch.nn as nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class LinearFiLMLayer(nn.Module):
        
    def __init__(
        self, 
        num_filters: int,
        cond_dim: int,
        **unused_kwargs
    ):
        super().__init__()
        self.num_filters = num_filters
        self.cond_dim = cond_dim
        self.fc = layer_init(nn.Linear(cond_dim, 2 * num_filters))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, cond_dim)
        # x: (B, num_filters, H, W)
        assert len(cond.shape) == 2 and len(x.shape) == 4
        gamma_beta = self.fc(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.view(-1, self.num_filters, 1, 1)  
        beta = beta.view(-1, self.num_filters, 1, 1)
        out = (1 + gamma) * x + beta
        return out
        
import random
import numpy as np
import torch
from torch import nn
from rich.console import Console
from typing import Dict, Optional, Union, List, Any, Tuple
from einops import rearrange, repeat
from rocket.arm.models.utils import FeedForward, ModalInput

class SymLog(nn.Module):
    def forward(self, x):
        return torch.sign(x) * torch.log(1 + torch.abs(x))

class Scalar(nn.Module):
    
    def __init__(self, hidsize: int, n_layer: int = 2, symlog: bool = False, **kwargs) -> None:
        super().__init__()
        self.preprocess = SymLog() if symlog else nn.Identity()
        self.updim = nn.Linear(1, hidsize)
        self.feedforwards = nn.ModuleList([
            FeedForward(hidsize, mult=2, **kwargs) for i in range(n_layer)
        ])
    
    def forward(self, returns: torch.Tensor, **kwargs) -> ModalInput:
        """
        :params scalar_feats: (B, ) indicates returns or rewards. 
        :returns: (B, 1, C) where C is the hidden size. 
        """
        x = rearrange(returns, 'b -> b 1')
        is_padding = torch.zeros_like(x).to(torch.bool)
        x = self.preprocess(x)
        x = self.updim(x)
        for ffn in self.feedforwards:
            x = ffn(x) + x
        x = rearrange(x, 'b c -> b 1 c')
        return {
            "tokens": x, 
            "is_padding": is_padding
        }

if __name__ == '__main__':
    B = 4
    scalar_module = Scalar(128, n_layer=2, symlog=True) 
    scalar_feats  = torch.randn((B, 1))
    output = scalar_module(scalar_feats)
    print(output.shape)
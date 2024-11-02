'''
author:        phython96 <1744260356@qq.com>
date:          2024-03-23 22:28:37
Copyright © Team CraftJarvis All rights reserved
'''

import numpy as np
import torch
from torch import nn
from torchvision import transforms as T
from einops import rearrange
from torchvision.models import swin_t
from typing import Dict, Optional, Union, List, Any, Tuple
from einops import rearrange

class CustomSwinTransformer(nn.Module):
    
    def __init__(self, weights: str = "IMAGENET1K_V1", out_dim: int = 1024, **kwargs):
        super().__init__()
        self.vision_encoder = swin_t(weights=weights)
        self.final_layer = nn.Linear(768, out_dim)
        self.transform = T.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
    
    def forward(self, imgs, **kwargs):
        imgs = imgs / 255.
        B, T = imgs.shape[:2]
        if imgs.shape[-1] == 3:
            x = rearrange(imgs, 'b t h w c -> (b t) c h w')
        else:
            x = rearrange(imgs, 'b t c h w -> (b t) c h w')
        x = self.transform(x)
        x = self.vision_encoder.features(x)
        x = self.vision_encoder.norm(x) # BxT, 7, 7, 768
        x = self.final_layer(x) 
        x = rearrange(x, '(B T) H W C -> B T C H W', B=B, T=T)
        return x

if __name__ == '__main__':
    model = CustomSwinTransformer().to("cuda")
    B, T = 4, 128
    example = torch.rand(B, T, 3, 224, 224).to("cuda")
    
    output = model(example)
    
    print(f"{output.shape = }")
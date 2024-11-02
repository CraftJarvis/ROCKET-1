'''
author:        caishaofei <1744260356@qq.com>
date:          2024-03-23 21:29:37
Copyright © Team CraftJarvis All rights reserved
'''
import numpy as np
import torch
from torch import nn
from einops import rearrange, repeat
from typing import Dict, Optional, Union, List, Any, Tuple
from rocket.arm.utils.efficientnet_lib import EfficientNet

class CustomEfficientNet(nn.Module):
    
    def __init__(
        self, 
        version: str, 
        resolution: int = 224, 
        out_dim: int = 1024, 
        pooling: bool = False, 
        accept_segment: bool = False,
        **kwargs, 
    ) -> None:
        super().__init__()
        self.version = version
        self.resoulution = resolution
        self.out_dim = out_dim
        
        if accept_segment:
            in_channels = 4
        else:
            in_channels = 3
        self.model = EfficientNet.from_pretrained(version, in_channels=in_channels)
        
        if 'b0' in version:
            self.mid_dim = 1280
        elif 'b4' in version:
            self.mid_dim = 1792
        
        if resolution == 360:
            self.feat_reso = (11, 11)
        elif resolution == 224:
            self.feat_reso = (7, 7)
        elif resolution == 128:
            self.feat_reso = (4, 4)

        self.final_layer = nn.Sequential(
            nn.GELU(), 
            nn.Conv2d(self.mid_dim, out_dim, 1)
        )
        
        if pooling:
            self.pooling_layer = nn.AdaptiveAvgPool2d(1)

    def forward(self, imgs, segment=None, **kwargs): 
        '''
        :params imgs: shape of (B, T, 3, H, W)
        :returns: shape of (B, T, C, R, R)
        '''
        imgs = imgs / 255.
        B, T = imgs.shape[:2]
        if imgs.shape[-1] == 3:
            x = rearrange(imgs, 'b t h w c -> (b t) c h w')
        else:
            x = rearrange(imgs, 'b t c h w -> (b t) c h w')
        if segment is not None:
            segment = segment / 255.
            y = rearrange(segment, 'b t h w -> (b t) 1 h w')
            x = torch.cat([x, y], dim=1)
        x = self.model.extract_features(x)
        if hasattr(self, 'pooling_layer'):
            x = self.pooling_layer(x)
            x = self.final_layer(x)
            x = rearrange(x, '(b t) c 1 1 -> b t c', b=B, t=T)
        else:
            x = self.final_layer(x)
            x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
        return x

if __name__ == '__main__':
    model = CustomEfficientNet(
        version='efficientnet-b0', 
        resolution=128, 
        out_dim=1024, 
        pooling=False,
    ).to("cuda")
    B, T = 4, 128
    example = torch.rand(B, T, 3, 128, 128).to("cuda")
    
    output = model(example)
    
    print(f"{output.shape = }")
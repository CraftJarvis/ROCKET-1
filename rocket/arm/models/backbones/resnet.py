'''
author:        caishaofei <1744260356@qq.com>
date:          2024-03-23 22:15:55
Copyright © Team CraftJarvis All rights reserved
'''

import numpy as np
import torch
from torch import nn
from einops import rearrange, repeat
import torchvision
from torchvision import transforms as T
from typing import Dict, Optional, Union, List, Any, Tuple

class CustomResNet(nn.Module):
    
    def __init__(self, version: str = '18', out_dim: int = 1024, pooling: bool = False, accept_segment: bool = False, **kwargs):
        super().__init__()
        if version == '18':
            self.model = torchvision.models.resnet18(pretrained=True)
            feature_dim = 512
        elif version == '50':
            self.model = torchvision.models.resnet50(pretrained=True)
            feature_dim = 2048
        elif version == '101':
            self.model = torchvision.models.resnet101(pretrained=True)
            feature_dim = 2048
        
        if accept_segment:
            ori_first_conv = self.model.conv1
            new_first_conv = nn.Conv2d(
                in_channels=4, 
                out_channels=ori_first_conv.out_channels, 
                kernel_size=ori_first_conv.kernel_size, 
                stride=ori_first_conv.stride, 
                padding=ori_first_conv.padding, 
                dilation=ori_first_conv.dilation,
                groups=ori_first_conv.groups,
                bias=ori_first_conv.bias,
            )
            new_first_conv.weight.data[:, :3] = ori_first_conv.weight.data
            new_first_conv.weight.data[:, 3] = 0.
            self.model.conv1 = new_first_conv
        
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.final_layer = nn.Sequential(
            nn.GELU(), 
            nn.Conv2d(feature_dim, out_dim, 1)
        )
        if pooling:
            self.pooling_layer = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, imgs, segment=None, **kwargs):
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
        x = self.model(x)
        if hasattr(self, 'pooling_layer'):
            x = self.pooling_layer(x)
            x = self.final_layer(x)
            x = rearrange(x, '(b t) c 1 1 -> b t c', b=B, t=T)
        else:
            x = self.final_layer(x)
            x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
        return x


if __name__ == '__main__':
    model = CustomResNet(
        version='50', 
        out_dim=1024, 
        pooling=False,
        accept_segment=True
    ).to("cuda")
    
    B, T = 4, 128
    example = torch.rand(B, T, 3, 224, 224).to("cuda")
    segment = torch.rand(B, T, 224, 224).to("cuda")
    output = model(imgs=example, segment=segment)
    
    print(f"{output.shape = }")
    
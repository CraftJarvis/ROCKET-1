'''
author:        caishaofei <1744260356@qq.com>
date:          2024-03-23 22:28:37
Copyright © Team CraftJarvis All rights reserved
'''
import numpy as np
import torch
from torch import nn
from torchvision import transforms as T
from einops import rearrange
from typing import Dict, Optional, Union, List, Any, Tuple
from transformers import CLIPVisionModel
from transformers import logging
logging.set_verbosity_error()

class CustomCLIPv(nn.Module):
    
    def __init__(
        self, 
        version: str = "openai/clip-vit-base-patch32", 
        out_dim: int = 1024, 
        freeze: bool = False, 
        accept_segment: bool = False, 
        **kwargs
    ):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(version)
        if accept_segment:
            #! begin of: override the first conv layer to accept 4 channels
            ori_first_conv = self.vision_encoder.vision_model.embeddings.patch_embedding
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
            new_first_conv.weight.data[:, :3] = ori_first_conv.weight.data # copy the weights of the first 3 channels
            new_first_conv.weight.data[:, 3] = 0. # zero conv for the 4th channel
            self.vision_encoder.vision_model.embeddings.patch_embedding = new_first_conv
        self.final_layer = nn.Linear(self.vision_encoder.config.hidden_size, out_dim)
        self.transform = T.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
        if freeze:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
    
    def forward(self, imgs, segment=None, **kwargs):
        B, T = imgs.shape[:2]
        imgs = imgs / 255.
        if imgs.shape[-1] == 3:
            x = rearrange(imgs, 'b t h w c -> (b t) c h w')
        else:
            x = rearrange(imgs, 'b t c h w -> (b t) c h w')
        x = self.transform(x)
        if segment is not None:
            segment = segment / 255.
            y = rearrange(segment, 'b t h w -> (b t) 1 h w')
            x = torch.cat([x, y], dim=1)
        x = self.vision_encoder(x).last_hidden_state
        x = self.final_layer(x) # x: (B*T, num_tokens, out_dim)
        x = x[:, 1:, :] # remove the CLS token
        r = int(np.sqrt(x.shape[1]))
        assert r * r == x.shape[1], f"the number of tokens should be a square number, but got {x.shape[1]}"
        x = rearrange(x, '(b t) (h w) c -> b t c h w', b=B, t=T, h=r, w=r)
        return x

if __name__ == '__main__':
    model = CustomCLIPv().to("cuda")
    B, T = 4, 128
    example = torch.rand(B, T, 224, 224, 3).to("cuda")
    
    output = model(example)
    
    print(f"{output.shape = }")
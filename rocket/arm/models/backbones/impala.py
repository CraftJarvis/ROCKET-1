'''
author:        caishaofei <1744260356@qq.com>
date:          2024-03-23 22:11:54
Copyright © Team CraftJarvis All rights reserved
'''

import numpy as np
import torch
from torch import nn
from typing import Dict, Optional, Union, List, Any, Tuple
from einops import rearrange, repeat
from rocket.arm.utils.vpt_lib.impala_cnn import ImpalaCNN
from rocket.arm.utils.vpt_lib.util import FanInInitReLULayer


class ImgObsProcess(nn.Module):
    """ImpalaCNN followed by a linear layer.

    :param cnn_outsize: impala output dimension
    :param output_size: output size of the linear layer.
    :param dense_init_norm_kwargs: kwargs for linear FanInInitReLULayer
    :param init_norm_kwargs: kwargs for 2d and 3d conv FanInInitReLULayer
    """

    def __init__(
        self,
        cnn_outsize: int,
        output_size: int,
        dense_init_norm_kwargs: Dict = {},
        init_norm_kwargs: Dict = {},
        pooling: bool = True, 
        **kwargs,
    ):
        super().__init__()
        self.cnn = ImpalaCNN(
            outsize=cnn_outsize,
            init_norm_kwargs=init_norm_kwargs,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            pooling=pooling, 
            **kwargs,
        )
        self.linear = FanInInitReLULayer(
            cnn_outsize,
            output_size,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )
        self.pooling = pooling

    def forward(self, imgs, cond=None, **kwargs):
        imgs = imgs / 255.
        if imgs.shape[-1] != 3:
            imgs = rearrange(imgs, 'b t c h w -> b t h w c')
        x = self.cnn(imgs, cond=cond)
        x = self.linear(x)
        if not self.pooling:
            x = rearrange(x, 'b t h w c -> b t c h w')
        return x

if __name__ == '__main__':
    
    kwargs = {
        'name': 'IMPALA', 
        'img_shape': [224, 224, 3], 
        'impala_chans': [16, 32, 32, 64], 
        'impala_kwargs':{
            'post_pool_groups': 1, 
            'pooling': True # do not pool in the last layer
        }, 
        'impala_width': 4, 
    }
    
    first_conv_norm = False
    impala_kwargs = kwargs.get('impala_kwargs', {})
    init_norm_kwargs = kwargs.get('init_norm_kwargs', {})
    dense_init_norm_kwargs = kwargs.get('dense_init_norm_kwargs', {})
    
    model = ImgObsProcess(
        cnn_outsize=256,
        output_size=1024,
        inshape=kwargs['img_shape'],
        chans=tuple(int(kwargs['impala_width'] * c) for c in kwargs['impala_chans']),
        nblock=2,
        dense_init_norm_kwargs=dense_init_norm_kwargs,
        init_norm_kwargs=init_norm_kwargs,
        first_conv_norm=first_conv_norm,
        **impala_kwargs, 
    ).to('cuda')

    B, T = 1, 128
    example = torch.rand(B, T, 3, 224, 224).to("cuda")
    
    output = model(example)
    
    print(f"{output.shape = }")
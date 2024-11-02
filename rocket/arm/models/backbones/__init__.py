'''
author:        caishaofei-mus1 <1744260356@qq.com>
date:          2023-08-17 12:15:31
Copyright © Team CraftJarvis All rights reserved
'''


import numpy as np
import torch
from torch import nn
from torchvision import transforms as T
from functools import partial
from typing import Dict, Optional, Union, List, Any, Tuple
from einops import rearrange

from rocket.arm.models.backbones.state import DictStateEncoder
from rocket.arm.models.backbones.clipv import CustomCLIPv
from rocket.arm.models.backbones.swin_transformer import CustomSwinTransformer
from rocket.arm.models.backbones.impala import ImgObsProcess
from rocket.arm.models.backbones.resnet import CustomResNet
from rocket.arm.models.backbones.efficient import CustomEfficientNet
from rocket.arm.models.backbones.nfnet import CustomNFNet
from rocket.arm.models.backbones.dinosiglip_vit import CustomDinoSigLIP

# def general_preprocessor(
#     image_tensor: torch.Tensor, 
#     normalize: bool = True, 
#     channel_last: bool = False,
#     image_shape: Optional[Tuple[int, ...]] = None,
#     image_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
#     image_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
# ) -> torch.Tensor:
#     if image_tensor.dtype == torch.uint8:
#         image_tensor = image_tensor.to(torch.float32)

#     assert image_tensor.dim() == 5, f"image_tensor shape: {image_tensor.shape}, should be 5 dims"
    
#     if image_tensor.shape[-1] == 3:
#         image_tensor = rearrange(image_tensor, 'B T H W C -> B T C H W')

#     if image_shape is not None:
#         H, W, C = image_shape
#         assert image_tensor.shape[-3] == C, "Input obs channel dim does not match."
#         assert image_tensor.shape[-2:] == (H, W), "Input obs shape does not match pre-training shape."

#     if normalize:
#         transform_list.append(T.Normalize(image_mean, image_std))
    
#     transform = T.Compose(transform_list)
#     x = rearrange(image_tensor, 'B T C H W -> (B T) C H W')
#     x = transform(x)
#     processed_images = rearrange(x, '(B T) C H W -> B T C H W', B=image_tensor.shape[0])
    
#     if channel_last:
#         processed_images = rearrange(processed_images, 'B T C H W -> B T H W C')
    
#     return processed_images

def build_backbone(name: str = 'IMPALA', **kwargs) -> Dict:
    
    result_modules = {}
    if name == 'IMPALA':
        first_conv_norm = False
        impala_kwargs = kwargs.get('impala_kwargs', {})
        init_norm_kwargs = kwargs.get('init_norm_kwargs', {})
        dense_init_norm_kwargs = kwargs.get('dense_init_norm_kwargs', {})
        result_modules['obsprocessing'] = ImgObsProcess(
            cnn_outsize=kwargs.get('cnn_outsize', 256),
            output_size=kwargs['hidsize'],
            inshape=kwargs['img_shape'],
            chans=tuple(int(kwargs['impala_width'] * c) for c in kwargs['impala_chans']),
            nblock=2,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            init_norm_kwargs=init_norm_kwargs,
            first_conv_norm=first_conv_norm,
            **impala_kwargs, 
        )
        if (film_kwargs := impala_kwargs.get('film_kwargs')) is not None:
            result_modules['uncond_embedding'] = nn.Embedding(1, film_kwargs["cond_dim"])
        
    elif name == 'CLIPv':
        model = CustomCLIPv(
            out_dim=kwargs['hidsize'],
            **kwargs,
        )
        result_modules['obsprocessing'] = model

    elif name == 'DINO_SIGLIP':
        model = CustomDinoSigLIP(
            out_dim=kwargs['hidsize'],
            **kwargs,
        )
        result_modules['obsprocessing'] = model

    elif name == 'SWIN':
        model = CustomSwinTransformer(
            out_dim=kwargs['hidsize'],
            **kwargs, 
        )
        result_modules['obsprocessing'] = model

    elif name == 'NFNET':
        model = CustomNFNet(
            out_dim=kwargs['hidsize'],
            **kwargs, 
        )
        result_modules['obsprocessing'] = model
    
    elif name == 'EFFICIENTNET':
        model = CustomEfficientNet(
            out_dim=kwargs['hidsize'],
            **kwargs, 
        )
        result_modules['obsprocessing'] = model
        
    elif name == 'RESNET':
        result_modules['obsprocessing'] = CustomResNet(
            out_dim=kwargs['hidsize'],
            **kwargs, 
        )
    return result_modules

def build_state_backbones(
    state_space: Dict[str, Any], hidsize: int, **kwargs
):
    return DictStateEncoder(state_space=state_space, hidsize=hidsize)

if __name__ == '__main__':
    pass 
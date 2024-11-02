import random
import numpy as np
import torch
from torch import nn
from typing import Dict, Optional, Union, List, Any, Tuple, Literal
from rich.console import Console
from einops import rearrange, repeat

from rocket.arm.utils.transformers import GPTConfig, GPT
from rocket.arm.models.utils import FeedForward, ModalInput

class Identity(nn.Module):
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x

class AttentionPooling(nn.Module):
    
    def __init__(self, hidsize: int, **transformer_kwargs) -> None:
        super().__init__()
        gpt_config = GPTConfig(
            bias = True,
            is_causal = False,
            is_ordered = True,
            n_embd = hidsize,
            ** transformer_kwargs
        )
        self.transformer = GPT(gpt_config)
    
    def forward(self, vision_feats: torch.Tensor, latent: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        :params vision_feats: (B, S, C), where S is the spatial dimension. 
        :params latent: (B, 1, C), the condition latent vector, optional to be used. 
        :return: (B, 1, C), the fused vision features.
        """
        assert len(vision_feats.shape) == 3
        if latent is not None:
            x = torch.cat([latent, vision_feats], dim=1)
        else:
            x = vision_feats
        x = self.transformer(x).mean(1)
        x = rearrange(x, 'b c -> b 1 c')
        return x

class MeanPooling(nn.Module):
    
    def __init__(self, hidsize: int, **kwargs) -> None:
        super().__init__()
        self.ffn = FeedForward(hidsize, mult=2)
    
    def forward(self, vision_feats: torch.Tensor, latent: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        :params vision_feats: (B, S, C), where S is the spatial dimension. 
        :return: (B, 1, C), the fused vision features.
        """
        assert len(vision_feats.shape) == 3, "Spatial dimension is required."
        assert latent is None, "Mean pooling method does not support latent input."
        x = vision_feats
        x = self.ffn(x) + x
        x = x.mean(dim=1)
        x = rearrange(x, 'b c -> b 1 c')
        return x

def make_pooling_layer(name: str, *args, **kwargs):
    if name == None:
        return None
    elif name == 'identity': 
        layer = Identity()
    elif name == 'attention':
        layer = AttentionPooling(*args, **kwargs)
    elif name == 'mean':
        layer = MeanPooling(*args, **kwargs)
    return layer

class Image(nn.Module):
    
    def __init__(
        self, 
        hidsize: int, 
        select: str = ':', 
        pooling: Literal['identity', 'attention', 'mean'] = 'identity', 
        aggregate: Optional[Literal['identity', 'attention', 'mean']] = None, 
        **pooling_kwargs, 
    ) -> None:
        """
        :params select: the slice index for the vision features, should result in a sequence not an element. 
                        e.g. select='0:2' is valid, select='0' is invalid. 
        :params pooling: whether to apply the spatial pooling. It can be `true` only when vision features is 5D. 
        """
        super().__init__()
        self.select = select
        self.pooling_layer = make_pooling_layer(pooling, hidsize=hidsize, **pooling_kwargs)
        self.aggregate_layer = make_pooling_layer(aggregate, hidsize=hidsize, **pooling_kwargs)
        self.post_ffn = FeedForward(hidsize, mult=2)
    
    def forward(self, vision_feats: torch.Tensor, ob_is_padding: torch.Tensor, latent: Optional[torch.Tensor] = None, **kwargs) -> ModalInput:
        """
        slice and squeeze the vision features. 
        :params vision_feats: (B, T, C, H, W) or (B, T, C) depends on the backbone. 
        :params latent: (B, 1, C), the condition latent vector.
        :return: (B, t, C), t=1 or t=T for squeeze=True, otherwise t=HxW or t=TxHxW. 
                 the apperance of T is determined by the select parameter. 
        """
        x = eval(f"vision_feats[:, {self.select}, ...]")
        ob_is_padding = eval(f"ob_is_padding[:, {self.select}, ...]")
        B, T = x.shape[:2]
        assert len(x.shape) == len(vision_feats.shape)
        if len(x.shape) == 5:
            x = rearrange(x, 'b t c h w -> (b t) (h w) c')
        elif len(x.shape) == 3:
            x = rearrange(x, 'b t c -> (b t) 1 c')
        if latent is not None:
            latent = repeat(latent, 'b n c -> (b t) n c', t=T)
        x = self.pooling_layer(x, latent=latent)
        x = self.post_ffn(x) + x
        is_padding = repeat(ob_is_padding, 'b t -> b (t m)', m=x.shape[1])
        x = rearrange(x, '(b t) m c -> b (t m) c', b=B, t=T)
        if self.aggregate_layer is not None:
            x = self.aggregate_layer(x)
            is_padding = torch.zeros_like(x[..., 0]).bool()
        return {
            "tokens": x,
            "is_padding": is_padding
        }

if __name__ == '__main__':
    B, T, C, H, W = 4, 16, 88, 7, 7
    vision_feats = torch.randn(B, T, C, H, W).to("cuda")
    
    print(f"{vision_feats.shape = }")
    
    image_module = Image(hidsize=88, select=':', squeeze=False).to("cuda")
    output = image_module(vision_feats)
    print(f"{output.shape = }")
    
    image_module = Image(hidsize=88, select=':', squeeze=True).to("cuda")
    output = image_module(vision_feats)
    print(f"{output.shape = }")
    
    image_module = Image(hidsize=88, select=':1', squeeze=False).to("cuda")
    output = image_module(vision_feats)
    print(f"{output.shape = }")
    
    image_module = Image(hidsize=88, select=':1', squeeze=True).to("cuda")
    output = image_module(vision_feats)
    print(f"{output.shape = }")
    
    vision_feats = torch.randn(B, T, C).to("cuda")
    print(f"{vision_feats.shape = }")
    
    image_module = Image(hidsize=88, select=':', squeeze=False).to("cuda")
    output = image_module(vision_feats)
    print(f"{output.shape = }")
    
    image_module = Image(hidsize=88, select=':1', squeeze=False).to("cuda")
    output = image_module(vision_feats)
    print(f"{output.shape = }")
    
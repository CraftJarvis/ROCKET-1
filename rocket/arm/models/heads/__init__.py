
import random
from functools import partial
from typing import Dict, Optional, Union, List, Any
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from rocket.arm.models.heads.recon_head import MinecraftReconHead, NLLReconHead

class BaseHead(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(self, latents: Dict[str, torch.Tensor], **kwargs) -> Any:
        '''
        Predict auxiliary task results based on the latents. 
        The returned results will be feed to the loss function as the `pred` term. 
        '''
        raise NotImplementedError
    
    def loss(self, obs, pred, mask=None, **kwargs) -> Any:
        '''
        `obs` terms refers to the original info that sampled from the dataset. 
        `pred` terms refers to the predicted results from the forward function. 
        You are supposed to return metric dict in this function. 
        '''
        raise NotImplementedError

def make_minecraft_recon_head(**kwargs) -> nn.Module:
    return MinecraftReconHead(**kwargs)

def make_nll_recon_head(**kwargs) -> nn.Module:
    return NLLReconHead(**kwargs)

register_heads = {
    'minecraft_recon_head': make_minecraft_recon_head,
    'nll_recon_head': make_nll_recon_head,
}

def build_auxiliary_heads(auxiliary_head_kwargs: List[Dict], **parent_kwargs) -> Dict[str, nn.Module]:
    
    auxilary_heads_dict = {}
    
    for head_kwargs in auxiliary_head_kwargs:
        if not head_kwargs['enable']:
            continue
        
        head_name = head_kwargs.pop('name')
        alias_name = head_kwargs.pop('alias')
        # support alias_name as a list of heads
        if isinstance(alias_name, str):
            alias_name = [alias_name]
        for name in alias_name:
            _head_kwargs = head_kwargs.copy()
            _head_kwargs['alias'] = name
            auxilary_heads_dict[name] = register_heads[head_name](**_head_kwargs, **parent_kwargs)
    
    return nn.ModuleDict(auxilary_heads_dict)


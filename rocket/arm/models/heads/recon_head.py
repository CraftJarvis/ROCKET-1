'''
author:        caishaofei <1744260356@qq.com>
date:          2024-01-30 14:52:05
Copyright © Team CraftJarvis All rights reserved
'''

import random
from typing import Dict, Literal, Optional, Union, List, Any, Literal
from rich.console import Console

import numpy as np
import torch
from torch import nn

class MinecraftReconHead(nn.Module):
    
    def __init__(
        self, 
        alias: str = "minecraft", 
        weight: float = 1.0, 
        reduction: Literal['sum', 'mean'] = 'sum', 
        # entropy_weight: float = 0.0, 
        **kwargs
    ) -> None:
        super().__init__()
        self.alias = alias
        self.weight = weight
        self.reduction = reduction
        # self.entropy_weight = entropy_weight
    
    def forward(self, latents, useful_heads, **kwargs):
        return {
            'pi_head': useful_heads['pi_head'],
            'pi_latent': latents['pi_latent'],
        }
    
    def loss(self, obs, pred, mask=None, **kwargs):
        pi_head = pred['pi_head']
        pi_logits = pi_head(pred['pi_latent'])
        nll_BC, nll_buttons, nll_camera, entropy = self.compute_hierarchical_logp(
            action_head=pi_head,
            agent_action=obs['minecraft_action'],
            pi_logits=pi_logits, 
            mask=mask, 
            reduction=self.reduction
        )
        
        sample_mask = torch.tensor([env == self.alias for env in obs['env']], device=nll_BC.device) * 1.0
        nll_BC = nll_BC * sample_mask
        nll_buttons = nll_buttons * sample_mask
        nll_camera = nll_camera * sample_mask
        entropy = entropy * sample_mask
        
        res = {
            f'({self.alias}) nll_BC': nll_BC,
            f'({self.alias}) nll_buttons': nll_buttons,
            f'({self.alias}) nll_camera': nll_camera,
            f'({self.alias}) recon_weight': self.weight,
            f'({self.alias}) recon_loss': self.weight * nll_BC,
            f'({self.alias}) entropy': entropy, 
            # f'({self.alias}) entropy_weight': self.entropy_weight, 
            # f'({self.alias}) entropy_loss': entropy * self.entropy_weight, 
        }
        
        if 'condition_info' in obs and 'informative' in obs['condition_info']:
            info_flag = obs['condition_info']['informative'].float()
            nll_BC_info = info_flag.float() * nll_BC
            nll_BC_video = (1.0 - info_flag.float()) * nll_BC
            res.update({
                f'({self.alias}) nll_BC_info': nll_BC_info,
                f'({self.alias}) nll_BC_video': nll_BC_video,
            })
        
        return res

    def compute_hierarchical_logp(
        self, 
        action_head: nn.Module, 
        agent_action: Dict, 
        pi_logits: Dict, 
        mask: Optional[torch.Tensor] = None,
        reduction: Literal['mean', 'sum'] = 'sum',
        eps: float = 1e-6, 
    ):
        log_prob = action_head.logprob(agent_action, pi_logits, return_dict=True)
        entropy  = action_head.entropy(pi_logits, return_dict=True)
        camera_mask = (agent_action['camera'] != 60).float().squeeze(-1)
        if mask is None:
            mask = torch.ones_like(camera_mask)
        logp_buttons = (log_prob['buttons'] * mask).sum(-1)
        logp_camera  = (log_prob['camera'] * mask * camera_mask).sum(-1)
        entropy_buttons = (entropy['buttons'] * mask).sum(-1)
        entropy_camera  = (entropy['camera'] * mask * camera_mask).sum(-1)
        if reduction == 'mean':
            logp_buttons = logp_buttons / (mask.sum(-1) + eps)
            logp_camera  = logp_camera / ((mask * camera_mask).sum(-1) + eps)
            entropy_buttons = entropy_buttons / (mask.sum(-1) + eps)
            entropy_camera  = entropy_camera / ((mask * camera_mask).sum(-1) + eps)
        logp_bc = logp_buttons + logp_camera
        entropy = entropy_buttons + entropy_camera
        return -logp_bc, -logp_buttons, -logp_camera, entropy


class NLLReconHead(nn.Module):
    
    def __init__(
        self, 
        alias: str, 
        weight: float = 1.0, 
        reduction: Literal['sum', 'mean'] = 'sum', 
        # entropy_weight: float = 0.0, 
        **kwargs
    ) -> None:
        super().__init__()
        self.alias = alias
        self.weight = weight
        self.reduction = reduction
        # self.entropy_weight = entropy_weight
    
    def forward(self, latents, useful_heads, **kwargs):
        return {
            'pi_head': useful_heads['auxiliary_pi_heads'][self.alias],
            'pi_latent': latents['pi_latent'],
        }
    
    def loss(self, obs, pred, mask=None, **kwargs):
        
        actions = obs[f'{self.alias}_action']
        pi_head = pred['pi_head']
        pi_logits = pi_head(pred['pi_latent'], actions=actions) # some action head may use `teacher forcing`
        nll_BC = -pi_head.logprob(actions, pi_logits)
        # entropy = pi_head.entropy(pi_logits)
        sampled_action = pi_head.sample(logits=pi_logits, pi_latent=pred['pi_latent'], deterministic=True)
        accuracy = (sampled_action == actions).float() # BxT
        if len(accuracy.shape) > 2:
            accuracy = accuracy.mean(-1)
        
        # #! for atari montezuma's revenge, using mask 
        # mask = 1 - ((actions == 0) * (torch.rand_like(actions.float()) <= 0.9).float())
        
        B, T = nll_BC.shape
        if mask is None:
            mask = torch.ones((B, T), device=nll_BC.device)
        nll_BC = (nll_BC * mask).sum(-1)
        # entropy = (entropy * mask).sum(-1)
        accuracy = (accuracy * mask).sum(-1)
        
        if self.reduction == 'mean':
            nll_BC = nll_BC / (mask.sum(-1) + 1e-8)
            # entropy = entropy / (mask.sum(-1) + 1e-8)
        
        #! accuracy must be averaged over temporal dimension
        accuracy = accuracy / (mask.sum(-1) + 1e-8)
        
        #? remove dummy action loss
        sample_mask = torch.tensor([env == self.alias for env in obs['env']], device=nll_BC.device) * 1.0
        nll_BC = nll_BC * sample_mask
        # entropy = entropy * sample_mask
        accuracy = accuracy * sample_mask
        
        res = {
            f'({self.alias}) nll_BC': nll_BC,
            f'({self.alias}) recon_weight': self.weight,
            f'({self.alias}) recon_loss': self.weight * nll_BC,
            f'({self.alias}) accuracy': accuracy,
            # f'({self.alias}) entropy': entropy, 
            # f'({self.alias}) entropy_weight': self.entropy_weight, 
            # f'({self.alias}) entropy_loss': self.entropy_weight*entropy, 
        }
        
        if 'condition_info' in obs and 'informative' in obs['condition_info']:
            info_flag = obs['condition_info']['informative'].float()
            nll_BC_info = info_flag.float() * nll_BC
            nll_BC_video = (1.0 - info_flag.float()) * nll_BC
            res.update({
                f'({self.alias}) nll_BC_info': nll_BC_info,
                f'({self.alias}) nll_BC_video': nll_BC_video,
            })
        
        return res
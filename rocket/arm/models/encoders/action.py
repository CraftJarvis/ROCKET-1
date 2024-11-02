import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from typing import Dict, Optional, Union, List, Any
from collections import OrderedDict
import gym
from gymnasium import spaces
from rocket.arm.models.utils import FeedForward

# from rocket.stark_tech.env_interface import MinecraftWrapper

# ACTION_KEY_DIM = OrderedDict({
#     'forward': {'type': 'one-hot', 'dim': 2}, 
#     'back': {'type': 'one-hot', 'dim': 2}, 
#     'left': {'type': 'one-hot', 'dim': 2}, 
#     'right': {'type': 'one-hot', 'dim': 2}, 
#     'jump': {'type': 'one-hot', 'dim': 2}, 
#     'sneak': {'type': 'one-hot', 'dim': 2}, 
#     'sprint': {'type': 'one-hot', 'dim': 2}, 
#     'attack': {'type': 'one-hot', 'dim': 2},
#     'use': {'type': 'one-hot', 'dim': 2}, 
#     'drop': {'type': 'one-hot', 'dim': 2},
#     'inventory': {'type': 'one-hot', 'dim': 2}, 
#     'camera': {'type': 'real', 'dim': 2}, 
#     'hotbar.1': {'type': 'one-hot', 'dim': 2}, 
#     'hotbar.2': {'type': 'one-hot', 'dim': 2}, 
#     'hotbar.3': {'type': 'one-hot', 'dim': 2}, 
#     'hotbar.4': {'type': 'one-hot', 'dim': 2}, 
#     'hotbar.5': {'type': 'one-hot', 'dim': 2}, 
#     'hotbar.6': {'type': 'one-hot', 'dim': 2}, 
#     'hotbar.7': {'type': 'one-hot', 'dim': 2}, 
#     'hotbar.8': {'type': 'one-hot', 'dim': 2}, 
#     'hotbar.9': {'type': 'one-hot', 'dim': 2}, 
# })

# class ActionEncoder(nn.Module):
    
#     def __init__(
#         self, 
#         num_channels: int = 512,
#         intermediate_dim: int = 64,
#         action_type: Union['decomposed', 'composed'] = 'decomposed', 
#         action_space: Optional[spaces.Space] = None, 
#     ) -> None:
#         super().__init__()
#         self.action_type = action_type
#         self.action_space = action_space
#         if self.action_type == 'decomposed': 
#             module_dict = dict()
#             for key, conf in ACTION_KEY_DIM.items():
#                 key = 'act_' + key.replace('.', '_')
#                 if conf['type'] == 'one-hot':
#                     module_dict[key] = nn.Embedding(conf['dim'], intermediate_dim)
#                 elif conf['type'] == 'real':
#                     module_dict[key] = nn.Linear(conf['dim'], intermediate_dim)
#             self.embedding_layer = nn.ModuleDict(module_dict)
            
#         elif self.action_type == 'composed':
#             module_dict = dict()
#             for key, space in action_space.items():
#                 module_dict[key] = nn.Embedding(space.nvec.item(), num_channels)
#             self.embedding_layer = nn.ModuleDict(module_dict)
        
#         else:
#             raise NotImplementedError
#         self.final_layer = nn.Linear(len(self.embedding_layer) * intermediate_dim, num_channels)
    
#     def forward_key_act(self, key: str, act: torch.Tensor) -> torch.Tensor:
#         key_embedding_layer = self.embedding_layer['act_'+key.replace('.', '_')]
#         if isinstance(key_embedding_layer, nn.Embedding):
#             return key_embedding_layer(act.long())
#         elif isinstance(key_embedding_layer, nn.Linear):
#             return key_embedding_layer(act.float())
    
#     def forward(self, action: Dict[str, torch.Tensor]) -> torch.Tensor:
        
#         if self.action_type == 'decomposed':
#             if len(action) != len(ACTION_KEY_DIM):
#                 # convert to decomposed action and launch to device
#                 npy_act = MinecraftWrapper.agent_action_to_env(action)
#                 device = next(self.parameters()).device
#                 action = {key: torch.from_numpy(act).to(device) for key, act in npy_act.items()}
#             return self.final_layer(torch.cat([
#                 self.forward_key_act(key, action[key]) for key in ACTION_KEY_DIM.keys()
#             ], dim=-1))
#         elif self.action_type == 'composed':
#             return self.final_layer(torch.cat([
#                 self.forward_key_act(key, action[key]) for key in self.action_space.keys()
#             ], dim=-1))

class ActEmbedding(nn.Module):
    
    def __init__(self, hidsize: int, action_space):
        super().__init__()
        if isinstance(action_space, spaces.Discrete) or isinstance(action_space, gym.spaces.Discrete):
            self.act_emb = nn.Embedding(action_space.n, hidsize)
            self.final_layer = None
        elif isinstance(action_space, spaces.MultiDiscrete) or isinstance(action_space, gym.spaces.MultiDiscrete):
            self.act_emb = nn.ModuleList([
                nn.Embedding(act, hidsize) for act in action_space.nvec
            ])
            self.final_layer = nn.Linear(hidsize*len(action_space.nvec), hidsize)
        elif isinstance(action_space, spaces.Dict):
            self.act_emb = nn.ModuleDict({
                name: ActEmbedding(hidsize, act) for name, act in action_space.items()
            })
            self.final_layer = nn.Linear(hidsize*len(action_space), hidsize)
        else:
            raise NotImplementedError("Not implemented for action space: ", action_space)
    
    def forward(self, actions):
        if isinstance(self.act_emb, nn.Embedding):
            assert len(actions.shape) == 2, "B, T"
            x = self.act_emb(actions)
        elif isinstance(self.act_emb, nn.ModuleList):
            assert len(actions.shape) == 3, "B, T, L"
            B, T, L = actions.shape
            x = torch.cat(
                [ self.act_emb[i](actions[..., i]) for i in range(L) ], dim=-1
            )
        elif isinstance(self.act_emb, nn.ModuleDict):
            x = torch.cat(
                [ self.act_emb[name](actions[name]) for name in self.act_emb.keys() ], dim=-1
            )
        if self.final_layer is not None:
            x = self.final_layer(x)
        return x

class Action(nn.Module):
    
    def __init__(self, hidsize: int, action_space: Dict, n_layer=2, **kwargs):
        super().__init__()
        self.act_emb = nn.ModuleDict({
            name: ActEmbedding(hidsize, act) for name, act in action_space.items()
        })
        self.act_names = sorted(list(action_space.keys()))
        self.updim = nn.Sequential(
            nn.GELU(), nn.Linear(hidsize*len(self.act_names), hidsize)
        )
        self.feedforwards = nn.ModuleList([
            FeedForward(hidsize, mult=2) for i in range(n_layer)
        ])
    
    def forward(self, actions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute joint action embeddings. 
        :params actions: 
        """
        acts = []
        for act_name in self.act_names:
            acts.append(self.act_emb[act_name](actions[act_name]))
        x = torch.cat(acts, dim=-1)
        x = self.updim(x)
        for ffn in self.feedforwards:
            x = ffn(x) + x
        return x

if __name__ == '__main__':
    pass 
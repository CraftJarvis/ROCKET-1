'''
author:        caishaofei <1744260356@qq.com>
date:          2024-02-05 21:44:53
Copyright © Team CraftJarvis All rights reserved
'''
import gym
import random
import numpy as np
import torch
from torch import nn
from rich.console import Console
from typing import Dict, Optional, Union, List, Any, Tuple

class StateEncoder(nn.Module):
    
    def forward(self, input_data) -> Any:
        raise NotImplementedError
    
class ContinuousStateEncoder(StateEncoder):
    
    def __init__(self, state_space: gym.spaces.Box, hidsize: int):
        super().__init__()
        self.input_size = np.prod(state_space.shape)
        self.output_size = hidsize
        self.net = nn.Sequential(
            nn.Linear(self.input_size, hidsize // 4),
            nn.ReLU(),
            nn.Linear(hidsize // 4, hidsize // 2),
            nn.ReLU(),
            nn.Linear(hidsize // 2, self.output_size),
        )
    
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.net(input_data)

class DictStateEncoder(nn.ModuleDict, StateEncoder):
    
    def __init__(self, state_space: Dict[str, gym.spaces.Box], hidsize: int):
        super().__init__()
        self.state_space = state_space
        self.hidsize = hidsize
        for key, space in state_space.items():
            self[key] = ContinuousStateEncoder(space, hidsize)
    
    def forward(self, input_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dic = {key: self[key](input_data[key]) for key in input_data}
        ret = torch.cat(list(dic.values()), dim=-1)
        return ret

if __name__ == '__main__':
    import d4rl
    state_space = {}
    env = gym.make('maze2d-umaze-v1')
    state_space['maze2d-umaze-v1'] = env.observation_space
    encoder = DictStateEncoder(state_space, 256)
    state = env.reset()
    import ipdb; ipdb.set_trace()
    state_dict = {
        'maze2d-umaze-v1': torch.from_numpy(state)[None].to(torch.float32)
    }
    y = encoder(state_dict)
    print(y.shape)
    print(y)
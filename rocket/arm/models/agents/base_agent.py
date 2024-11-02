from abc import ABC, abstractmethod
import typing
import torch
from rocket.arm.utils.vpt_lib.action_head import ActionHead
from typing import Dict, List, Optional, Tuple, Any, Union
from omegaconf import DictConfig, OmegaConf
import numpy as np

def dict_map(fn, d):
    if isinstance(d, Dict) or isinstance(d, DictConfig):
        return {k: dict_map(fn, v) for k, v in d.items()}
    else:
        return fn(d)

T = typing.TypeVar("T")
def recursive_tensor_op(fn, d: T) -> T:
    if isinstance(d, torch.Tensor):
        return fn(d)
    elif isinstance(d, list):
        return [recursive_tensor_op(fn, elem) for elem in d] # type: ignore
    elif isinstance(d, tuple):
        return tuple(recursive_tensor_op(fn, elem) for elem in d) # type: ignore
    elif isinstance(d, dict):
        return {k: recursive_tensor_op(fn, v) for k, v in d.items()} # type: ignore
    elif d is None:
        return None # type: ignore
        
    else:
        raise ValueError(f"Unexpected type {type(d)}")
    
class BaseAgent(torch.nn.Module, ABC):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def _batchify(self, elem):
        if isinstance(elem, (int, float)):
            elem = torch.tensor(elem, device=self.device)
        if isinstance(elem, np.ndarray):
            return torch.from_numpy(elem).unsqueeze(0).unsqueeze(0).to(self.device)
        elif isinstance(elem, torch.Tensor):
            return elem.unsqueeze(0).unsqueeze(0).to(self.device)
        elif isinstance(elem, str):
            return [[elem]]
        else:
            raise NotImplementedError
    
    @abstractmethod
    def initial_state(self, batch_size: Optional[int] = None) -> List[torch.Tensor]:
        pass

    # @property
    @abstractmethod
    def action_head(self) -> ActionHead:
        pass

    @abstractmethod
    def forward(self, 
                obs: Dict[str, Any], 
                state_in: Optional[List[torch.Tensor]],
                first: Optional[torch.Tensor],
                **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
        pass

    def set_var(self, key, val):
        self.policy.set_var(key, val)
    
    def get_var(self, key):
        return self.policy.get_var(key)

    @torch.inference_mode()
    def get_action(self,
                   obs: Dict[str, Any],
                   state_in: Optional[List[torch.Tensor]],
                   first: Optional[torch.Tensor],
                   deterministic: bool = False,
                   input_shape: str = "BT*",
                   **kwargs, 
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        '''
        Get actions from raw observations (no batch and temporal dims).
        '''
        if input_shape == "*":
            obs = dict_map(self._batchify, obs)
            if state_in is not None:
                state_in = recursive_tensor_op(lambda x: x.unsqueeze(0), state_in)
        elif input_shape != "BT*":
            raise NotImplementedError
        
        result, state_out, latent_out = self.forward(obs, state_in, first=first, stage='rollout', **kwargs)
        '''
        Explain: if mixed_agent is used, pi_logits will be in result, 
        otherwise, use pi_latent in latent_out to compute pi_logits. 
        '''
        action_head = self.action_head()
        pi_latent = latent_out['pi_latent']
        
        if 'pi_logits' not in result: # this is for naive VPT
            pi_logits = action_head(pi_latent)
        else:
            pi_logits = result['pi_logits']
        action = action_head.sample(pi_logits, deterministic)
        
        if input_shape == "BT*":
            return action, state_out
        elif input_shape == "*":
            return dict_map(lambda tensor: tensor[0][0], action), recursive_tensor_op(lambda x: x[0], state_out)
        else:
            raise NotImplementedError
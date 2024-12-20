
from typing import (
    List, Dict, Optional, Callable, Union, Tuple, Any
)

import typing
import os
import pickle
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from pathlib import Path
import hydra
import torch
from torch import nn

from rocket.arm.utils.vpt_lib.action_head import make_action_head
from rocket.arm.utils.vpt_lib.normalize_ewma import NormalizeEwma
from rocket.arm.utils.vpt_lib.scaled_mse_head import ScaledMSEHead
from rocket.arm.utils.vpt_lib.tree_util import tree_map
from rocket.arm.models.policys.rocket_policy import RocketPolicy
from rocket.arm.models.heads import build_auxiliary_heads

from omegaconf import DictConfig, OmegaConf
import gymnasium.spaces.dict as dict_spaces

RELATIVE_POLICY_CONFIG_DIR = '../../configs/policy'

def _make_policy(policy_name: str, **kwargs):
    if policy_name == 'ROCKET':
        return MinecraftAgentPolicy(policy_cls=RocketPolicy, **kwargs)
    else:
        raise ValueError(f'Unknown policy name: {policy_name}')
    
def load_policy_cfg(cfg_name: str) -> DictConfig:
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    config_path = Path(RELATIVE_POLICY_CONFIG_DIR) / f"{cfg_name}.yaml"
    hydra.initialize(config_path=str(config_path.parent), version_base='1.3')
    policy_cfg = hydra.compose(config_name=config_path.stem)
    OmegaConf.resolve(policy_cfg)
    return policy_cfg

def load_state_dict_non_strict(model: nn.Module, state_dict: Dict[str, torch.Tensor], verbose: bool = False):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    missing_keys = set (missing_keys)
    unexpected_keys = set (unexpected_keys)

    if verbose:
        table = Table(title="Missing Keys")
        for key in missing_keys:
            table.add_row(f"[bold red]{key}[/bold red]")
        
        if len(missing_keys) > 0:
            Console().print(Panel(table))

        table = Table(title="Unexpected Keys")
        for key in unexpected_keys:
            table.add_row(f"[bold yellow]{key}[/bold yellow]")
        
        if len(unexpected_keys) > 0:
            Console().print(Panel(table))

def make_policy(
    policy_cfg: Union[DictConfig, str, Dict[str, Any]], 
    state_space: Dict[str, Any], 
    action_space: Dict[str, Any], 
    weights_dict: Optional[Any] = None
):
    if isinstance(policy_cfg, str):
        policy_cfg = load_policy_cfg(policy_cfg)

    if not isinstance(policy_cfg, dict):
        assert isinstance(policy_cfg, DictConfig), f"policy_cfg must be a string, a DictConfig or a dict, got {type(policy_cfg)}"
        build_kwargs = typing.cast(Dict[str, Any], OmegaConf.to_container(policy_cfg, resolve=True))
    else:
        build_kwargs = policy_cfg
    
    policy_name = build_kwargs['policy_name']
    
    building_info = {}
    # Parse the model structure file
    model_path = policy_cfg['from'].get('model', None)
    if model_path and Path(model_path).is_file(): 
        Console().log(f"Loading predefined model from {model_path}. ")
        agent_parameters = pickle.load(Path(model_path).open("rb"))
        policy_body_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    else:
        policy_body_kwargs = build_kwargs['policy_kwargs']
        pi_head_kwargs = build_kwargs['pi_head_kwargs']
    auxiliary_head_kwargs = build_kwargs.get('auxiliary_head_kwargs', {})
    
    policy_kwargs = dict(
        state_space=state_space,
        action_space=action_space, 
        policy_body_kwargs=policy_body_kwargs, 
        pi_head_kwargs=pi_head_kwargs, 
        auxiliary_head_kwargs=auxiliary_head_kwargs, 
    )
    policy = _make_policy(policy_name, **policy_kwargs)
    
    # Load weights from file if weights not provided
    weights_path = build_kwargs['from'].get('weights', None)
    if weights_dict is None and weights_path is not None:
        Console().log(f'Load from checkpoint file {weights_path}')
        if Path(weights_path).is_dir():
            weights_path = os.path.join(weights_path, 'model')
        checkpoint = torch.load(weights_path, map_location='cpu')
        weights_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    if weights_dict is not None:
        filter_weights_dict = {}
        for k, v in weights_dict.items():
            k = k.replace('agent.', '') 
            if k.startswith('policy.'):
                filter_weights_dict[k.replace('policy.', '')] = v
            else:
                filter_weights_dict[k] = v
        building_info['ckpt_parameters'] = filter_weights_dict
        load_state_dict_non_strict(policy, filter_weights_dict, verbose=True)

    return policy, building_info
    

class MinecraftAgentPolicy(nn.Module):
    
    def __init__(
        self, 
        policy_cls,
        state_space: Dict[str, Any] = {},
        action_space: Dict[str, Any] = {}, 
        policy_body_kwargs: Dict = {}, 
        pi_head_kwargs: Dict = {}, 
        auxiliary_head_kwargs: List[Dict] = [], 
    ):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.net = policy_cls(state_space=state_space, action_space=action_space, **policy_body_kwargs)
        self.useful_heads = {}
        self.make_value_head(self.net.output_latent_size())
        self.make_action_heads(self.net.output_latent_size(), **pi_head_kwargs)

    def make_value_head(self, v_out_size: int, norm_type: str = "ewma", norm_kwargs: Optional[Dict] = None):
        self.value_head = ScaledMSEHead(v_out_size, 1, norm_type=norm_type, norm_kwargs=norm_kwargs)
        self.useful_heads['value_head'] = self.value_head

    def make_action_heads(self, pi_out_size: int, **pi_head_opts):
        if 'minecraft' in self.action_space:
            space = self.action_space.pop('minecraft')
            self.pi_head = make_action_head(space, pi_out_size, **pi_head_opts)
            self.useful_heads['pi_head'] = self.pi_head
        res = {}
        for key, space in self.action_space.items():
            res[key] = make_action_head(space, pi_out_size, **pi_head_opts)
        self.auxiliary_pi_heads = nn.ModuleDict(res)
        self.useful_heads['auxiliary_pi_heads'] = self.auxiliary_pi_heads

    def initial_state(self, batch_size: int):
        return self.net.initial_state(batch_size)

    def reset_parameters(self):
        super().reset_parameters()
        self.net.reset_parameters()
        self.pi_head.reset_parameters()
        self.value_head.reset_parameters()
        for pi_head in self.auxiliary_pi_heads.values():
            pi_head.reset_parameters()

    def encode_condition(self, *args, **kwargs):
        return self.net.encode_condition(*args, **kwargs)

    def is_conditioned(self):
        return self.net.is_conditioned()

    def action_head(self, env_name: str):
        if env_name == 'minecraft':
            return self.pi_head
        else:
            return self.auxiliary_pi_heads[env_name]

    def forward(
        self, 
        obs: Dict, 
        first: torch.Tensor, 
        state_in: List[torch.Tensor], 
        stage: str = 'train', 
        ice_latent: Optional[torch.Tensor] = None, 
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        obs = obs.copy()
        
        latents, state_out, internal_loss = self.net(
            obs=obs, 
            state_in=state_in, 
            context={"first": first}, 
            ice_latent=ice_latent,
        )
        
        result = {
            'obs_mask': latents.get('obs_mask', None),
            'internal_loss': internal_loss,
        }
        
        return result, state_out, latents

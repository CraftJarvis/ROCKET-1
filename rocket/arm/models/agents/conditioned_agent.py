import functools
import torch
import re
import av
import cv2
import numpy as np
import typing

from rich import print
from rich.console import Console
from typing import Union, Dict, Optional, List, Tuple, Any
from omegaconf import DictConfig, ListConfig
import gymnasium
import gymnasium.spaces.dict as dict_spaces
from huggingface_hub import PyTorchModelHubMixin

from rocket.arm.utils import fit_img_space
from rocket.arm.utils.vpt_lib.action_head import ActionHead
from rocket.arm.models.policys import make_policy, load_policy_cfg
from rocket.arm.models.agents.base_agent import BaseAgent

def convert_to_normal(obj):
    if isinstance(obj, DictConfig) or isinstance(obj, Dict):
        return {key: convert_to_normal(value) for key, value in obj.items()}
    elif isinstance(obj, ListConfig) or isinstance(obj, List):
        return [convert_to_normal(item) for item in obj]
    else:
        return obj

def beautify_tensor(tensor: torch.Tensor) -> str:
    ele = [f'{x:.3f}' for x in tensor]
    string = ' '.join(ele)
    return string

class ROCKET1(
    BaseAgent, 
    PyTorchModelHubMixin, 
    repo_url="https://huggingface.co/phython96/ROCKET-1",
    license="mit",
):
    def __init__(self, 
        policy_config: Union[DictConfig, str] = {}, 
        weights_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.action_space = {"minecraft": gymnasium.spaces.Dict({
            "buttons": gymnasium.spaces.MultiDiscrete([8641]),
            "camera": gymnasium.spaces.MultiDiscrete([121]), 
        })}
        self.policy_config = policy_config
        self.infer_env = "minecraft"

        if isinstance(self.policy_config, str):
            self.policy_config = load_policy_cfg(self.policy_config)
        self.policy, self.policy_building_info = make_policy(
            policy_cfg=self.policy_config, 
            state_space={},
            action_space=self.action_space, 
            weights_dict=weights_dict, 
        )

        self.timesteps = self.policy_config['policy_kwargs']['timesteps']
        self.resolution = self.policy_config['policy_kwargs']['backbone_kwargs']['img_shape'][:2]
        
        self.cached_init_states = {}
        self.cached_first = {}

    def wrapped_forward(self, 
                        obs: Dict[str, Any], 
                        state_in: Optional[List[torch.Tensor]],
                        first: Optional[torch.Tensor] = None, 
                        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
        '''Wrap state and first arguments if not specified. '''
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                B, T = v.shape[:2]
                break

        state_in = self.initial_state(B) if state_in is None else state_in

        if first is None:
            first = self.cached_first.get((B, T), torch.tensor([[False]], device=self.device).repeat(B, T))
            self.cached_first[(B, T)] = first
        
        return self.policy(
            obs=obs, 
            first=first, 
            state_in=state_in, 
            **kwargs
        )

    # @property
    def action_head(self) -> ActionHead:
        return self.policy.action_head(self.infer_env)

    @property
    def value_head(self) -> torch.nn.Module:
        return self.policy.value_head
    
    def initial_state(self, batch_size: Optional[int] = None) -> List[torch.Tensor]:
        if batch_size is None:
            return [t.squeeze(0).to(self.device) for t in self.policy.initial_state(1)]
        else:
            if batch_size not in self.cached_init_states:
                self.cached_init_states[batch_size] = [t.to(self.device) for t in self.policy.initial_state(batch_size)]
            return self.cached_init_states[batch_size]

    def forward(self, 
                obs: Dict[str, Any], 
                state_in: Optional[List[torch.Tensor]],
                first: Optional[torch.Tensor] = None,
                **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
        forward_result, state_out, latents = self.wrapped_forward(obs=obs, state_in=state_in, first=first, **kwargs)
        return forward_result, state_out, latents

    def decorate_obs(self, obs: np.ndarray) -> Dict[str, torch.Tensor]:
        '''
        Convert the observation from environment to the format that the policy can understand.
        For example, numpy.array[84, 84] -> { torch.Tensor[128, 128, 3] cuda }
        Arguments:
            obs: np.ndarray, the observation from the environment.
        Returns:
            obs: Dict[str, torch.Tensor], the observation policy received. 
        '''
        assert isinstance(obs, np.ndarray), 'The observation should be a numpy array. '
        if len(obs.shape) > 1:
            # image type (such as minecraft and ataris)
            fit_img = fit_img_space([obs], resolution=self.resolution, to_torch=True, device=self.device)[0]
            return {'img': fit_img}
        else:
            # state type (such as mujoco and meta-world)
            return {f'{self.infer_env}_state': torch.from_numpy(obs).to(torch.float32).to(self.device)}

    # @staticmethod
    # def from_pretrained(ckpt_path: str, **kwargs):
    #     '''
    #     Load the agent from the checkpoint.
    #     '''
    #     Console().log("Loading agent from checkpoint: ", ckpt_path)
    #     checkpoint = torch.load(ckpt_path, map_location='cpu')
    #     assert 'state_dict' in checkpoint, 'The checkpoint does not contain the state_dict. '
    #     build_helper_hparams = convert_to_normal(checkpoint['hyper_parameters']['build_helper_hparams'])
    #     build_helper_hparams.update(kwargs)
    #     instance = ROCKET1(**build_helper_hparams, weights_dict=checkpoint['state_dict'])
    #     return instance

if __name__ == '__main__':
    ckpt_path = "/nfs-shared/shaofei/jarvisbase/output/BOYA/2024-10-06/16-08-47/weights/weight-epoch=5-step=110000-EMA.ckpt"
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    build_helper_hparams = convert_to_normal(checkpoint['hyper_parameters']['build_helper_hparams'])
    
    policy_config = {
        'name': 'rocket_minecraft',
        'policy_name': 'ROCKET',
        'from': {'model': None, 'weights': None},
        'policy_kwargs': {
            'attention_heads': 8,
            'attention_mask_style': 'clipped_causal',
            'attention_memory_size': 256,
            'hidsize': 1024,
            'init_norm_kwargs': {'batch_norm': False, 'group_norm_groups': 1},
            'n_recurrence_layers': 4,
            'only_img_input': True,
            'pointwise_ratio': 4,
            'pointwise_use_activation': False,
            'recurrence_is_residual': True,
            'recurrence_type': 'transformer',
            'timesteps': 128,
            'use_pointwise_layer': True,
            'use_pre_lstm_ln': False,
            'word_dropout': 0.0,
            'backbone_kwargs': {'name': 'EFFICIENTNET', 'img_shape': [224, 224, 3], 'version': 'efficientnet-b0', 'pooling': False, 'accept_segment': True}
        },
        'pi_head_kwargs': {'temperature': 1.0},
        'auxiliary_head_kwargs': [{'name': 'minecraft_recon_head', 'alias': 'minecraft', 'enable': True}]
    }
    weights_dict = checkpoint['state_dict']
    # model = ROCKET1(policy_config=policy_config, weights_dict=weights_dict)
    # model.save_pretrained("ROCKET-1")
    # model.push_to_hub("phython96/ROCKET-1")
    
    import ipdb; ipdb.set_trace()
    model = ROCKET1.from_pretrained("phython96/ROCKET-1")
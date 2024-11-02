'''
author:        caishaofei-mus1 <1744260356@qq.com>
date:          2023-05-05 15:44:33
Copyright © Team CraftJarvis All rights reserved
'''
import av
import re
import os
import time
import argparse
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict
from typing import Dict, List, Tuple, Union, Sequence, Mapping, Any, Optional

import cv2
import torch
import numpy as np

from rocket.stark_tech.entry import env_generator
from rocket.arm.utils.vpt_lib.actions import ActionTransformer
from rocket.arm.utils.vpt_lib.action_mapping import CameraHierarchicalMapping
from rocket.stark_tech.utils.hierarchical_action_space import HierarchicalActionSpace

os.environ["FORCE_CPU_RENDER"] = "1"

ENV_CONFIG_DIR = Path(__file__).parent.parent / "env_configs" 
RELATIVE_ENV_CONFIG_DIR = "../env_configs"


ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)

KEYS_TO_INFO = ['pov', 'inventory', 'equipped_items', 'life_stats', 'location_stats', 'use_item', 'drop', 'pickup', 'break_item', 'craft_item', 'mine_block', 'damage_dealt', 'entity_killed_by', 'kill_entity', 'full_stats', 'player_pos', 'is_gui_open']

def resize_image(img, target_resolution = (224, 224)):
    return cv2.resize(img, dsize=target_resolution, interpolation=cv2.INTER_LINEAR)


class MinecraftWrapper(gym.Env):
    
    ACTION_SPACE_TYPE = 'Hierarchical'
    action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
    action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

    @classmethod
    def get_obs_space(cls, width=640, height=360):
        return {
            'img': spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8),
            'text': str, 
            'obs_conf': Dict, 
        }
    
    @classmethod
    def get_action_space(cls):
        '''
        Convert the action space to the type of 'spaces.Tuple', 
        since spaces.Dict is not supported by ray.rllib. 
        '''
        if MinecraftWrapper.ACTION_SPACE_TYPE == 'Hierarchical':
            return HierarchicalActionSpace(
                action_mapper=cls.action_mapper,
                spaces=cls.action_mapper.get_action_space_update()
            )
        elif MinecraftWrapper.ACTION_SPACE_TYPE == 'Dict':
            return spaces.Dict(cls.action_mapper.get_action_space_update())
        elif MinecraftWrapper.ACTION_SPACE_TYPE == 'Tuple':
            original_action_space = cls.action_mapper.get_action_space_update()
            return spaces.Tuple((original_action_space['buttons'], original_action_space['camera']))
        else:
            raise ValueError(f'Unsupported action space type: {ACTION_SPACE_TYPE}')

    @classmethod
    def get_dummy_action(cls, B: int, T: int, device="cpu"):
        '''
        Get a dummy action for the environment.
        '''
        ac_space = cls.get_action_space()
        action = ac_space.sample()
        
        dummy_action = {}
        if isinstance(action, OrderedDict):
            for key, val in action.items():
                dummy_action[key] = (
                    torch.from_numpy(val)
                    .reshape(1, 1, -1)
                    .repeat(B, T, 1)
                    .to(device)
                )
        elif isinstance(action, tuple):
            dummy_action = (
                torch.from_numpy(action)
                .reshape(1, 1, -1)
                .repeat(B, T, 1)
                .to(device)
            )
        else:
            raise NotImplementedError
        
        return dummy_action

    @classmethod
    def agent_action_to_env(cls, agent_action):
        """Turn output from policy into action for MineRL"""
        # This is quite important step (for some reason).
        # For the sake of your sanity, remember to do this step (manual conversion to numpy)
        # before proceeding. Otherwise, your agent might be a little derp.
        action = agent_action
        # First, convert the action to the type of dict
        if isinstance(action, tuple):
            action = {
                'buttons': action[0], 
                'camera': action[1], 
            }
        # Second, convert the action to the type of numpy
        if isinstance(action["buttons"], torch.Tensor):
            action = {
                "buttons": action["buttons"].cpu().numpy(),
                "camera": action["camera"].cpu().numpy()
            }
        # Here, the action is the type of dict, and the value is the type of numpy
        minerl_action = cls.action_mapper.to_factored(action)
        minerl_action_transformed = cls.action_transformer.policy2env(minerl_action)
        return minerl_action_transformed

    
    @classmethod
    def env_action_to_agent(cls, minerl_action_transformed, to_torch=True, device: Union[str, torch.device]="cpu"):
        """
        Turn action from MineRL to model's action.

        Note that this will add batch dimensions to the action.
        Returns numpy arrays, unless `to_torch` is True, in which case it returns torch tensors.

        If `check_if_null` is True, check if the action is null (no action) after the initial
        transformation. This matches the behaviour done in OpenAI's VPT work.
        If action is null, return "None" instead
        """
        if isinstance(minerl_action_transformed["attack"], torch.Tensor):
            minerl_action_transformed = {key: val.cpu().numpy() for key, val in minerl_action_transformed.items()}

        minerl_action = cls.action_transformer.env2policy(minerl_action_transformed)

        # Add batch dims if not existant
        if minerl_action["camera"].ndim == 1:
            minerl_action = {k: v[None] for k, v in minerl_action.items()}
        
        # Merge temporal and batch dimension
        if minerl_action["camera"].ndim == 3:
            B, T = minerl_action["camera"].shape[:2]
            minerl_action = {k: v.reshape(B*T, -1) for k, v in minerl_action.items()}
            action = cls.action_mapper.from_factored(minerl_action)
            action = {key: val.reshape(B, T, -1) for key, val in action.items()}
        else:
            action = cls.action_mapper.from_factored(minerl_action)
            
        if to_torch:
            action = {k: torch.from_numpy(v).to(device) for k, v in action.items()}

        return action


    def __init__(self, env_config: Union[str, Dict, DictConfig], prev_action_obs = False, disable_text = False) -> None:
        super().__init__()
        self.started = False
        self.disable_text = disable_text
        self.prev_action_obs = prev_action_obs
        self.last_pov = None
        self.last_info = {}
        if isinstance(env_config, str):
            hydra.core.global_hydra.GlobalHydra.instance().clear()
            config_path = Path(RELATIVE_ENV_CONFIG_DIR) / f"{env_config}.yaml"
            initialize(config_path=str(config_path.parent), version_base='1.3')
            self.env_config = compose(config_name=config_path.stem)
        elif isinstance(env_config, Dict) or isinstance(env_config, DictConfig):
            self.env_config = env_config
        else:
            raise ValueError("env_config must be a string or a dict")
        
        self._env, self.additional_info = env_generator(self.env_config)
        
        width, height = self.env_config['resize_resolution'] # 224x224
        self.resize_resolution = (width, height)
        self.action_space = MinecraftWrapper.get_action_space()
        self.observation_space = MinecraftWrapper.get_obs_space(width=width, height=height)
        
    
    def set_current_task(self, task: str):
        '''Manually change the current task.'''
        return self._env.set_current_task(task)
    
    def manual_set_task(self, text: Optional[str] = None, obs_conf: Optional[Dict] = None):
        '''
        Set the text/obs_conf to be returned by the environment. 
        However, Recommanded to use `set_current_task` instead. 
        '''
        self.override_task_conf = {}
        if text is not None:
            self.override_task_conf['text'] = text
        if obs_conf is not None:
            self.override_task_conf['obs_conf'] = obs_conf
    
    def _build_obs(self, input_obs: Dict, info: Dict) -> Dict:
        output_obs = {
            'img': resize_image( input_obs['pov'], self.resize_resolution ),
        }
        if self.prev_action_obs:
            output_obs['prev_action'] = self.prev_action
        if not self.disable_text:
            info.update(getattr(self, 'override_task_conf', {}))
            output_obs['text'] = info['text']
            output_obs['obs_conf'] = info['obs_conf']
        return output_obs

    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        '''Takes three kinds of actions as environment inputs. '''
        if isinstance(action, dict) and 'attack' in action.keys():
            minerl_action = action
        else:
            # Hierarchical action space to factored action space
            minerl_action = MinecraftWrapper.agent_action_to_env(action)
        if self.prev_action_obs:
            self.prev_action = minerl_action.copy()
        obs, reward, terminated, info = self._env.step(minerl_action)
        truncated = terminated

        if 'event_info' in info and len(info['event_info']) > 0:
            print("env info:", info['event_info'])
        self.last_pov = obs['pov']
        self.last_info = info
        
        return (
            self._build_obs(obs, info), 
            reward, 
            terminated, 
            truncated, 
            info,
        )

    def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:
        self.started = True
        obs, info = self._env.reset()
        if self.prev_action_obs:
            self.prev_action = self.noop_action()
        self.last_pov = obs['pov']
        self.last_info = info
        return self._build_obs(obs, info), info

    def noop_action(self):
        return MinecraftWrapper.agent_action_to_env({"camera": torch.tensor([60]), "buttons": torch.tensor([0])})

    def seed(self, seed: int) -> None:
        self._env.seed(seed)

    def close(self):
        print('Simulator is being closed.')
        return self._env.close()

    def render(self) -> np.ndarray:
        assert self.last_pov is not None
        img = self.last_pov.copy()
        event_info = self.last_info['event_info']
        cnt = 0
        for event_type in event_info.keys():
            for key in event_info[event_type].keys():
                message = f"{event_type}: {key} x {int(event_info[event_type][key])}"
                cv2.putText(img, message, (150, 275 + cnt * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (67, 188, 205), 2)
                cnt += 1
        
        return img

    def __del__(self):
        if self.started:
            self.close()


class PrefixMinecraftWrapper(MinecraftWrapper):
    
    def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:
        obs, self.cache_info = super().reset()
        self.prefix_len = 128
        text = self._env.current_task_conf['text'] # example: ''traj:[/home/video.mp5] - [/home/video2.mp5]''
        print('[video text]', text)
        
        # recognize the path pattern
        video_path = re.findall('(/.*?mp4)', text)[0]
        print('[video path]', video_path)
        self.frames = []
        width, height = self.resize_resolution 
        with av.open(video_path, "r") as container: 
            for frame in container.decode(video=0): 
                frame = frame.to_ndarray(format="rgb24") 
                self.frames.append(frame) 
        
        info = self.cache_info.copy()
        info['pov'] = self.frames[0]
        obs = super()._build_obs({'pov': self.frames[0]}, info)
        self.prefix_point = 1
        return obs, info
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        if self.prefix_point >= self.prefix_len:
            return super().step(action)
        else:
            info = self.cache_info.copy()
            info['pov'] = self.frames[self.prefix_point]
            obs = super()._build_obs({'pov': self.frames[self.prefix_point]}, info)
            self.prefix_point += 1
            return obs, 0, False, False, info
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='rocket/collect_wood')
    args = parser.parse_args()
    env_name = args.env
    
    # env = MinecraftWrapper(env_name)
    env = MinecraftWrapper(env_name, prev_action_obs=True)
    obs, info = env.reset()
    print(env.observation_space)
    print(env.action_space)
    # print(info.keys())
    # print(info['player_pos'])
    container = av.open("env_test.mp4", mode='w', format='mp4')
    stream = container.add_stream('h264', rate=20)
    stream.width = 224 
    stream.height = 224 
    stream.pix_fmt = 'yuv420p'
    # import queue 
    from queue import Queue
    fps_queue = Queue()
    for i in range(4000):
        
        time_start = time.time()
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        # import ipdb; ipdb.set_trace()
        if terminated:
            obs, info = env.reset()

        time_end = time.time()
        curr_fps = 1/(time_end-time_start)
        fps_queue.put(curr_fps)
        if fps_queue.qsize() > 200:
            fps_queue.get()
        average_fps = sum(list(fps_queue.queue))/fps_queue.qsize()
        text = f"frame: {i}, fps: {curr_fps:.2f}, avg_fps: {average_fps:.2f}"
        if i % 50 == 0:
            print(text)
        frame = resize_image(info['pov'], (640, 360))
        
        if 'attack' not in action:
            action = MinecraftWrapper.agent_action_to_env(action)
        
        for row, (k, v) in enumerate(action.items()):
            color = (234, 53, 70) if (v != 0).any() else (249, 200, 14) 
            if k == 'camera':
                v = "[{:.2f}, {:.2f}]".format(v[0], v[1])
            cv2.putText(frame, f"{k}: {v}", (10, 25 + row*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, text, (150, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (67, 188, 205), 2)
        
        frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)
    
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    env.close()

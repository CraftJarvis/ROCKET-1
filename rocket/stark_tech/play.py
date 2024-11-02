import os
import argparse
from rocket.stark_tech.human_play_interface.human_play_interface import RecordHumanPlay
from pathlib import Path
import hydra
from hydra import compose, initialize
from typing import (
    Dict, Optional, Union, List, Tuple, Any, Callable
)

from rocket.assembly.marks import GROOT

def prompt_by_condition(info: Dict):
    dig_down_traj = '/home/caishaofei/workspace/CODE_BASE/JarvisBase/jarvis/stark_tech/prog_trajs/ref_digdown_0.mp4'
    mine_diamond_traj  = '/home/caishaofei/workspace/CODE_BASE/JarvisBase/jarvis/stark_tech/prog_trajs/ref_mine_0.mp4'
    climbup_traj = '/home/caishaofei/workspace/CODE_BASE/JarvisBase/jarvis/stark_tech/prog_trajs/ref_climbup_fast.mp4'
    
    if info['player_pos']['y'] > 12:
        return dig_down_traj
    elif info['player_pos']['y'] < 5:
        return climbup_traj
    else:
        return mine_diamond_traj

def vpt_on_switch_bot(play_loop, info):
    print("Native VPT")
    play_loop.human_interface.env.manual_set_text("vpt")

def plan_on_after_bot_step(play_loop, info):
    lucky_traj = prompt_by_condition(info)
    current_timestep = play_loop.human_interface.timestep
    print("Current timestep:", current_timestep)
    if (
        lucky_traj != getattr(play_loop, "last_traj") and 
        current_timestep - getattr(play_loop, 'last_set_time', 0) > 100
    ):
        print("Use reference video:", lucky_traj, "current timestep:", current_timestep)
        play_loop.human_interface.env.manual_set_text(f"traj:[{lucky_traj}]")
        play_loop.last_traj = lucky_traj
        play_loop.last_set_time = current_timestep

def imitate_on_switch_bot(play_loop, info):
    latest_video = play_loop.human_interface.latest_video
    print(f"Imitate latest video: <{latest_video}>")
    prompt = f"traj:[{latest_video}]"
    play_loop.human_interface.env.manual_set_text(prompt)

def record_on_switch_bot(play_loop, info):
    print("Start recording bot behaviors...")
    play_loop.human_interface.trajectory = []

def record_on_switch_human(play_loop, info):
    print("Stop recording bot behaviors...")
    # play_loop.human_interface._save_trajectory(dir_name='bot', change_latest=False)

class PlayLoop:
    
    def __init__(
        self, 
        env: str, 
        output_dir: str,
        enable_bot: bool = False,
        policy: Optional[str] = None, 
        ckpt_path: Optional[str] = None, 
        *, 
        on_after_step: Optional[Callable] = [], 
        on_after_human_step: Optional[Callable] = [], 
        on_after_bot_step: Optional[Callable] = [],
        on_switch_human: Optional[Callable] = [],
        on_switch_bot: Optional[Callable] = [],
        on_terminated: Optional[Callable] = [],
    ):
        self.enable_bot = enable_bot
        self.on_after_step = on_after_step
        self.on_after_human_step = on_after_human_step
        self.on_after_bot_step = on_after_bot_step
        self.on_switch_human = on_switch_human
        self.on_switch_bot = on_switch_bot
        self.on_terminated = on_terminated
        output_dir_path = Path(output_dir)
        if not output_dir_path.exists():
            print("Creating output directory: ", output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            
        self.human_interface = RecordHumanPlay(env_config=env, output_dir=output_dir)
        self.human_interface.reset()
        
        if self.enable_bot:
            policy_configs = {
                
            }
            self.agent = GROOT(
                env=self.human_interface, 
                policy_configs=policy_configs,
                # ('../arm/configs/policy', policy), 
                checkpoint=ckpt_path
            )
            self.agent.reset()
        
        self.switch = 'human'
        self.last_traj = None
    
    def run(self):
        
        while True:
            # On Step
            if self.switch == 'human':
                action = None
                obs, reward, terminated, truncated, info = self.human_interface.step(action)
                for callback in self.on_after_human_step:
                    callback(self, info=info)
            else:
                action = self.agent.compute_action(obs)
                obs, reward, terminated, truncated, info = self.human_interface.step(action)
                for callback in self.on_after_bot_step:
                    callback(self, info=info)
            
            for callback in self.on_after_step:
                callback(self, info=info)
            
            next_switch = info.get('switch', 'human')
            if next_switch != self.switch:
                if next_switch == 'human':
                    print("Switch to human")
                    for callback in self.on_switch_human:
                        callback(self, info=info)
                    self.switch = next_switch
                elif next_switch == 'bot':
                    if not self.enable_bot:
                        print("Bot is not enabled. ")
                    else:
                        print("Switch to bot")
                        for callback in self.on_switch_bot:
                            callback(self, info=info)
                        self.agent.reset()
                        self.switch = next_switch
                # if next_switch, step noop action
                obs, reward, terminated, truncated, info = self.human_interface.step(
                    self.human_interface.noop_action()
                )
            
            if terminated:
                self.human_interface.reset()
                if self.enable_bot:
                    self.agent.reset()

def get_and_eval(inp: List[str]) -> List[Callable]:
    return [eval(x) for x in inp]

@hydra.main(config_path="configs", config_name="human")
def main(cfg):
    play_loop = PlayLoop(
        env=cfg.env,
        output_dir=cfg.output_dir,
        enable_bot=getattr(cfg, 'enable_bot', False),
        policy=getattr(cfg, 'policy', None),
        ckpt_path=getattr(cfg, 'ckpt_path', None),
        on_after_step=get_and_eval(getattr(cfg, 'on_after_step', [])),
        on_after_human_step=get_and_eval(getattr(cfg, 'on_after_human_step', [])),
        on_after_bot_step=get_and_eval(getattr(cfg, 'on_after_bot_step', [])),
        on_switch_bot=get_and_eval(getattr(cfg, 'on_switch_bot', [])),
        on_switch_human=get_and_eval(getattr(cfg, 'on_switch_human', [])),
    )
    play_loop.run()

if __name__ == "__main__":
    main()

    
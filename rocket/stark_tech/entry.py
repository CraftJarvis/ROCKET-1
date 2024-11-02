import gym
import os
import json
import math
import random
import numpy as np
from pathlib import Path
from collections import OrderedDict, defaultdict
from typing import Optional, Sequence, List, Tuple, Dict, Union, Callable
from omegaconf import DictConfig, OmegaConf

from rocket.stark_tech.herobraine.env_specs.human_survival_specs import HumanSurvival
from rocket.stark_tech.env._singleagent import _SingleAgentEnv
from rocket.stark_tech.utils.inventory import map_slot_number_to_cmd_slot

from rocket.assets import *

if not os.path.exists(os.path.join(os.path.dirname(__file__), "MCP-Reborn")):
    raise ValueError("Please download and extract `jarvis-MCP-Reborn.zip` to `jarvis/stark_tech/MCP-Reborn`")

class CommandsGenerator(object):
    
    def __init__(
        self, 
        summon_mobs: Optional[DictConfig] = None, 
        summon_items: Optional[DictConfig] = None, 
        random_fill_inventory: Optional[DictConfig] = None, 
    ):
        self.summon_mobs = summon_mobs
        self.summon_items = summon_items
        self.random_fill_inventory = random_fill_inventory
    
    def __call__(self):
        # commands = ['/execute at @s facing ~ ~ ~-1 run tp ~ ~ ~']
        commands = ['/gamerule sendCommandFeedback false']
        if self.summon_mobs:
            for mob_conf in self.summon_mobs:
                mob_name = mob_conf.mob_name
                range_x = mob_conf.range_x
                range_z = mob_conf.range_z
                number = mob_conf.number
                for i in range(number):
                    command = '/execute as @p at @p run summon minecraft:{} ~{} ~ ~{} {{Age:0}}'.format(mob_name, str(random.randint(range_x[0], range_x[1])), str(random.randint(range_z[0], range_z[1])))
                    commands.append(command)
                    
        if self.summon_items:
            if self.summon_items['random_equip']:
                for equip_slot in EQUIP_SLOTS:
                    if random.random() > self.summon_items['random_equip_ratio']:
                        continue
                    equip_item = random.choice(EQUIPABLE_ITEMS[equip_slot])
                    commands.append(f'/replaceitem entity @p armor.{equip_slot} minecraft:{equip_item} 1')
        
            if self.summon_items['items'] is not None:
                candidate_inventory_items = self.summon_items['items']
            else:
                candidate_inventory_items = ALL_ITEMS_IDX_TO_NAME
            slot_lo = self.summon_items['slot_lo']
            slot_up = self.summon_items['slot_up']
            for i in range(slot_lo, slot_up+1):
                if random.random() < self.summon_items['summon_ratio']:
                    item_name = random.choice(candidate_inventory_items)
                    item_number = random.randint(1, ALL_ITEMS[item_name]['stackSize'])
                    commands.append(f'/replaceitem entity @p container.{i} minecraft:{item_name} {item_number}')
        
        if self.random_fill_inventory:
            slot_range = self.random_fill_inventory['slot_range']
            for dic in self.random_fill_inventory['requires']:
                item_name = dic['type']
                quantity_range = dic['quantity_range']
                slot = random.randint(slot_range[0], slot_range[1])
                quantity = random.randint(quantity_range[0], quantity_range[1])
                commands.append(f'/replaceitem entity @p container.{slot} minecraft:{item_name} {quantity}')
        
        return commands


class ScriptExecuter(object):
    
    def __init__(self, script_conf: List = []) -> None:
        '''
        For example:
            script_conf = [
                {
                    'action':
                        'inventory': 1, 
                        'camera': [-5, -1], 
                    'repeat': 1 
                }, 
                {
                    'action':
                        'attack': 1, 
                    'repeat': 5 
                }, 
                }, 
                ...
            ]
        '''
        self.script_conf = script_conf
    
    def execute(self, env) -> None:
        for action_conf in self.script_conf:
            repeat = action_conf.get('repeat', 1)
            for _ in range(repeat):
                action = env.action_space.no_op()
                for key, val in action_conf['action'].items():
                    if key in action:
                        action[key] = np.array(val)
                env.step(action)
        return env.step(env.action_space.no_op())

class RewardGenerator(object):

    def __init__(self, reward_conf: Union[Dict, DictConfig] = {}) -> None:
        '''
        For example: 
            reward_conf = {
                'craft_item': {
                    planks: {
                        quantity_rewarded: 10,
                        reward: 0.5,
                        objs: ["oak_planks"]
                    },
                    crafting_table: {
                        quantity_rewarded: 1,
                        reward: 1.0,
                        objs: ["crafting_table"]
                    },
                },
                'kill_entity': {
                    'cow': {
                        quantity_rewarded: 10,
                        reward: 0.5,
                        objs: ["cow"]
                    }, 
                    'sheep': {
                        quantity_rewarded: 10,
                        reward: 0.5,
                        objs: ["sheep"]
                    }
                }
            }
        '''
        self.reward_conf = reward_conf
        self.reset()

    def reset(self):
        self.rewarded_count = defaultdict(float)
        if not hasattr(self, 'stored_info'):
            self.stored_info = {}

    def _get_obj_num(self, info: Dict, event_type: str, obj: str) -> float:
        if event_type not in info:
            return 0.
        if obj not in info[event_type]:
            return 0.
        res = info[event_type][obj]
        return res.item() if isinstance(res, np.ndarray) else res 

    def step(self, info: Dict) -> float:
        cum_reward = 0
        event_info = {}
        terminated = False
        for event_type in self.reward_conf:
            if event_type not in info:
                continue
            for key, conf in self.reward_conf[event_type].items():
                delta = 0
                for o in conf["objs"]:
                    if o not in info[event_type]:
                        continue
                    delta += self._get_obj_num(info, event_type, o) - self._get_obj_num(self.stored_info, event_type, o)
                if delta > 0:
                    c_rewarded = min(delta, conf['quantity_rewarded'] - self.rewarded_count[(event_type, key)])
                    self.rewarded_count[(event_type, key)] += c_rewarded
                    cum_reward += conf['reward'] * c_rewarded
                    # record event info 
                    if event_type not in event_info:
                        event_info[event_type] = {}
                    event_info[event_type][key] = delta
        self.stored_info = info
        return cum_reward, terminated, event_info


class Minecraft(gym.Env):
    """
    frameskip: Number of frames to skip per step.

    resolution: [width, height] of the observed game frame.

    init_inventory: Initial inventory of the agent.
            Example: {
                0: {'type':'dirt', 'quantity':10},
                1: {'type':'planks', 'quantity':5},
                5: {'type':'log', 'quantity':1},
                6: {'type':'log', 'quantity':2},
                32: {'type':'iron_ore', 'quantity':4}
            }

    fast_reset: Whether to use fast reset features.

    slow_reset_interval: If `fast_reset == True`, the interval to trigger slow reset to completely refresh
                         the environment.

    random_tp_range: Maximum distance to teleport randomly at fast reset.

    start_time: Start game time of the simulator.

    start_weather: Start weather of the simulator.

    custom_init_commands: Commands executed after every `env.reset()`.

    compute_delta_inventory: Whether to return delta inventory in the obs dict.

    preferred_spawn_biome: Specify biome to spawn agent.
            Example: 'plains', 'extreme_hills', 'forest'
    """
    def __init__(
        self, 
        close_ended: bool = False,
        seed: Optional[int] = None,
        frameskip = 1, 
        resolution = [640,360],
        init_inventory: Optional[Dict[int, Dict[str, Union[str, int]]]] = None,
        fast_reset: bool = True, slow_reset_interval: int = 50,
        random_tp_range: int = 400,
        start_time: int = 0, start_weather: str = "clear",
        custom_init_commands: Optional[Sequence[str]] = [],
        custom_init_script: ScriptExecuter = None,
        commands_generator: Optional[Callable] = None,
        compute_delta_inventory: bool = True,
        # preferred_spawn_biome: Optional[str] = "plains",
        candidate_weather: List[str] = [],
        candidate_preferred_spawn_biome: List[str] = [], 
        time_limit: int = math.inf,
        task_conf: Optional[Dict[str, Dict]] = {},
        enable_tasks: Optional[List[str]] = [], 
        reset_inventory_open: bool = False,
        masked_actions: Optional[Dict] = {},
        **unused_kwargs,
    ) -> None:
        super().__init__()

        assert slow_reset_interval >= 1
        self.close_ended = close_ended
        self.nseed = seed
        
        self.fast_reset = fast_reset
        self.slow_reset_interval = slow_reset_interval

        self.custom_init_commands = custom_init_commands
        self.custom_init_script = custom_init_script
        self.compute_delta_inventory = compute_delta_inventory
        self.commands_generator = commands_generator
        self.candidate_weather = candidate_weather
        self.candidate_preferred_spawn_biome = candidate_preferred_spawn_biome
        # self.preferred_spawn_biome = preferred_spawn_biome
        self.time_limit = time_limit
        self.task_conf = task_conf
        self.enable_tasks = enable_tasks
        self.reset_inventory_open = reset_inventory_open
        self.masked_actions = masked_actions

        self.nb_steps = 0
        
        if self.fast_reset:
            self.fast_reset_commands = [
                "/kill"
            ]
            self.fast_reset_commands.extend([
                " " for _ in range(5)
            ])
            self.fast_reset_commands.extend([
                f"/time set {start_time or 0}",
                f"/weather {start_weather or 'clear'}"
            ])
            if init_inventory is not None:
                for slot, item_dict in init_inventory.items():
                    mc_slot = map_slot_number_to_cmd_slot(slot)
                    item_type = item_dict["type"]
                    item_quantity = item_dict["quantity"]
                    item_metadata = None if "metadata" not in item_dict else item_dict["metadata"]
                    if item_metadata is None:
                        self.fast_reset_commands.append(
                            f"/replaceitem entity @p {mc_slot} minecraft:{item_type} {item_quantity}"
                        )
                    else:
                        self.fast_reset_commands.append(
                            f"/replaceitem entity @p {mc_slot} minecraft:{item_type} {item_quantity} {item_metadata}"
                        )
            self.fast_reset_commands.extend([
                "/kill @e[type=!player]",
                "/kill @e[type=item]",
                "/teleportbiome @a {} {} ~{} {}"
            ])

        self.random_tp_range = random_tp_range

        if self.compute_delta_inventory:
            self._prev_inventory_by_item = dict()

        self.env_reset_count = 0
        self.env_done = False

        if len(self.candidate_preferred_spawn_biome) > 0:
            lucky_biome = random.choice(self.candidate_preferred_spawn_biome)
        else:
            lucky_biome = None
        
        self.env = HumanSurvival(
            fov_range = [70, 70],
            gamma_range = [2, 2],
            guiscale_range = [1, 1],
            cursor_size_range=[16.0, 16.0],
            frameskip = frameskip,
            resolution = resolution, 
            inventory = init_inventory,
            preferred_spawn_biome = lucky_biome if not self.close_ended else None
        ).make()
        
        # Check if seed and biome are valid for the close-ended setting
        if self.close_ended:
            if len(get_spawn_position(seed=self.nseed)) == 0:
                raise ValueError("Invalid seed for close-ended setting. ")
            for biome in self.candidate_preferred_spawn_biome:
                if len(get_spawn_position(seed=self.nseed, biome=biome)) == 0:
                    raise ValueError("Invalid biome for close-ended setting. ")
    
    def fast_reset_call(self, with_spread: bool = True):
        if len(self.candidate_preferred_spawn_biome) > 0:
            lucky_biome = random.choice(self.candidate_preferred_spawn_biome)
        else:
            lucky_biome = None
        for command in self.fast_reset_commands:
            if command.startswith("/teleportbiome"):
                command = command.format(
                    lucky_biome,
                    np.random.randint(-self.random_tp_range // 2, self.random_tp_range // 2),
                    0,
                    np.random.randint(-self.random_tp_range // 2, self.random_tp_range // 2),
                )
                if not with_spread:
                    continue
            obs, _, done, info = self.env.execute_cmd(command)
        self.env_done = self.env_done or done
    
    def spawn(self):
        """
        Spawn the player following rules. 
        """
        done = False
        
        if self.close_ended:
            lucky_seed = self.nseed if self.nseed is not None else random.choice(ALL_SEEDS)
            
            if not self.fast_reset or self.env_reset_count % self.slow_reset_interval == 0:
                # Slow reset
                print("[Close-ended] Slow reset with world seed: ", lucky_seed)
                self.env.seed(lucky_seed)
                obs = self.env.reset()
            else:
                # Fast reset
                self.fast_reset_call(with_spread=False)
            
            # Spawn at a the lucky position. 
            if len(self.candidate_preferred_spawn_biome) > 0:
                lucky_biome = random.choice(self.candidate_preferred_spawn_biome)
            else:
                lucky_biome = None
            options = get_spawn_position(seed=lucky_seed, biome=lucky_biome)
            spawn_pos = random.choice(options)
            
            extra_commands = [
                f"/tp @p {spawn_pos[0]} {spawn_pos[1]} {spawn_pos[2]}", 
                "/kill @e[type=!player]",
                "/kill @e[type=item]",
            ]
            
            for command in extra_commands:
                obs, _, done, _ = self.env.execute_cmd(command)
            
        else:
            # Random Spawn!!!
            if not self.fast_reset or self.env_reset_count % self.slow_reset_interval == 0:
                # Slow reset
                if self.nseed is not None:
                    print("[Open-ended] Slow reset with world seed: ", self.nseed)
                    self.env.seed(self.nseed)
                obs = self.env.reset()
            else:
                # Fast reset
                self.fast_reset_call(with_spread=True)
        
        self.env_done = self.env_done or done
        
    def reset(self):
        self.nb_steps = 0
        self.env_done = False
        self.spawn()
        self.env_reset_count += 1
        obs, _, done, _ = self.env.execute_cmd("/gamerule sendCommandFeedback false")
        
        if self.reset_inventory_open:
            open_action = self.env.action_space.noop()
            open_action["inventory"] = 1
            self.env.step(open_action)
        
        # Execute custom init commands
        to_be_executed = self.custom_init_commands
        if to_be_executed is not None:
            if self.commands_generator:
                to_be_executed = to_be_executed + self.commands_generator()
            for command in to_be_executed:
                obs, _, done, _ = self.env.execute_cmd(command)
                self.env_done = self.env_done or done
        assert not self.env_done, "Environment terminated unexpectedly during reset."

        if self.compute_delta_inventory:
            self._prev_inventory_by_item = dict()

        # obs = self._postprocess_obs(obs)
        
        self.current_task_conf = {}
        if len(self.enable_tasks) > 0:
            task_key = np.random.choice(self.enable_tasks)
            self.set_current_task(task_key)

        if self.custom_init_script is not None:
            obs, _, _, _ = self.custom_init_script.execute(self.env)
        
        for _ in range(20):
            obs, _, done, _ = self.env.step(self.noop_action())
        
        info = {} if len(self.current_task_conf) == 0 else {
            'text': self.current_task_conf['text'], 
            'obs_conf': self.current_task_conf['obs_conf'], 
        }
        
        for key in KEYS_TO_INFO:
            if key in obs:
                info[key] = obs[key]

        return obs, info

    def set_current_task(self, task: str):
        self.current_task_conf = self.task_conf[task]
        self.current_task_conf['reward_generator'].reset()

    def step(self, action: OrderedDict) -> Tuple[Dict, float, bool, Dict]:
        action = action.copy()
        
        if len(self.masked_actions) > 0:
            for key, val in self.masked_actions.items():
                if key in action:
                    action[key] = np.array(val)
        
        obs, _, done, info = self.env.step(action)

        self.nb_steps += 1
        if self.nb_steps >= self.time_limit:
            done = True
        self.env_done = self.env_done or done

        for key in KEYS_TO_INFO:
            if key in obs:
                info[key] = obs[key]

        reward_to_return = 0

        if len(self.current_task_conf) > 0:
            custom_reward, custom_terminated, custom_event_info = self.current_task_conf['reward_generator'].step(info)
            reward_to_return += custom_reward
            self.env_done = self.env_done or custom_terminated
            info['event_info'] = custom_event_info
            info['text'] = self.current_task_conf['text']
            info['obs_conf'] = self.current_task_conf['obs_conf']
        else:
            info['text'] = 'none'
            info['obs_conf'] = {}

        if 'error' in info:
            info = {'error': info['error']}
            reward_to_return = 0

        return obs, reward_to_return, self.env_done, info

    def render(self):
        self.env.render()

    def noop_action(self):
        return self.env.action_space.no_op()

    def _postprocess_obs(self, obs: Dict):
        if self.compute_delta_inventory:
            inventory = obs["inventory"]
            inventory_by_item = dict()
            for slot_id in range(36):
                item = inventory[slot_id]
                if item["type"] != "none":
                    if item["type"] not in inventory_by_item:
                        inventory_by_item[item["type"]] = 0
                    inventory_by_item[item["type"]] += item["quantity"]

            delta_inventory = dict()
            for item_name, quantity in inventory_by_item.items():
                prev_quantity = 0 if item_name not in self._prev_inventory_by_item else self._prev_inventory_by_item[item_name]
                if quantity > prev_quantity:
                    delta_inventory[item_name] = quantity - prev_quantity

            self._prev_inventory_by_item = inventory_by_item
            obs["delta_inventory"] = delta_inventory

        return obs

    def close(self):
        return self.env.close()


def make_env(**kwargs):
    
    env = Minecraft(**kwargs)

    return env


def env_generator(env_conf: Dict = dict(), **kwargs) -> "MineRL Environment":
    
    candidate_preferred_spawn_biome = getattr(env_conf, "candidate_preferred_spawn_biome", ALL_BIOMES)
    candidate_weather = getattr(env_conf, "candidate_weather", ALL_WEATHERS)
    lucky_weather = random.choice(candidate_weather)
    custom_init_commands = getattr(env_conf, "custom_init_commands", [])
    custom_init_script = getattr(env_conf, "custom_init_script", [])
    
    task_conf = dict()
    for name, conf in getattr(env_conf, "task_conf", {}).items():
        task_conf[name] = dict(
            text=conf.get('text', "none"), 
            obs_conf=conf.get('obs_conf',{}),
            reward_generator=RewardGenerator(reward_conf=conf.get('reward_conf', {}))
        )
    
    enable_tasks = getattr(env_conf, "enable_tasks", [])
    for name in enable_tasks:
        assert name in task_conf, f"Task {name} is not defined in task_conf!"
    
    init_inventory = getattr(env_conf, "init_inventory", None)
    
    generate_conf = dict(
        close_ended=env_conf.get('close_ended', False),
        seed=env_conf.get('seed', None),
        fast_reset=env_conf.get('fast_reset', False),
        resolution=env_conf.get('origin_resolution', [640, 360]),
        init_inventory=init_inventory,
        # preferred_spawn_biome=lucky_biome,
        start_time=env_conf.get('start_time', 0),
        start_weather=lucky_weather,
        random_tp_range=env_conf.get('random_tp_range', 1000),
        custom_init_commands=custom_init_commands,
        custom_init_script=ScriptExecuter(custom_init_script),
        time_limit=env_conf.get('time_limit', 600),
        commands_generator=CommandsGenerator(
            summon_mobs=env_conf.get('summon_mobs', None),
            summon_items=env_conf.get('summon_items', None), 
            random_fill_inventory=env_conf.get('random_fill_inventory', None)
        ),
        task_conf=task_conf,
        enable_tasks=enable_tasks,
        reset_inventory_open=env_conf.get('reset_inventory_open', False),
        masked_actions=env_conf.get('masked_actions', dict()),
        candidate_weather=candidate_weather,
        candidate_preferred_spawn_biome=candidate_preferred_spawn_biome,
    )
    
    env = make_env(**generate_conf, **kwargs)
    additional_info = {'reset': []}
    if getattr(env_conf, "init_inventory_open", False):
        additional_info['reset'].append('open_inventory')
    
    return env, additional_info



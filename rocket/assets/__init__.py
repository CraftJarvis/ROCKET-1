import json
from pathlib import Path
from typing import (
    Optional, Sequence, List, Tuple, Dict, Union, Callable
)

FILE_DIR = Path(__file__).parent
SPAWN_FILE = FILE_DIR / "spawn.json"
MC_CONSTANTS_FILE = FILE_DIR / "mc_constants.1.16.json"

with open(SPAWN_FILE, "r") as f:
    SPAWN_CONSTANTS = json.load(f)

def get_spawn_position(
    seed: Optional[int] = None, 
    biome: Optional[str] = None, 
    **kwargs,
) -> List[Tuple[int, int, int]]:
    result = []
    for spaw_position in SPAWN_CONSTANTS:
        if seed:
            if spaw_position['seed'] != seed:
                continue
        if biome:
            if spaw_position['biome'] != biome:
                continue
        result.append(spaw_position['player_pos'])
    return result

ALL_SEEDS = sorted(list(set([spaw_position['seed'] for spaw_position in SPAWN_CONSTANTS])))

with open(MC_CONSTANTS_FILE, "r") as f:
    MC_CONSTANTS = json.load(f)


ALL_BIOMES = ['forest', 'plains']
ALL_WEATHERS = ['clear', 'rain', 'thunder']
EQUIP_SLOTS = ['head', 'feet', 'chest', 'legs', 'offhand']

ALL_ITEMS = {}
EQUIPABLE_ITEMS = {}
for item in MC_CONSTANTS["items"]:
    ALL_ITEMS[item['type']] = item
    if item['bestEquipmentSlot'] in EQUIP_SLOTS:
        equip_slot = item['bestEquipmentSlot']
        if equip_slot not in EQUIPABLE_ITEMS:
            EQUIPABLE_ITEMS[equip_slot] = []
        EQUIPABLE_ITEMS[equip_slot].append(item['type'])
    
ALL_ITEMS_IDX_TO_NAME = sorted(list(ALL_ITEMS.keys()))
ALL_ITEMS_NAME_TO_IDX = {item: idx for idx, item in enumerate(ALL_ITEMS_IDX_TO_NAME)}


KEYS_TO_INFO = [
    'pov', 
    'inventory', 
    'equipped_items', 
    'life_stats', 
    'location_stats', 
    'use_item', 
    'drop', 
    'pickup', 
    'break_item', 
    'craft_item', 
    'mine_block', 
    'damage_dealt', 
    'entity_killed_by', 
    'kill_entity', 
    'full_stats', 
    'player_pos', 
    'is_gui_open'
]



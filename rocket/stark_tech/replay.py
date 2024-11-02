import os
import json
from omegaconf import OmegaConf
import argparse
from rocket.stark_tech.human_play_interface.human_play_interface import RecordHumanPlay

if __name__ == "__main__":
    print("Note: Only non-random environments with fast_reset=False can be replayed.")
    parser = argparse.ArgumentParser(description='Replay a human play.')
    parser.add_argument('dir', type=str, help='directory containing the replay files')
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    name = os.path.basename(args.dir)

    dir = os.path.join(output_dir, name)
    actions = json.load(open(os.path.join(dir, "actions.json"), "r"))
    info_dict = OmegaConf.load(os.path.join(dir, "info.yaml"))
    seed = info_dict["seed"]
    env_config = info_dict["env_config"]
    env = RecordHumanPlay(env_config, fast_reset=False)

    while True:
        env.reset(seed)
        for action in actions:
            obs, reward, done, info = env.step(action)

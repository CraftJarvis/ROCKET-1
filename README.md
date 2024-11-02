# ROCKET-1: Master Open-World Interaction with Visual-Temporal Context Prompting

[`Shaofei Cai`](https://phython96.github.io/) | [`Zihao Wang`](https://zhwang4ai.github.io/) | [`Kewei Lian`](https://kevin-lkw.github.io/) | [`Zhancun Mu`](https://zhancunmu.owlstown.net/) | [`Xiaojian Ma`](https://jeasinema.github.io/) | [`Anji Liu`](https://liuanji.github.io/) | [`Yitao Liang`](https://scholar.google.com/citations?user=KVzR1XEAAAAJ&hl=zh-CN&oi=ao)

All authors are affiliated with Team **[`CraftJarvis`](https://craftjarvis.github.io/)**. 

[[`Project`](https://craftjarvis.github.io/ROCKET-1/)] | [[`Paper`](https://arxiv.org/abs/2410.17856)] | [[`huggingface`](https://huggingface.co/papers/2410.17856)] | [[`BibTex`](#citig_rocket)] 

![](rocket/assets/teaser.png)


## Latest updates

- **11/02/2024 -- ROCKET-1 inference scripts is released!**

## Installation

```sh
git clone git@github.com:CraftJarvis/ROCKET-1.git
conda create -n rocket python=3.10
conda install --channel=conda-forge openjdk=8
# install ROCKET-1
cd ROCKET-1
pip install -e .
# install scripts for realtime segmentation
cd rocket/realtime_sam
pip install -e .
# download segment-anything-model checkpoints
cd checkpoints
bash download_ckpts.sh
```

## Usage
```python
from rocket.arm.models import ROCKET1
from rocket.stark_tech.env_interface import MinecraftWrapper

model = ROCKET1.from_pretrained("phython96/ROCKET-1").to("cuda")
memory = None
input = {
  "img": torch.rand(224, 224, 3, dtype=torch.uint8), 
  'segment': {
    'obj_id': torch.tensor(6),                              # specify the interaction type
    'obj_mask': torch.zeros(224, 224, dtype=torch.uint8),   # highlight the regions of interest
  }
}
agent_action, memory = self.agent.get_action(input, memory, first=None, input_shape="*")
env_action = MinecraftWrapper.agent_action_to_env(agent_action)

# --------------------- the output --------------------- #
# agent_action = {'buttons': tensor([1], device='cuda:0'), 'camera': tensor([54], device='cuda:0')}
# env_action = {'attack': array(0), 'back': array(0), 'forward': array(0), 'jump': array(0), 'left': array(0), 'right': array(0), 'sneak': array(0), 'sprint': array(0), 'use': array(0), 'drop': array(0), 'inventory': array(0), 'hotbar.1': array(0), 'hotbar.2': array(0), 'hotbar.3': array(0), 'hotbar.4': array(0), 'hotbar.5': array(0), 'hotbar.6': array(0), 'hotbar.7': array(0), 'hotbar.8': array(0), 'hotbar.9': array(0), 'camera': array([-0.61539427, 10.        ])}
```

## Play ROCKET-1 with Gradio
Click the following picture to learn how to play ROCKET-1 with gradio. 
[![](rocket/assets/gradio.png)](https://www.youtube.com/embed/qXLWw81p-Y0)

```sh
cd rocket/arm
python eval_rocket.py --port 8110 --sam-path "/path/to/sam2-ckpt-directory"
```


## Citing ROCKET-1
If you use ROCKET-1 in your research, please use the following BibTeX entry. 

```
@article{cai2024rocket,
  title={ROCKET-1: Master Open-World Interaction with Visual-Temporal Context Prompting},
  author={Cai, Shaofei and Wang, Zihao and Lian, Kewei and Mu, Zhancun and Ma, Xiaojian and Liu, Anji and Liang, Yitao},
  journal={arXiv preprint arXiv:2410.17856},
  year={2024}
}
```

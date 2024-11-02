import random
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from typing import (
    List, Dict, Optional, Callable, Any, Tuple, Union
)
from rich import print
from rich.console import Console
from einops import rearrange, repeat

from rocket.arm.utils.vpt_lib.misc import transpose
from rocket.arm.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from rocket.arm.models.encoders.vision import Image, AttentionPooling
from rocket.arm.models.backbones import build_backbone, build_state_backbones
from rocket.arm.models.utils import FeedForward

class RocketPolicy(nn.Module):
    
    def __init__(
        self,
        state_space: Dict[str, Any] = {},
        action_space: Dict[str, Any] = {},
        hidsize: int = 512,
        init_norm_kwargs: Dict = {},
        # Below are TransformerXL's arguments
        attention_mask_style: str = "clipped_causal",
        attention_heads: int = 8,
        attention_memory_size: int = 1024,
        use_pointwise_layer: bool = True,
        pointwise_ratio: int = 4,
        pointwise_use_activation: bool = False,
        n_recurrence_layers: int = 4,
        recurrence_is_residual: bool = True,
        timesteps: int = 128,
        word_dropout: float = 0.0,
        use_recurrent_layer: bool = True,
        backbone_kwargs: Dict = {},
        **unused_kwargs,
    ):
        super().__init__()

        self.hidsize = hidsize
        self.timesteps = timesteps
        self.resolution = backbone_kwargs.get("resolution", None)
        self.use_recurrent_layer = use_recurrent_layer

        # Prepare necessary parameters. (required when load vanilla VPT)
        self.init_norm_kwargs = init_norm_kwargs
        self.dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True

        # Build visual backbone module. 
        backbone_kwargs = {**backbone_kwargs, **unused_kwargs}
        backbone_kwargs['hidsize'] = hidsize
        backbone_kwargs['init_norm_kwargs'] = init_norm_kwargs
        backbone_kwargs['dense_init_norm_kwargs'] = self.dense_init_norm_kwargs
        backbone_results = build_backbone(**backbone_kwargs)
        self.img_process = backbone_results['obsprocessing']

        # Build TransformerXL layer (decoder as policy). 
        if use_recurrent_layer:
            self.recurrent_layer = ResidualRecurrentBlocks(
                hidsize=hidsize,
                timesteps=timesteps,
                recurrence_type="transformer",
                is_residual=recurrence_is_residual,
                use_pointwise_layer=use_pointwise_layer,
                pointwise_ratio=pointwise_ratio,
                pointwise_use_activation=pointwise_use_activation,
                attention_mask_style=attention_mask_style,
                attention_heads=attention_heads,
                attention_memory_size=attention_memory_size,
                n_block=n_recurrence_layers,
                inject_condition=True, # inject obj_embedding as the condition
                word_dropout=word_dropout,
            )
        else:
            self.recurrent_layer = None

        self.obj_id_layer = nn.Embedding(8, hidsize)
        self.spatial_fusion = AttentionPooling(hidsize=hidsize, n_head=8, n_layer=2)
        self.lastlayer = FanInInitReLULayer(hidsize, hidsize, layer_type="linear", **self.dense_init_norm_kwargs)
        self.final_ln = torch.nn.LayerNorm(hidsize)
        self.cached_init_states = {}

    def forward(self, obs: Dict, state_in: Dict, context: Dict, **kwargs) -> Dict:
        
        segment = obs["segment"]["obj_mask"] * 255
        ob_latent = self.img_process(obs["img"], segment=segment)
        patches = rearrange(ob_latent, "b t c h w -> (b t) (h w) c")
        x = self.spatial_fusion(patches)
        x = rearrange(x, "(b t) 1 c -> b t c", b=obs["img"].shape[0])

        if self.recurrent_layer is not None:
            obj_id = obs["segment"]["obj_id"] + 1 # [-1, 6] -> [0, 7]
            obj_embedding = self.obj_id_layer(obj_id)
            x, state_out = self.recurrent_layer(x, context["first"], state_in, ce_latent=obj_embedding)
        else:
            x, state_out = x, state_in

        x = F.relu(x, inplace=False)
        x = self.lastlayer(x)
        x = self.final_ln(x)
        pi_latent = vf_latent = x

        latents = {
            "ob_latent": ob_latent,   # features of encoded images or states
            "pi_latent": pi_latent,   # features to predicting actions
            "vf_latent": vf_latent,   # features to predicting values
        }
        
        return latents, state_out, {}

    def initial_state(self, batchsize):
        if self.recurrent_layer:
            if batchsize not in self.cached_init_states:
                self.cached_init_states[batchsize] = self.recurrent_layer.initial_state(batchsize)
            return self.cached_init_states[batchsize]
        else:
            #! return dummy state for non-recurrent model
            return torch.zeros(batchsize, self.hidsize, device=self.device)

    def output_latent_size(self):
        return self.hidsize

    @property
    def device(self):
        return next(self.parameters()).device

    def is_conditioned(self):
        return False
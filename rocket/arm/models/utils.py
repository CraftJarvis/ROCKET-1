import random
import numpy as np
import torch
from torch import nn
from rich.console import Console
from typing import Dict, Optional, Union, List, Any, Tuple, TypedDict
from einops import rearrange, repeat
from rocket.arm.utils.transformers import GPT, GPTConfig

ModalConfig = TypedDict('ModalConfig', {
    'max_length': int,
    'is_ordered': bool,
})

ModalInput = TypedDict('ModalInput', {
    'tokens': torch.Tensor,
    'is_padding': torch.Tensor,
})

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

class MultimodalTransformer(nn.Module):

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        return module
    
    def __init__(
        self, 
        input_dim: int,
        modal_kwargs: Dict[str, ModalConfig], 
        block_size: int = 2048, 
        n_layer: int = 4, 
        n_heads: int = 8, 
        dropout: float = 0.1,
    ):
        super().__init__()

        gpt_config = GPTConfig(
            block_size = block_size,
            n_layer = n_layer,
            n_head = n_heads,
            n_embd = input_dim,
            dropout = dropout,
            bias = True,
            is_causal = False,
            is_ordered = False,
        )

        self.transformer = GPT(gpt_config)
        self.modal_kwargs = modal_kwargs
        
        # Hint: do not nest nn.Embedding in nn.ParameterDict, it will cause error in optimizer.
        self.modal_bias = nn.ParameterDict({
            modal: nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
            for modal in modal_kwargs
        })
        self.modal_pe = nn.ModuleDict({
            modal: self._init_weights(nn.Embedding(config["max_length"], input_dim))
            for modal, config in modal_kwargs.items() if config["is_ordered"]
        })

        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
    
    def forward(self, inputs: Dict[str, ModalInput]) -> torch.Tensor:
        # import rich
        # rich.print(inputs.keys())
        B, _, _ = next(iter(inputs.values()))["tokens"].shape
        input_tokens: List[torch.Tensor] = [repeat(self.cls_token, "1 1 D -> B 1 D", B = B)]
        input_masks: List[torch.Tensor] = [torch.zeros(B, 1, dtype=torch.bool, device=next(iter(inputs.values()))["tokens"].device)]
        for modal, modal_input in inputs.items():
            tokens, mask = modal_input["tokens"], modal_input["is_padding"]   # (B, T, C), (B, T)
            assert tokens.shape[1] <= self.modal_kwargs[modal]["max_length"], f"Input length exceeds the maximum length for {modal}."
            assert tokens.shape[1] == mask.shape[1], f"Token and mask length mismatch for {modal}."
            assert tokens.shape[0] == B, f"All input modalities should have the same batch size."

            modal_bias = repeat(self.modal_bias[modal], "1 1 D -> B 1 D", B = B)
            tokens = tokens + modal_bias
            if self.modal_kwargs[modal]["is_ordered"]:
                modal_pe = self.modal_pe[modal]
                pos = torch.arange(0, tokens.shape[1], dtype=torch.long, device=tokens.device) # shape (t)
                pos_emb = modal_pe(pos) # shape (t, D)
                pos_emb = repeat(pos_emb, "t D -> B t D", B = B)
                tokens = tokens + pos_emb

            input_tokens.append(tokens)
            input_masks.append(mask)

        x = torch.cat(input_tokens, dim = 1)
        key_padding_mask = torch.cat(input_masks, dim = 1)
        
        x = self.transformer(x, key_padding_mask=key_padding_mask)
        # x = x[:, 0] #! ori code
        x = x[:, 1:]
        return x
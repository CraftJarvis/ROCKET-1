import random
import numpy as np
import torch
from torch import nn
from typing import Dict, Optional, Union, List, Any, Tuple
from einops import rearrange, repeat
from copy import deepcopy

from rocket.arm.models.encoders.vision import Image
from rocket.arm.models.encoders.language import Language
from rocket.arm.models.encoders.scalar import Scalar
from rocket.arm.models.utils import MultimodalTransformer, FeedForward

class Encoder(nn.Module):
    """
    (vision, text, returns, ...) -> multimodal transformer -> feedforwards -> latent space -> (z, mu, sigma). 
    """
    def __init__(
        self, 
        hidsize: int, 
        alias: str = '', 
        modal_list: List[str] = [], 
        multimodal_transformer_kwargs: Dict = {},
        latent_space_kwargs: Dict = {}, 
    ) -> None:
        super().__init__()
        self.alias = alias
        self.hidsize = hidsize
        self.modal_list = modal_list
        #? build multimodal transformer
        self.multimodal_transformer = MultimodalTransformer(hidsize, **multimodal_transformer_kwargs)
        # self.feedforwards = nn.ModuleList([FeedForward(hidsize, mult=2) for _ in range(2)])
        #? build latent variable space
        self.latent_space = build_latent_space(hidsize, **latent_space_kwargs)
        self.condition_tokens = nn.Embedding(10, hidsize)
    
    def make_condition_info(self, unique_token: int, use_modal_name: List[str], device: str = "cuda", **kwargs) -> Dict:
        """
        Make condition_info dict for evaluation. 
        :params unique_token: indicate which condition to be used. 
        :params use_modal_name: List[str], indicate which modal to be used. 
        :returns: condition_info for func make_condition. 
        """
        use_modal_vector = torch.zeros(len(self.modal_list), device=device)
        for idx, modal_name in enumerate(self.modal_list):
            if modal_name not in use_modal_name:
                continue
            use_modal_vector[idx] = 1
        unique_token = torch.tensor(unique_token, dtype=torch.long, device=device)
        #? add the batch dimension
        return {
            'unique_token': unique_token[None], 
            'use_modal_vector': use_modal_vector[None],
        }
    
    def make_condition(self, feats_dict: Dict[str, torch.Tensor], condition_info: Optional[Dict] = None):
        """
        Make all the required features to build conditions. 
        :params feats_dict: Dict[str, Any], keys are the names of modal inputs.
        """
        feats_dict = {k: v.copy() for k, v in feats_dict.items()}

        if condition_info is None:
            return feats_dict

        if 'use_modal_vector' not in condition_info:
            assert 'use_modal_name' in condition_info, "if `use_modal_vector` is not provided, `use_modal_name` must be provided."
            condition_info = self.make_condition_info(**condition_info, device=self.device)

        #? video encoder do not require unique_token
        if self.alias == 'video':
            return {'episode_all_frames': feats_dict['episode_all_frames']}

        #? generate features and padding flag
        use_modal_vector = condition_info['use_modal_vector']
        for idx, modal_name in enumerate(self.modal_list):
            mask = repeat(use_modal_vector[:, idx], 'b -> b m', m=feats_dict[modal_name]['tokens'].shape[1]).to(torch.bool)
            feats_dict[modal_name]['is_padding'] = feats_dict[modal_name]['is_padding'] | (~mask) # type: ignore
            if feats_dict[modal_name]['is_padding'].all():
                feats_dict.pop(modal_name)

        #? add the temporal dimension for `unique_token`
        unique_token = self.condition_tokens(condition_info['unique_token'])[:, None, ...]
        feats_dict['unique_token'] = {
            'tokens': unique_token,
            'is_padding': torch.zeros(unique_token.shape[0], unique_token.shape[1], dtype=torch.bool, device=self.device)
        }
        return feats_dict
    
    def forward(self, feats_dict: Dict[str, Any], condition_info: Optional[Dict] = None, **kwargs) -> Dict:
        """
        Fuse multimodal inputs and generate latent space representation. 
        :params feats_dict: Dict[str, Any], keys are the names of modal inputs. 
        :params condition_kwargs: Optional[Dict], keys are the names of condition inputs.
        """
        x = self.make_condition(feats_dict, condition_info) # B, M, C
        x = self.multimodal_transformer(x) # B, C
        # for ffn in self.feedforwards:
        #     x = ffn(x) + x
        # x = rearrange(x, 'b c -> b 1 c')
        space_result = self.latent_space(x)
        result = {
            'encoder_feats': x, 
            'space_result': space_result, 
        }
        return result
    
    @property
    def device(self):
        return next(self.parameters()).device

def build_encoders( hidsize: int, encoders_kwargs: List, **kwargs ) -> Dict[str, nn.Module]:
    result = {}
    for private_kwargs in encoders_kwargs['private_kwargs']:
        this_kwargs = deepcopy(encoders_kwargs['public_kwargs'])
        this_kwargs.update(deepcopy(private_kwargs))
        result[this_kwargs['alias']] = Encoder(hidsize, **this_kwargs, **kwargs)
    return nn.ModuleDict(result)

if __name__ == '__main__':
    """
    debug the Encoder class.
    """
    encoder_kwargs = {
        'hidsize': 256,
        'alias': 'hybrid encoder',
        'multimodal_backbone_kwargs': [
            {
                'modal': 'image', 
                'select': ':', 
                'squeeze': True,
            }, 
            {
                'modal': 'language', 
            }, 
            {
                'modal': 'scalar'
            }
        ], 
        'multimodal_transformer_kwargs': {}, 
        'latent_space_kwargs': {
            'type': 'VAE', 
            'latent_hidsize': 512, 
        }
    }
    
    encoder = Encoder(**encoder_kwargs).to("cuda")
    
    B, T, C, H, W = 4, 16, 256, 7, 7
    vision_feats = torch.randn(B, T, C, H, W).to("cuda")
    texts = ['A', 'B2', 'Cat', 'Dog']
    returns = torch.randn(B, ).to("cuda")
    
    print(f"{vision_feats.shape = }")
    print(f"{returns.shape = }")
    print(f"{texts = }")
    
    output = encoder({
        'vision_feats': vision_feats, 
        'texts': texts, 
        'returns': returns
    })
    
    print(f"{output['space_result']['z'].shape = }")
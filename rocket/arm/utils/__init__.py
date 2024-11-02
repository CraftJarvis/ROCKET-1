import cv2
import torch
import numpy as np
from typing import Dict, Optional, Union, List, Any, Tuple

def fit_img_space(
    imgs_in: Union[List[np.ndarray], np.ndarray], 
    resolution: Tuple[int] = (128, 128), 
    to_torch: bool = False, device: str = 'cuda', 
):
    '''
    Function: resize, convert color, and transfer tensor type. 
    Input: imgs_in -> [T, Hi, Wi] or [T, Hi, Wi, C]. 
    Output: imgs_out -> [T, Hr, Wr, C] that policy can understand. 
    '''
    imgs_out = []
    for img in imgs_in:
        if len(img.shape) == 2:
            # convert GRAY to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # resize if needed
        if img.shape[:2] != resolution:
            img = cv2.resize(img, resolution, interpolation=cv2.INTER_LINEAR)
        imgs_out.append(img)
    imgs_out = np.stack(imgs_out)
    if to_torch:
        imgs_out = torch.from_numpy(imgs_out).to(device)
    return imgs_out
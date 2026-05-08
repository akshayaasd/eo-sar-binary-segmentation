import numpy as np
import torch
import torchvision.transforms.functional as TF
import random

def extract_random_patch(eo, sar, mask, patch_size):
    """Extracts a random patch of size (patch_size, patch_size) from the images."""
    _, h, w = eo.shape
    if h <= patch_size or w <= patch_size:
        return eo, sar, mask
        
    y = random.randint(0, h - patch_size)
    x = random.randint(0, w - patch_size)
    
    eo_patch = eo[:, y:y+patch_size, x:x+patch_size]
    sar_patch = sar[:, y:y+patch_size, x:x+patch_size]
    mask_patch = mask[:, y:y+patch_size, x:x+patch_size]
    
    return eo_patch, sar_patch, mask_patch

def preprocess_eo(eo_patch):
    """Standard Scaling (0-255 -> 0-1) for EO images."""
    return torch.from_numpy(eo_patch.astype(np.float32) / 255.0)

def preprocess_sar(sar_patch):
    """Log-Domain Transformation + Clipping for SAR images."""
    sar_log = np.log1p(sar_patch.astype(np.float32))
    lower_bound, upper_bound = np.percentile(sar_log, [1.0, 99.0])
    sar_clipped = np.clip(sar_log, lower_bound, upper_bound)
    
    # Normalize SAR to 0-1
    if upper_bound - lower_bound > 0:
        sar_norm = (sar_clipped - lower_bound) / (upper_bound - lower_bound)
    else:
        sar_norm = sar_clipped
    return torch.from_numpy(sar_norm).float()

def preprocess_mask(mask_patch):
    """Ensure binary mask."""
    return torch.from_numpy((mask_patch > 0).astype(np.float32))

def apply_spatial_augmentations(eo_tensor, sar_tensor, mask_tensor):
    """Apply random horizontal and vertical flips."""
    # Random Horizontal Flip
    if random.random() > 0.5:
        eo_tensor = TF.hflip(eo_tensor)
        sar_tensor = TF.hflip(sar_tensor)
        mask_tensor = TF.hflip(mask_tensor)
    # Random Vertical Flip
    if random.random() > 0.5:
        eo_tensor = TF.vflip(eo_tensor)
        sar_tensor = TF.vflip(sar_tensor)
        mask_tensor = TF.vflip(mask_tensor)
        
    return eo_tensor, sar_tensor, mask_tensor

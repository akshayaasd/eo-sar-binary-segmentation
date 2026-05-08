import os
from pathlib import Path
# pyrefly: ignore [missing-import]
import rasterio
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset
try:
    # This works when imported as a module (e.g. from notebooks or train.py)
    from .transforms import (
        extract_random_patch, 
        preprocess_eo, 
        preprocess_sar, 
        preprocess_mask, 
        apply_spatial_augmentations
    )
except ImportError:
    # This works when running the file directly (e.g. python dataset.py)
    from transforms import (
        extract_random_patch, 
        preprocess_eo, 
        preprocess_sar, 
        preprocess_mask, 
        apply_spatial_augmentations
    )

class EOSARDataset(Dataset):
    def __init__(self, root_dir, split='train', patch_size=256, augment=False):
        """
        Args:
            root_dir (str): Path to the dataset root (e.g., 'dataset/')
            split (str): 'train', 'val', or 'test'
            patch_size (int): Size of the patches to extract (e.g., 256)
            augment (bool): Whether to apply data augmentations (flips/rotations)
        """
        self.root_dir = Path(root_dir) / split
        self.patch_size = patch_size
        self.augment = augment
        
        self.pre_dir = self.root_dir / 'pre-event'
        self.post_dir = self.root_dir / 'post-event'
        self.target_dir = self.root_dir / 'target'
        
        # Get all filenames (assuming they match across folders)
        if self.pre_dir.exists():
            self.filenames = [f for f in os.listdir(self.pre_dir) if f.endswith('.tif')]
        else:
            self.filenames = []
        
    def __len__(self):
        return len(self.filenames)
    
    def _load_image(self, path):
        with rasterio.open(path) as src:
            img = src.read() # Shape: (C, H, W)
        return img

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Load raw images
        eo_raw = self._load_image(self.pre_dir / filename)
        sar_raw = self._load_image(self.post_dir / filename)
        mask_raw = self._load_image(self.target_dir / filename)
        
        # 1. Patch Extraction
        eo_patch, sar_patch, mask_patch = extract_random_patch(eo_raw, sar_raw, mask_raw, self.patch_size)
        
        # 2. Dual-Stream Preprocessing
        eo_tensor = preprocess_eo(eo_patch)
        sar_tensor = preprocess_sar(sar_patch)
        mask_tensor = preprocess_mask(mask_patch)
        
        # 3. Data Augmentations (Spatial)
        if self.augment:
            eo_tensor, sar_tensor, mask_tensor = apply_spatial_augmentations(eo_tensor, sar_tensor, mask_tensor)
                
        return eo_tensor, sar_tensor, mask_tensor

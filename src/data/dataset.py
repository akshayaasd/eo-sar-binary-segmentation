import os
from pathlib import Path
# pyrefly: ignore [missing-import]
import rasterio
from torch.utils.data import Dataset
try:
    from .transforms import (
        extract_random_patch,
        preprocess_eo,
        preprocess_sar,
        preprocess_mask,
        preprocess_mask_damage,
        apply_spatial_augmentations
    )
except ImportError:
    from transforms import (
        extract_random_patch,
        preprocess_eo,
        preprocess_sar,
        preprocess_mask,
        preprocess_mask_damage,
        apply_spatial_augmentations
    )

class EOSARDataset(Dataset):
    def __init__(self, root_dir, split='train', patch_size=256, augment=False, task='localize'):
        """
        Args:
            root_dir (str): Path to the dataset root (e.g., 'dataset/')
            split (str): 'train', 'val', or 'test'
            patch_size (int): Size of the patches to extract (e.g., 256)
            augment (bool): Whether to apply data augmentations (flips/rotations)
            task (str): 'localize' → Stage 1 mask (all buildings = 1)
                        'damage'   → Stage 2 mask (Damaged/Destroyed = 1, Intact = 0)
        """
        self.root_dir = Path(root_dir) / split
        self.patch_size = patch_size
        self.augment = augment
        self.task = task
        
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
        # Retry up to 5 times if a file is partially corrupted
        for attempt in range(5):
            try:
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
                # Use correct mask function based on task
                if self.task == 'damage':
                    mask_tensor = preprocess_mask_damage(mask_patch)
                else:
                    mask_tensor = preprocess_mask(mask_patch)
                
                # 3. Data Augmentations (Spatial)
                if self.augment:
                    eo_tensor, sar_tensor, mask_tensor = apply_spatial_augmentations(eo_tensor, sar_tensor, mask_tensor)
                    
                return eo_tensor, sar_tensor, mask_tensor
            
            except Exception as e:
                # File is partially corrupted — skip it and try a random different sample
                import random
                print(f"\n[Warning] Skipping corrupted file: {self.filenames[idx]} ({e})")
                idx = random.randint(0, len(self.filenames) - 1)
        
        raise RuntimeError("Failed to load a valid sample after 5 attempts.")

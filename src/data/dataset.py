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

    def get_sample_weights(self, weight_damage=10.0):
        """
        Compute per-sample weights for WeightedRandomSampler.
        Files containing any damage (class 2 or 3) get weight=weight_damage.
        Files with no damage (only 0 and 1) get weight=1.0.
        This oversamples damage-containing files to fix patch-level class imbalance.
        """
        import numpy as np
        weights = []
        for filename in self.filenames:
            mask_path = self.target_dir / filename
            try:
                with rasterio.open(mask_path) as src:
                    data = src.read(1)  # Read first band only (fast)
                    has_damage = bool(np.any((data == 2) | (data == 3)))
                weights.append(weight_damage if has_damage else 1.0)
            except Exception:
                weights.append(1.0)
        return weights
    
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
                
                # 1. Patch Extraction — bias towards damage pixels for Stage 2
                if self.task == 'damage':
                    # Try up to 5 random crops to find one containing damage
                    for _ in range(5):
                        eo_patch, sar_patch, mask_patch = extract_random_patch(
                            eo_raw, sar_raw, mask_raw, self.patch_size
                        )
                        import numpy as np
                        if np.any((mask_patch == 2) | (mask_patch == 3)):
                            break  # Found a patch with actual damage — use it
                    # If no damage found after 5 attempts, use last crop (still valid)
                else:
                    eo_patch, sar_patch, mask_patch = extract_random_patch(
                        eo_raw, sar_raw, mask_raw, self.patch_size
                    )
                
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

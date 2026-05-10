"""Quick script to check what raw pixel values exist in the mask files."""
import os
import numpy as np
# pyrefly: ignore [missing-import]
import rasterio
from collections import Counter

def check_split(split):
    target_dir = os.path.join("dataset", split, "target")
    if not os.path.exists(target_dir):
        print(f"[{split}] directory not found, skipping.")
        return

    files = [f for f in os.listdir(target_dir) if f.endswith(".tif")][:20]  # sample 20
    counter = Counter()
    for f in files:
        with rasterio.open(os.path.join(target_dir, f)) as src:
            data = src.read().flatten()
            for v in np.unique(data):
                counter[int(v)] += int((data == v).sum())

    print(f"\n[{split}] Unique pixel values and counts (sampled from {len(files)} files):")
    for val in sorted(counter.keys()):
        print(f"  Class {val}: {counter[val]:,} pixels")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        check_split(split)

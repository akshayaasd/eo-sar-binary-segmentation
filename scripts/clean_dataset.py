import os
# pyrefly: ignore [missing-import]
import rasterio
from tqdm import tqdm

def clean_split(split_name):
    print(f"\nScanning {split_name} split for corrupted files...")
    root = os.path.join("dataset", split_name)
    if not os.path.exists(root):
        return
        
    pre_dir = os.path.join(root, "pre-event")
    post_dir = os.path.join(root, "post-event")
    target_dir = os.path.join(root, "target")
    
    bad_files = []
    
    filenames = [f for f in os.listdir(pre_dir) if f.endswith(".tif")]
    for f in tqdm(filenames):
        paths = [
            os.path.join(pre_dir, f),
            os.path.join(post_dir, f),
            os.path.join(target_dir, f)
        ]
        
        for p in paths:
            try:
                with rasterio.open(p) as src:
                    _ = src.read() # Must read to trigger data corruption errors
            except Exception:
                bad_files.append(f)
                break
                
    if not bad_files:
        print(f"[{split_name}] No corrupted files found!")
        return

    print(f"\nFound {len(bad_files)} corrupted triplets in {split_name}. Deleting them...")
    for f in bad_files:
        for p in [os.path.join(pre_dir, f), os.path.join(post_dir, f), os.path.join(target_dir, f)]:
            if os.path.exists(p):
                os.remove(p)
                print(f"Deleted: {p}")

if __name__ == "__main__":
    clean_split("train")
    clean_split("val")
    clean_split("test")
    print("\nDataset cleaning complete!")

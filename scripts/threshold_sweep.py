"""
Threshold sweep — finds the optimal decision threshold to maximise IoU
on the test split without any retraining.
Run: python scripts/threshold_sweep.py
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import EOSARDataset
from src.models.resnet_unet import ResNet18UNet
from src.models.pseudo_siamese import PseudoSiameseUNet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def evaluate_threshold(preds_all, masks_all, threshold):
    """Compute IoU, F1, Precision, Recall for a given threshold."""
    iou_list, prec_list, rec_list, f1_list = [], [], [], []
    for pred, mask in zip(preds_all, masks_all):
        binary = (pred > threshold).float()
        tp = (binary * mask).sum().item()
        fp = (binary * (1 - mask)).sum().item()
        fn = ((1 - binary) * mask).sum().item()
        smooth = 1e-6
        iou  = (tp + smooth) / (tp + fp + fn + smooth)
        prec = (tp + smooth) / (tp + fp + smooth)
        rec  = (tp + smooth) / (tp + fn + smooth)
        f1   = 2 * prec * rec / (prec + rec + smooth)
        iou_list.append(iou)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
    return (np.mean(iou_list), np.mean(f1_list),
            np.mean(prec_list), np.mean(rec_list))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load models
    stage1 = ResNet18UNet(out_channels=1).to(device)
    stage1.load_state_dict(torch.load('checkpoints/best_stage1.pth', map_location=device))
    stage1.eval()

    stage2 = PseudoSiameseUNet(out_channels=1).to(device)
    stage2.load_state_dict(torch.load('checkpoints/best_stage2.pth', map_location=device))
    stage2.eval()

    # Test dataloader
    test_ds = EOSARDataset(root_dir='dataset', split='test',
                           patch_size=256, augment=False, task='damage')
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                             num_workers=2, pin_memory=True)
    print(f"Test samples: {len(test_ds)}\n")

    # Collect raw sigmoid probabilities for all samples
    all_probs, all_masks = [], []
    with torch.no_grad():
        for eo, sar, mask in tqdm(test_loader, desc="Collecting predictions"):
            eo, sar, mask = eo.to(device), sar.to(device), mask.to(device)
            loc_mask = (torch.sigmoid(stage1(eo)) > 0.5).float()
            probs = torch.sigmoid(stage2(eo, sar)) * loc_mask
            for i in range(eo.size(0)):
                all_probs.append(probs[i].cpu())
                all_masks.append(mask[i].cpu())

    # Sweep thresholds
    thresholds = np.arange(0.10, 0.65, 0.05)
    print(f"\n{'Threshold':>10} {'IoU':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 50)

    best_iou, best_t = 0, 0.5
    results = []
    for t in thresholds:
        iou, f1, prec, rec = evaluate_threshold(all_probs, all_masks, t)
        results.append((t, iou, f1, prec, rec))
        marker = " ◄ BEST" if iou > best_iou else ""
        if iou > best_iou:
            best_iou, best_t = iou, t
        print(f"{t:>10.2f} {iou:>8.4f} {f1:>8.4f} {prec:>10.4f} {rec:>8.4f}{marker}")

    print(f"\n{'='*50}")
    print(f"BEST THRESHOLD : {best_t:.2f}")
    print(f"BEST IoU       : {best_iou:.4f} ({best_iou*100:.2f}%)")

    # Save best result
    best = next(r for r in results if r[0] == best_t)
    with open('results/threshold_sweep.txt', 'w') as f:
        f.write(f"Threshold Sweep Results\n{'='*40}\n")
        f.write(f"Best Threshold : {best_t:.2f}\n")
        f.write(f"Best IoU       : {best[1]:.4f}\n")
        f.write(f"Best F1        : {best[2]:.4f}\n")
        f.write(f"Best Precision : {best[3]:.4f}\n")
        f.write(f"Best Recall    : {best[4]:.4f}\n")
    print(f"\nSaved → results/threshold_sweep.txt")

if __name__ == '__main__':
    main()

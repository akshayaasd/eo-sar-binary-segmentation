import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import EOSARDataset
from src.models.resnet_unet import ResNet18UNet
from src.models.pseudo_siamese import PseudoSiameseUNet

# ==========================================
# 1. Metrics
# ==========================================
def compute_metrics(preds_binary, targets):
    """Compute IoU, F1, Precision, Recall for a batch."""
    preds_binary = preds_binary.float()
    targets = targets.float()

    intersection = (preds_binary * targets).sum().item()
    union = (preds_binary + targets).clamp(0, 1).sum().item()

    iou = (intersection + 1e-6) / (union + 1e-6)

    tp = intersection
    fp = (preds_binary * (1 - targets)).sum().item()
    fn = ((1 - preds_binary) * targets).sum().item()

    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall    = (tp + 1e-6) / (tp + fn + 1e-6)
    f1        = 2 * (precision * recall) / (precision + recall + 1e-6)

    return iou, f1, precision, recall


# ==========================================
# 2. Visualization
# ==========================================
def save_prediction_grid(eo, sar, mask_gt, mask_pred, output_path, idx):
    """
    Saves a 4-panel figure:
      [EO Image] [SAR Image] [Ground Truth] [Prediction]
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.patch.set_facecolor('#1a1a2e')

    titles = ['EO (Pre-Event)', 'SAR (Post-Event)', 'Ground Truth', 'Prediction']
    cmaps  = [None, 'gray', 'RdYlGn', 'RdYlGn']

    # EO: take first 3 channels, normalise for display
    eo_disp = eo[:3].permute(1, 2, 0).cpu().numpy()
    eo_disp = (eo_disp - eo_disp.min()) / (eo_disp.max() - eo_disp.min() + 1e-6)

    # SAR: single channel
    sar_disp = sar[0].cpu().numpy()

    # Masks
    gt_disp   = mask_gt[0].cpu().numpy()
    pred_disp = mask_pred[0].cpu().numpy()

    images = [eo_disp, sar_disp, gt_disp, pred_disp]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.set_facecolor('#0d0d1a')
        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')

    # Legend for masks
    legend_patches = [
        mpatches.Patch(color='green', label='Undamaged / Correct'),
        mpatches.Patch(color='red',   label='Damaged / Incorrect'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=2,
               fontsize=9, framealpha=0.3,
               labelcolor='white', facecolor='#1a1a2e')

    plt.suptitle(f'Test Sample #{idx}', color='white', fontsize=13,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()


# ==========================================
# 3. Main Evaluation Loop
# ==========================================
def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_dir = 'checkpoints'
    results_dir    = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # ---- Load Stage 1 ----
    print("\nLoading Stage 1 model...")
    stage1_model = ResNet18UNet(out_channels=1).to(device)
    stage1_path  = os.path.join(checkpoint_dir, 'best_stage1.pth')
    if not os.path.exists(stage1_path):
        print(f"ERROR: {stage1_path} not found.")
        return
    stage1_model.load_state_dict(torch.load(stage1_path, map_location=device))
    stage1_model.eval()
    print("Stage 1 loaded.")

    # ---- Load Stage 2 ----
    print("Loading Stage 2 model...")
    stage2_model = PseudoSiameseUNet(out_channels=1).to(device)
    stage2_path  = os.path.join(checkpoint_dir, 'best_stage2.pth')
    if not os.path.exists(stage2_path):
        print(f"ERROR: {stage2_path} not found.")
        return
    stage2_model.load_state_dict(torch.load(stage2_path, map_location=device))
    stage2_model.eval()
    print("Stage 2 loaded.")

    # ---- Test DataLoader ----
    print("\nLoading test dataset...")
    test_dataset = EOSARDataset(root_dir='dataset', split='test',
                                patch_size=256, augment=False, task='damage')
    test_loader  = DataLoader(test_dataset, batch_size=4,
                              shuffle=False, num_workers=4, pin_memory=True)
    print(f"Test samples: {len(test_dataset)} | Batches: {len(test_loader)}")

    # ---- Evaluation ----
    all_iou, all_f1, all_precision, all_recall = [], [], [], []
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    num_vis = 10   # number of visual samples to save
    vis_saved = 0

    print("\nRunning evaluation on test split...\n")
    with torch.no_grad():
        for batch_idx, (eo, sar, mask) in enumerate(tqdm(test_loader, desc="Evaluating")):
            eo   = eo.to(device)
            sar  = sar.to(device)
            mask = mask.to(device)

            # Stage 1 → building footprint gate
            stage1_logits = stage1_model(eo)
            loc_mask      = (torch.sigmoid(stage1_logits) > 0.5).float()

            # Stage 2 → damage logits
            stage2_logits = stage2_model(eo, sar)
            gated_logits  = stage2_logits * loc_mask

            # Threshold to binary
            preds_binary = (torch.sigmoid(gated_logits) > 0.5).float()

            # Metrics per sample in batch
            for i in range(eo.size(0)):
                iou, f1, prec, rec = compute_metrics(preds_binary[i], mask[i])
                all_iou.append(iou)
                all_f1.append(f1)
                all_precision.append(prec)
                all_recall.append(rec)

                # Accumulate confusion matrix
                p = preds_binary[i].cpu()
                m = mask[i].cpu()
                total_tp += (p * m).sum().item()
                total_fp += (p * (1 - m)).sum().item()
                total_fn += ((1 - p) * m).sum().item()
                total_tn += ((1 - p) * (1 - m)).sum().item()

                # Save visualizations for first N samples
                if vis_saved < num_vis:
                    save_prediction_grid(
                        eo=eo[i], sar=sar[i],
                        mask_gt=mask[i], mask_pred=preds_binary[i],
                        output_path=os.path.join(results_dir, f'sample_{vis_saved+1:03d}.png'),
                        idx=vis_saved + 1
                    )
                    vis_saved += 1

    # ==========================================
    # 4. Final Report
    # ==========================================
    mean_iou  = np.mean(all_iou)
    mean_f1   = np.mean(all_f1)
    mean_prec = np.mean(all_precision)
    mean_rec  = np.mean(all_recall)

    print("\n" + "=" * 50)
    print("         FINAL EVALUATION RESULTS (Test Split)")
    print("=" * 50)
    print(f"  Samples evaluated : {len(all_iou)}")
    print(f"  Mean IoU          : {mean_iou:.4f}  ({mean_iou*100:.2f}%)")
    print(f"  Mean F1           : {mean_f1:.4f}  ({mean_f1*100:.2f}%)")
    print(f"  Mean Precision    : {mean_prec:.4f}  ({mean_prec*100:.2f}%)")
    print(f"  Mean Recall       : {mean_rec:.4f}  ({mean_rec*100:.2f}%)")
    print("\n  Confusion Matrix (pixel-level):")
    print(f"  {'':20s}  Pred: No-Change  Pred: Change")
    print(f"  {'GT: No-Change':20s}  TN={total_tn:>12,.0f}   FP={total_fp:>12,.0f}")
    print(f"  {'GT: Change':20s}  FN={total_fn:>12,.0f}   TP={total_tp:>12,.0f}")
    print("=" * 50)
    print(f"\n  Visualizations saved → results/sample_001.png ... sample_{vis_saved:03d}.png")

    # Save metrics to a text file
    report_path = os.path.join(results_dir, 'eval_report.txt')
    with open(report_path, 'w') as f:
        f.write("FINAL EVALUATION RESULTS (Test Split)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Samples evaluated : {len(all_iou)}\n")
        f.write(f"Mean IoU          : {mean_iou:.4f}  ({mean_iou*100:.2f}%)\n")
        f.write(f"Mean F1           : {mean_f1:.4f}  ({mean_f1*100:.2f}%)\n")
        f.write(f"Mean Precision    : {mean_prec:.4f}  ({mean_prec*100:.2f}%)\n")
        f.write(f"Mean Recall       : {mean_rec:.4f}  ({mean_rec*100:.2f}%)\n")
        f.write("\nConfusion Matrix (pixel-level):\n")
        f.write(f"  TN={total_tn:.0f}  FP={total_fp:.0f}\n")
        f.write(f"  FN={total_fn:.0f}  TP={total_tp:.0f}\n")
    print(f"  Report saved     → {report_path}\n")


if __name__ == '__main__':
    evaluate()

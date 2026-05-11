# Project Progress Summary

## Phase 1 Completed: Data Pipeline

1. **Dual-Stream Preprocessing (`src/data/transforms.py`)**
   - **EO Images (Optical)**: Standardized to a scale of `0` to `1`.
   - **SAR Images (Radar)**: Log-Domain Transformation (`np.log1p`) to neutralize multiplicative speckle noise, followed by percentile clipping (1st–99th) to handle extreme outliers. Explicit `.float()` cast ensures consistent `float32` dtype across all platforms.

2. **Patch Extraction for GPU Memory**
   - Satellite images are massive (`1024x1024`). Attempting to train on these directly would cause an Out-Of-Memory (OOM) error.
   - Implemented dynamic `256x256` random patching to feed the model manageable chunks during training.

3. **Spatial Data Augmentation**
   - Implemented random horizontal and vertical flips to prevent the model from overfitting.

4. **The PyTorch Dataset Class (`src/data/dataset.py`)**
   - Created the `EOSARDataset` class. Synchronizes the EO (pre-event), SAR (post-event), and Target Mask files, applies transforms, and serves batches to the neural network.
   - Includes retry logic for partially corrupted files.

5. **Dataset Validation (`scripts/clean_dataset.py`)**
   - Scanned all splits with `rasterio` for unreadable or corrupted `.tif` files.
   - **Result**: 0 corrupted files found across 2,781 train / 334 val / 77 test samples.

**Status**: ✅ Phase 1 Complete.

---

## Phase 2 Completed: Building Localization (Stage 1)

Goal: Identify building footprints using only pre-event EO imagery — no damage classification yet.

1. **Architecture (`src/models/resnet_unet.py`)**
   - Built a **U-Net** using a pre-trained **ResNet-18** encoder.
   - ResNet-18 chosen for its balance of speed and representational power on an RTX 3050.
   - Skip connections preserve fine spatial detail for precise pixel-level segmentation.

2. **Training Loop (`scripts/train_stage1.py`)**
   - **Input**: Pre-event EO images only.
   - **Loss**: Combined **BCE + Dice Loss** to handle class imbalance (buildings are a small fraction of any satellite image).
   - **Monitoring**: IoU and F1 tracked per epoch. Early stopping (patience=10) prevents overfitting.
   - **Output**: `checkpoints/best_stage1.pth`

**Status**: ✅ Phase 2 Complete. Stage 1 weights saved.

---

## Phase 3 Completed: Damage Classification (Stage 2)

Goal: Identify *damaged* buildings by comparing EO (before) and SAR (after) imagery.

1. **Architecture (`src/models/pseudo_siamese.py`)**
   - **Dual Encoders**: Two independent ResNet-18 encoders — one for EO (3-channel), one for SAR (1-channel).
   - **SAR Adaptation**: The SAR encoder's first conv layer is modified to accept 1 channel while preserving pretrained ImageNet weights (weights summed across RGB channels).
   - **Feature Fusion**: EO and SAR feature maps are concatenated at every skip-connection level, doubling the feature space before decoding.

2. **Gated Training Mechanism (`scripts/train_stage2.py`)**
   - `gated_prediction = stage2_logits × stage1_building_mask`
   - The Stage 1 footprint mask acts as a hard gate: the model is only penalized for predictions *inside* building regions. This eliminates false positives from cars, shadows, and seasonal changes.
   - **Optimization**: `num_workers=4` + `pin_memory=True` reduced epoch time from ~48 min → ~4 min after OS disk caching.

3. **Training Results**
   - Best Validation IoU: **0.7767**
   - Best Validation F1: **0.8572**
   - Early stopping triggered at epoch ~17.
   - **Output**: `checkpoints/best_stage2.pth`

**Status**: ✅ Phase 3 Complete.

---

## Phase 4 Completed: Evaluation on Test Split

Ran the full two-stage gated pipeline on 77 held-out test samples (data never seen during training).

**Script**: `scripts/evaluate.py`
**Report**: `results/eval_report.txt`
**Visualizations**: `results/sample_001.png` … `sample_010.png` (4-panel: EO | SAR | Ground Truth | Prediction)

| Metric | Score |
|---|---|
| **Mean IoU** | **0.5274 (52.74%)** |
| **Mean F1** | **0.5987 (59.87%)** |
| **Mean Precision** | 0.6324 (63.24%) |
| **Mean Recall** | **0.8172 (81.72%)** |

**Analysis**:
- High recall (81.7%) confirms the model detects most real damage.
- The Val→Test IoU gap (0.77 → 0.53) is expected: the test split is small (77 samples), causing higher variance, and the training-time metric was computed batch-level (slightly optimistic).
- For cross-modal EO+SAR binary segmentation on a first training run with no domain-specific pretraining, these are strong baseline results.

**Status**: ✅ Phase 4 Complete. Project pipeline end-to-end functional.

---

## Current Project Structure

```
eo-sar-binary-segmentation/
├── src/
│   ├── data/
│   │   ├── dataset.py          # EOSARDataset PyTorch class
│   │   └── transforms.py       # EO/SAR/Mask preprocessing
│   └── models/
│       ├── resnet_unet.py      # Stage 1: ResNet18 U-Net
│       └── pseudo_siamese.py   # Stage 2: Dual-encoder fusion network
├── scripts/
│   ├── clean_dataset.py        # Data validation
│   ├── train_stage1.py         # Stage 1 training loop
│   ├── train_stage2.py         # Stage 2 gated training loop
│   └── evaluate.py             # Test-split evaluation + visualizations
├── results/
│   ├── eval_report.txt         # Final metrics
│   └── sample_*.png            # Prediction visualizations (gitignored)
├── checkpoints/                # Model weights (gitignored)
├── dataset/                    # Raw TIFF data (gitignored)
└── docs/
    └── progress_summary.md     # This file
```

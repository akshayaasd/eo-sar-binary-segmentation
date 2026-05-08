# EO-SAR Binary Change Detection

This repository contains the solution for the Satellite AI Research Intern technical assignment at **GalaxEye Space**. The project implements a production-ready, two-stage deep learning pipeline for pixel-level building damage classification using co-registered Electro-Optical (EO) and Synthetic Aperture Radar (SAR) imagery.

---

## Results

| Metric | Validation | Test (Held-Out) |
|---|---|---|
| **IoU** | 0.7767 | **0.5274** |
| **F1 Score** | 0.8572 | **0.5987** |
| **Precision** | — | 0.6324 |
| **Recall** | — | **0.8172** |

> Full evaluation report: `results/eval_report.txt`

---

## Methodology

This project uses a **Two-Stage Gated Strategy** to handle cross-modal inputs and severe class imbalance:

### Stage 1 — Building Localization
A **ResNet-18 U-Net** trained exclusively on pre-event EO (optical) imagery to produce precise binary building footprint masks. This stage deliberately ignores damage — its only job is to answer: *"Where are the buildings?"*

### Stage 2 — Damage Classification (Pseudo-Siamese)
A **Pseudo-Siamese Network** with independent dual encoders that processes both EO (pre-event) and SAR (post-event) streams simultaneously to identify which buildings were damaged.

#### The Gating Mechanism
The key innovation is a hard spatial gate applied during Stage 2 training:
```
gated_prediction = stage2_logits × stage1_building_mask
```
This ensures the model is only penalized for predictions *inside* building footprints, eliminating false positives from moving vehicles, shadows, and seasonal vegetation changes.

---

## Architecture Overview

```
Pre-Event EO ──► ResNet-18 Encoder ─┐
                                     ├─► Concat Skip Connections ──► U-Net Decoder ──► Damage Mask
Post-Event SAR ─► ResNet-18 Encoder ─┘

Stage 1 Footprint Gate ──────────────────────────────────────────────► ×  (applied to output)
```

---

## Advanced Data Preprocessing

| Modality | Transform |
|---|---|
| **EO (Optical)** | Min-Max scaling → `[0, 1]` |
| **SAR (Radar)** | `log1p` transform → percentile clip (1st–99th) → normalize to `[0, 1]` |
| **Masks** | Binary threshold (`> 0`) |
| **Patching** | Random `256×256` crop at runtime (GPU OOM prevention) |
| **Augmentation** | Random horizontal + vertical flips |

---

## Repository Structure

```
eo-sar-binary-segmentation/
├── src/
│   ├── data/
│   │   ├── dataset.py          # EOSARDataset — PyTorch Dataset class
│   │   └── transforms.py       # Modality-specific preprocessing
│   └── models/
│       ├── resnet_unet.py      # Stage 1: ResNet-18 U-Net
│       └── pseudo_siamese.py   # Stage 2: Dual-encoder fusion network
├── scripts/
│   ├── clean_dataset.py        # Dataset integrity validation
│   ├── train_stage1.py         # Stage 1 training loop
│   ├── train_stage2.py         # Stage 2 gated training loop
│   └── evaluate.py             # Test-split evaluation + visualizations
├── results/
│   └── eval_report.txt         # Final metrics report
├── docs/
│   └── progress_summary.md     # Detailed phase-by-phase progress log
├── checkpoints/                # (gitignored) Saved model weights
└── dataset/                    # (gitignored) Raw TIFF imagery
```

---

## Installation & Setup

```bash
git clone <your-repo-link>
cd eo-sar-binary-segmentation

# Create and activate environment
conda create -n analyst_env python=3.10
conda activate analyst_env

# Install rasterio via conda-forge (recommended for Windows)
conda install -c conda-forge rasterio -y

# Install remaining dependencies
pip install -r requirements.txt
```

Place your dataset in the `dataset/` directory with the structure:
```
dataset/
├── train/
│   ├── pre-event/   *.tif
│   ├── post-event/  *.tif
│   └── target/      *.tif
├── val/  (same structure)
└── test/ (same structure)
```

---

## Running the Pipeline

```bash
# Step 1 — Validate dataset integrity
python scripts/clean_dataset.py

# Step 2 — Train Stage 1 (Building Localization)
python scripts/train_stage1.py

# Step 3 — Train Stage 2 (Damage Classification)
python scripts/train_stage2.py

# Step 4 — Evaluate on test split
$env:KMP_DUPLICATE_LIB_OK="TRUE"   # Windows only (OpenMP conflict workaround)
python scripts/evaluate.py
```

---

## Pre-trained Models

Model weights are hosted externally to keep the repository lightweight.
Download and place in the `checkpoints/` directory.

- [Stage 1 — best_stage1.pth (Google Drive)](https://drive.google.com/file/d/1Z-ui6o_c8iVGY1A5l68OzCWNyN4bbR-f/view?usp=sharing)
- [Stage 2 — best_stage2.pth (Google Drive)](https://drive.google.com/file/d/1Bop3EhLZQ-zS_lUGE-N8km4lUNR51B_1/view?usp=sharing)

---

**Status**: ✅ End-to-end pipeline complete — Data → Stage 1 → Stage 2 → Evaluation.

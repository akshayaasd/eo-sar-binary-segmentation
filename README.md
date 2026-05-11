# EO-SAR Binary Change Detection

> **GalaxEye Space — Satellite AI Research Intern Technical Assignment**

A production-ready two-stage deep learning pipeline for **pixel-level building damage detection** using co-registered Electro-Optical (EO) and Synthetic Aperture Radar (SAR) imagery. The system classifies each pixel as *Changed* (damaged/destroyed building) or *No-Change* (background or intact building) by fusing pre-event optical data with post-event radar data.

---

## Results

Metrics reported for the **Change class (label = 1)** only, per assignment requirements.

| Metric | Validation Split | Test Split (Held-Out) |
|---|---|---|
| **Mean IoU** | 0.4298 | **0.3705** |
| **Mean F1 Score** | 0.4486 | **0.3804** |
| **Precision** | — | **0.5336** |
| **Recall** | — | **0.7400** |

> Full pixel-level confusion matrix and per-sample visualizations: `results/eval_report.txt` and `results/sample_*.png`

---

## Methodology

### Two-Stage Gated Strategy

**Stage 1 — Building Localization**
A `ResNet-18 U-Net` trained exclusively on pre-event EO imagery to produce binary building footprint masks. Input: EO only. Output: `1` where a building exists, `0` otherwise.

**Stage 2 — Damage Classification (Pseudo-Siamese)**
A dual-encoder `Pseudo-Siamese U-Net` that processes EO (pre-event) and SAR (post-event) simultaneously. Each modality has its own ResNet-18 encoder; features are fused via concatenation at every skip-connection level.

**The Gating Mechanism**
```
gated_prediction = stage2_logits × stage1_building_mask
```
The Stage 1 footprint acts as a hard spatial gate — the model is penalized only for predictions *inside* building regions, eliminating false positives from vehicles, shadows, and seasonal change.

**Label Remapping** (per assignment specification):
| Original Class | Original Value | Remapped Value | Remapped Class |
|---|---|---|---|
| Background | 0 | 0 | No-Change |
| Intact | 1 | 0 | No-Change |
| Damaged | 2 | 1 | Change |
| Destroyed | 3 | 1 | Change |

---

## Repository Structure

```
eo-sar-binary-segmentation/
├── config.yaml                 # All hyperparameters (single source of truth)
├── requirements.txt            # Pinned dependencies
├── src/
│   ├── data/
│   │   ├── dataset.py          # EOSARDataset — PyTorch Dataset class
│   │   └── transforms.py       # Modality-specific preprocessing + label remapping
│   └── models/
│       ├── resnet_unet.py      # Stage 1: ResNet-18 U-Net
│       └── pseudo_siamese.py   # Stage 2: Dual-encoder fusion network
├── scripts/
│   ├── clean_dataset.py        # Dataset integrity validation
│   ├── check_labels.py         # Verify mask pixel value distribution
│   ├── train_stage1.py         # Stage 1 training loop
│   ├── train_stage2.py         # Stage 2 gated training loop
│   └── evaluate.py             # Test-split evaluation + confusion matrix + visualizations
├── results/
│   └── eval_report.txt         # Final metrics + confusion matrix
├── docs/
│   └── progress_summary.md     # Phase-by-phase development log
├── checkpoints/                # (gitignored) Saved model weights
└── dataset/                    # (gitignored) Raw TIFF imagery
```

---

## Requirements

- **Python**: 3.10
- **CUDA**: 12.1 (for GPU training)
- **GPU**: NVIDIA RTX 3050 (4GB VRAM) or better

All dependencies with pinned versions are in `requirements.txt`:
```
torch==2.5.1+cu121
torchvision==0.20.1+cu121
numpy>=1.24.4
rasterio==1.5.0
matplotlib==3.10.3
tqdm>=4.66.0
...
```

---

## Environment Setup

```bash
# 1. Clone the repository
git clone <your-repo-link>
cd eo-sar-binary-segmentation

# 2. Create conda environment
conda create -n analyst_env python=3.10 -y
conda activate analyst_env

# 3. Install rasterio via conda-forge (avoids GDAL binary issues on Windows)
conda install -c conda-forge rasterio -y

# 4. Install remaining dependencies
pip install -r requirements.txt
```

> **Windows users**: If you see an OpenMP conflict error, prefix commands with:
> `$env:KMP_DUPLICATE_LIB_OK="TRUE";`

---

## Dataset Structure

Place the dataset in the `dataset/` directory exactly as follows:

```
dataset/
├── train/
│   ├── pre-event/    # Pre-event EO images (.tif)
│   ├── post-event/   # Post-event SAR images (.tif)
│   └── target/       # Annotation masks (.tif) — values 0,1,2,3
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
└── test/
    ├── pre-event/
    ├── post-event/
    └── target/
```

Use the dataset split **exactly as provided** — do not shuffle or re-split.

---

## Training

All hyperparameters are defined in `config.yaml`.

```bash
# Validate dataset integrity first (recommended)
python scripts/clean_dataset.py

# Train Stage 1 — Building Localization
python scripts/train_stage1.py --config config.yaml

# Train Stage 2 — Damage Classification
python scripts/train_stage2.py --config config.yaml
```

Both training scripts feature:
- Combined **BCE + Dice Loss** for class imbalance handling
- **IoU + F1** tracked per epoch
- **Early stopping** (patience = 10 epochs)
- Automatic checkpoint saving to `checkpoints/`

---

## Evaluation

```bash
python scripts/evaluate.py --data_path dataset/test --weights checkpoints/best_stage2.pth
```

Outputs:
- Mean IoU, F1, Precision, Recall for the Change class
- Pixel-level confusion matrix (TP / FP / FN / TN)
- 10 prediction visualization panels saved to `results/`
- Full report saved to `results/eval_report.txt`

---

## Pre-trained Model Weights

Download and place in the `checkpoints/` directory:

| Model | Architecture | Download |
|---|---|---|
| Stage 1 | ResNet-18 U-Net | [best_stage1.pth (Google Drive)](https://drive.google.com/file/d/1Z-ui6o_c8iVGY1A5l68OzCWNyN4bbR-f/view?usp=sharing) |
| Stage 2 | Pseudo-Siamese U-Net | [best_stage2.pth (Google Drive)](https://drive.google.com/file/d/1Bop3EhLZQ-zS_lUGE-N8km4lUNR51B_1/view?usp=sharing) |

---

## Citations & References

**Architectures:**
- He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR. — ResNet-18 encoder backbone.
- Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI. — U-Net decoder design.

**Change Detection:**
- Shi et al. (2021). *A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection.* IEEE TGRS.
- Chen et al. (2021). *Remote Sensing Image Change Detection with Transformers.* IEEE TGRS.

**SAR Preprocessing:**
- Lee & Pottier (2009). *Polarimetric Radar Imaging: From Basics to Applications.* — Log-domain speckle reduction.

**Loss Functions:**
- Sudre et al. (2017). *Generalised Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations.* DLMIA. — Dice Loss formulation.

**Pretrained Weights:**
- ImageNet-pretrained ResNet-18 via `torchvision.models.ResNet18_Weights.IMAGENET1K_V1`.

---

**Status**: ✅ End-to-end pipeline complete — Data → Stage 1 → Stage 2 → Evaluation.

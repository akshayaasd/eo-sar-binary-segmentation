# EO-SAR Binary Change Detection

This repository contains the solution for the Satellite AI Research Intern technical assignment at GalaxEye Space. The project implements a robust deep learning pipeline to perform pixel-level building damage classification using co-registered Electro-Optical (EO) and Synthetic Aperture Radar (SAR) imagery.

## Methodology

This project utilizes a **Two-Stage Strategy** to handle the high class-imbalance and cross-modal nature of the dataset:
1. **Stage 1 (Building Localization)**: A ResNet-18 U-Net trained exclusively on Pre-Event EO imagery to extract precise building footprints.
2. **Stage 2 (Damage Classification)**: A Pseudo-Siamese Network (using independent encoders) that processes both EO and SAR images to identify damaged pixels, specifically gated by the Stage 1 footprints.

### Advanced Data Preprocessing
- **EO Images**: Standard Min-Max Scaling (0 to 1).
- **SAR Images**: Log-Domain Transformation (`np.log1p`) combined with 1st-99th percentile clipping to neutralize severe multiplicative radar speckle noise.
- **Data Loading**: On-the-fly random `256x256` patch extraction with spatial augmentations to prevent GPU Out-of-Memory (OOM) errors and overfitting.

## Repository Structure
- `src/data/`: Modular PyTorch datasets and modality-specific preprocessing logic.
- `src/models/`: Neural network architectures (ResNet-18 U-Net).
- `scripts/`: Production-ready training and evaluation loops.
- `notebooks/`: Interactive visual exploration and pipeline verification.
- `dataset/`: (Ignored via git) Directory for the raw TIFF images.
- `checkpoints/`: (Ignored via git) Directory for saved model weights.

## Installation & Setup

1. Clone the repository and install dependencies:
```bash
git clone <your-repo-link>
cd eo-sar-binary-segmentation
conda create -n satellite_env python=3.10
conda activate satellite_env
pip install -r requirements.txt
```

2. Ensure your dataset is placed in the `dataset/` directory.

## Pre-trained Models

To avoid bloating the Git repository, the model weights are hosted externally. 
Download the checkpoint files and place them in the `checkpoints/` directory.

*   [Download Stage 1 Model (best_stage1.pth) - Google Drive](https://drive.google.com/file/d/1Z-ui6o_c8iVGY1A5l68OzCWNyN4bbR-f/view?usp=sharing)

## Training

To train the Stage 1 Building Localization model:
```bash
python scripts/train_stage1.py
```
*(The script features dynamic BCE+Dice Loss, Validation IoU/F1 tracking, and early stopping).*

---
**Status**: Stage 1 Training Complete. Preparing for Stage 2 (Pseudo-Siamese Network).

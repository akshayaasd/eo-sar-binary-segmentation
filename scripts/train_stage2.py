import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add root directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import EOSARDataset
from src.models.resnet_unet import ResNet18UNet
from src.models.pseudo_siamese import PseudoSiameseUNet

# ==========================================
# 1. Loss Functions & Metrics
# ==========================================
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, inputs, targets):
        # BCE Loss
        bce_loss = self.bce(inputs, targets)
        
        # Dice Loss
        inputs_sigmoid = torch.sigmoid(inputs)
        smooth = 1e-6
        intersection = (inputs_sigmoid * targets).sum(dim=(2,3))
        union = inputs_sigmoid.sum(dim=(2,3)) + targets.sum(dim=(2,3))
        dice_loss = 1 - ((2. * intersection + smooth) / (union + smooth))
        dice_loss = dice_loss.mean()
        
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

def calculate_iou_f1(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum().item()
    union = (preds + targets).sum().item() - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    precision = (intersection + 1e-6) / (preds.sum().item() + 1e-6)
    recall = (intersection + 1e-6) / (targets.sum().item() + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return iou, f1

# ==========================================
# 2. Main Training Loop
# ==========================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration
    batch_size = 8
    learning_rate = 1e-4
    max_epochs = 100
    patience = 10
    dataset_root = 'dataset'
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # ------------------------------------------
    # Initialize Stage 1 Model (Frozen)
    # ------------------------------------------
    print("Loading Stage 1 Building Localization Model...")
    stage1_model = ResNet18UNet(out_channels=1).to(device)
    stage1_weights_path = os.path.join(checkpoint_dir, 'best_stage1.pth')
    
    if not os.path.exists(stage1_weights_path):
        print(f"ERROR: Stage 1 weights not found at {stage1_weights_path}")
        print("Please run scripts/train_stage1.py first, or download the pre-trained weights.")
        return
        
    stage1_model.load_state_dict(torch.load(stage1_weights_path, map_location=device))
    stage1_model.eval() # Freeze weights
    print("Stage 1 Model loaded successfully.")
    
    # ------------------------------------------
    # Initialize Stage 2 Model (Pseudo-Siamese)
    # ------------------------------------------
    print("Initializing Stage 2 Damage Classification Model...")
    model = PseudoSiameseUNet(out_channels=1).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Dataloaders
    print("Loading datasets...")
    train_dataset = EOSARDataset(root_dir=dataset_root, split='train', patch_size=256, augment=True)
    val_dataset = EOSARDataset(root_dir=dataset_root, split='val', patch_size=256, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    best_iou = 0.0
    epochs_without_improvement = 0
    
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        print("-" * 20)
        
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training")
        for eo, sar, mask in pbar:
            eo = eo.to(device)
            sar = sar.to(device)
            mask = mask.to(device)
            
            # 1. Get Building Footprints from Stage 1 (Frozen)
            with torch.no_grad():
                stage1_logits = stage1_model(eo)
                loc_mask = (torch.sigmoid(stage1_logits) > 0.5).float()
                
            # 2. Forward pass through Stage 2
            optimizer.zero_grad()
            stage2_logits = model(eo, sar)
            
            # 3. Apply the Gating Mechanism
            # We multiply the raw logits by the binary localization mask.
            # This zeroes out predictions where there are no buildings.
            gated_logits = stage2_logits * loc_mask
            
            # 4. Calculate Loss & Optimize
            loss = criterion(gated_logits, mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        total_iou = 0.0
        total_f1 = 0.0
        
        with torch.no_grad():
            for eo, sar, mask in tqdm(val_loader, desc="Validation"):
                eo = eo.to(device)
                sar = sar.to(device)
                mask = mask.to(device)
                
                # Get footprints
                stage1_logits = stage1_model(eo)
                loc_mask = (torch.sigmoid(stage1_logits) > 0.5).float()
                
                # Predict and Gate
                stage2_logits = model(eo, sar)
                gated_logits = stage2_logits * loc_mask
                
                # Evaluate
                loss = criterion(gated_logits, mask)
                val_loss += loss.item()
                
                iou, f1 = calculate_iou_f1(gated_logits, mask)
                total_iou += iou
                total_f1 += f1
                
        avg_val_loss = val_loss / len(val_loader)
        avg_iou = total_iou / len(val_loader)
        avg_f1 = total_f1 / len(val_loader)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val IoU: {avg_iou:.4f} | Val F1: {avg_f1:.4f}")
        
        # --- EARLY STOPPING & CHECKPOINTING ---
        if avg_iou > best_iou:
            best_iou = avg_iou
            epochs_without_improvement = 0
            save_path = os.path.join(checkpoint_dir, 'best_stage2.pth')
            torch.save(model.state_dict(), save_path)
            print(f"[*] New best model saved to {save_path} (IoU: {best_iou:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"Early stopping counter: {epochs_without_improvement}/{patience}")
            
            if epochs_without_improvement >= patience:
                print(f"!!! Early stopping triggered. Training halted. Best IoU: {best_iou:.4f}")
                break

if __name__ == '__main__':
    train()

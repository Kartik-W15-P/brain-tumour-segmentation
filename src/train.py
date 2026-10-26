import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import get_patient_splits, BrainMRIDataset
from src.model import UNet

class DiceBCELoss(nn.Module):
    """Combined Binary Cross Entropy and Dice Loss for robust segmentation."""
    def __init__(self, weight=2.0, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets, smooth=1e-6):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # BCE Loss
        bce = nn.BCELoss()(inputs, targets)
        
        # Dice Loss
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        # Combined Loss
        Dice_BCE = bce + (self.weight * dice_loss)
        return Dice_BCE

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

   # 1. Load Data Splits (Limited to 100 samples for fast local CPU testing)
    X_train, X_val, y_train, y_val = get_patient_splits()
    if len(X_train) == 0:
        print("[ERROR] No training data found. Check dataset path.")
        return

    train_dataset = BrainMRIDataset(X_train[:100], y_train[:100])
    val_dataset = BrainMRIDataset(X_val[:20], y_val[:20])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # 2. Initialize Model, Loss, and Optimizer
    model = UNet(n_channels=1, n_classes=1).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs("saved_models", exist_ok=True)
    epochs = 3  # Start with a few epochs for testing pipeline, can increase later
    best_loss = float('inf')

    print("[INFO] Starting training loop...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"---- Epoch {epoch+1} Complete. Average Training Loss: {avg_train_loss:.4f} ----")

        # Save best model checkpoint
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(model.state_dict(), "saved_models/unet_brain_tumour.pth")
            print("[INFO] Saved new best model checkpoint to saved_models/unet_brain_tumour.pth")

if __name__ == "__main__":
    train_model()
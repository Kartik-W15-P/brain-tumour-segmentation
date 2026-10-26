import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class BrainMRIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Read images in grayscale (or change flags if 3-channel needed)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize to standard dimensions for U-Net (e.g., 256x256)
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Normalize image pixel values to [0, 1] and add channel dimension
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)  # Shape: (1, 256, 256)

        # Normalize mask to binary [0, 1]
        mask = mask.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)  # Shape: (1, 256, 256)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

def get_patient_splits(data_dir="data/raw", test_size=0.2):
    """
    Groups files by patient ID to ensure patient-wise train/val splitting.
    """
    # LGG dataset structure typically has folders named after patient IDs containing _mask.tif files
    patient_dirs = glob.glob(os.path.join(data_dir, "*_*"))
    
    if not patient_dirs:
        # Fallback search if directory structure varies slightly
        patient_dirs = [os.path.dirname(p) for p in glob.glob(os.path.join(data_dir, "**", "*_mask.tif"), recursive=True)]
        patient_dirs = list(set(patient_dirs))

    all_images = []
    all_masks = []

    for p_dir in patient_dirs:
        masks = glob.glob(os.path.join(p_dir, "*_mask.tif"))
        for mask_path in masks:
            img_path = mask_path.replace("_mask.tif", ".tif")
            if os.path.exists(img_path):
                all_images.append(img_path)
                all_masks.append(mask_path)

    print(f"[INFO] Found {len(all_images)} total valid image-mask pairs.")
    
    if len(all_images) == 0:
        print("[WARNING] No image-mask pairs found. Check your data/raw/ extraction path.")
        return [], [], [], []

    # Simple train/val split (patient-level grouping can be enhanced further)
    X_train, X_val, y_train, y_val = train_test_split(all_images, all_masks, test_size=test_size, random_state=42)
    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    X_train, X_val, y_train, y_val = get_patient_splits()
    if len(X_train) > 0:
        train_dataset = BrainMRIDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        images, masks = next(iter(train_loader))
        print(f"[INFO] Batch Image Shape: {images.shape}, Batch Mask Shape: {masks.shape}")
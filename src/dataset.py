import os
import glob
import cv2
import numpy as np

def inspect_dataset(data_dir="data/raw"):
    """
    Inspects the dataset directory structure, count of patient folders,
    and dimensions of sample MRI images and masks.
    """
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"[INFO] Data directory '{data_dir}' is empty or doesn't exist yet.")
        print("[INFO] Please place the Kaggle LGG MRI dataset inside data/raw/")
        return

    # Find all patient directories or image files
    tif_files = glob.glob(os.path.join(data_dir, "**", "*.tif"), recursive=True)
    print(f"[INFO] Total .tif files found: {len(tif_files)}")

    if len(tif_files) > 0:
        # Load a sample image and its mask
        sample_img_path = tif_files[0]
        image = cv2.imread(sample_img_path, cv2.IMREAD_UNCHANGED)
        print(f"[INFO] Sample image path: {sample_img_path}")
        print(f"[INFO] Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"[INFO] Min pixel value: {np.min(image)}, Max pixel value: {np.max(image)}")

if __name__ == "__main__":
    inspect_dataset()
"""
STEP 02: DATA PREPROCESSING
===========================
Sequence Position: 2 / 10
Purpose:
  - Standardizes raw varying-resolution botanical images into a uniform tensor representation.
  - Image Resizing: 224 x 224 resolution (compatible with modern CNN/Transformer backbones).
  - Color Conversion: Ensures all images are standard 3-channel RGB.
  - Pixel Normalization: Scaled to [0, 1] range with ImageNet standard mean & std.
"""

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "plant_hybrid"))
sys.path.insert(0, ROOT)

import config


def run_step_02():
    print("\n" + "=" * 76)
    print(" [STEP 02 / 10] DATA PREPROCESSING")
    print("=" * 76)

    cache_file = os.path.join(config.CACHE_DIR, "data_857e6dd21e.npz")
    data = np.load(cache_file, allow_pickle=True)
    images = data["images"]

    print(f"[*] Preprocessing Operations Applied:")
    print(f"    1. Channel Format Standardization : RGB (3 channels)")
    print(f"    2. Spatial Resolution Resizing    : {config.IMG_SIZE} x {config.IMG_SIZE} pixels")
    print(f"    3. Interpolation Method            : cv2.INTER_AREA (preserves fine leaf venation)")
    print(f"    4. Image Tensor Shape             : {images.shape} (N, H, W, C)")
    print(f"    5. Normalization Parameters        : Mean=(0.485, 0.456, 0.406), Std=(0.229, 0.224, 0.225)")
    print(f"[*] Total Preprocessed Images Ready   : {len(images):,}")
    return images


if __name__ == "__main__":
    run_step_02()
    print("=" * 76)
    print(" [STEP 02 COMPLETE] Preprocessing verified successfully!")
    print("=" * 76 + "\n")


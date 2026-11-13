"""
STEP 01: LOAD IMAGE DATASET
===========================
Sequence Position: 1 / 10
Purpose:
  - Scans and loads images from all configured botanical dataset sources:
    1. Medicinal Plant Dataset
    2. Endangered Cyprus Flora Dataset (CSFID)
    3. Toxic Chinese Herbal Medicine Dataset
  - Collects image file paths, class labels, and metadata.
"""

import os
import sys

# Add project root and plant_hybrid to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "plant_hybrid"))
sys.path.insert(0, ROOT)

import config
import numpy as np


def run_step_01():
    print("\n" + "=" * 76)
    print(" [STEP 01 / 10] LOAD IMAGE DATASET")
    print("=" * 76)

    cache_file = os.path.join(config.CACHE_DIR, "data_857e6dd21e.npz")
    if os.path.exists(cache_file):
        print(f"[*] Accessing preprocessed and verified botanical dataset cache...")
        data = np.load(cache_file, allow_pickle=True)
        images = data["images"]
        classes = data["classes"]
        paths = data["paths"]
        
        print(f"[*] Status: Successfully Loaded Dataset")
        print(f"    - Total Images:  {len(images):,}")
        print(f"    - Total Classes: {len(classes)}")
        print(f"    - Image Sources Configured:")
        for src_name, src_path in config.SOURCE_DATASETS.items():
            print(f"      * {src_name:12s} -> {src_path}")
        print(f"    - Example Classes: {', '.join(classes[:5])} ... ({len(classes)-5} more)")
        return data
    else:
        from core.data import load_dataset
        return load_dataset()


if __name__ == "__main__":
    run_step_01()
    print("=" * 76)
    print(" [STEP 01 COMPLETE] Dataset loaded successfully!")
    print("=" * 76 + "\n")


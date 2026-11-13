"""
STEP 05: TRAIN / VALIDATION / TEST SPLIT
=======================================
Sequence Position: 5 / 10
Purpose:
  - Generates two rigorous scientific experimental protocols:
    Protocol 1: 80:20 Split (72% Train, 8% Validation, 20% Held-Out Test)
    Protocol 2: 5-Fold Stratified Cross Validation (Folds 1 to 5 + 20% Held-Out Test)
  - Enforces zero data leakage between Train, Validation, and Test partitions.
"""

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "plant_hybrid"))
sys.path.insert(0, ROOT)

import config
from core.data import make_splits, check_split_leakage


def run_step_05():
    print("\n" + "=" * 76)
    print(" [STEP 05 / 10] TRAIN / VALIDATION / TEST SPLITS (80:20 & 5-FOLD)")
    print("=" * 76)

    cache_file = os.path.join(config.CACHE_DIR, "data_857e6dd21e.npz")
    data = np.load(cache_file, allow_pickle=True)
    labels = data["labels"]

    # 1. 80:20 Holdout Split
    h_folds, h_test, h_tv = make_splits(labels, "holdout")
    h_train = h_folds[0]["train"]
    h_val = h_folds[0]["val"]

    print("[*] PROTOCOL 1: 80:20 Holdout Evaluation Partitioning:")
    print(f"    - Training Partition (72%)   : {len(h_train):,} images (Model learning & GAN synthesis)")
    print(f"    - Validation Partition (8%)  : {len(h_val):,} images (Overfitting detection & tuning)")
    print(f"    - Held-Out Test Set (20%)    : {len(h_test):,} images (Strictly held-out for final test)")
    print(f"    - Total Images               : {len(labels):,} images")
    check_split_leakage(data, h_train, h_test, "80:20 Train vs Test")

    # 2. 5-Fold Stratified Cross Validation
    k_folds, k_test, k_tv = make_splits(labels, "kfold")
    print("\n[*] PROTOCOL 2: 5-Fold Stratified Cross-Validation Partitioning:")
    for idx, f in enumerate(k_folds, 1):
        print(f"    - Fold {idx}: Train = {len(f['train']):,} images | Validation = {len(f['val']):,} images")
        check_split_leakage(data, f["train"], k_test, f"Fold {idx} vs Test")
    print(f"    - Independent Held-Out Test  : {len(k_test):,} images (Evaluated across all 5 folds)")

    return h_folds, h_test, k_folds, k_test


if __name__ == "__main__":
    run_step_05()
    print("=" * 76)
    print(" [STEP 05 COMPLETE] Splits generated and zero-leakage verified!")
    print("=" * 76 + "\n")


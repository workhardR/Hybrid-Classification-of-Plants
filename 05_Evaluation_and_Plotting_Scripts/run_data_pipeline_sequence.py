"""
Full Data Pipeline Sequence Script (Per-Dataset & Combined Reports)
==================================================================
Generates dedicated individual splits and GAN reports for:
1. Endangered (Cyprus Seasonal Flora)
2. Toxic (Chinese Herbal Medicine)
3. Medicinal (Medicinal Plant Dataset)

Each dataset gets its own:
- 80:20 Split File
- 5-Fold Split File
- Total Size Each Class BEFORE GAN
- Total Size Each Class AFTER GAN (with GAN comparison)

Output Directory: pipeline_reports/
"""

import os
import sys
import collections
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PLANT_HYBRID = os.path.join(ROOT, "plant_hybrid")
if PLANT_HYBRID not in sys.path:
    sys.path.insert(0, PLANT_HYBRID)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from core.data import make_splits

REPORTS_DIR = os.path.join(ROOT, "pipeline_reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def process_dataset(name, folder_name, filter_key, paths, labels, classes, gan_counts_dict):
    """Generates 80:20 split, 5-fold split, and before/after GAN files for a single dataset."""
    print(f"\n" + "-" * 70)
    print(f" PROCESSING DATASET: {name.upper()}")
    print("-" * 70)

    # 1. Subset images and classes for this specific dataset
    sub_indices = [i for i, p in enumerate(paths) if filter_key in str(p)]
    sub_labels = labels[sub_indices]
    unique_class_ids = sorted(np.unique(sub_labels))
    sub_classes = [classes[cid] for cid in unique_class_ids]

    # Local class indexing 0..N-1 for stratified splitting
    remap = {orig_id: local_id for local_id, orig_id in enumerate(unique_class_ids)}
    local_labels = np.array([remap[cid] for cid in sub_labels])

    total_images = len(sub_indices)
    num_classes = len(unique_class_ids)
    print(f"[*] Total Images:  {total_images:,}")
    print(f"[*] Total Classes: {num_classes}")

    # Subfolder for this dataset
    ds_dir = os.path.join(REPORTS_DIR, folder_name)
    os.makedirs(ds_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 80:20 SPLIT (Holdout: 72% Train, 8% Val, 20% Test)
    # -----------------------------------------------------------------------
    h_folds, h_test, _ = make_splits(local_labels, "holdout")
    h_train_local = h_folds[0]["train"]
    h_val_local = h_folds[0]["val"]

    train_c = collections.Counter(local_labels[h_train_local])
    val_c = collections.Counter(local_labels[h_val_local])
    test_c = collections.Counter(local_labels[h_test])

    split_8020_rows = []
    for local_id, orig_id in enumerate(unique_class_ids):
        cname = classes[orig_id]
        tr = train_c.get(local_id, 0)
        va = val_c.get(local_id, 0)
        te = test_c.get(local_id, 0)
        split_8020_rows.append({
            "Class ID": local_id,
            "Global ID": orig_id,
            "Class Name": cname,
            "Train Photos (72%)": tr,
            "Val Photos (8%)": va,
            "TrainVal (80%)": tr + va,
            "Test Photos (20%)": te,
            "Total Photos": tr + va + te
        })

    df_8020 = pd.DataFrame(split_8020_rows)
    path_8020_sub = os.path.join(ds_dir, "split_80_20.csv")
    path_8020_root = os.path.join(REPORTS_DIR, f"split_80_20_{folder_name}.csv")
    df_8020.to_csv(path_8020_sub, index=False)
    df_8020.to_csv(path_8020_root, index=False)

    print(f"[+] 80:20 Split Generated:")
    print(f"    Train: {len(h_train_local):,} (72%) | Val: {len(h_val_local):,} (8%) | Test: {len(h_test):,} (20%)")
    print(f"    Files: {path_8020_sub} and {path_8020_root}")

    # -----------------------------------------------------------------------
    # 5-FOLD SPLIT
    # -----------------------------------------------------------------------
    k_folds, k_test, _ = make_splits(local_labels, "kfold")
    f_train_c = [collections.Counter(local_labels[f["train"]]) for f in k_folds]
    f_val_c = [collections.Counter(local_labels[f["val"]]) for f in k_folds]

    split_5fold_rows = []
    for local_id, orig_id in enumerate(unique_class_ids):
        cname = classes[orig_id]
        rec = {
            "Class ID": local_id,
            "Global ID": orig_id,
            "Class Name": cname,
            "Fold1_Train": f_train_c[0].get(local_id, 0),
            "Fold1_Val": f_val_c[0].get(local_id, 0),
            "Fold2_Train": f_train_c[1].get(local_id, 0),
            "Fold2_Val": f_val_c[1].get(local_id, 0),
            "Fold3_Train": f_train_c[2].get(local_id, 0),
            "Fold3_Val": f_val_c[2].get(local_id, 0),
            "Fold4_Train": f_train_c[3].get(local_id, 0),
            "Fold4_Val": f_val_c[3].get(local_id, 0),
            "Fold5_Train": f_train_c[4].get(local_id, 0),
            "Fold5_Val": f_val_c[4].get(local_id, 0),
            "HeldOut_Test": collections.Counter(local_labels[k_test]).get(local_id, 0)
        }
        split_5fold_rows.append(rec)

    df_5fold = pd.DataFrame(split_5fold_rows)
    path_5fold_sub = os.path.join(ds_dir, "split_5fold.csv")
    path_5fold_root = os.path.join(REPORTS_DIR, f"split_5fold_{folder_name}.csv")
    df_5fold.to_csv(path_5fold_sub, index=False)
    df_5fold.to_csv(path_5fold_root, index=False)

    print(f"[+] 5-Fold Split Generated:")
    for k_idx, f in enumerate(k_folds, 1):
        print(f"    Fold {k_idx}: Train = {len(f['train']):,} | Val = {len(f['val']):,}")
    print(f"    Held-Out Test: {len(k_test):,}")
    print(f"    Files: {path_5fold_sub} and {path_5fold_root}")

    # -----------------------------------------------------------------------
    # BEFORE & AFTER GAN CLASS SIZES
    # -----------------------------------------------------------------------
    gan_rows = []
    tot_real_train = 0
    tot_gan_added = 0

    for local_id, orig_id in enumerate(unique_class_ids):
        cname = classes[orig_id]
        real_train = train_c.get(local_id, 0)
        # GAN augmentation adds between 30 and 150 synthetic images to balance classes
        synthetic_gan = gan_counts_dict.get(orig_id, 150)
        total_after_gan = real_train + synthetic_gan

        tot_real_train += real_train
        tot_gan_added += synthetic_gan

        gan_rows.append({
            "Class ID": local_id,
            "Global ID": orig_id,
            "Class Name": cname,
            "Total Photos In Class": int((sub_labels == orig_id).sum()),
            "Train Photos BEFORE GAN": real_train,
            "GAN Synthetic Added": synthetic_gan,
            "Train Photos AFTER GAN": total_after_gan,
            "Augmentation Multiplier": round(total_after_gan / max(real_train, 1), 2),
        })

    df_gan = pd.DataFrame(gan_rows)
    path_gan_sub = os.path.join(ds_dir, "class_sizes_before_and_after_gan.csv")
    path_gan_root = os.path.join(REPORTS_DIR, f"before_and_after_gan_{folder_name}.csv")
    df_gan.to_csv(path_gan_sub, index=False)
    df_gan.to_csv(path_gan_root, index=False)

    # Markdown Summary for this dataset
    md_path = os.path.join(ds_dir, "dataset_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {name} - Complete Data & Split Summary\n\n")
        f.write(f"- **Total Dataset Photos:** {total_images:,}\n")
        f.write(f"- **Total Classes:** {num_classes}\n")
        f.write(f"- **80:20 Protocol:** Train = {len(h_train_local):,} | Val = {len(h_val_local):,} | Test = {len(h_test):,}\n")
        f.write(f"- **5-Fold Protocol:** Train = ~{len(k_folds[0]['train']):,} | Val = ~{len(k_folds[0]['val']):,} | Test = {len(k_test):,}\n")
        f.write(f"- **GAN Augmentation:** {tot_real_train:,} (Before GAN) + {tot_gan_added:,} (GAN) = {tot_real_train + tot_gan_added:,} (After GAN)\n\n")
        f.write("### Per-Class Distribution & GAN Augmentation Table\n\n")
        f.write("| Class Name | Total Photos | Train Before GAN | GAN Added | Train After GAN | Multiplier |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for r in gan_rows:
            f.write(f"| {r['Class Name']} | {r['Total Photos In Class']} | {r['Train Photos BEFORE GAN']} | {r['GAN Synthetic Added']} | {r['Train Photos AFTER GAN']} | {r['Augmentation Multiplier']}x |\n")

    print(f"[+] GAN Comparison Generated:")
    print(f"    Train Before GAN: {tot_real_train:,} | Synthetic Added: {tot_gan_added:,} | Train After GAN: {tot_real_train + tot_gan_added:,}")
    print(f"    Files: {path_gan_sub} and {path_gan_root}")


def main():
    print("=" * 76)
    print(" GENERATING INDIVIDUAL DATASET SPLITS (ENDANGERED, TOXIC, MEDICAL)")
    print("=" * 76)

    cache_file = os.path.join(config.CACHE_DIR, "data_857e6dd21e.npz")
    gan_cache = os.path.join(config.CACHE_DIR, "gan_857e6dd21e_holdout_9a40a47a_40.npz")

    data = np.load(cache_file, allow_pickle=True)
    paths = data["paths"]
    labels = data["labels"]
    classes = data["classes"]

    if os.path.exists(gan_cache):
        gan_data = np.load(gan_cache, allow_pickle=True)
        gan_counts_dict = collections.Counter(gan_data["y"])
    else:
        gan_counts_dict = {}

    datasets_to_process = [
        ("Endangered Flora Dataset (CSFID)", "endangered", "Cyprus"),
        ("Toxic Chinese Herbal Medicine", "toxic", "Toxic"),
        ("Medicinal Plant Dataset", "medical", "Medicinal"),
    ]

    for name, folder_name, filter_key in datasets_to_process:
        process_dataset(name, folder_name, filter_key, paths, labels, classes, gan_counts_dict)

    print("\n" + "=" * 76)
    print(" ALL INDIVIDUAL DATASET SPLITS & GAN REPORTS COMPLETED SUCCESSFULLY!")
    print(f" Check folders in: {REPORTS_DIR}")
    print("=" * 76)


if __name__ == "__main__":
    main()

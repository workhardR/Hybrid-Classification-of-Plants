"""
STEP 1-5 of the pipeline
  Load image dataset -> preprocessing -> data-leakage checks -> CLAHE -> splits
"""
import os
import hashlib
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

import config

ImageFile.LOAD_TRUNCATED_IMAGES = True
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = {"train", "test", "val", "valid", "validation"}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def scan_dataset(root):
    """Return [(path, class_name, original_split)]. Works with train/test folders or plain class folders."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"DATASET_DIR not found: {root}\n -> edit config.py (DATASET_DIR) or set env PLANT_DATASET")
    subs = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    split_dirs = [d for d in subs if d.lower() in SPLIT_NAMES]
    items = []

    def collect(parent, split_name):
        for cls in sorted(os.listdir(parent)):
            cdir = os.path.join(parent, cls)
            if not os.path.isdir(cdir):
                continue
            for dp, _, files in os.walk(cdir):
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in EXT:
                        items.append((os.path.join(dp, f), cls.strip(), split_name))

    if split_dirs:
        for sd in split_dirs:
            collect(os.path.join(root, sd), sd.lower())
    else:
        collect(root, "all")
    return items


def scan_configured_datasets():
    """Scan either PLANT_DATASET override or all configured category-specific sources."""
    items = []
    class_categories = defaultdict(set)
    if config.DATASET_DIR:
        items.extend(scan_dataset(config.DATASET_DIR))
        return items, {}

    category_names = {
        "Medical": "Medicinal",
        "Endangered": "Endangered",
        "Poisonous": "Toxic (Poisonous)",
    }
    for source_category, root in config.SOURCE_DATASETS.items():
        found = scan_dataset(root)
        if not found:
            raise RuntimeError(f"No images found for {source_category} source: {root}")
        category = category_names.get(source_category, source_category)
        items.extend(found)
        for _, cls, _ in found:
            class_categories[cls].add(category)

    ordered = ("Medicinal", "Endangered", "Toxic (Poisonous)")
    mapped = {cls: "; ".join(c for c in ordered if c in cats)
              for cls, cats in class_categories.items()}
    return items, mapped


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dhash(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    return np.packbits((s[:, 1:] > s[:, :-1]).flatten()).tobytes().hex()


def apply_clahe(rgb, clip=None, tile=None):
    """CLAHE on the L channel of LAB (keeps colours natural)."""
    clip = clip or config.CLAHE_CLIP
    tile = tile or config.CLAHE_TILE
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


# ----------------------------------------------------------------------------
# DATA LEAKAGE CHECKS
# ----------------------------------------------------------------------------
def leakage_checks(rows):
    """
    1. exact duplicates (MD5)            -> removed (conflicting labels -> all copies removed)
    2. near duplicates (perceptual hash) -> removed when same class
    3. target leakage via file names     -> warning (we never use file names as a feature)
    4. suspicious metadata features      -> a tiny decision-tree must NOT be able to predict the
                                            class from width/height/file-size/brightness/contrast
    5. tiny classes                      -> removed
    """
    rep = []
    n = len(rows)
    keep = np.ones(n, bool)

    def groups(key):
        g = defaultdict(list)
        for i, r in enumerate(rows):
            g[r[key]].append(i)
        return [v for v in g.values() if len(v) > 1]

    # 1 + 2 duplicates
    for key, title in (("md5", "Exact duplicates (MD5)"), ("dh", "Near duplicates (perceptual hash)")):
        removed = conflicts = cross = 0
        for g in groups(key):
            alive = [i for i in g if keep[i]]
            if len(alive) < 2:
                continue
            labs = {rows[i]["cls"] for i in alive}
            sps = {rows[i]["split"] for i in alive}
            if len(sps) > 1:
                cross += 1
            if len(labs) > 1:
                conflicts += 1
                if key == "md5":                       # same file under different labels -> label is unreliable
                    for i in alive:
                        keep[i] = False
                        removed += 1
            else:
                for i in alive[1:]:
                    keep[i] = False
                    removed += 1
        rep.append(f"[Duplicate leakage] {title}: removed {removed} images | groups spanning original train/test: {cross} "
                   f"| groups with conflicting labels: {conflicts}")

    # 5 tiny classes
    cnt = defaultdict(int)
    for i, r in enumerate(rows):
        if keep[i]:
            cnt[r["cls"]] += 1
    small = [c for c, k in cnt.items() if k < config.MIN_IMAGES_PER_CLASS]
    for i, r in enumerate(rows):
        if r["cls"] in small:
            keep[i] = False
    rep.append(f"[Class check] dropped {len(small)} class(es) with < {config.MIN_IMAGES_PER_CLASS} images: {small if small else 'none'}")

    kept = [r for i, r in enumerate(rows) if keep[i]]

    # 3 file-name leakage
    hit = np.mean([r["cls"].lower() in os.path.basename(r["path"]).lower() for r in kept if len(r["cls"]) >= 4] or [0])
    if hit > 0.5:
        rep.append(f"[Target leakage] WARNING: {hit*100:.0f}% of file names contain the class name. "
                   "This code never uses file names as input, so the model cannot cheat with them.")
    else:
        rep.append("[Target leakage] file names do not reveal the label -> OK")

    # 4 suspicious metadata features
    try:
        X = np.array([[r["w"], r["h"], r["w"] / max(r["h"], 1), r["bytes"], r["bright"], r["contrast"], r["ext"] == ".png"] for r in kept], float)
        y = np.array([r["cls"] for r in kept])
        chance = max(np.unique(y, return_counts=True)[1]) / len(y)
        sc = cross_val_score(DecisionTreeClassifier(max_depth=6, random_state=config.SEED), X, y, cv=3).mean()
        flag = "WARNING - metadata alone predicts the class (images of each class were probably collected differently)" \
            if sc > chance + 0.25 else "OK"
        rep.append(f"[Suspicious features] class predicted from width/height/file-size/brightness only: "
                   f"{sc*100:.1f}% (chance {chance*100:.1f}%) -> {flag}")
    except Exception as e:  # pragma: no cover
        rep.append(f"[Suspicious features] check skipped ({e})")

    return keep, rep


def check_split_leakage(data, train_idx, test_idx, name=""):
    """After splitting: no identical image may be in both train and test."""
    a = set(data["hashes"][train_idx])
    b = set(data["hashes"][test_idx])
    overlap = a & b
    assert not overlap, f"LEAKAGE: {len(overlap)} identical images in train and test!"
    print(f"   [Split leakage check {name}] train/test overlap = 0 images -> OK")


# ----------------------------------------------------------------------------
# MAIN LOADER
# ----------------------------------------------------------------------------
def load_dataset():
    S = config.IMG_SIZE
    items, class_categories = scan_configured_datasets()
    if not items:
        raise RuntimeError("No images found in the configured dataset source(s)")
    source_key = config.DATASET_DIR or repr(config.SOURCE_DATASETS)
    item_key = "\n".join("|".join(row) for row in items)
    sig = hashlib.md5(f"{source_key}|{item_key}|{S}|{config.USE_CLAHE}|{config.CLAHE_CLIP}|{config.CLAHE_TILE}|{config.MIN_IMAGES_PER_CLASS}".encode()).hexdigest()[:10]
    cache = os.path.join(config.CACHE_DIR, f"data_{sig}.npz")

    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        cached_classes = [str(c) for c in z["classes"]]
        cached_categories = ([str(c) for c in z["class_categories"]]
                             if "class_categories" in z.files else [class_categories.get(c, "") for c in cached_classes])
        data = dict(images=z["images"], labels=z["labels"], classes=cached_classes, paths=z["paths"],
                    hashes=z["hashes"], orig_split=z["orig_split"], clahe=bool(z["clahe"]), report=str(z["report"]), sig=sig,
                    class_categories=dict(zip(cached_classes, cached_categories)))
        print("\n(loaded preprocessed dataset from cache)")
        print(data["report"])
        return data

    print(f"\nSTEP 1: Loading {len(items)} images from configured source(s)")
    rows, imgs, bad = [], [], 0
    for path, cls, sp in tqdm(items, desc="Loading + preprocessing"):
        try:
            with Image.open(path) as im:
                w, h = im.size
                arr = np.asarray(im.convert("RGB"))
            rgb = cv2.resize(arr, (S, S), interpolation=cv2.INTER_AREA)       # STEP 2: preprocessing (RGB + resize)
            rows.append(dict(path=path, cls=cls, split=sp, w=w, h=h, bytes=os.path.getsize(path), md5=_md5(path),
                             dh=_dhash(rgb), bright=float(rgb.mean()), contrast=float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).std()),
                             ext=os.path.splitext(path)[1].lower()))
            imgs.append(rgb)
        except Exception:
            bad += 1
    rep = [f"Images found: {len(items)} | unreadable/corrupt skipped: {bad}"]

    print("STEP 3: Data-leakage checks")
    keep, lrep = leakage_checks(rows)
    rep += lrep
    rows = [r for i, r in enumerate(rows) if keep[i]]
    imgs = np.stack([im for i, im in enumerate(imgs) if keep[i]])

    # STEP 4: CLAHE (only when contrast enhancement is useful)
    low = float(np.mean([r["contrast"] < config.LOW_CONTRAST_STD for r in rows]))
    if config.USE_CLAHE == "auto":
        use = low > config.AUTO_CLAHE_FRACTION
    else:
        use = bool(config.USE_CLAHE)
    rep.append(f"[CLAHE] {low*100:.1f}% of images are low-contrast -> CLAHE {'APPLIED' if use else 'NOT applied (not needed)'}")
    if use:
        print("STEP 4: Applying CLAHE")
        imgs = np.stack([apply_clahe(im) for im in tqdm(imgs, desc="CLAHE")])

    classes = sorted({r["cls"] for r in rows})
    c2i = {c: i for i, c in enumerate(classes)}
    labels = np.array([c2i[r["cls"]] for r in rows], dtype=np.int64)
    cnts = np.bincount(labels)
    rep.append(f"Final dataset: {len(rows)} images, {len(classes)} classes | images/class min={cnts.min()} max={cnts.max()}")
    report = "\n".join(rep)
    print(report)

    class_categories = {c: class_categories.get(c, "") for c in classes}
    data = dict(images=imgs, labels=labels, classes=classes, paths=np.array([r["path"] for r in rows]),
                hashes=np.array([r["md5"] for r in rows]), orig_split=np.array([r["split"] for r in rows]),
                clahe=use, report=report, sig=sig, class_categories=class_categories)
    np.savez_compressed(cache, images=imgs, labels=labels, classes=np.array(classes),
                        class_categories=np.array([class_categories[c] for c in classes]), paths=data["paths"], hashes=data["hashes"],
                        orig_split=data["orig_split"], clahe=use, report=report)
    return data


# ----------------------------------------------------------------------------
# SPLITS  (80:20  and  5-fold)
# ----------------------------------------------------------------------------
def _strat_split(idx, y, test_size):
    try:
        return train_test_split(idx, test_size=test_size, stratify=y[idx], random_state=config.SEED)
    except ValueError:
        return train_test_split(idx, test_size=test_size, random_state=config.SEED)


def make_splits(labels, mode):
    """
    The 20 % TEST set is always held out first and never touched until the very end.
      mode 'holdout': 80 % -> (train 90 % / validation 10 %)           => one split
      mode 'kfold'  : 80 % -> StratifiedKFold(5) (train 4/5, val 1/5)  => five splits
    """
    idx = np.arange(len(labels))
    trainval, test = _strat_split(idx, labels, config.TEST_SIZE)
    if mode == "holdout":
        tr, va = _strat_split(trainval, labels, config.VAL_FRACTION)
        return [dict(tag="holdout", train=np.sort(tr), val=np.sort(va))], np.sort(test), np.sort(trainval)
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    folds = []
    for k, (a, b) in enumerate(skf.split(trainval, labels[trainval]), 1):
        folds.append(dict(tag=f"fold{k}", train=np.sort(trainval[a]), val=np.sort(trainval[b])))
    return folds, np.sort(test), np.sort(trainval)

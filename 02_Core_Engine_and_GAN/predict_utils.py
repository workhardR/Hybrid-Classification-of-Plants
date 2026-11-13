"""Upload an image -> predict plant + category (Medicinal / Endangered / Toxic)."""
import os

import cv2
import joblib
import numpy as np
from PIL import Image

import config
from core.category import get_category
from core.data import apply_clahe
from core.features import build_backbone, extract

_BB = {}


def load_saved(cfg, mode_key):
    path = os.path.join(cfg["dir"], "outputs", mode_key, "model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained model at {path}. Run run.py first (choose this mode).")
    return joblib.load(path)


def predict_image(cfg, saved, path, topk=3):
    S = saved["img_size"]
    rgb = np.asarray(Image.open(path).convert("RGB"))
    rgb = cv2.resize(rgb, (S, S), interpolation=cv2.INTER_AREA)
    if saved["clahe"]:
        rgb = apply_clahe(rgb)
    if cfg["timm"] not in _BB:
        _BB[cfg["timm"]] = build_backbone(cfg)
    f = extract(_BB[cfg["timm"]], rgb[None], cfg)
    proba = np.mean([h.predict_proba(f) for h in saved["heads"]], axis=0)[0]
    order = np.argsort(proba)[::-1][:topk]
    return [(saved["classes"][i], float(proba[i])) for i in order]


def pick_image():
    try:                                              # Google Colab
        from google.colab import files
        up = files.upload()
        return list(up.keys())[0] if up else None
    except ImportError:
        pass
    try:                                              # desktop file dialog
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        p = filedialog.askopenfilename(title="Select a plant image",
                                       filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff")])
        root.destroy()
        if p:
            return p
    except Exception:
        pass
    p = input("Type the full path of the image (or press Enter to quit): ").strip().strip('"')
    return p or None


def interactive(cfg):
    print(f"\n===== {cfg['name']} : plant image prediction =====")
    print("Use the model trained with:\n  1) 80:20 split\n  2) 5-fold cross-validation")
    key = {"1": "80_20", "2": "5_fold"}.get(input("Enter 1 or 2: ").strip(), "80_20")
    saved = load_saved(cfg, key)
    while True:
        print("\nSelect / upload an image ...")
        p = pick_image()
        if not p or not os.path.exists(p):
            print("No valid image selected. Bye!")
            break
        res = predict_image(cfg, saved, p)
        name, conf = res[0]
        print("\n" + "=" * 60)
        print(f" Image      : {os.path.basename(p)}")
        print(f" Prediction : {name}   (confidence {conf*100:.1f}%)")
        print(f" CATEGORY   : {get_category(name).upper()}")
        print(" Top-3      : " + " | ".join(f"{n} {c*100:.1f}%" for n, c in res))
        print("=" * 60)
        if input("Predict another image? (y/n): ").strip().lower() != "y":
            break

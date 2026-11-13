"""
Full pipeline (same order for every model):

 Load dataset -> preprocessing -> leakage checks -> CLAHE        (core/data.py)
 -> 80:20 split  OR  5-fold CV   (20 % test is always held out)
 -> GAN augmentation on TRAINING data only                        (core/gan.py)
 -> initial training
 -> over / under-fitting detection
 -> correction technique + retrain
 -> final test evaluation
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

import config
from core import data as D
from core import gan
from core.category import ensure_category_map
from core.features import FeatureStore
from core.heads import make_head
from core.report import metrics, show_table, plot_curve, confusion_and_report


def diagnose(train_acc, val_acc):
    gap = train_acc - val_acc
    if gap > 100 * config.OVERFIT_GAP:
        return "OVERFITTING"
    if train_acc < 100 * config.UNDERFIT_ACC and val_acc < 100 * config.UNDERFIT_ACC:
        return "UNDERFITTING"
    return "GOOD FIT"


def train_one_split(cfg, data, store, F_real, split, out_dir):
    """Returns (final_head, row_dict)."""
    y = data["labels"]
    n_classes = len(data["classes"])
    tr, va, tag = split["train"], split["val"], split["tag"]

    # ---- GAN augmentation on TRAINING data only
    simgs, sy = gan.get_synthetic(data, tr, tag)
    Xs = store.synthetic(tag, tr, simgs)
    Xr, yr = F_real[tr], y[tr]
    Xtr = np.concatenate([Xr, Xs]) if len(Xs) else Xr
    ytr = np.concatenate([yr, sy]) if len(Xs) else yr
    Xva, yva = F_real[va], y[va]
    in_dim = Xtr.shape[-1]

    def evaluate(head):
        tra = 100 * (head.predict(Xr) == yr).mean()       # accuracy on REAL training images
        vaa = 100 * (head.predict(Xva) == yva).mean()
        return tra, vaa

    # ---- initial training
    print(f"   [{tag}] initial training: {len(Xr)} real + {len(Xs)} GAN train images | {len(Xva)} validation images")
    h1 = make_head(cfg["head"], n_classes, in_dim)
    technique2_fold = cfg["name"] == "EfficientNet-B4 + XGBoost" and tag.startswith("fold")
    if technique2_fold:
        h1.params["n_estimators"] = 200
    epochs = 60 if cfg["name"] == "EfficientNet-B4 + Attention" and tag == "holdout" else config.HEAD_EPOCHS
    if h1.kind == "torch":
        h1.params["epochs"] = epochs
    h1.fit(Xtr, ytr, Xva, yva)
    tr1, va1 = evaluate(h1)
    diag = diagnose(tr1, va1)
    print(f"   [{tag}] train acc {tr1:.2f}% | val acc {va1:.2f}% | gap {tr1 - va1:.2f}% -> {diag}")
    plot_curve(h1.curve(), os.path.join(out_dir, f"curve_{tag}_initial.png"), f"{tag} initial")

    final, correction, tr2, va2 = h1, "none (good fit)", tr1, va1
    if diag != "GOOD FIT":
        # ---- apply correction technique and retrain
        h2 = make_head(cfg["head"], n_classes, in_dim)
        h2.params = dict(h1.params)
        if diag == "OVERFITTING":
            h2.regularize()
            correction = "regularization (more dropout / weight-decay / smaller capacity)"
        else:
            h2.boost()
            correction = "more capacity / longer training"
        if technique2_fold:
            h2.params["n_estimators"] = 200
        print(f"   [{tag}] applying correction: {correction} -> retraining")
        if h2.kind == "torch":
            h2.params["epochs"] = epochs
        h2.fit(Xtr, ytr, Xva, yva)
        tr2, va2 = evaluate(h2)
        d2 = diagnose(tr2, va2)
        print(f"   [{tag}] after correction: train {tr2:.2f}% | val {va2:.2f}% | gap {tr2 - va2:.2f}% -> {d2}")
        plot_curve(h2.curve(), os.path.join(out_dir, f"curve_{tag}_corrected.png"), f"{tag} corrected")
        if va2 >= va1 - 0.5:
            final = h2
        else:
            correction += " (rejected: validation got worse, kept initial model)"
            tr2, va2 = tr1, va1
    mv = metrics(yva, final.predict(Xva))
    row = dict(tag=tag, n_train_real=len(Xr), n_train_gan=len(Xs), n_val=len(Xva),
               train_acc_initial=tr1, val_acc_initial=va1, gap_initial=tr1 - va1, diagnosis=diag,
               correction=correction, train_acc_final=tr2, val_acc_final=va2, gap_final=tr2 - va2,
               val_precision=mv["precision"], val_recall=mv["recall"], val_f1=mv["f1"])
    return final, row


def final_test(cfg, data, heads, F_real, test_idx, out_dir, title):
    Xte, yte = F_real[test_idx], data["labels"][test_idx]
    proba = np.mean([h.predict_proba(Xte) for h in heads], axis=0)
    pred = proba.argmax(1)
    m = metrics(yte, pred)
    confusion_and_report(yte, pred, data["classes"], out_dir, title)
    return m


def save_model(cfg, data, heads, out_dir):
    for h in heads:
        h.cpu()
    joblib.dump(dict(heads=heads, classes=data["classes"], cfg=cfg, clahe=data["clahe"], img_size=config.IMG_SIZE),
                os.path.join(out_dir, "model.joblib"))


# -----------------------------------------------------------------------------
def run_holdout(cfg, data, store, F_real):
    out_dir = os.path.join(cfg["dir"], "outputs", "80_20")
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "#" * 100 + f"\n  {cfg['name']}  |  MODE 1: 80:20 SPLIT\n" + "#" * 100)
    splits, test_idx, trainval = D.make_splits(data["labels"], "holdout")
    s = splits[0]
    D.check_split_leakage(data, np.concatenate([s["train"], s["val"]]), test_idx, "80:20")
    head, row = train_one_split(cfg, data, store, F_real, s, out_dir)
    m = final_test(cfg, data, [head], F_real, test_idx, out_dir, f"{cfg['name']} - test confusion matrix (80:20)")

    sizes = (f"Total {len(data['labels'])} images | TRAIN 80% = {len(trainval)} "
             f"(train {len(s['train'])} real + {row['n_train_gan']} GAN | validation {len(s['val'])}) | TEST 20% = {len(test_idx)}")
    print("\n" + sizes)
    t = pd.DataFrame([
        dict(Stage="Initial training", **{"Train Acc %": row["train_acc_initial"], "Val Acc %": row["val_acc_initial"],
             "Gap %": row["gap_initial"], "Diagnosis": row["diagnosis"], "Correction": "-"},
             **{"Test Acc %": np.nan, "Precision %": np.nan, "Recall %": np.nan, "F1 %": np.nan}),
        dict(Stage="After correction / FINAL", **{"Train Acc %": row["train_acc_final"], "Val Acc %": row["val_acc_final"],
             "Gap %": row["gap_final"], "Diagnosis": diagnose(row["train_acc_final"], row["val_acc_final"]),
             "Correction": row["correction"]},
             **{"Test Acc %": m["acc"], "Precision %": m["precision"], "Recall %": m["recall"], "F1 %": m["f1"]}),
    ]).round(2).fillna("-")
    show_table(t, f"{cfg['name']}  -  80:20 SPLIT RESULT TABLE", out_dir, "table_80_20")
    save_model(cfg, data, [head], out_dir)
    summary = dict(model=cfg["name"], mode="80:20", **{k: round(v, 2) for k, v in m.items()},
                   train_acc=round(row["train_acc_final"], 2), val_acc=round(row["val_acc_final"], 2))
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    return summary


def run_kfold(cfg, data, store, F_real):
    out_dir = os.path.join(cfg["dir"], "outputs", "5_fold")
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "#" * 100 + f"\n  {cfg['name']}  |  MODE 2: {config.N_FOLDS}-FOLD CROSS VALIDATION\n" + "#" * 100)
    splits, test_idx, trainval = D.make_splits(data["labels"], "kfold")
    D.check_split_leakage(data, trainval, test_idx, "5-fold pool vs test")
    heads, rows = [], []
    for k, s in enumerate(splits, 1):
        print(f"\n--- FOLD {k}/{len(splits)} ---")
        D.check_split_leakage(data, s["train"], s["val"], f"fold{k} train/val")
        h, r = train_one_split(cfg, data, store, F_real, s, out_dir)
        heads.append(h)
        rows.append(r)
    m = final_test(cfg, data, heads, F_real, test_idx, out_dir, f"{cfg['name']} - test confusion matrix (5-fold ensemble)")

    t = pd.DataFrame([{
        "Fold": i + 1, "Train(real+GAN)": f"{r['n_train_real']}+{r['n_train_gan']}", "Val n": r["n_val"],
        "Train Acc %": r["train_acc_initial"], "Val Acc %": r["val_acc_initial"], "Gap %": r["gap_initial"],
        "Diagnosis": r["diagnosis"], "Correction": r["correction"].split(" (")[0].split(" ")[0],
        "Final Val Acc %": r["val_acc_final"], "Val Prec %": r["val_precision"], "Val Rec %": r["val_recall"],
        "Val F1 %": r["val_f1"]} for i, r in enumerate(rows)])
    num = ["Train Acc %", "Val Acc %", "Gap %", "Final Val Acc %", "Val Prec %", "Val Rec %", "Val F1 %"]
    mean = {c: t[c].mean() for c in num}
    std = {c: t[c].std(ddof=0) for c in num}
    t = pd.concat([t, pd.DataFrame([dict(Fold="MEAN", **mean), dict(Fold="STD", **std)])], ignore_index=True)
    t = t.round(2).fillna("")
    show_table(t, f"{cfg['name']}  -  {config.N_FOLDS}-FOLD CROSS-VALIDATION TABLE", out_dir, "table_5_fold")

    ft = pd.DataFrame([{"Final TEST (ensemble of 5 fold models)": cfg["name"], "Test Acc %": m["acc"], "Precision %": m["precision"],
                        "Recall %": m["recall"], "F1 %": m["f1"]}]).round(2)
    show_table(ft, f"{cfg['name']}  -  FINAL TEST EVALUATION (20% held-out test set)", out_dir, "table_5_fold_final_test")
    save_model(cfg, data, heads, out_dir)
    summary = dict(model=cfg["name"], mode="5-fold", **{k: round(v, 2) for k, v in m.items()},
                   cv_val_acc_mean=round(mean["Final Val Acc %"], 2), cv_val_acc_std=round(std["Final Val Acc %"], 2))
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    return summary


# -----------------------------------------------------------------------------
def main(cfg):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["1", "2", "3"], help="1 = 80:20 | 2 = 5-fold | 3 = both")
    args = ap.parse_args()
    mode = args.mode
    if not mode:
        print(f"\n===== {cfg['name']} =====")
        print("  1) 80:20 split table")
        print("  2) 5-fold cross-validation table")
        print("  3) both")
        mode = input("Enter 1, 2 or 3: ").strip()
    t0 = time.time()
    data = D.load_dataset()
    ensure_category_map(data["classes"], data.get("class_categories", {}))
    store = FeatureStore(cfg, data)
    F_real = store.real()
    if mode in ("1", "3"):
        run_holdout(cfg, data, store, F_real)
    if mode in ("2", "3"):
        run_kfold(cfg, data, store, F_real)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results are in: {os.path.join(cfg['dir'], 'outputs')}")



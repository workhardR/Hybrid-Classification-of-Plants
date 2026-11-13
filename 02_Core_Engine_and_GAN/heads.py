"""
Classifier heads that sit on top of the frozen backbone features.

  svm        -> RBF-SVM
  xgb        -> XGBoost
  mlp        -> 3-layer MLP
  attention  -> Transformer-style self-attention + attention pooling over feature-map tokens

Every head has:
  fit(Xtr, ytr, Xva, yva) / predict_proba(X) / predict(X)
  regularize()  -> used when OVERFITTING is detected
  boost()       -> used when UNDERFITTING is detected
  curve()       -> learning-curve dict (or None)
"""
import copy

import numpy as np
import torch
import torch.nn as nn

import config
from core.features import get_device


def _full_proba(raw, classes_, n_classes):
    out = np.zeros((raw.shape[0], n_classes), dtype=np.float32)
    out[:, np.asarray(classes_, int)] = raw
    return out


# =============================================================================
# SVM
# =============================================================================
class SVMHead:
    kind = "svm"

    def __init__(self, n_classes, in_dim=None):
        self.n_classes = n_classes
        self.params = dict(C=10.0, gamma="scale")

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler().fit(Xtr)
        self.clf = SVC(kernel="rbf", probability=True, random_state=config.SEED, **self.params)
        self.clf.fit(self.scaler.transform(Xtr), ytr)
        return self

    def predict_proba(self, X):
        return _full_proba(self.clf.predict_proba(self.scaler.transform(X)), self.clf.classes_, self.n_classes)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def regularize(self):          # smaller C = wider margin = less over-fitting
        self.params["C"] = max(self.params["C"] / 10.0, 0.05)

    def boost(self):               # bigger C = more flexible
        self.params["C"] = self.params["C"] * 10.0

    def curve(self):
        return None

    def cpu(self):
        return self


# =============================================================================
# XGBoost
# =============================================================================
class XGBHead:
    kind = "xgb"

    def __init__(self, n_classes, in_dim=None):
        self.n_classes = n_classes
        # L2 REGULARIZATION (reg_lambda): Ridge L2 penalty on leaf weights to restrict tree complexity
        self.params = dict(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9,
                           colsample_bytree=0.5, reg_lambda=1.0, min_child_weight=1, gamma=0.0)

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        import xgboost as xgb
        kw = dict(objective="multi:softprob", tree_method="hist", n_jobs=-1, random_state=config.SEED,
                  eval_metric="mlogloss", **self.params)
        if int(xgb.__version__.split(".")[0]) >= 2 and torch.cuda.is_available():
            kw["device"] = "cuda"
        if Xva is not None:
            kw["early_stopping_rounds"] = 20
        self.clf = xgb.XGBClassifier(**kw)
        self.clf.fit(Xtr, ytr, eval_set=[(Xva, yva)] if Xva is not None else None, verbose=False)
        return self

    def predict_proba(self, X):
        return _full_proba(self.clf.predict_proba(X), self.clf.classes_, self.n_classes)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def regularize(self):          # shallower trees, stronger penalties, more sub-sampling
        p = self.params
        p["max_depth"] = max(2, p["max_depth"] - 3)
        p["subsample"] = 0.7
        p["colsample_bytree"] = 0.3
        p["reg_lambda"] = p["reg_lambda"] * 10  # L2 REGULARIZATION BOOST: 10x penalty to prevent overfitting
        p["min_child_weight"] = p["min_child_weight"] + 3
        p["gamma"] = 1.0
        p["learning_rate"] = p["learning_rate"] * 0.5

    def boost(self):               # deeper trees, more rounds
        p = self.params
        p["max_depth"] = p["max_depth"] + 3
        p["n_estimators"] = p["n_estimators"] * 2
        p["colsample_bytree"] = 0.8
        p["reg_lambda"] = 0.5
        p["learning_rate"] = 0.15

    def curve(self):
        try:
            r = self.clf.evals_result()["validation_0"]["mlogloss"]
            return dict(val_loss=list(r))
        except Exception:
            return None

    def cpu(self):
        return self


# =============================================================================
# Torch heads (MLP / Attention)
# =============================================================================
class AttnNet(nn.Module):
    """tokens [B,16,C] -> LayerNorm -> Linear -> +pos -> multi-head self-attention -> FFN -> attention pooling -> classifier"""

    def __init__(self, in_dim, n_classes, d=256, heads=4, drop=0.3, n_tokens=16):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d))
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(drop), nn.Linear(d * 2, d))
        self.n2 = nn.LayerNorm(d)
        self.score = nn.Linear(d, 1)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(d, n_classes)

    def forward(self, x):
        x = self.proj(self.norm_in(x)) + self.pos
        a, _ = self.attn(x, x, x)
        x = self.n1(x + a)
        x = self.n2(x + self.ffn(x))
        w = torch.softmax(self.score(x), dim=1)          # attention weights over tokens
        return self.fc(self.drop((w * x).sum(1)))


class MLPNet(nn.Module):
    def __init__(self, in_dim, n_classes, d=512, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d // 2, n_classes))

    def forward(self, x):
        return self.net(x)


class TorchHead:
    kind = "torch"

    def __init__(self, n_classes, in_dim):
        self.n_classes, self.in_dim = n_classes, in_dim
        self.params = dict(lr=1e-3, wd=1e-2, drop=0.3, ls=0.0, noise=0.0, epochs=config.HEAD_EPOCHS,
                           patience=config.HEAD_PATIENCE, bs=64, d=256)
        self.history = []
        self.net = None

    def _build(self):
        raise NotImplementedError

    def _dev(self):
        return next(self.net.parameters()).device

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        p, dev = self.params, get_device()
        torch.manual_seed(config.SEED)
        self.net = self._build().to(dev)
        # L2 REGULARIZATION (WEIGHT DECAY): AdamW applies L2 penalty (weight_decay) to stabilize network weights
        opt = torch.optim.AdamW(self.net.parameters(), lr=p["lr"], weight_decay=p["wd"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p["epochs"])
        crit = nn.CrossEntropyLoss(label_smoothing=p["ls"])
        Xt, yt = torch.from_numpy(np.ascontiguousarray(Xtr)), torch.from_numpy(ytr).long()
        best_loss, best_state, bad = 1e9, None, 0
        self.history = []
        for ep in range(p["epochs"]):
            self.net.train()
            perm = torch.randperm(len(Xt))
            tl = corr = 0
            for i in range(0, len(Xt), p["bs"]):
                idx = perm[i:i + p["bs"]]
                if len(idx) < 2:
                    continue
                xb, yb = Xt[idx].to(dev).float(), yt[idx].to(dev)
                if p["noise"] > 0:
                    xb = xb + torch.randn_like(xb) * p["noise"] * xb.std()
                opt.zero_grad()
                out = self.net(xb)
                loss = crit(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tl += loss.item() * len(idx)
                corr += (out.argmax(1) == yb).sum().item()
            sched.step()
            rec = dict(epoch=ep + 1, train_loss=tl / len(Xt), train_acc=corr / len(Xt))
            if Xva is not None:
                pr = self.predict_proba(Xva)
                vl = float(-np.log(pr[np.arange(len(yva)), yva] + 1e-9).mean())
                rec.update(val_loss=vl, val_acc=float((pr.argmax(1) == yva).mean()))
                if vl < best_loss - 1e-4:
                    best_loss, bad = vl, 0
                    best_state = copy.deepcopy(self.net.state_dict())
                else:
                    bad += 1
                    if bad >= p["patience"]:
                        self.history.append(rec)
                        break
            self.history.append(rec)
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        self.net.eval()
        dev = self._dev()
        out = []
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + 256])).to(dev).float()
            out.append(torch.softmax(self.net(xb), 1).cpu().numpy())
        return np.concatenate(out)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def regularize(self):
        p = self.params
        p["drop"] = min(0.6, p["drop"] + 0.2)
        p["wd"] = p["wd"] * 10  # L2 REGULARIZATION BOOST: 10x L2 weight decay penalty to cure overfitting
        p["ls"] = 0.1
        p["noise"] = 0.15
        p["lr"] = p["lr"] * 0.5
        p["d"] = max(64, p["d"] // 2)

    def boost(self):
        p = self.params
        p["drop"] = max(0.1, p["drop"] - 0.2)
        p["wd"] = p["wd"] / 5
        p["epochs"] = int(p["epochs"] * 1.5)
        p["patience"] = p["patience"] + 6
        p["d"] = p["d"] * 2
        p["lr"] = p["lr"] * 2

    def curve(self):
        if not self.history:
            return None
        keys = self.history[0].keys()
        return {k: [h.get(k) for h in self.history] for k in keys if k != "epoch"}

    def cpu(self):
        self.net.cpu()
        return self


class AttentionHead(TorchHead):
    def _build(self):
        return AttnNet(self.in_dim, self.n_classes, d=self.params["d"], heads=4, drop=self.params["drop"])


class MLPHead(TorchHead):
    def _build(self):
        return MLPNet(self.in_dim, self.n_classes, d=max(self.params["d"] * 2, 128), drop=self.params["drop"])


HEADS = {"svm": SVMHead, "xgb": XGBHead, "mlp": MLPHead, "attention": AttentionHead}


def make_head(name, n_classes, in_dim):
    return HEADS[name](n_classes, in_dim)

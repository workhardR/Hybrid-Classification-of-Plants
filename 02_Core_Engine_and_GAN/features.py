"""
Frozen pre-trained backbones (timm) used as feature extractors.
feature = "pooled"  -> one vector per image             (SVM / XGBoost / MLP heads)
feature = "tokens"  -> 4x4=16 feature-map tokens/image  (Attention head)
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_backbone(cfg):
    bname = cfg.get("backbone") or cfg.get("timm", "")
    if bname in ["efficientnet_b4", "efficientnet-b4"]:
        try:
            import torchvision.models as models
            weights = models.EfficientNet_B4_Weights.DEFAULT if config.PRETRAINED else None
            model = models.efficientnet_b4(weights=weights)
            model.forward_features = model.features
            model.classifier = torch.nn.Identity()
            return model.to(get_device()).eval()
        except Exception:
            pass

    import timm

    model = timm.create_model(
        cfg["timm"],
        pretrained=config.PRETRAINED,
        num_classes=0,
    )
    return model.to(get_device()).eval()


@torch.no_grad()
def extract(model, imgs, cfg, bs=None, desc="   extracting features"):
    """imgs: uint8 [N,H,W,3] (any size, resized to IMG_SIZE) -> float features."""
    dev = get_device()
    bs = bs or config.BATCH_EXTRACT
    S = config.IMG_SIZE

    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    out = []
    rng = range(0, len(imgs), bs)

    for i in (tqdm(rng, desc=desc) if len(imgs) > bs else rng):
        x = (
            torch.from_numpy(np.ascontiguousarray(imgs[i:i + bs]))
            .to(dev)
            .permute(0, 3, 1, 2)
            .float()
            / 255.0
        )

        if x.shape[-1] != S or x.shape[-2] != S:
            x = F.interpolate(
                x,
                size=(S, S),
                mode="bicubic",
                align_corners=False,
            ).clamp(0, 1)

        x = (x - mean) / std

        with torch.autocast(device_type="cuda", enabled=dev.type == "cuda"):
            if cfg["feature"] == "tokens":
                f = model.forward_features(x)  # B,C,h,w
                f = F.adaptive_avg_pool2d(f.float(), 4).flatten(2).transpose(1, 2)
                out.append(f.half().cpu().numpy())  # B,16,C
            else:
                out.append(model(x).float().cpu().numpy())

    return np.concatenate(out) if out else np.zeros((0,))


class FeatureStore:
    """Computes features once per backbone and caches them on disk."""

    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data
        self._model = None

    @property
    def model(self):
        if self._model is None:
            bname = self.cfg.get("backbone") or self.cfg.get("timm", "")
            print(
                f"   loading backbone {bname} "
                f"(pretrained={config.PRETRAINED})"
            )
            self._model = build_backbone(self.cfg)
        return self._model

    def _path(self, name):
        return os.path.join(
            config.CACHE_DIR,
            f"feat_{self.cfg['timm']}_{self.cfg['feature']}_"
            f"{int(config.PRETRAINED)}_{name}.npy",
        )

    def real(self):
        p = self._path(self.data["sig"])

        if os.path.exists(p):
            return np.load(p)

        f = extract(
            self.model,
            self.data["images"],
            self.cfg,
            desc="   backbone features (all real images)",
        )
        np.save(p, f)
        return f

    def synthetic(self, tag, key_arr, imgs):
        if len(imgs) == 0:
            real = self.real()
            return np.zeros((0,) + real[:1].shape[1:], dtype=real.dtype)

        import hashlib

        key = hashlib.md5(key_arr.tobytes()).hexdigest()[:8]
        p = self._path(
            f"gan_{self.data['sig']}_{tag}_{key}_{config.GAN_EPOCHS}"
        )

        if os.path.exists(p):
            return np.load(p)

        f = extract(
            self.model,
            imgs,
            self.cfg,
            desc="   backbone features (GAN images)",
        )
        np.save(p, f)
        return f
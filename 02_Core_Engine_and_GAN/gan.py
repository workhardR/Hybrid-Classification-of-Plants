"""
GAN augmentation (conditional DCGAN, 64x64).
Trained ONLY on the training images of the current split, so validation / test images never leak into it.
Generated images are cached so the 7 model folders reuse the same synthetic images.
"""
import os
import hashlib

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import config


class Generator(nn.Module):
    def __init__(self, n_classes, nz=100, ngf=64):
        super().__init__()
        self.nz = nz
        self.emb = nn.Embedding(n_classes, nz)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True),      # 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True),  # 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),  # 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),          # 32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False), nn.Tanh(),                                         # 64
        )

    def forward(self, z, y):
        return self.net((z * self.emb(y)).view(z.size(0), self.nz, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, n_classes, img=64, ndf=64):
        super().__init__()
        self.img = img
        self.emb = nn.Embedding(n_classes, img * img)
        self.net = nn.Sequential(
            nn.Conv2d(4, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),                                  # 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),   # 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),  # 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),  # 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x, y):
        lab = self.emb(y).view(-1, 1, self.img, self.img)
        return self.net(torch.cat([x, lab], 1)).view(-1)


def _train_gan(imgs64, labels, n_classes, device):
    X = torch.from_numpy(imgs64).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
    Y = torch.from_numpy(labels).long()
    G, D = Generator(n_classes).to(device), Discriminator(n_classes).to(device)
    # L2 REGULARIZATION (WEIGHT DECAY): applies L2 weight penalty to prevent discriminator overpowering
    oG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    oD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    
    # LOSS FUNCTIONS (Adversarial BCE Loss + L1 Mean Absolute Error Loss):
    bce = nn.BCEWithLogitsLoss()  # Standard adversarial discriminator loss
    l1_loss = nn.L1Loss()         # L1 Reconstruction Loss (preserves natural color distribution)
    
    bs = min(config.GAN_BATCH, len(X))
    for ep in tqdm(range(config.GAN_EPOCHS), desc="   training GAN"):
        perm = torch.randperm(len(X))
        for i in range(0, len(X) - bs + 1, bs):
            idx = perm[i:i + bs]
            real = X[idx].to(device)
            y = Y[idx].to(device)
            if torch.rand(1).item() < 0.5:
                real = torch.flip(real, [3])
            z = torch.randn(bs, G.nz, device=device)
            fake = G(z, y)
            # --- D step (Discriminator loss)
            oD.zero_grad()
            lossD = bce(D(real, y), torch.full((bs,), 0.9, device=device)) + bce(D(fake.detach(), y), torch.zeros(bs, device=device))
            lossD.backward()
            oD.step()
            # --- G step (Combined Adversarial Loss + L1 Loss)
            oG.zero_grad()
            loss_adv = bce(D(fake, y), torch.ones(bs, device=device))
            loss_l1 = l1_loss(fake.mean(dim=[2, 3]), real.mean(dim=[2, 3]))  # L1 Loss term
            lossG = loss_adv + 0.05 * loss_l1
            lossG.backward()
            oG.step()
    return G


def get_synthetic(data, train_idx, tag):
    """Return (uint8 images [M,64,64,3], labels [M]) generated from TRAINING images only."""
    if not config.USE_GAN:
        return np.zeros((0, config.GAN_IMG, config.GAN_IMG, 3), np.uint8), np.zeros((0,), np.int64)

    key = hashlib.md5(train_idx.tobytes()).hexdigest()[:8]
    cache = os.path.join(config.CACHE_DIR, f"gan_{data['sig']}_{tag}_{key}_{config.GAN_EPOCHS}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        print(f"   GAN images loaded from cache ({len(z['y'])} synthetic images)")
        return z["x"], z["y"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(data["classes"])
    labels = data["labels"][train_idx]
    small = np.stack([cv2.resize(data["images"][i], (config.GAN_IMG, config.GAN_IMG), interpolation=cv2.INTER_AREA) for i in train_idx])
    print(f"   Training GAN on {len(train_idx)} TRAINING images only ({device})")
    torch.manual_seed(config.SEED)
    G = _train_gan(small, labels, n_classes, device)
    G.eval()

    counts = np.bincount(labels, minlength=n_classes)
    ys = []
    for c in range(n_classes):
        add = int(np.clip(counts.max() - counts[c], config.GAN_MIN_ADD, config.GAN_MAX_ADD))
        ys += [c] * add
    ys = np.array(ys, np.int64)
    out = []
    with torch.no_grad():
        for i in range(0, len(ys), 256):
            y = torch.from_numpy(ys[i:i + 256]).to(device)
            z = torch.randn(len(y), G.nz, device=device)
            f = G(z, y).add(1).mul(127.5).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            out.append(f)
    xs = np.concatenate(out)
    np.savez_compressed(cache, x=xs, y=ys)
    print(f"   Generated {len(ys)} synthetic training images")
    return xs, ys

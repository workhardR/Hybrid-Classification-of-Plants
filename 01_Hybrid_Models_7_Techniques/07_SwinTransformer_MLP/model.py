"""Complete Swin Transformer feature extractor + MLP classifier."""
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

CFG = {"name": "Swin Transformer + MLP", "model": "swin_t", "pretrained": True,
       "num_classes": 4, "batch_size": 32, "dropout": 0.3}


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron (MLP) head with GELU activations and LayerNorm."""
    def __init__(self, in_features=768, hidden_dim=512, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(CFG["dropout"]),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(CFG["dropout"]),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SwinTransformerMLP:
    """Hybrid Model: Extract pooled Swin Transformer features, then fit/predict with an MLP."""
    def __init__(self, num_classes=CFG["num_classes"], device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. SWIN TRANSFORMER CALL (MODEL INITIALIZATION): load pretrained Swin-T without timm.
        try:
            weights = models.Swin_T_Weights.DEFAULT if CFG["pretrained"] else None
            self.backbone = models.swin_t(weights=weights).to(self.device).eval()
        except Exception:
            self.backbone = models.swin_t(weights=None).to(self.device).eval()
        in_features = self.backbone.head.in_features  # 768
        self.backbone.head = nn.Identity()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # 2. MLP CALL (CLASSIFIER INITIALIZATION): multilayer perceptron classifier head.
        self.classifier = MLPClassifier(in_features=in_features, num_classes=num_classes).to(self.device)

    @torch.no_grad()
    # 3. INPUT OF SWIN TRANSFORMER & FEATURE EXTRACTION:
    # Takes raw input images [Batch, 3, 224, 224] -> outputs pooled feature vectors [Batch, 768].
    def extract_features(self, images):
        chunks = []
        for batch in images.split(CFG["batch_size"]):
            # SWIN TRANSFORMER CALL (FORWARD PASS): passes input images through shifted-window self-attention blocks.
            features = self.backbone(batch.to(self.device))
            chunks.append(features.cpu().numpy())
        return np.concatenate(chunks, axis=0)

    # 4. WHERE BOTH ARE MERGED (TRAINING):
    # Feeds extracted Swin Transformer features into the MLP classifier head.
    def fit(self, images, labels, epochs=5, lr=0.001):
        # MERGED CALL: MLP trains on the feature representations extracted by Swin Transformer.
        features = torch.from_numpy(self.extract_features(images)).to(self.device)
        targets = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        self.classifier.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.classifier(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        return self

    # 5. FINAL PREDICTION OUTPUT:
    # Classifies Swin Transformer features and returns predicted class IDs.
    def predict(self, images):
        proba = self.predict_proba(images)
        return np.argmax(proba, axis=1)

    # 6. PROBABILITY PREDICTION OUTPUT:
    # Classifies Swin Transformer features and returns class probability distribution.
    def predict_proba(self, images):
        self.classifier.eval()
        with torch.no_grad():
            features = torch.from_numpy(self.extract_features(images)).to(self.device)
            logits = self.classifier(features)
            return torch.softmax(logits, dim=1).cpu().numpy()


if __name__ == "__main__":
    # DEVICE SELECTION: use the GPU when available; otherwise use the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinTransformerMLP(device=device)
    dummy_images = torch.randn(2, 3, 224, 224)
    features = model.extract_features(dummy_images)
    print("Feature shape:", features.shape)
    print("Device:", device)
    print("Fit model with model.fit(training_images, training_labels).")

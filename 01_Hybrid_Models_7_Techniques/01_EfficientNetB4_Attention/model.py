"""Complete EfficientNet-B4 feature extractor + Attention classifier."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

CFG = {"name": "EfficientNet-B4 + Attention", "model": "efficientnet_b4", "pretrained": True,
       "num_classes": 4, "batch_size": 32}


class AttentionClassifier(nn.Module):
    """Multi-Head Self-Attention network that classifies feature tokens."""
    def __init__(self, in_features=1792, num_classes=4, d=256, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.proj = nn.Linear(in_features, d)
        self.pos = nn.Parameter(torch.zeros(1, 16, d))
        # ATTENTION CALL: MultiheadAttention lets spatial tokens exchange relationships.
        self.attention = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(0.3), nn.Linear(2 * d, d))
        self.norm2 = nn.LayerNorm(d)
        self.score = nn.Linear(d, 1)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, num_classes))

    def forward(self, tokens):
        # ATTENTION FORWARD CALL: run self-attention over tokens and pool into class scores.
        tokens = self.proj(self.norm(tokens)) + self.pos
        attended, _ = self.attention(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attended)
        tokens = self.norm2(tokens + self.ffn(tokens))
        weights = torch.softmax(self.score(tokens), dim=1)
        pooled = (weights * tokens).sum(dim=1)
        return self.fc(pooled)


class EfficientNetB4Attention:
    """Extract spatial EfficientNet-B4 tokens, then fit/predict with an Attention classifier."""
    def __init__(self, num_classes=CFG["num_classes"], device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # EFFICIENTNET-B4 CALL (MODEL INITIALIZATION): loads pretrained EfficientNet-B4 backbone
        try:
            weights = models.EfficientNet_B4_Weights.DEFAULT if CFG["pretrained"] else None
            self.backbone = models.efficientnet_b4(weights=weights).to(self.device).eval()
        except Exception:
            self.backbone = models.efficientnet_b4(weights=None).to(self.device).eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        # ATTENTION CLASSIFIER CALL: create Attention head to learn from EfficientNet-B4 tokens.
        self.classifier = AttentionClassifier(in_features=1792, num_classes=num_classes).to(self.device)

    @torch.no_grad()
    # FEATURE EXTRACTION: run image batches through EfficientNet-B4 before Attention classification.
    def extract_features(self, images):
        chunks = []
        for batch in images.split(CFG["batch_size"]):
            # EFFICIENTNET-B4 CALL (FORWARD PASS): takes input image batch and extracts feature maps
            fmap = self.backbone.features(batch.to(self.device))
            tokens = F.adaptive_avg_pool2d(fmap, 4).flatten(2).transpose(1, 2)
            chunks.append(tokens.cpu().numpy())
        return np.concatenate(chunks, axis=0)

    # CLASSIFIER TRAINING CALL: extract EfficientNet-B4 tokens, then fit the Attention head on labels.
    def fit(self, images, labels, epochs=5, lr=0.001):
        # ATTENTION .fit CALL: train Attention head using EfficientNet-B4 tokens and target labels.
        features = torch.from_numpy(self.extract_features(images)).to(self.device)
        targets = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        # L2 REGULARIZATION (weight_decay): applies L2 weight penalty to prevent overfitting
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()

        self.classifier.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.classifier(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        return self

    # PREDICTION CALL: extract EfficientNet-B4 tokens, then ask Attention for class IDs.
    def predict(self, images):
        # ATTENTION .predict CALL: classify EfficientNet-B4 tokens and return predicted class IDs.
        proba = self.predict_proba(images)
        return np.argmax(proba, axis=1)

    # PROBABILITY PREDICTION CALL: extract EfficientNet-B4 tokens and get Attention class probabilities.
    def predict_proba(self, images):
        # ATTENTION .predict_proba CALL: return class probabilities from EfficientNet-B4 tokens.
        self.classifier.eval()
        with torch.no_grad():
            features = torch.from_numpy(self.extract_features(images)).to(self.device)
            logits = self.classifier(features)
            return torch.softmax(logits, dim=1).cpu().numpy()


if __name__ == "__main__":
    # DEVICE SELECTION: use the GPU when available; otherwise use the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetB4Attention(device=device)
    dummy_images = torch.randn(2, 3, 224, 224)
    features = model.extract_features(dummy_images)
    print("Feature shape:", features.shape)
    print("Device:", device)
    print("Fit model with model.fit(training_images, training_labels).")

"""Complete ConvNeXt-Tiny feature extractor + XGBoost classifier."""
import numpy as np
import torch
from torchvision import models
import xgboost as xgb

CFG = {"name": "ConvNeXt + XGBoost", "model": "convnext_tiny", "pretrained": True,
       "num_classes": 4, "batch_size": 32}


class ConvNeXtXGBoost:
    """Hybrid Model: Extract pooled ConvNeXt features, then fit/predict with XGBoost."""
    def __init__(self, num_classes=CFG["num_classes"], device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. CONVNEXT CALL (MODEL INITIALIZATION): load pretrained ConvNeXt-Tiny without timm.
        try:
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if CFG["pretrained"] else None
            self.backbone = models.convnext_tiny(weights=weights).to(self.device).eval()
        except Exception:
            self.backbone = models.convnext_tiny(weights=None).to(self.device).eval()
        self.backbone.classifier = torch.nn.Identity()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # 2. XGBOOST CALL (CLASSIFIER INITIALIZATION): tree-based gradient boosted classifier.
        self.classifier = xgb.XGBClassifier(objective="multi:softprob", num_class=num_classes,
            n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.5, reg_lambda=1.0, min_child_weight=1,
            tree_method="hist", eval_metric="mlogloss", n_jobs=-1, random_state=42)

    @torch.no_grad()
    # 3. INPUT OF CONVNEXT & FEATURE EXTRACTION:
    # Takes raw input images [Batch, 3, 224, 224] -> outputs pooled feature vectors [Batch, 768].
    def extract_features(self, images):
        chunks = []
        for batch in images.split(CFG["batch_size"]):
            # CONVNEXT CALL (FORWARD PASS): passes input images through ConvNeXt modernized conv blocks.
            chunks.append(self.backbone(batch.to(self.device)).flatten(1).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    # 4. WHERE BOTH ARE MERGED (TRAINING):
    # Feeds extracted ConvNeXt features into the XGBoost classifier head.
    def fit(self, images, labels):
        # MERGED CALL: XGBoost trains on the feature representations extracted by ConvNeXt.
        features = self.extract_features(images)
        self.classifier.fit(features, np.asarray(labels))
        return self

    # 5. FINAL PREDICTION OUTPUT:
    # Classifies ConvNeXt features and returns predicted class IDs.
    def predict(self, images):
        features = self.extract_features(images)
        return self.classifier.predict(features)

    # 6. PROBABILITY PREDICTION OUTPUT:
    # Classifies ConvNeXt features and returns class probability distribution.
    def predict_proba(self, images):
        features = self.extract_features(images)
        return self.classifier.predict_proba(features)


if __name__ == "__main__":
    # DEVICE SELECTION: use the GPU when available; otherwise use the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtXGBoost(device=device)
    dummy_images = torch.randn(2, 3, 224, 224)
    features = model.extract_features(dummy_images)
    print("Feature shape:", features.shape)
    print("Device:", device)
    print("Fit model with model.fit(training_images, training_labels).")

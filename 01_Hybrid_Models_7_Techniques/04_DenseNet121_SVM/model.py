"""Complete DenseNet121 feature extractor + RBF-SVM classifier."""
import numpy as np
import torch
from torchvision import models
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CFG = {"name": "DenseNet121 + SVM", "model": "densenet121", "pretrained": True,
       "num_classes": 4, "batch_size": 32}


class DenseNet121SVM:
    """Hybrid Model: Extract pooled DenseNet121 features, then fit/predict with an RBF SVM."""
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. DENSENET-121 CALL (MODEL INITIALIZATION): load pretrained DenseNet-121 without timm.
        try:
            weights = models.DenseNet121_Weights.DEFAULT if CFG["pretrained"] else None
            self.backbone = models.densenet121(weights=weights).to(self.device).eval()
        except Exception:
            self.backbone = models.densenet121(weights=None).to(self.device).eval()
        self.backbone.classifier = torch.nn.Identity()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # 2. SVM CALL (CLASSIFIER INITIALIZATION): scale features and classify with RBF SVM.
        self.classifier = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0,
            gamma="scale", probability=True, random_state=42))

    @torch.no_grad()
    # 3. INPUT OF DENSENET-121 & FEATURE EXTRACTION:
    # Takes raw input images [Batch, 3, 224, 224] -> outputs pooled feature vectors [Batch, 1024].
    def extract_features(self, images):
        chunks = []
        for batch in images.split(CFG["batch_size"]):
            # DENSENET-121 CALL (FORWARD PASS): passes input images through densely connected convolutional blocks.
            chunks.append(self.backbone(batch.to(self.device)).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    # 4. WHERE BOTH ARE MERGED (TRAINING):
    # Feeds extracted DenseNet-121 features into the RBF-SVM classifier.
    def fit(self, images, labels):
        # MERGED CALL: SVM trains on the feature representations extracted by DenseNet-121.
        features = self.extract_features(images)
        self.classifier.fit(features, np.asarray(labels))
        return self

    # 5. FINAL PREDICTION OUTPUT:
    # Classifies DenseNet-121 features and returns predicted class IDs.
    def predict(self, images):
        features = self.extract_features(images)
        return self.classifier.predict(features)

    # 6. PROBABILITY PREDICTION OUTPUT:
    # Classifies DenseNet-121 features and returns class probability distribution.
    def predict_proba(self, images):
        features = self.extract_features(images)
        return self.classifier.predict_proba(features)


if __name__ == "__main__":
    # DEVICE SELECTION: use the GPU when available; otherwise use the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121SVM(device=device)
    dummy_images = torch.randn(2, 3, 224, 224)
    features = model.extract_features(dummy_images)
    print("Feature shape:", features.shape)
    print("Device:", device)
    print("Fit model with model.fit(training_images, training_labels).")

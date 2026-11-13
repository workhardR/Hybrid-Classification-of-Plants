import os

# Technique 1 Model Configuration
# Backbone: EfficientNet-B4 (torchvision.models.efficientnet_b4)
# Classifier Head: Multi-Head Self-Attention
CFG = dict(
    name="EfficientNet-B4 + Attention",
    backbone="efficientnet_b4",
    model="efficientnet_b4",
    timm="efficientnet_b4",  # Fallback compatibility
    feature="tokens",
    head="attention",
    dir=os.path.dirname(os.path.abspath(__file__)),
)


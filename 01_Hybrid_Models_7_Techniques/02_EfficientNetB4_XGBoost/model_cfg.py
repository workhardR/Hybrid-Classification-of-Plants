import os

CFG = dict(
    name="EfficientNet-B4 + XGBoost",
    backbone="efficientnet_b4",
    model="efficientnet_b4",
    timm="efficientnet_b4",
    feature="pooled",
    head="xgb",
    dir=os.path.dirname(os.path.abspath(__file__)),
)

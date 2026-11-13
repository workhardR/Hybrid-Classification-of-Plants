import os
CFG = dict(name="ConvNeXt + XGBoost", timm="convnext_tiny", feature="pooled", head="xgb",
           dir=os.path.dirname(os.path.abspath(__file__)))

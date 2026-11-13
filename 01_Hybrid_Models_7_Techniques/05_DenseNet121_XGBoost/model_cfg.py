import os
CFG = dict(name="DenseNet121 + XGBoost", timm="densenet121", feature="pooled", head="xgb",
           dir=os.path.dirname(os.path.abspath(__file__)))

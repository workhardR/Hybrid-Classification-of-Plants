import os
CFG = dict(name="Swin Transformer + MLP", timm="swin_tiny_patch4_window7_224", feature="pooled", head="mlp",
           dir=os.path.dirname(os.path.abspath(__file__)))

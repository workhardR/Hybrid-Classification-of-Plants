import os
CFG = dict(name="ResNet50 + SVM", timm="resnet50", feature="pooled", head="svm",
           dir=os.path.dirname(os.path.abspath(__file__)))

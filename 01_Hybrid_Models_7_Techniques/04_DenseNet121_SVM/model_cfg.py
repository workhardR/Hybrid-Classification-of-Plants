import os
CFG = dict(name="DenseNet121 + SVM", timm="densenet121", feature="pooled", head="svm",
           dir=os.path.dirname(os.path.abspath(__file__)))

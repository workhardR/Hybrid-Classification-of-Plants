import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_cfg import CFG
from core.predict_utils import interactive

if __name__ == "__main__":
    interactive(CFG)

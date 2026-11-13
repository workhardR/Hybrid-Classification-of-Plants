"""
Root Launcher for Complete 10-Step Pipeline
Run: python run_all_pipeline_steps.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS_DIR = os.path.join(ROOT, "pipeline_steps")
sys.path.insert(0, STEPS_DIR)

from run_all_steps import main

if __name__ == "__main__":
    main()


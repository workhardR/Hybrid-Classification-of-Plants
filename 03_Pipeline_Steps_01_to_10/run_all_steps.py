"""
MASTER PIPELINE RUNNER: RUN ALL 10 STEPS SEQUENTIALLY
=====================================================
Executes the full image classification pipeline in exact required sequence:
  Step 01: Load Image Dataset
  Step 02: Data Preprocessing
  Step 03: Data Leakage Checks (Target leakage, duplicate leakage, suspicious features)
  Step 04: CLAHE (Contrast-Limited Adaptive Histogram Equalization)
  Step 05: Train / Validation / Test Split (80:20 and 5-Fold Cross Validation)
  Step 06: GAN Augmentation on TRAINING Data Only
  Step 07: Initial Model Training
  Step 08: Overfitting / Underfitting Detection and Rectification
  Step 09: Apply Correction Technique and Retrain
  Step 10: Final Test Evaluation
"""

import os
import sys

# Add project root and pipeline_steps to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS_DIR = os.path.join(ROOT, "pipeline_steps")
sys.path.insert(0, STEPS_DIR)
sys.path.insert(0, ROOT)

from step01_load_dataset import run_step_01
from step02_data_preprocessing import run_step_02
from step03_data_leakage_checks import run_step_03
from step04_clahe_enhancement import run_step_04
from step05_train_val_test_splits import run_step_05
from step06_gan_augmentation import run_step_06
from step07_initial_model_training import run_step_07
from step08_overfit_underfit_detection import run_step_08
from step09_correction_and_retrain import run_step_09
from step10_final_test_evaluation import run_step_10


def main():
    print("#" * 80)
    print(" EXECUTING COMPLETE 10-STEP IMAGE CLASSIFICATION PIPELINE")
    print("#" * 80)

    # Step 1
    run_step_01()

    # Step 2
    run_step_02()

    # Step 3
    run_step_03()

    # Step 4
    run_step_04()

    # Step 5
    run_step_05()

    # Step 6
    run_step_06()

    # Step 7
    run_step_07()

    # Step 8
    run_step_08()

    # Step 9
    run_step_09()

    # Step 10
    run_step_10()

    print("\n" + "#" * 80)
    print(" ALL 10 PIPELINE STEPS SUCCESSFULLY EXECUTED AND VALIDATED!")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()


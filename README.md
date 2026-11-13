# Hybrid Classification of Plants: Medical, Endangered, and Toxic Botanical Species

An end-to-end Machine Learning and Deep Learning research framework for fine-grained multi-class botanical classification covering **111 plant species** across three critical domains: **Medicinal**, **Endangered**, and **Poisonous/Toxic** flora.

---

## 📌 Overview

Accurate identification of plant species is vital for biodiversity preservation, ethnopharmacological discovery, and public safety (preventing accidental ingestion of toxic plants). This repository implements a robust machine learning pipeline and benchmarks **7 Hybrid Deep Learning Architectures** combining state-of-the-art vision backbones (CNNs and Vision Transformers) with advanced classification heads (Self-Attention, Support Vector Machines, XGBoost, and Multilayer Perceptrons).

### Key Features
- **111 Botanical Classes:** Full taxonomic mapping across medicinal, endangered, and toxic plant categories.
- **Image Preprocessing & Enhancement:** CLAHE (Contrast Limited Adaptive Histogram Equalization) to emphasize subtle venation patterns, textural cues, and morphological leaf structures.
- **Class Balancing:** Synthetic data augmentation via Deep Convolutional GANs (DCGAN) alongside SMOTE feature-space comparisons.
- **Rigorously Evaluated:** 5-Fold Stratified Cross-Validation, 80:20 independent test splits, data leakage validation, and automated overfit/underfit detection.

---

## 🏗️ Repository Architecture

```
Hybrid-Classification-of-Plants/
├── 01_Hybrid_Models_7_Techniques/         # Implementations & outputs for the 7 hybrid models
│   ├── 01_EfficientNetB4_Attention/       # Model 1: EfficientNet-B4 + Multi-Head Self-Attention
│   ├── 02_EfficientNetB4_XGBoost/         # Model 2: EfficientNet-B4 + XGBoost
│   ├── 03_ResNet50_SVM/                   # Model 3: ResNet-50 + Support Vector Machine
│   ├── 04_DenseNet121_SVM/                # Model 4: DenseNet-121 + Support Vector Machine
│   ├── 05_DenseNet121_XGBoost/            # Model 5: DenseNet-121 + XGBoost
│   ├── 06_ConvNeXt_XGBoost/               # Model 6: ConvNeXt-Tiny + XGBoost
│   └── 07_SwinTransformer_MLP/            # Model 7: Swin Transformer-Tiny + MLP
├── 02_Core_Engine_and_GAN/                # Shared utilities, architectures & data loaders
│   ├── category.py                        # Botanical class mapping & taxonomic taxonomy
│   ├── category_map.csv                   # 111 classes catalog
│   ├── config.py                          # Global training hyperparameters
│   ├── data.py                            # Multi-domain dataset loader & CLAHE processing
│   ├── features.py                        # Feature extraction backbone
│   ├── gan.py                             # DCGAN training & synthetic sample generation
│   ├── heads.py                           # Classification heads (Attention, SVM, XGBoost, MLP)
│   ├── pipeline.py                        # 5-fold cross-validation & evaluation engine
│   ├── predict_utils.py                   # Single-image diagnostic inference
│   ├── report.py                          # Metrics computation & tabular generation
│   └── smote_augmentation.py              # SMOTE baseline module
├── 03_Pipeline_Steps_01_to_10/            # Step-by-step modular pipeline
│   ├── step01_load_dataset.py
│   ├── step02_data_preprocessing.py
│   ├── step03_data_leakage_checks.py
│   ├── step04_clahe_enhancement.py
│   ├── step05_train_val_test_splits.py
│   ├── step06_gan_augmentation.py
│   ├── step07_initial_model_training.py
│   ├── step08_overfit_underfit_detection.py
│   ├── step09_correction_and_retrain.py
│   ├── step10_final_test_evaluation.py
│   └── run_all_steps.py                   # Orchestrator for all 10 steps
├── 04_Google_Colab_Notebook/              # Interactive Jupyter / Colab notebooks
│   └── Plant_Hybrid_7_Algorithms_Complete_Colab.ipynb
├── 05_Evaluation_and_Plotting_Scripts/    # Visualization, ROC, PR curves & Confusion Matrices
│   ├── compare_all.py
│   ├── generate_7_confusion_grid.py
│   ├── generate_all_7_learning_curves.py
│   ├── plot_all_algorithm_outputs.py
│   └── run_all_pipeline_steps.py
├── 06_Configurations_and_Setup/           # Requirements & environment specifications
│   ├── requirements.txt
│   └── README.txt
├── dataset/
│   └── Dataset Links.txt                  # Kaggle sources for raw datasets
├── HOW_TO_RUN_PROGRAMS.txt                # Detailed CLI execution guide
├── README_FOLDER_STRUCTURE.txt            # Folder organization reference
└── README.md
```

---

## ⚡ The 7 Hybrid Techniques

| # | Architecture | Feature Extractor | Classifier Head | Primary Advantage |
|---|---|---|---|---|
| **1** | **EfficientNet-B4 + Self-Attention** | EfficientNet-B4 | Multi-Head Self-Attention | High contextual feature weighting across distinct floral regions |
| **2** | **EfficientNet-B4 + XGBoost** | EfficientNet-B4 | Gradient Boosted Trees | Strong non-linear boundary separation on deep embeddings |
| **3** | **ResNet-50 + SVM** | ResNet-50 | Support Vector Machine (RBF) | Optimal hyperplane margin maximization on residual features |
| **4** | **DenseNet-121 + SVM** | DenseNet-121 | Support Vector Machine | Rich feature reuse through dense connections paired with max-margin classification |
| **5** | **DenseNet-121 + XGBoost** | DenseNet-121 | XGBoost | High feature diversity combined with gradient boosting ensemble |
| **6** | **ConvNeXt-Tiny + XGBoost** | ConvNeXt-Tiny | XGBoost | Modernized ConvNet receptive field dynamics |
| **7** | **Swin Transformer-Tiny + MLP** | Swin Transformer-T | Multi-Layer Perceptron | Shifted-window self-attention baseline benchmark |

---

## 🚀 Getting Started

### 1. Environment Setup

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/workhardR/Hybrid-Classification-of-Plants.git
cd Hybrid-Classification-of-Plants

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r 06_Configurations_and_Setup/requirements.txt
```

### 2. Dataset Acquisition

The datasets used in this work are publicly hosted on Kaggle. Refer to [`dataset/Dataset Links.txt`](dataset/Dataset%20Links.txt) to download:
1. *Medicinal Plants Dataset*
2. *Toxic Chinese Herbal Medicine Dataset*
3. *Plant Species Classification in Real-World Environments (Cyprus Flora)*

Place downloaded images in the `dataset/` directory according to `02_Core_Engine_and_GAN/config.py`.

---

## 💻 Running the Experiments

### Execute the Full 10-Step Pipeline:
```bash
python 03_Pipeline_Steps_01_to_10/run_all_steps.py
```

### Train / Evaluate an Individual Hybrid Model:
```bash
python 01_Hybrid_Models_7_Techniques/01_EfficientNetB4_Attention/train.py
```

### Generate Confusion Matrices and Comparison Curves:
```bash
python 05_Evaluation_and_Plotting_Scripts/generate_7_confusion_grid.py
python 05_Evaluation_and_Plotting_Scripts/compare_all.py
```

### Run via Google Colab:
Open `04_Google_Colab_Notebook/Plant_Hybrid_7_Algorithms_Complete_Colab.ipynb` directly in Google Colab for accelerated GPU execution (T4 / A100).

---

## 📜 License & Citation

Distributed for academic research and educational purposes. If you build upon this project or use the hybrid classification pipeline, please reference this repository.

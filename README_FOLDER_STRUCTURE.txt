# Plant Hybrid Multi-Class Classification - Clean Codes Archive
===================================================================
Pure machine learning source code package ready for Google Drive upload.
Organized folder-by-folder.

FOLDER STRUCTURE:
-----------------
├── 01_Hybrid_Models_7_Techniques/
│   ├── 01_EfficientNetB4_Attention/       (Model 1: EfficientNet-B4 + Multi-Head Self-Attention)
│   ├── 02_EfficientNetB4_XGBoost/         (Model 2: EfficientNet-B4 + XGBoost)
│   ├── 03_ResNet50_SVM/                   (Model 3: ResNet-50 + Support Vector Machine)
│   ├── 04_DenseNet121_SVM/                (Model 4: DenseNet-121 + Support Vector Machine)
│   ├── 05_DenseNet121_XGBoost/            (Model 5: DenseNet-121 + XGBoost)
│   ├── 06_ConvNeXt_XGBoost/               (Model 6: ConvNeXt-Tiny + XGBoost)
│   └── 07_SwinTransformer_MLP/            (Model 7: Swin Transformer-Tiny + MLP - Base Benchmark)
│
├── 02_Core_Engine_and_GAN/
│   ├── category.py          (111 botanical class mapping: Medical, Endangered, Toxic)
│   ├── config.py            (Global training and dataset hyperparameters)
│   ├── data.py              (Multi-domain dataset loader and CLAHE enhancement)
│   ├── features.py          (Feature extraction engine)
│   ├── gan.py               (Deep Convolutional GAN training & synthetic augmentation)
│   ├── heads.py             (Classifier heads: Attention, XGBoost, SVM, MLP)
│   ├── pipeline.py          (5-fold stratified cross-validation & 80:20 evaluation)
│   ├── predict_utils.py     (Single-image diagnostic inference routines)
│   ├── report.py            (Metrics calculation and tabular generation)
│   ├── smote_augmentation.py(SMOTE baseline comparison module)
│   └── category_map.csv     (111 botanical classes catalog)
│
├── 03_Pipeline_Steps_01_to_10/
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
│   └── run_all_steps.py
│
├── 04_Google_Colab_Notebook/
│   └── Plant_Hybrid_7_Algorithms_Complete_Colab.ipynb
│
├── 05_Evaluation_and_Plotting_Scripts/
│   ├── compare_all.py
│   ├── generate_7_confusion_grid.py
│   ├── generate_all_7_learning_curves.py
│   ├── plot_all_algorithm_outputs.py
│   ├── run_all_pipeline_steps.py
│   └── run_data_pipeline_sequence.py
│
└── 06_Configurations_and_Setup/
    ├── requirements.txt
    └── README.txt

EXCLUSIONS ENFORCED:
--------------------
- NO Excel files (.xlsx, .xls)
- NO Word (.docx, .doc) files
- NO templates or static folders
- NO app.py or web upload scripts
- NO create-colab or create-doc scripts
- NO Techniques 8, 9, 10
- NO raw dataset samples or binary cache dumps

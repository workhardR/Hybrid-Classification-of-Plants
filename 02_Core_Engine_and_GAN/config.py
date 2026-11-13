"""
GLOBAL SETTINGS  (shared by all 7 model folders)
=================================================
The three SOURCE_DATASETS below are combined automatically. Set PLANT_DATASET only to override them with one dataset.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# 1) PUT YOUR DATASET PATH HERE  (or set env var PLANT_DATASET)
#    Expected layout (class-name folders inside train/ and test/):
#        dataset/
#           train/<class_1>/*.jpg ...
#           test/<class_1>/*.jpg  ...
#    (a dataset with only class folders and no train/test also works)
# ------------------------------------------------------------------
SOURCE_DATASETS = {
    "Medical": r"C:\Users\ronit\OneDrive\Documents\dataset new\3_2026_A Deep Learning Model for Identification of_3\archive\Medicinal plant dataset",
    "Endangered": r"C:\Users\ronit\OneDrive\Documents\dataset new\13_2026_A CNN and Transformer-Based Framework for species\The Cyprus Seasonal Flora Image Dataset (CSFID)\Plant_Dataset",
    "Poisonous": r"C:\Users\ronit\OneDrive\Documents\dataset new\7_2026_Toxic Chinese Herbal Medicine Recognition in Real-World Images via Multi-Scale and Attention-Enhanced EfficientNetV2_7\dataset\dataset",
}
DATASET_DIR = os.environ.get("PLANT_DATASET", "")  # Optional single-dataset override; blank means combine SOURCE_DATASETS

# ---------------- image / preprocessing ----------------
IMG_SIZE = 224            # all backbones use 224x224 (raise to 320/380 for EfficientNet-B4 if you have a GPU)
USE_CLAHE = "auto"        # True / False / "auto" (auto = apply only if many images are low-contrast)
LOW_CONTRAST_STD = 50     # grayscale std below this = "low contrast" image
AUTO_CLAHE_FRACTION = 0.15
CLAHE_CLIP = 2.0
CLAHE_TILE = 8
MIN_IMAGES_PER_CLASS = 10 # classes with fewer images are dropped (needed for 5-fold + stratified split)

# ---------------- splits ----------------
SEED = 42
TEST_SIZE = 0.20          # 80 : 20
VAL_FRACTION = 0.10       # part of the 80% used as validation in 80:20 mode (early stopping + over/under-fit detection)
N_FOLDS = 5

# ---------------- GAN augmentation (TRAINING DATA ONLY) ----------------
USE_GAN = True
GAN_IMG = 64              # GAN works at 64x64, outputs are up-scaled to IMG_SIZE
GAN_EPOCHS = 40           # more epochs = better fakes (slower). 150+ on a GPU if you have time
GAN_BATCH = 64
GAN_MIN_ADD = 30          # synthetic images added to every class (at least)
GAN_MAX_ADD = 150         # ... and at most this many (used to balance small classes)

# ---------------- over / under-fitting detection ----------------
OVERFIT_GAP = 0.10        # train_acc - val_acc  > 10 %  -> OVERFITTING
UNDERFIT_ACC = 0.70       # train_acc and val_acc both < 70 % -> UNDERFITTING

# ---------------- classifier-head training ----------------
HEAD_EPOCHS = 20
HEAD_PATIENCE = 8
BATCH_EXTRACT = 32        # batch size for feature extraction (lower if GPU/RAM runs out)

# ---------------- misc ----------------
PRETRAINED = os.environ.get("PRETRAINED", "1") == "1"   # ImageNet weights (needs internet the first time)
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
CATEGORY_CSV = os.path.join(PROJECT_ROOT, "category_map.csv")
os.makedirs(CACHE_DIR, exist_ok=True)



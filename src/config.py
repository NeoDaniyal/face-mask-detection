from pathlib import Path
import os

if os.path.exists("/content/drive"):
    BASE_DIR = Path("/content/drive/MyDrive/face_mask_detection/face-mask-detection")
else:
    BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR/"data"
RAW_DATA_DIR = DATA_DIR/"raw"
PROCCESSED_DATA_DIR = DATA_DIR/"processed"
SPLIT_DATA_DIR = PROCCESSED_DATA_DIR / "split"

MODEL_DIR = BASE_DIR / "models"

OUPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUPUT_DIR / "plots"
PREDICTION_DIR = OUPUT_DIR / "predictions"
RANDOM_SEED = 42
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

CLASS_TO_IDX = {"with_mask": 0, "without_mask": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
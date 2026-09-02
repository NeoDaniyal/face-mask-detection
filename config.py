from pathlib import Path

DRIVE_PROJECT_ROOT = Path(
    "/content/drive/MyDrive/face_mask_detection/face-mask-detection"
)

LOCAL_PROJECT_ROOT = Path(__file__).resolve().parent

if DRIVE_PROJECT_ROOT.exists():
    BASE_DIR = DRIVE_PROJECT_ROOT
else:
    BASE_DIR = LOCAL_PROJECT_ROOT


DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLIT_DATA_DIR = PROCESSED_DATA_DIR / "split"

MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

RANDOM_SEED = 42

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

CLASS_TO_IDX = {
    "with_mask": 0,
    "without_mask": 1
}

IDX_TO_CLASS = {
    v: k for k, v in CLASS_TO_IDX.items()
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
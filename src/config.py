from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR/"data"
RAW_DATA_DIR = DATA_DIR/"raw"
PROCCESSED_DATA_DIR = DATA_DIR/"processed"


MODEL_DIR = BASE_DIR / "models"

OUPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUPUT_DIR / "plots"
PREDICTION_DIR = OUPUT_DIR / "predictions"
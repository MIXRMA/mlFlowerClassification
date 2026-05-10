import torch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR         = ROOT_DIR / "data"
RAW_DIR          = DATA_DIR / "raw"
PROCESSED_DIR    = DATA_DIR / "processed"

OXFORD102_DIR    = RAW_DIR / "oxford102"
ORGANIZED_DIR    = OXFORD102_DIR / "organized"   # class sub-folder root
CLASS_MAP_PATH   = OXFORD102_DIR / "classMap.json"

CHECKPOINTS_DIR  = ROOT_DIR / "models" / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

CNN_CHECKPOINT   = CHECKPOINTS_DIR / "cnnFlower.pth"
LABEL_MAP_CACHE  = CHECKPOINTS_DIR / "labelMap.json"
                                                           

def getDevice() -> torch.device:

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = getDevice()

IMAGE_SIZE      = 224
IMAGE_CHANNELS  = 3

NORMALIZE_MEAN  = [0.485, 0.456, 0.406]
NORMALIZE_STD   = [0.229, 0.224, 0.225]

TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15

BATCH_SIZE      = 32
NUM_WORKERS     = 0
PIN_MEMORY      = False

NUM_CLASSES     = 102
NUM_EPOCHS      = 30
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
                               
LR_STEP_SIZE    = 10
LR_GAMMA        = 0.5
EARLY_STOP_PATIENCE = 7

CONV_CHANNELS   = [32, 64, 128]
FC_HIDDEN_DIM   = 512
DROPOUT_RATE    = 0.5

FLASK_HOST      = "0.0.0.0"
FLASK_PORT      = 8080
FLASK_DEBUG     = False

MAX_CONTENT_LENGTH = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "tiff", "gif"}

UPLOAD_TEMP_DIR = ROOT_DIR / "app" / "static" / "uploads"
UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)

TOP_K_PREDICTIONS = 5

LOG_INTERVAL    = 10
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════╗")
    print("║   Flower Species Recognition — Config Check  ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Project root  : {ROOT_DIR}")
    print(f"  Device        : {DEVICE}  {'← Apple Silicon MPS ✓' if str(DEVICE) == 'mps' else ''}")
    print(f"  Image size    : {IMAGE_SIZE}×{IMAGE_SIZE}px, {IMAGE_CHANNELS} channels")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Num classes   : {NUM_CLASSES}")
    print(f"  Epochs        : {NUM_EPOCHS}")
    print(f"  Learning rate : {LEARNING_RATE}")
    print(f"  CNN channels  : {CONV_CHANNELS}")
    print(f"  FC hidden dim : {FC_HIDDEN_DIM}")
    print(f"  Dropout       : {DROPOUT_RATE}")
    print(f"  Dataset split : {int(TRAIN_SPLIT*100)}/{int(VAL_SPLIT*100)}/{int(TEST_SPLIT*100)} (train/val/test)")
    print(f"  Oxford 102 dir: {OXFORD102_DIR}")
    print(f"  Checkpoints   : {CHECKPOINTS_DIR}")
    print(f"  Flask         : http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"  Top-K preds   : {TOP_K_PREDICTIONS}")
    print()
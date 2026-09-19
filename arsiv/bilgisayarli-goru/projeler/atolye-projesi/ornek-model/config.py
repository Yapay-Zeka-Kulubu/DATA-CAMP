import torch
from pathlib import Path

#Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

#Dataset
IMG_SIZE = 224
NUM_CLASSES = 1000
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

#Training
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

#Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Model
MODEL_NAME = "resnet18"
PRETRAINED = True

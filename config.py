import torch

DATA_FLAG = "bloodmnist"
BATCH_SIZE = 64
NUM_CLASSES = 8  # BloodMNIST ma 8 klas
IMG_SIZE = 28    # Rozmiar obrazów BloodMNIST
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

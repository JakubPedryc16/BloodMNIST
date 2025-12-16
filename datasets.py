# datasets.py
import os
import random
from typing import Tuple
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import Config

# download URL for BloodMNIST npz file
BLOODMNIST_URL = "https://zenodo.org/record/5208230/files/bloodmnist.npz?download=1"

# short labels for BloodMNIST classes
LABELS_BLOODMNIST_SHORT = {
    "0": "baso",
    "1": "eos",
    "2": "ery",
    "3": "imm. gran.",
    "4": "lymph",
    "5": "mono",
    "6": "neut",
    "7": "plt",
}

# full labels with Polish names (used in plots / GUI)
LABELS_BLOODMNIST_FULL = {
    "0": "Basophil (Bazofil)",
    "1": "Eosinophil (Eozynofil)",
    "2": "Erythroblast (Erytroblaście)",
    "3": "Immature Granulocytes (Mieolocyty / Metamieolocyty / Promielocyty)",
    "4": "Lymphocyte (Limfocyt)",
    "5": "Monocyte (Monocyt)",
    "6": "Neutrophil (Neutrofil)",
    "7": "Platelet (Płytka krwi)"
}


def set_seed(seed: int = 42):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_bloodmnist(root: str) -> str:
    
    os.makedirs(root, exist_ok=True)
    filepath = os.path.join(root, "bloodmnist.npz")

    if not os.path.exists(filepath):
        print(f"Pobieram BloodMNIST z {BLOODMNIST_URL} ...")
        urllib.request.urlretrieve(BLOODMNIST_URL, filepath)
        print("Pobrano do:", filepath)
    else:
        print("Znaleziono istniejący plik:", filepath)

    return filepath


class BloodMNISTNPZ(Dataset):

    def __init__(self, npz_path: str, split: str = "train", transform=None):
        
        super().__init__()
        self.transform = transform

        # load npz file once
        data = np.load(npz_path)
        images_key = f"{split}_images"
        labels_key = f"{split}_labels"

        # images: shape [N, 28, 28, 3], uint8
        self.images = data[images_key]

        # labels: shape [N], int64
        self.labels = data[labels_key].reshape(-1).astype(np.int64)

        assert self.images.shape[0] == self.labels.shape[0]
        print(
            f"Split={split}: images={self.images.shape}, "
            f"labels={self.labels.shape}"
        )

    def __len__(self):
        """Return number of samples in this split."""
        return self.images.shape[0]

    def __getitem__(self, idx):
        """
        Get single sample:
          - img: numpy array [28, 28, 3], uint8
          - label: int
        Returns:
          - img_tensor: torch.Tensor [3, 28, 28], float
          - label: int
        """
        img = self.images[idx]   # shape (28, 28, 3), uint8
        label = int(self.labels[idx])

        # apply transform if provided
        if self.transform is not None:
            # transform expects H x W x C numpy array
            img = self.transform(img)
        else:
            # fallback: manual conversion [0,255] -> [0,1] and HWC -> CHW
            img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        return img, label


def get_transforms(use_augment: bool):
    """
    Create and return (train_transform, test_transform).
    Both transforms work on numpy arrays H x W x C (uint8).
    """
    # base transform: ToTensor + Normalize
    base_transform = transforms.Compose([
        transforms.ToTensor(),  # [0,255] -> [0,1], HWC -> CHW
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    # if no augmentation, use same transform for train and test
    if not use_augment:
        return base_transform, base_transform

    # train transform with basic augmentations
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    return train_transform, base_transform


def get_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create train/val/test DataLoaders and info dict for BloodMNIST.
    Uses settings from Config (batch_size, num_workers, seed, use_augment).
    """
    # fix randomness
    set_seed(cfg.seed)

    # sanity check: we only support BloodMNIST in this project
    if cfg.data_flag != "bloodmnist":
        raise ValueError("Config.data_flag should be 'bloodmnist' in this project.")

    # 1) download or reuse bloodmnist.npz file
    root = os.path.join(cfg.output_dir, "data")
    npz_path = download_bloodmnist(root)

    # 2) get image transforms
    train_transform, test_transform = get_transforms(cfg.use_augment)

    # 3) create Dataset objects for each split
    train_dataset = BloodMNISTNPZ(npz_path, split="train", transform=train_transform)
    val_dataset   = BloodMNISTNPZ(npz_path, split="val",   transform=test_transform)
    test_dataset  = BloodMNISTNPZ(npz_path, split="test",  transform=test_transform)

    # 4) wrap datasets in DataLoader (mini-batches)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,              # shuffle only for training
        num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    # 5) collect dataset info (similar to medmnist INFO dict)
    info = {
        "flag": "bloodmnist",
        "n_channels": 3,
        "n_classes": 8,
        "label": LABELS_BLOODMNIST_SHORT,
        "n_samples": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "description": "BloodMNIST – 17 092 images, 8 classes, RGB 3x28x28.",
    }

    return train_loader, val_loader, test_loader, info

# models.py
import torch.nn as nn
import torch.nn.functional as F


class SimpleBloodCNN(nn.Module):
    """
    Prosty model CNN: 3 convy + 2 fc.
    """
    def __init__(self, n_classes: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # 3x28x28 -> po 3 poolingach: 128x3x3
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 32x14x14
        x = self.pool(F.relu(self.conv2(x)))   # 64x7x7
        x = self.pool(F.relu(self.conv3(x)))   # 128x3x3
        x = x.view(x.size(0), -1)              # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DeepBloodCNN(nn.Module):
    """
    Trochę głębszy model: więcej filtrów + BatchNorm.
    """
    def __init__(self, n_classes: int = 8):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        # 3x28x28 -> 32x28 -> pool -> ...
        # Po 3 poolach: 256x3x3
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x14x14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 64x7x7
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 128x3x3
        x = F.relu(self.bn4(self.conv4(x)))             # 256x3x3 (bez pool)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def build_model(model_type: str, n_classes: int):
    if model_type == "simple_cnn":
        return SimpleBloodCNN(n_classes=n_classes)
    elif model_type == "deep_cnn":
        return DeepBloodCNN(n_classes=n_classes)
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}")

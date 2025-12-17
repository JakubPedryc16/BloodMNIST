from typing import List
from io import BytesIO

import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

import torch
import torch.nn.functional as F

from src.config import Config
from src.models.cnn_models import build_model
from src.data.datasets import LABELS_BLOODMNIST_FULL
from src.utils.train_utils import get_device
from src.data.datasets import get_transforms


cfg = Config()

app = FastAPI(title="BloodMNIST API")


def load_model(model_type: str, ckpt_path: str, n_classes: int = 8):
    device = get_device()

    model = build_model(model_type, n_classes=n_classes)

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model, device


def preprocess_image_bytes(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    image = image.resize((28, 28))

    _, test_transform = get_transforms(use_augment=False)

    x = test_transform(np.array(image)) 

    x = x.unsqueeze(0)
    return x


class PredictionResponse(BaseModel):
    predicted_class_idx: int
    predicted_class_name: str
    probabilities: List[float]


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model_type: str = "simple_cnn",
):
    
    if model_type == "simple_cnn":
        ckpt_path = f"{cfg.output_dir}/simple_cnn_adam_aug.pt"
    else:
        ckpt_path = f"{cfg.output_dir}/deep_cnn_adam_noaug.pt"

    model, device = load_model(model_type, ckpt_path)

    image_bytes = await file.read()

    x = preprocess_image_bytes(image_bytes).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        probs = probs.cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))

    class_names = [
        LABELS_BLOODMNIST_FULL[str(i)]
        for i in range(len(LABELS_BLOODMNIST_FULL))
    ]
    pred_name = class_names[pred_idx]

    return PredictionResponse(
        predicted_class_idx=pred_idx,
        predicted_class_name=pred_name,
        probabilities=[float(p) for p in probs],
    )
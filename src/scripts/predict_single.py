import argparse

import torch
from torchvision import transforms
from PIL import Image
import medmnist
from medmnist import INFO

from src.config import Config
from src.models.cnn_models import build_model


def load_model(ckpt_path: str, cfg: Config):
    info = INFO[cfg.data_flag]
    n_classes = len(info["label"])

    model = build_model(cfg.model_type, n_classes=n_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model, device, info


def get_transform():
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])


def predict_image(image_path: str, ckpt_path: str, cfg: Config):
    model, device, info = load_model(ckpt_path, cfg)
    transform = get_transform()

    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = probs.argmax()
    pred_class = info["label"][str(pred_idx)]

    print("Plik:", image_path)
    print("Predykcja:", pred_class)
    print("Prawdopodobieństwa:")
    for i, p in enumerate(probs):
        print(f"  {i} ({info['label'][str(i)]}): {p:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="Ścieżka do obrazu krwinki")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Ścieżka do modelu .pt")
    parser.add_argument("--model_type", type=str, default="simple_cnn",
                        choices=["simple_cnn", "deep_cnn"])
    args = parser.parse_args()

    cfg = Config()
    cfg.model_type = args.model_type

    predict_image(args.image, args.ckpt, cfg)
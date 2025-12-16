# debug_model_direct.py

import torch
import torch.nn.functional as F

from config import Config
from datasets import get_dataloaders, LABELS_BLOODMNIST_FULL
from models import build_model
from train_utils import get_device


def load_model_direct(model_type: str, ckpt_path: str, n_classes: int = 8):
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


def main():
    cfg = Config()

    print("Ładuję dane testowe...")
    train_loader, val_loader, test_loader, info = get_dataloaders(cfg)
    class_names = [LABELS_BLOODMNIST_FULL[str(i)] for i in range(8)]

    # WYBIERZ MODEL I CHECKPOINT
    model_type = "deep_cnn"   # albo "simple_cnn"
    ckpt_path = f"{cfg.output_dir}/deep_cnn_adam_noaug.pt"

    print(f"Ładuję model {model_type} z {ckpt_path}")
    model, device = load_model_direct(model_type, ckpt_path)

    # Weź jedną paczkę z test loadera
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)

    print("\n=== PIERWSZE 16 PRÓBEK Z TESTU ===")
    with torch.no_grad():
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

    for i in range(16):
        true_idx = int(labels[i].item())
        pred_idx = int(preds[i].item())
        true_name = class_names[true_idx]
        pred_name = class_names[pred_idx]
        max_prob = float(probs[i, pred_idx].item()) * 100

        print(
            f"Sample {i:02d}: "
            f"Y_true={true_idx} ({true_name}), "
            f"Y_pred={pred_idx} ({pred_name}), "
            f"p={max_prob:.2f}%"
        )


if __name__ == "__main__":
    main()

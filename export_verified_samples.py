import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from config import Config
from datasets import download_bloodmnist, LABELS_BLOODMNIST_FULL
from models import build_model
from train_utils import get_device


def main():
    cfg = Config()
    os.makedirs("outputs/verified_samples", exist_ok=True)

    # ----- load dataset -----
    npz_path = download_bloodmnist(cfg.output_dir + "/data")
    data = np.load(npz_path)
    X = data["test_images"]
    y = data["test_labels"].reshape(-1)

    # ----- load model -----
    device = get_device()
    model = build_model("deep_cnn", n_classes=8)

    ckpt_path = f"{cfg.output_dir}/deep_cnn_adam_noaug.pt"
    print(">>> Loading checkpoint:", ckpt_path)
    state = torch.load(ckpt_path, map_location=device)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.to(device).eval()

    print(">>> Model loaded.")

    saved_per_class = {i: 0 for i in range(8)}
    target_per_class = 5   # ZMIEŃ NA WIĘCEJ, jeśli chcesz np. 10 obrazów na klasę

    idx = 0
    total = len(X)

    while True:
        if all(saved_per_class[c] >= target_per_class for c in range(8)):
            break

        img_arr = X[idx]
        true_class = int(y[idx])

        # preprocess (identycznie jak w debug)
        img_tensor = torch.tensor(img_arr).permute(2, 0, 1).float() / 255
        img_tensor = (img_tensor - 0.5) / 0.5
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        if pred == true_class and conf > 0.80:
            fname = (
                f"true{true_class}_"
                f"pred{pred}_"
                f"conf{int(conf*100):02d}_"
                f"sample{idx}.png"
            )
            img_pil = Image.fromarray(img_arr.astype(np.uint8))
            img_pil.save(os.path.join("outputs/verified_samples", fname))

            saved_per_class[true_class] += 1

            print(
                f"[OK] saved: {fname} "
                f"({LABELS_BLOODMNIST_FULL[str(true_class)]})"
            )

        idx += 1
        if idx >= total:
            break

    print("DONE: saved verified samples into outputs/verified_samples/")


if __name__ == "__main__":
    main()

import torch
import numpy as np
from PIL import Image

from config import Config
from models.cnn_models import build_model
from data.datasets import download_bloodmnist
from utils.train_utils import get_device
from api import preprocess_image_bytes


print("\n=========== TEST ROUNDTRIP API ===========")

cfg = Config()

npz = download_bloodmnist(cfg.output_dir + "/data")
data = np.load(npz)
X = data["test_images"]
y = data["test_labels"].reshape(-1)

idx = 0
img_arr = X[idx]
true_class = int(y[idx])

device = get_device()

model = build_model("deep_cnn", n_classes=8)

ckpt_path = f"{cfg.output_dir}/deep_cnn_adam_noaug.pt"
print(">>> Ładuję checkpoint:", ckpt_path)

state = torch.load(ckpt_path, map_location=device)

if "model_state" in state:
    model.load_state_dict(state["model_state"])
else:
    model.load_state_dict(state)

model.to(device).eval()

img_tensor = torch.tensor(img_arr).permute(2, 0, 1).float() / 255.0
img_tensor = (img_tensor - 0.5) / 0.5
img_tensor = img_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(img_tensor)
    probs_direct = torch.softmax(logits, dim=1)[0].cpu().numpy()

pred_direct = int(np.argmax(probs_direct))

print(f"\n>>> direct_model pred = {pred_direct}, probs={np.round(probs_direct, 3)}")
print(f"    true class = {true_class}\n")

img_pil = Image.fromarray(img_arr.astype(np.uint8))
img_pil.save("test_roundtrip.png")

with open("test_roundtrip.png", "rb") as f:
    img_bytes = f.read()

api_tensor = preprocess_image_bytes(img_bytes).to(device)

with torch.no_grad():
    logits = model(api_tensor)
    probs_api = torch.softmax(logits, dim=1)[0].cpu().numpy()

pred_api = int(np.argmax(probs_api))

print(f">>> API preprocess pred = {pred_api}, probs={np.round(probs_api, 3)}")

print("\n------------------------------------------")
print("true =", true_class, "| direct =", pred_direct, "| api =", pred_api)
print("------------------------------------------")
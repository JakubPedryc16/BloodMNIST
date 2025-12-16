import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

from config import Config
from models.cnn_models import build_model
from utils.train_utils import get_device
from medmnist import INFO
import os


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, x, class_idx=None):
        self.model.zero_grad()
        out = self.model(x)

        if class_idx is None:
            class_idx = out.argmax(dim=1).item()

        score = out[:, class_idx]
        score.backward()

        grads = self.gradients
        activations = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


def get_transform():
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])


def visualize_grad_cam(image_path: str,
                       ckpt_path: str,
                       cfg: Config,
                       use_deep_layer: bool = False,
                       output_path: str = "grad_cam_overlay.png"):
    device = get_device()
    info = INFO[cfg.data_flag]
    n_classes = len(info["label"])

    model = build_model(cfg.model_type, n_classes)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
        
    model.to(device)
    model.eval()

    if cfg.model_type == "simple_cnn":
        target_layer = model.conv3
    else:
        target_layer = model.conv4 if use_deep_layer else model.conv3

    cam_gen = GradCAM(model, target_layer)

    transform = get_transform()
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()

    cam = cam_gen.generate(img_t, class_idx=pred_idx)
    cam = cv2.resize(cam, (img.size[0], img.size[1]))

    img_np = np.array(img)
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.4 * heatmap + 0.6 * img_np).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("Zapisano Grad-CAM do:", output_path)
    print("Predykcja:", info["label"][str(pred_idx)])
    for i, p in enumerate(probs):
        print(f"  {i} ({info['label'][str(i)]}): {p:.4f}")


if __name__ == "__main__":
    cfg = Config()
    cfg.model_type = "deep_cnn"
    visualize_grad_cam(
        image_path="jakis_obrazek.png",
        ckpt_path="outputs/deep_cnn_adam_noaug.pt",
        cfg=cfg,
        output_path="grad_cam_example.png"
    )
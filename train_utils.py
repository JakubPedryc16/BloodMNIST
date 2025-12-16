# train_utils.py
import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from config import Config


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(cfg: Config, model: torch.nn.Module):
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(),
                                lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
    elif cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(model.parameters(),
                               lr=cfg.lr,
                               momentum=0.9,
                               weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Nieznany optimizer: {cfg.optimizer}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device) -> Dict[str, Any]:
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    cm = confusion_matrix(all_labels, all_preds)

    return {
        "loss": running_loss / len(loader.dataset),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "labels": all_labels,
        "preds": all_preds,
        "confusion_matrix": cm,
    }


def train_model(model,
                train_loader,
                val_loader,
                cfg: Config):

    device = get_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg, model)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "lr": []
    }

    best_val_f1 = 0.0
    best_state = None

    os.makedirs(cfg.output_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.output_dir, f"{cfg.experiment_name}.pt")

    for epoch in range(1, cfg.n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        scheduler.step(val_metrics["macro_f1"])

        print(
            f"Epoch {epoch:02d}/{cfg.n_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.1e}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = model.state_dict()
            torch.save(best_state, ckpt_path)
            print(f"  >> Zapisano nowy najlepszy model (val_macro_f1={best_val_f1:.4f})")

    # Załaduj najlepszy model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, ckpt_path

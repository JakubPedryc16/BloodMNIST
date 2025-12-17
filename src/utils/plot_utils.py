import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import seaborn as sns

from src.config import Config


def plot_training_curves(history: Dict[str, List[float]],
                         cfg: Config,
                         show: bool = False):
    os.makedirs(cfg.output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoka")
    plt.ylabel("Strata")
    plt.title(f"Loss – {cfg.experiment_name}")
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(cfg.output_dir, f"{cfg.experiment_name}_loss.png")
    plt.savefig(loss_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    plt.figure()
    plt.plot(epochs, history["val_accuracy"], label="val_accuracy")
    plt.plot(epochs, history["val_macro_f1"], label="val_macro_f1")
    plt.xlabel("Epoka")
    plt.ylabel("Wartość")
    plt.title(f"Metryki walidacyjne – {cfg.experiment_name}")
    plt.legend()
    plt.grid(True)
    metrics_path = os.path.join(cfg.output_dir, f"{cfg.experiment_name}_metrics.png")
    plt.savefig(metrics_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    plt.figure()
    plt.plot(epochs, history["lr"])
    plt.xlabel("Epoka")
    plt.ylabel("Learning rate")
    plt.title(f"LR – {cfg.experiment_name}")
    plt.grid(True)
    lr_path = os.path.join(cfg.output_dir, f"{cfg.experiment_name}_lr.png")
    plt.savefig(lr_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_confusion_matrix(cm: np.ndarray,
                          class_names,
                          cfg: Config,
                          normalize: bool = True,
                          show: bool = False):
    os.makedirs(cfg.output_dir, exist_ok=True)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm,
                annot=True,
                fmt=".2f" if normalize else "d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predykcja")
    plt.ylabel("Prawdziwa klasa")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    norm_str = "norm" if normalize else "raw"
    plt.title(f"Macierz pomyłek ({norm_str}) – {cfg.experiment_name}")
    cm_path = os.path.join(cfg.output_dir, f"{cfg.experiment_name}_cm_{norm_str}.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def save_classification_report(y_true,
                               y_pred,
                               class_names,
                               cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    path = os.path.join(cfg.output_dir, f"{cfg.experiment_name}_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_str)
    print("Zapisano classification report do", path)
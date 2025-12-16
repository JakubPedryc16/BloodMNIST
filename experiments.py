# experiments.py
from copy import deepcopy

from config import Config
from datasets import get_dataloaders
from models import build_model
from train_utils import train_model, evaluate, get_device
from plot_utils import plot_training_curves, plot_confusion_matrix, save_classification_report
from datasets import LABELS_BLOODMNIST_SHORT


def run_experiment(cfg: Config):
    print("=" * 80)
    print("Eksperyment:", cfg.experiment_name)
    print("=" * 80)

    train_loader, val_loader, test_loader, info = get_dataloaders(cfg)
    n_classes = len(info["label"])

    # pełne nazwy – do raportu tekstowego
    class_names_full = [info["label"][str(i)] for i in range(n_classes)]
    # krótkie nazwy – do macierzy pomyłek
    class_names_short = [LABELS_BLOODMNIST_SHORT[str(i)] for i in range(n_classes)]
    n_classes = len(info["label"])


    model = build_model(cfg.model_type, n_classes=n_classes)

    # trening
    model, history, ckpt_path = train_model(model, train_loader, val_loader, cfg)

    # wykresy treningu
    plot_training_curves(history, cfg, show=False)

    # ewaluacja na test
    device = get_device()
    import torch
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\nWyniki na teście:")
    print("  test_loss    :", test_metrics["loss"])
    print("  test_accuracy:", test_metrics["accuracy"])
    print("  test_macro_f1:", test_metrics["macro_f1"])

    # macierz pomyłek + classification report
    plot_confusion_matrix(test_metrics["confusion_matrix"], class_names_short, cfg,
                      normalize=True, show=False)
    plot_confusion_matrix(test_metrics["confusion_matrix"], class_names_short, cfg,
                      normalize=False, show=False)

    save_classification_report(test_metrics["labels"],
                           test_metrics["preds"],
                           class_names_full,
                           cfg)



if __name__ == "__main__":
    base_cfg = Config()

    # Eksperyment 1: prosty CNN, Adam, augmentacja
    cfg1 = deepcopy(base_cfg)
    cfg1.experiment_name = "simple_cnn_adam_aug"
    cfg1.model_type = "simple_cnn"
    cfg1.optimizer = "adam"
    cfg1.use_augment = True
    run_experiment(cfg1)

    # Eksperyment 2: prosty CNN, SGD, augmentacja
    cfg2 = deepcopy(base_cfg)
    cfg2.experiment_name = "simple_cnn_sgd_aug"
    cfg2.model_type = "simple_cnn"
    cfg2.optimizer = "sgd"
    cfg2.use_augment = True
    run_experiment(cfg2)

    # Eksperyment 3: deep CNN, Adam, bez augmentacji
    cfg3 = deepcopy(base_cfg)
    cfg3.experiment_name = "deep_cnn_adam_noaug"
    cfg3.model_type = "deep_cnn"
    cfg3.optimizer = "adam"
    cfg3.use_augment = False
    run_experiment(cfg3)

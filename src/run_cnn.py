from copy import deepcopy

from src.config import Config
from src.data.datasets import get_dataloaders
from src.models.cnn_models import build_model
from src.utils.train_utils import train_model, evaluate, get_device
from src.utils.plot_utils import plot_training_curves, plot_confusion_matrix, save_classification_report
from src.data.datasets import LABELS_BLOODMNIST_SHORT
import torch
import torch.nn as nn


def run_experiment(cfg: Config):
    print("=" * 80)
    print("Eksperyment:", cfg.experiment_name)
    print("=" * 80)

    train_loader, val_loader, test_loader, info = get_dataloaders(cfg)
    n_classes = len(info["label"])

    class_names_full = [info["label"][str(i)] for i in range(n_classes)]
    class_names_short = [LABELS_BLOODMNIST_SHORT[str(i)] for i in range(n_classes)]
    n_classes = len(info["label"])


    model = build_model(cfg.model_type, n_classes=n_classes)

    model, history, ckpt_path = train_model(model, train_loader, val_loader, cfg)

    plot_training_curves(history, cfg, show=False)

    device = get_device()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\nWyniki na teście:")
    print("  test_loss    :", test_metrics["loss"])
    print("  test_accuracy:", test_metrics["accuracy"])
    print("  test_macro_f1:", test_metrics["macro_f1"])

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

    cfg1 = deepcopy(base_cfg)
    cfg1.experiment_name = "simple_cnn_adam_aug"
    cfg1.model_type = "simple_cnn"
    cfg1.optimizer = "adam"
    cfg1.use_augment = True
    run_experiment(cfg1)

    cfg2 = deepcopy(base_cfg)
    cfg2.experiment_name = "simple_cnn_sgd_aug"
    cfg2.model_type = "simple_cnn"
    cfg2.optimizer = "sgd"
    cfg2.use_augment = True
    run_experiment(cfg2)

    cfg3 = deepcopy(base_cfg)
    cfg3.experiment_name = "deep_cnn_adam_noaug"
    cfg3.model_type = "deep_cnn"
    cfg3.optimizer = "adam"
    cfg3.use_augment = False
    run_experiment(cfg3)
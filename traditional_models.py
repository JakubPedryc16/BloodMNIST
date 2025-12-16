# traditional_models.py
import os
import time
from typing import Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import GridSearchCV

from config import Config
from datasets import download_bloodmnist, LABELS_BLOODMNIST_SHORT


# ==============================
# Ładowanie i przygotowanie danych
# ==============================

def load_flattened_data(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ładuje dane z bloodmnist.npz, spłaszcza obrazy do wektorów (N, 28*28*3)
    i zwraca X_train, y_train, X_val, y_val, X_test, y_test (wszystko jako numpy).
    """
    root = os.path.join(cfg.output_dir, "data")
    npz_path = download_bloodmnist(root)

    data = np.load(npz_path)

    X_train = data["train_images"]  # [N, 28, 28, 3]
    y_train = data["train_labels"].reshape(-1)

    X_val = data["val_images"]
    y_val = data["val_labels"].reshape(-1)

    X_test = data["test_images"]
    y_test = data["test_labels"].reshape(-1)

    # Spłaszczenie: (N, 28, 28, 3) -> (N, 2352)
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_val = X_val.reshape(X_val.shape[0], -1).astype(np.float32)
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

    print("Shapes after flatten:")
    print("  X_train:", X_train.shape, "y_train:", y_train.shape)
    print("  X_val  :", X_val.shape, "y_val  :", y_val.shape)
    print("  X_test :", X_test.shape, "y_test :", y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test


def scale_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Standaryzacja cech (średnia 0, odchylenie 1) na podstawie zbioru treningowego.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ==============================
# Wizualizacja – macierz pomyłek + podsumowanie modeli
# ==============================

def plot_confusion_matrix(cm: np.ndarray, class_names, title: str, output_path: str, normalize: bool = True):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predykcja")
    plt.ylabel("Prawdziwa klasa")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print("Zapisano macierz pomyłek do:", output_path)


def plot_results_summary(results: List[Dict], cfg: Config):
    """
    Prosta wizualizacja porównująca accuracy i macro-F1 dla wszystkich 4 modeli.
    """
    out_dir = os.path.join(cfg.output_dir, "traditional_models")
    os.makedirs(out_dir, exist_ok=True)

    names = [r["name"] for r in results]
    accs = [r["test_accuracy"] for r in results]
    f1s = [r["test_macro_f1"] for r in results]

    x = np.arange(len(names))

    # Wykres accuracy
    plt.figure(figsize=(8, 4))
    plt.bar(x, accs)
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("Test accuracy")
    plt.title("Porównanie accuracy klasycznych modeli (BloodMNIST)")
    plt.tight_layout()
    acc_path = os.path.join(out_dir, "traditional_models_test_accuracy.png")
    plt.savefig(acc_path, dpi=200)
    plt.close()
    print("Zapisano wykres accuracy do:", acc_path)

    # Wykres macro-F1
    plt.figure(figsize=(8, 4))
    plt.bar(x, f1s)
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("Test macro-F1")
    plt.title("Porównanie macro-F1 klasycznych modeli (BloodMNIST)")
    plt.tight_layout()
    f1_path = os.path.join(out_dir, "traditional_models_test_macro_f1.png")
    plt.savefig(f1_path, dpi=200)
    plt.close()
    print("Zapisano wykres macro-F1 do:", f1_path)


# ==============================
# Optymalizacja hiperparametrów (GridSearchCV – walidacja krzyżowa)
# ==============================

def tune_with_grid_search(
    name: str,
    base_estimator,
    param_grid: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 3,
):
    """
    Prosta optymalizacja hiperparametrów z użyciem GridSearchCV.
    Używamy f1_macro jako metryki i walidacji krzyżowej cv-krotnej.
    """
    print("-" * 80)
    print(f"[{name}] GridSearchCV – walidacja krzyżowa (cv={cv})")
    print("Param grid:", param_grid)
    grid = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
    )
    t0 = time.time()
    grid.fit(X_train, y_train)
    dt = time.time() - t0
    print(f"[{name}] Zakończono GridSearchCV w {dt:.2f} s")
    print(f"[{name}] Najlepsze parametry: {grid.best_params_}")
    print(f"[{name}] Najlepszy wynik (f1_macro): {grid.best_score_:.4f}")
    print("-" * 80)
    return grid.best_estimator_


# ==============================
# Trenowanie i ewaluacja pojedynczego modelu
# ==============================

def run_model(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Config,
    class_names,
) -> Dict:
    """
    Trenuje dany model, liczy metryki na walidacji i teście, zapisuje raporty i wykresy.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_dir = os.path.join(cfg.output_dir, "traditional_models")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(f"Model: {name}")
    print("=" * 80)

    # Trening na pełnym zbiorze treningowym
    t0 = time.time()
    model.fit(X_train, y_train)
    t_train = time.time() - t0
    print(f"Czas trenowania: {t_train:.2f} s")

    # Walidacja
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")
    print(f"Walidacja: accuracy={val_acc:.4f}, macro-F1={val_f1:.4f}")

    # Test
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")
    print(f"Test:       accuracy={test_acc:.4f}, macro-F1={test_f1:.4f}")

    # Macierz pomyłek
    cm = confusion_matrix(y_test, y_test_pred)
    cm_norm_path = os.path.join(out_dir, f"{name}_cm_norm.png")
    cm_raw_path = os.path.join(out_dir, f"{name}_cm_raw.png")

    plot_confusion_matrix(
        cm,
        class_names,
        title=f"Macierz pomyłek (norm) – {name}",
        output_path=cm_norm_path,
        normalize=True,
    )
    plot_confusion_matrix(
        cm,
        class_names,
        title=f"Macierz pomyłek (raw) – {name}",
        output_path=cm_raw_path,
        normalize=False,
    )

    # Raport tekstowy
    report = classification_report(y_test, y_test_pred, target_names=class_names, digits=4)
    report_path = os.path.join(out_dir, f"{name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("Zapisano classification report do:", report_path)

    return {
        "name": name,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "train_time_sec": t_train,
    }


# ==============================
# Główny skrypt – 4 klasyczne modele
# ==============================

if __name__ == "__main__":
    cfg = Config()
    class_names = [LABELS_BLOODMNIST_SHORT[str(i)] for i in range(8)]

    # 1. Załaduj dane
    X_train, y_train, X_val, y_val, X_test, y_test = load_flattened_data(cfg)

    # 2. Standaryzacja
    X_train_s, X_val_s, X_test_s, scaler = scale_data(X_train, X_val, X_test)

    results = []

    # ======================
    # 3. Logistic Regression
    # ======================
    # Uwaga: minimalna wersja (bez multi_class, n_jobs – żeby działało na każdej wersji sklearn)
    lr_base = LogisticRegression(max_iter=500)
    lr_param_grid = {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    }
    lr_best = tune_with_grid_search(
        name="logistic_regression",
        base_estimator=lr_base,
        param_grid=lr_param_grid,
        X_train=X_train_s,
        y_train=y_train,
        cv=3,
    )
    results.append(
        run_model(
            name="logistic_regression",
            model=lr_best,
            X_train=X_train_s,
            y_train=y_train,
            X_val=X_val_s,
            y_val=y_val,
            X_test=X_test_s,
            y_test=y_test,
            cfg=cfg,
            class_names=class_names,
        )
    )

    # =========
    # 4. SVM RBF
    # =========
    svm_base = SVC(kernel="rbf")
    svm_param_grid = {
        "C": [0.5, 1.0, 5.0],
        "gamma": ["scale", 0.01, 0.001],
    }
    svm_best = tune_with_grid_search(
        name="svm_rbf",
        base_estimator=svm_base,
        param_grid=svm_param_grid,
        X_train=X_train_s,
        y_train=y_train,
        cv=3,
    )
    results.append(
        run_model(
            name="svm_rbf",
            model=svm_best,
            X_train=X_train_s,
            y_train=y_train,
            X_val=X_val_s,
            y_val=y_val,
            X_test=X_test_s,
            y_test=y_test,
            cfg=cfg,
            class_names=class_names,
        )
    )

    # ===============
    # 5. Random Forest
    # ===============
    rf_base = RandomForestClassifier(random_state=cfg.seed)
    rf_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 20, 40],
        "min_samples_split": [2, 5],
    }
    rf_best = tune_with_grid_search(
        name="random_forest",
        base_estimator=rf_base,
        param_grid=rf_param_grid,
        X_train=X_train_s,
        y_train=y_train,
        cv=3,
    )
    results.append(
        run_model(
            name="random_forest",
            model=rf_best,
            X_train=X_train_s,
            y_train=y_train,
            X_val=X_val_s,
            y_val=y_val,
            X_test=X_test_s,
            y_test=y_test,
            cfg=cfg,
            class_names=class_names,
        )
    )

    # =====================
    # 6. MLPClassifier (NN)
    # =====================
    mlp_base = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=50,
        batch_size=256,
        random_state=cfg.seed,
    )
    mlp_param_grid = {
        "hidden_layer_sizes": [(128,), (256, 128)],
        "alpha": [1e-4, 1e-3],
        "learning_rate_init": [1e-3, 5e-4],
    }
    mlp_best = tune_with_grid_search(
        name="mlp_sklearn",
        base_estimator=mlp_base,
        param_grid=mlp_param_grid,
        X_train=X_train_s,
        y_train=y_train,
        cv=3,
    )
    results.append(
        run_model(
            name="mlp_sklearn",
            model=mlp_best,
            X_train=X_train_s,
            y_train=y_train,
            X_val=X_val_s,
            y_val=y_val,
            X_test=X_test_s,
            y_test=y_test,
            cfg=cfg,
            class_names=class_names,
        )
    )

    # 7. Podsumowanie wyników + wizualizacja
    print("\nPodsumowanie wyników klasycznych modeli:")
    for res in results:
        print(
            f"{res['name']:>18} | "
            f"val_acc={res['val_accuracy']:.4f}, "
            f"val_macro_f1={res['val_macro_f1']:.4f}, "
            f"test_acc={res['test_accuracy']:.4f}, "
            f"test_macro_f1={res['test_macro_f1']:.4f}, "
            f"train_time={res['train_time_sec']:.2f}s"
        )

    plot_results_summary(results, cfg)

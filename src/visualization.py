import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch.utils.data import DataLoader
import pandas as pd

label_map = {
    0: "NEUTROPHIL",
    1: "EOSINOPHIL",
    2: "BASOPHIL",
    3: "LYMPHOCYTE",
    4: "MONOCYTE",
    5: "IG",
    6: "RBC",
    7: "PLATELET"
}

def plot_class_distribution(dataset, title="Rozkład klas"):

    labels = [label_map[label.item()] for _, label in dataset]
    df = pd.DataFrame({"label": labels})
    
    plt.figure(figsize=(10, 5))
    sns.countplot(x="label", data=df, palette="Set2", hue="label", dodge=False, legend=False)
    plt.xticks(rotation=45)
    plt.ylabel("Liczba próbek")
    plt.xlabel("Typ komórki")
    plt.title(title)
    plt.show()


def show_sample_images(dataset, n=10):
    rows = 2
    cols = n // 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(n):
        img, label = dataset[i]
        img = img * 0.5 + 0.5
        img_np = img.numpy().transpose((1, 2, 0))
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].set_title(label_map[label.item()])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()



def dataset_stats(dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mean = 0.0
    std = 0.0
    nb_samples = 0
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    return mean, std

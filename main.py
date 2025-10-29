from src.dataset import get_bloodmnist_dataloaders
from src.visualization import plot_class_distribution, show_sample_images, dataset_stats

train_loader, val_loader, test_loader, dataset_sizes, train_dataset, val_dataset, test_dataset = get_bloodmnist_dataloaders()

plot_class_distribution(train_dataset, "Rozkład klas w zbiorze treningowym")
plot_class_distribution(val_dataset, "Rozkład klas w zbiorze walidacyjnym")
plot_class_distribution(test_dataset, "Rozkład klas w zbiorze testowym")

show_sample_images(train_dataset)

mean, std = dataset_stats(train_dataset)
print("Mean:", mean)
print("Std:", std)

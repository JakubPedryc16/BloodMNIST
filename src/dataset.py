
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import BloodMNIST
from config import BATCH_SIZE

def get_bloodmnist_dataloaders(batch_size=64):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = BloodMNIST(split='train', transform=data_transform, download=True)
    val_dataset = BloodMNIST(split='val', transform=data_transform, download=True)
    test_dataset = BloodMNIST(split='test', transform=data_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    dataset_sizes = {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset)
    }

    return train_loader, val_loader, test_loader, dataset_sizes, train_dataset, val_dataset, test_dataset

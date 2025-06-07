import torch
from torchvision import transforms
from datasets.jewellery_dataset import JewelleryDataset
from torch.utils.data import DataLoader

BATCH_SIZE = 32

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

val_test_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor()])

# Paths
train_dir = "data/train"
val_dir = "data/val"
test_dir = "data/test"

# Datasets
train_dataset = JewelleryDataset(train_dir, transform=train_transform)
val_dataset = JewelleryDataset(val_dir, transform=val_test_transform)
test_dataset = JewelleryDataset(test_dir, transform=val_test_transform)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 🔁 Sample usage
if __name__ == "__main__":
    print("Classes:", train_dataset.classes)
    for images, labels in train_loader:
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        break

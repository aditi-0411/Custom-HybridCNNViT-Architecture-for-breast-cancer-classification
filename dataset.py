"""
dataset.py

BreakHis dataset loading, transforms, and train/val/test splitting.

TODO (known limitation): the current split uses `random_split` over the full
image list, which does not guarantee patients/slides are kept entirely within
one split. BreakHis has multiple images per patient, so this risks leakage.
For a rigorous evaluation, replace `random_split` with a patient-stratified
split (group by patient ID parsed from the filename, then split groups).
"""

import os

import kagglehub
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class BreakHisDataset(Dataset):
    """Recursively loads benign/malignant images from a BreakHis root dir."""

    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for label, class_name in enumerate(["benign", "malignant"]):
            class_path = os.path.join(root_dir, class_name)
            for root, _, files in os.walk(class_path):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.samples.append((os.path.join(root, file), label))

        print(f"Loaded {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def find_data_root(base_path):
    """Walks a downloaded dataset dir to find the folder containing
    'benign' and 'malignant' subfolders."""
    for root, dirs, _ in os.walk(base_path):
        if "benign" in dirs and "malignant" in dirs:
            return root
    return None


def download_breakhis():
    path = kagglehub.dataset_download("ambarish/breakhis")
    data_dir = find_data_root(path)
    if data_dir is None:
        raise ValueError("Could not find dataset root with 'benign' and 'malignant' folders")
    return data_dir


def get_dataloaders(batch_size=16, train_frac=0.7, val_frac=0.15, seed=42, num_workers=0):
    """Downloads BreakHis (if needed) and returns train/val/test DataLoaders."""
    data_dir = download_breakhis()

    full_dataset = BreakHisDataset(data_dir, transform=train_transform)

    train_size = int(train_frac * len(full_dataset))
    val_size = int(val_frac * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Val/test should use deterministic (non-augmented) transforms.
    val_dataset.dataset.transform = eval_transform
    test_dataset.dataset.transform = eval_transform

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)} | Test size: {len(test_dataset)}")
    return train_loader, val_loader, test_loader

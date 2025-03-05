import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold

class CoffeeDataset(Dataset):
    def __init__(self, root_dir="dataset_kopi", transform=None):
        """
        Args:
            root_dir (string): Directory with all the images organized in class folders
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform

        # Get all class folders (labels)
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.classes.sort()  # Sort to ensure consistent class indices

        # Create class to index mapping
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Collect all image paths and their corresponding labels
        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_kfold_data_loaders(dataset, batch_size=32, num_workers=4, n_splits=5, seed=42):
    """
    Create data loaders for K-Fold Cross Validation
    
    Args:
        dataset (CoffeeDataset): The full dataset
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        n_splits (int): Number of folds
        seed (int): Random seed for reproducibility
        
    Returns:
        List of tuples: Each tuple contains (train_loader, val_loader) for each fold
    """
    # Define transforms for training and validation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize KFold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Store all data loaders
    fold_loaders = []

    # Split dataset into folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"  Training samples: {len(train_idx)}")
        print(f"  Validation samples: {len(val_idx)}")

        # Create Subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Apply transforms
        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = val_transform

        # Create DataLoaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        fold_loaders.append((train_loader, val_loader))

    return fold_loaders


# Example usage
if __name__ == "__main__":
    # Create the full dataset
    full_dataset = CoffeeDataset()

    # Get K-Fold data loaders
    fold_loaders = get_kfold_data_loaders(full_dataset, n_splits=5)

    # Iterate through folds
    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"\nProcessing Fold {fold + 1}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")

        # Example: Get a sample batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")

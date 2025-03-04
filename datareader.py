
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class CoffeeDataset(Dataset):
    def __init__(self, root_dir="dataset_kopi", transform=None, split="train", train_ratio=0.8, seed=42):
        """
        Args:
            root_dir (string): Directory with all the images organized in class folders
            transform (callable, optional): Optional transform to be applied on a sample
            split (string): 'train' or 'val' to specify the dataset split
            train_ratio (float): Ratio of data to use for training
            seed (int): Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        
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
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Split dataset into train and validation
        indices = np.arange(len(self.image_paths))
        np.random.shuffle(indices)
        
        split_idx = int(len(indices) * self.train_ratio)
        
        if self.split == "train":
            self.indices = indices[:split_idx]
        else:  # validation set
            self.indices = indices[split_idx:]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Use the mapped index
        actual_idx = self.indices[idx]
        
        img_path = self.image_paths[actual_idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[actual_idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(batch_size=32, num_workers=4):
    """
    Create data loaders for training and validation
    
    Args:
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
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
    
    # Create datasets
    train_dataset = CoffeeDataset(split="train", transform=train_transform)
    val_dataset = CoffeeDataset(split="val", transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.classes


# Example usage
if __name__ == "__main__":
    train_loader, val_loader, classes = get_data_loaders()
    print(f"Classes: {classes}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Get a sample batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

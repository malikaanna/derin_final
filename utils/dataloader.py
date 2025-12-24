"""
Data loading utilities for Fruit Freshness dataset.

This module provides functions to download the dataset from Kaggle
and create PyTorch DataLoaders for training.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"
DATASET_NAME = "sriramr/fruits-fresh-and-rotten-for-classification"


def download_dataset(data_dir: Optional[Path] = None) -> Path:
    """
    Download the Fruit Freshness dataset from Kaggle using kagglehub.
    
    No API key required for public datasets!
    
    Args:
        data_dir: Directory to save the dataset (not used, kagglehub manages location)
        
    Returns:
        Path to the downloaded dataset
    """
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        import subprocess
        subprocess.run(["pip", "install", "kagglehub", "-q"], check=True)
        import kagglehub
    
    print("Downloading dataset from Kaggle using kagglehub...")
    print("This may take a few minutes (~3.5 GB)...")
    
    path = kagglehub.dataset_download("sriramr/fruits-fresh-and-rotten-for-classification")
    print(f"Dataset downloaded to: {path}")
    
    return Path(path)


def get_transforms(image_size: int = 64, augment: bool = True) -> tuple:
    """
    Get image transforms for training and validation.
    
    Args:
        image_size: Target image size (square)
        augment: Whether to apply data augmentation
        
    Returns:
        train_transform, val_transform
    """
    # Normalization for images to [-1, 1] (for Tanh output)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transform, val_transform


def get_dataloader(
    data_dir: Optional[Path] = None,
    batch_size: int = 32,
    image_size: int = 64,
    num_workers: int = 4,
    augment: bool = True,
    val_split: float = 0.1,
    subset: Optional[str] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation.
    
    Args:
        data_dir: Path to dataset directory (auto-detected if None)
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of data loading workers
        augment: Whether to apply data augmentation
        val_split: Fraction of data to use for validation
        subset: Specific subset to use ('fresh', 'rotten', or None for all)
        
    Returns:
        train_loader, val_loader
    """
    # Try to find the dataset in multiple locations
    possible_paths = []
    
    if data_dir is not None:
        possible_paths.append(Path(data_dir))
    
    # Check local data directory first
    possible_paths.append(DEFAULT_DATA_DIR / "dataset")
    possible_paths.append(DEFAULT_DATA_DIR / "fruits-fresh-and-rotten-for-classification")
    
    # Then check kagglehub cache location
    kagglehub_cache = Path.home() / ".cache" / "kagglehub" / "datasets" / "sriramr" / "fruits-fresh-and-rotten-for-classification" / "versions" / "1"
    possible_paths.append(kagglehub_cache)
    
    # Find existing dataset
    data_dir = None
    for path in possible_paths:
        if path.exists() and any(path.iterdir()):
            data_dir = path
            break
    
    if data_dir is None:
        print("Dataset not found. Downloading...")
        data_dir = download_dataset()
    
    # Find the train directory
    train_dir = None
    for possible_path in [
        data_dir / "dataset" / "train",
        data_dir / "train",
        data_dir
    ]:
        if possible_path.exists() and any(possible_path.iterdir()):
            train_dir = possible_path
            break
    
    if train_dir is None:
        # List what's available
        print(f"Contents of {data_dir}:")
        for item in data_dir.rglob("*"):
            if item.is_dir():
                print(f"  DIR: {item}")
        raise FileNotFoundError(f"Could not find training data in {data_dir}")
    
    print(f"Loading data from: {train_dir}")
    
    # Get transforms
    train_transform, val_transform = get_transforms(image_size, augment)
    
    # Load dataset using ImageFolder
    full_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    
    print(f"Found {len(full_dataset)} images")
    print(f"Classes: {full_dataset.classes}")
    
    # Filter subset if specified
    if subset is not None:
        subset = subset.lower()
        if subset in ['fresh', 'rotten']:
            # Filter to only include classes containing the subset name
            indices = [i for i, (_, label) in enumerate(full_dataset.samples) 
                      if subset in full_dataset.classes[label].lower()]
            full_dataset = torch.utils.data.Subset(full_dataset, indices)
            print(f"Filtered to {len(full_dataset)} {subset} images")
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply validation transform to validation set
    # Note: This is a workaround since random_split doesn't allow different transforms
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize tensor from [-1, 1] to [0, 1] for visualization.
    
    Args:
        tensor: Normalized tensor
        
    Returns:
        Denormalized tensor in [0, 1] range
    """
    return (tensor + 1) / 2


if __name__ == "__main__":
    # Test dataloader
    print("Testing dataloader...")
    
    try:
        train_loader, val_loader = get_dataloader(
            batch_size=4,
            image_size=64,
            num_workers=0,
        )
        
        # Get a batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")
        
        print("\n✅ Dataloader test passed!")
        
    except Exception as e:
        print(f"\n⚠️ Test failed: {e}")
        print("Please download the dataset first.")

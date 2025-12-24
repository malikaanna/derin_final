"""
Visualization utilities for GAN and VAE training.

This module provides functions to visualize generated images,
plot training losses, and save results.
"""

import os
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize tensor from [-1, 1] to [0, 1] for visualization.
    
    Args:
        tensor: Normalized tensor
        
    Returns:
        Denormalized tensor in [0, 1] range
    """
    return torch.clamp((tensor + 1) / 2, 0, 1)


def show_images(
    images: torch.Tensor,
    nrow: int = 8,
    title: Optional[str] = None,
    figsize: tuple = (12, 12),
    save_path: Optional[str] = None,
) -> None:
    """
    Display a grid of images.
    
    Args:
        images: Tensor of images (N, C, H, W)
        nrow: Number of images per row
        title: Optional title for the plot
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Denormalize if needed
    if images.min() < 0:
        images = denormalize(images)
    
    # Create grid
    grid = make_grid(images.cpu(), nrow=nrow, padding=2, normalize=False)
    
    # Convert to numpy
    grid_np = grid.permute(1, 2, 0).numpy()
    
    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(grid_np)
    plt.axis('off')
    
    if title:
        plt.title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved image to {save_path}")
    
    plt.show()
    plt.close()


def save_generated_images(
    images: torch.Tensor,
    save_path: Union[str, Path],
    nrow: int = 8,
    title: Optional[str] = None,
) -> None:
    """
    Save a grid of generated images to file.
    
    Args:
        images: Tensor of images (N, C, H, W)
        save_path: Path to save the image
        nrow: Number of images per row
        title: Optional title for the plot
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Denormalize if needed
    if images.min() < 0:
        images = denormalize(images)
    
    # Create grid
    grid = make_grid(images.cpu(), nrow=nrow, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).numpy()
    
    # Plot and save
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(grid_np)
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=14)
    
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    print(f"Saved generated images to {save_path}")


def plot_losses(
    losses: dict,
    title: str = "Training Losses",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> None:
    """
    Plot training losses.
    
    Args:
        losses: Dictionary of loss names to loss values (lists)
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, len(losses), figsize=figsize)
    
    if len(losses) == 1:
        axes = [axes]
    
    colors = plt.cm.tab10.colors
    
    for idx, (name, values) in enumerate(losses.items()):
        ax = axes[idx]
        ax.plot(values, color=colors[idx % len(colors)], linewidth=1.5)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(name, fontsize=12)
        ax.set_title(name, fontsize=14)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved loss plot to {save_path}")
    
    plt.show()
    plt.close()


def plot_comparison(
    real_images: torch.Tensor,
    vae_images: torch.Tensor,
    gan_images: torch.Tensor,
    nrow: int = 4,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 12),
) -> None:
    """
    Plot comparison between real, VAE-generated, and GAN-generated images.
    
    Args:
        real_images: Real images tensor
        vae_images: VAE-generated images tensor
        gan_images: GAN-generated images tensor
        nrow: Number of images per row
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    images_list = [
        (real_images, "Gerçek Görüntüler (Real)"),
        (vae_images, "VAE Üretimi (VAE Generated)"),
        (gan_images, "GAN Üretimi (GAN Generated)"),
    ]
    
    for ax, (images, title) in zip(axes, images_list):
        if images.min() < 0:
            images = denormalize(images)
        
        grid = make_grid(images.cpu()[:nrow*2], nrow=nrow, padding=2, normalize=False)
        grid_np = grid.permute(1, 2, 0).numpy()
        
        ax.imshow(grid_np)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    fig.suptitle("Model Karşılaştırması", fontsize=18, fontweight='bold', y=0.98)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved comparison to {save_path}")
    
    plt.show()
    plt.close()


def plot_latent_space(
    model,
    dataloader,
    device: torch.device,
    num_samples: int = 1000,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize the latent space of a VAE using t-SNE or PCA.
    
    Args:
        model: VAE model
        dataloader: DataLoader with images
        device: Device to use
        num_samples: Number of samples to use
        save_path: Optional path to save the figure
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("sklearn not installed. Skipping latent space visualization.")
        return
    
    model.eval()
    latent_vectors = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(latent_vectors) * images.size(0) >= num_samples:
                break
            
            images = images.to(device)
            mu, _ = model.encode(images)
            latent_vectors.append(mu.cpu().numpy())
            labels_list.append(labels.numpy())
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)[:num_samples]
    labels_array = np.concatenate(labels_list, axis=0)[:num_samples]
    
    # Apply PCA
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        latent_2d[:, 0], 
        latent_2d[:, 1], 
        c=labels_array, 
        cmap='tab10',
        alpha=0.6,
        s=10
    )
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('VAE Latent Space (PCA)', fontsize=14)
    
    plt.colorbar(scatter, ax=ax, label='Class')
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved latent space plot to {save_path}")
    
    plt.show()
    plt.close()


def create_interpolation(
    model,
    z1: torch.Tensor,
    z2: torch.Tensor,
    num_steps: int = 10,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create interpolation between two latent vectors.
    
    Args:
        model: VAE or Generator model
        z1: First latent vector
        z2: Second latent vector
        num_steps: Number of interpolation steps
        device: Device to use
        
    Returns:
        Tensor of interpolated images
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Create interpolation weights
    alphas = torch.linspace(0, 1, num_steps, device=device)
    
    interpolated_images = []
    
    with torch.no_grad():
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            
            # Handle different model types
            if hasattr(model, 'decode'):
                # VAE
                img = model.decode(z)
            else:
                # Generator
                img = model(z)
            
            interpolated_images.append(img)
    
    return torch.cat(interpolated_images, dim=0)


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization functions...")
    
    # Create dummy images
    dummy_images = torch.randn(16, 3, 64, 64)
    
    # Test denormalize
    denorm = denormalize(dummy_images)
    assert denorm.min() >= 0 and denorm.max() <= 1, "Denormalization failed"
    
    print("✅ Visualization tests passed!")

"""
VAE Training Script for Fruit Freshness Dataset

This script trains a Variational Autoencoder on fruit images
and saves generated samples during training.

Usage:
    python train_vae.py --epochs 50 --batch_size 32 --lr 0.0002
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.vae import VAE, vae_loss
from utils.dataloader import get_dataloader
from utils.visualization import save_generated_images, plot_losses


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE on Fruit Freshness Dataset")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent space dimension")
    parser.add_argument("--kl_weight", type=float, default=1.0, help="KL divergence weight (beta-VAE)")
    
    # Data parameters
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="outputs/vae", help="Output directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--sample_every", type=int, default=5, help="Generate samples every N epochs")
    
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, device, kl_weight):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_recon = 0
    total_kl = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for images, _ in pbar:
        images = images.to(device)
        
        # Forward pass
        recon_images, mu, logvar = model(images)
        
        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(recon_images, images, mu, logvar, kl_weight)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}'
        })
    
    return {
        'total': total_loss / num_batches,
        'recon': total_recon / num_batches,
        'kl': total_kl / num_batches
    }


def validate(model, dataloader, device, kl_weight):
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            loss, _, _ = vae_loss(recon_images, images, mu, logvar, kl_weight)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    try:
        train_loader, val_loader = get_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nPlease download the dataset first:")
        print("  1. Install kaggle: pip install kaggle")
        print("  2. Configure API: ~/.kaggle/kaggle.json")
        print("  3. Run: kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification -p data --unzip")
        return
    
    # Create model
    print(f"Creating VAE with latent_dim={args.latent_dim}...")
    model = VAE(latent_dim=args.latent_dim).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Training history
    history = {
        'train_loss': [],
        'train_recon': [],
        'train_kl': [],
        'val_loss': []
    }
    
    # Fixed noise for consistent sample generation
    fixed_noise = torch.randn(64, args.latent_dim, device=device)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"KL weight (beta): {args.kl_weight}")
    print("-" * 50)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.kl_weight)
        
        # Validate
        val_loss = validate(model, val_loader, device, args.kl_weight)
        
        # Record history
        history['train_loss'].append(train_metrics['total'])
        history['train_recon'].append(train_metrics['recon'])
        history['train_kl'].append(train_metrics['kl'])
        history['val_loss'].append(val_loss)
        
        # Print progress
        print(f"Epoch [{epoch}/{args.epochs}] "
              f"Train: {train_metrics['total']:.4f} (Recon: {train_metrics['recon']:.4f}, KL: {train_metrics['kl']:.4f}) "
              f"Val: {val_loss:.4f}")
        
        # Generate samples
        if epoch % args.sample_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                samples = model.decode(fixed_noise)
            save_generated_images(
                samples,
                output_dir / f"samples_epoch_{epoch:03d}.png",
                nrow=8,
                title=f"VAE Samples - Epoch {epoch}"
            )
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"vae_epoch_{epoch:03d}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['total'],
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / "vae_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, best_path)
    
    # Save final model
    final_path = checkpoint_dir / "vae_final.pt"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, final_path)
    print(f"\nSaved final model: {final_path}")
    
    # Plot training curves
    plot_losses(
        {
            'Total Loss': history['train_loss'],
            'Reconstruction': history['train_recon'],
            'KL Divergence': history['train_kl'],
        },
        title="VAE Training Losses",
        save_path=output_dir / "training_losses.png"
    )
    
    # Generate final samples
    print("\nGenerating final samples...")
    model.eval()
    with torch.no_grad():
        final_samples = model.sample(64, device)
    save_generated_images(
        final_samples,
        output_dir / "final_samples.png",
        nrow=8,
        title="VAE Final Generated Samples"
    )
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Outputs saved to: {output_dir}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()

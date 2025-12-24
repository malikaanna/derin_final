"""
DCGAN Training Script for Fruit Freshness Dataset

This script trains a Deep Convolutional GAN on fruit images
and saves generated samples during training.

Usage:
    python train_dcgan.py --epochs 50 --batch_size 32 --lr 0.0002
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.dcgan import Generator, Discriminator, weights_init
from utils.dataloader import get_dataloader
from utils.visualization import save_generated_images, plot_losses


def parse_args():
    parser = argparse.ArgumentParser(description="Train DCGAN on Fruit Freshness Dataset")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--noise_dim", type=int, default=100, help="Noise dimension")
    parser.add_argument("--ngf", type=int, default=64, help="Generator feature maps")
    parser.add_argument("--ndf", type=int, default=64, help="Discriminator feature maps")
    
    # GAN training tricks
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--d_steps", type=int, default=1, help="Discriminator steps per generator step")
    
    # Data parameters
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="outputs/dcgan", help="Output directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--sample_every", type=int, default=5, help="Generate samples every N epochs")
    
    return parser.parse_args()


def train_epoch(
    generator, 
    discriminator, 
    dataloader, 
    optimizer_g, 
    optimizer_d,
    criterion,
    device,
    noise_dim,
    label_smoothing,
    d_steps
):
    """Train for one epoch."""
    generator.train()
    discriminator.train()
    
    total_d_loss = 0
    total_g_loss = 0
    total_d_real = 0
    total_d_fake = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (real_images, _) in enumerate(pbar):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        # Labels with smoothing
        real_label = 1.0 - label_smoothing
        fake_label = 0.0
        
        real_labels = torch.full((batch_size, 1, 1, 1), real_label, device=device)
        fake_labels = torch.full((batch_size, 1, 1, 1), fake_label, device=device)
        
        # ========================
        # Train Discriminator
        # ========================
        for _ in range(d_steps):
            optimizer_d.zero_grad()
            
            # Train on real images
            output_real = discriminator(real_images)
            loss_d_real = criterion(output_real, real_labels)
            
            # Train on fake images
            noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach())
            loss_d_fake = criterion(output_fake, fake_labels)
            
            # Combined discriminator loss
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()
        
        # ========================
        # Train Generator
        # ========================
        optimizer_g.zero_grad()
        
        # Generate fake images
        noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
        fake_images = generator(noise)
        
        # Try to fool discriminator
        output_fake = discriminator(fake_images)
        loss_g = criterion(output_fake, real_labels)  # Generator wants discriminator to say "real"
        
        loss_g.backward()
        optimizer_g.step()
        
        # Statistics
        total_d_loss += loss_d.item()
        total_g_loss += loss_g.item()
        total_d_real += output_real.mean().item()
        total_d_fake += output_fake.mean().item()
        num_batches += 1
        
        pbar.set_postfix({
            'D_loss': f'{loss_d.item():.4f}',
            'G_loss': f'{loss_g.item():.4f}',
            'D(x)': f'{output_real.mean().item():.3f}',
            'D(G(z))': f'{output_fake.mean().item():.3f}'
        })
    
    return {
        'd_loss': total_d_loss / num_batches,
        'g_loss': total_g_loss / num_batches,
        'd_real': total_d_real / num_batches,
        'd_fake': total_d_fake / num_batches
    }


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
        train_loader, _ = get_dataloader(
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
    
    # Create models
    print(f"Creating DCGAN with noise_dim={args.noise_dim}, ngf={args.ngf}, ndf={args.ndf}...")
    generator = Generator(noise_dim=args.noise_dim, ngf=args.ngf).to(device)
    discriminator = Discriminator(ndf=args.ndf).to(device)
    
    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    print(f"Total parameters: {g_params + d_params:,}")
    
    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Training history
    history = {
        'd_loss': [],
        'g_loss': [],
        'd_real': [],
        'd_fake': []
    }
    
    # Fixed noise for consistent sample generation
    fixed_noise = torch.randn(64, args.noise_dim, 1, 1, device=device)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Label smoothing: {args.label_smoothing}")
    print("-" * 50)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        metrics = train_epoch(
            generator, discriminator, train_loader,
            optimizer_g, optimizer_d, criterion,
            device, args.noise_dim, args.label_smoothing, args.d_steps
        )
        
        # Record history
        for key in history:
            history[key].append(metrics[key])
        
        # Print progress
        print(f"Epoch [{epoch}/{args.epochs}] "
              f"D_loss: {metrics['d_loss']:.4f} G_loss: {metrics['g_loss']:.4f} "
              f"D(x): {metrics['d_real']:.3f} D(G(z)): {metrics['d_fake']:.3f}")
        
        # Generate samples
        if epoch % args.sample_every == 0 or epoch == 1:
            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_noise)
            save_generated_images(
                samples,
                output_dir / f"samples_epoch_{epoch:03d}.png",
                nrow=8,
                title=f"DCGAN Samples - Epoch {epoch}"
            )
            generator.train()
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"dcgan_epoch_{epoch:03d}.pt"
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'metrics': metrics,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = checkpoint_dir / "dcgan_final.pt"
    torch.save({
        'epoch': args.epochs,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'history': history,
    }, final_path)
    print(f"\nSaved final model: {final_path}")
    
    # Plot training curves
    plot_losses(
        {
            'Discriminator Loss': history['d_loss'],
            'Generator Loss': history['g_loss'],
        },
        title="DCGAN Training Losses",
        save_path=output_dir / "training_losses.png"
    )
    
    # Plot D(x) and D(G(z))
    plot_losses(
        {
            'D(x) - Real': history['d_real'],
            'D(G(z)) - Fake': history['d_fake'],
        },
        title="Discriminator Outputs",
        save_path=output_dir / "discriminator_outputs.png"
    )
    
    # Generate final samples
    print("\nGenerating final samples...")
    generator.eval()
    with torch.no_grad():
        final_samples = generator.generate(64, device)
    save_generated_images(
        final_samples,
        output_dir / "final_samples.png",
        nrow=8,
        title="DCGAN Final Generated Samples"
    )
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Final D_loss: {history['d_loss'][-1]:.4f}")
    print(f"Final G_loss: {history['g_loss'][-1]:.4f}")
    print(f"Outputs saved to: {output_dir}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()

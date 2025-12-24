"""
Model Evaluation Script

This script loads trained VAE and DCGAN models and generates sample images
for visual inspection and comparison.

Usage:
    python evaluate.py --vae_checkpoint checkpoints/vae_final.pt --gan_checkpoint checkpoints/dcgan_final.pt
"""

import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.vae import VAE
from models.dcgan import Generator
from utils.dataloader import get_dataloader, denormalize
from utils.visualization import save_generated_images, show_images


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    
    parser.add_argument("--vae_checkpoint", type=str, default="checkpoints/vae_final.pt",
                        help="Path to VAE checkpoint")
    parser.add_argument("--gan_checkpoint", type=str, default="checkpoints/dcgan_final.pt",
                        help="Path to DCGAN checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation",
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=64,
                        help="Number of samples to generate")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="VAE latent dimension")
    parser.add_argument("--noise_dim", type=int, default=100,
                        help="GAN noise dimension")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to dataset for real samples")
    
    return parser.parse_args()


def load_vae(checkpoint_path, latent_dim, device):
    """Load trained VAE model."""
    model = VAE(latent_dim=latent_dim).to(device)
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded VAE from {checkpoint_path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    else:
        print(f"Warning: VAE checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model.")
    
    model.eval()
    return model


def load_generator(checkpoint_path, noise_dim, device):
    """Load trained Generator model."""
    model = Generator(noise_dim=noise_dim).to(device)
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['generator_state_dict'])
        print(f"Loaded Generator from {checkpoint_path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
    else:
        print(f"Warning: DCGAN checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model.")
    
    model.eval()
    return model


def generate_samples(model, num_samples, device, model_type='vae'):
    """Generate samples from a model."""
    with torch.no_grad():
        if model_type == 'vae':
            samples = model.sample(num_samples, device)
        else:  # gan
            samples = model.generate(num_samples, device)
    return samples


def get_real_samples(dataloader, num_samples):
    """Get real samples from dataloader."""
    images_list = []
    count = 0
    
    for images, _ in dataloader:
        images_list.append(images)
        count += images.size(0)
        if count >= num_samples:
            break
    
    all_images = torch.cat(images_list, dim=0)
    return all_images[:num_samples]


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("\n" + "=" * 50)
    print("Loading models...")
    print("=" * 50)
    
    vae = load_vae(args.vae_checkpoint, args.latent_dim, device)
    generator = load_generator(args.gan_checkpoint, args.noise_dim, device)
    
    # Generate samples
    print("\n" + "=" * 50)
    print("Generating samples...")
    print("=" * 50)
    
    vae_samples = generate_samples(vae, args.num_samples, device, 'vae')
    gan_samples = generate_samples(generator, args.num_samples, device, 'gan')
    
    print(f"VAE samples shape: {vae_samples.shape}")
    print(f"GAN samples shape: {gan_samples.shape}")
    
    # Save individual model samples
    save_generated_images(
        vae_samples,
        output_dir / "vae_samples.png",
        nrow=8,
        title="VAE Generated Samples"
    )
    
    save_generated_images(
        gan_samples,
        output_dir / "gan_samples.png",
        nrow=8,
        title="DCGAN Generated Samples"
    )
    
    # Load real samples if available
    real_samples = None
    try:
        train_loader, _ = get_dataloader(
            data_dir=args.data_dir,
            batch_size=args.num_samples,
            num_workers=0
        )
        real_samples = get_real_samples(train_loader, args.num_samples)
        
        save_generated_images(
            real_samples,
            output_dir / "real_samples.png",
            nrow=8,
            title="Real Samples"
        )
        print(f"Real samples shape: {real_samples.shape}")
    except Exception as e:
        print(f"Could not load real samples: {e}")
    
    # Create comparison figure
    print("\nCreating comparison figure...")
    
    num_show = min(16, args.num_samples)
    
    fig, axes = plt.subplots(3 if real_samples is not None else 2, 1, figsize=(15, 12))
    
    from torchvision.utils import make_grid
    
    def plot_grid(ax, images, title):
        if images.min() < 0:
            images = denormalize(images)
        grid = make_grid(images[:num_show].cpu(), nrow=8, padding=2)
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    idx = 0
    if real_samples is not None:
        plot_grid(axes[idx], real_samples, "Gerçek Görüntüler (Real Images)")
        idx += 1
    
    plot_grid(axes[idx], vae_samples, "VAE Üretimi (VAE Generated)")
    idx += 1
    plot_grid(axes[idx], gan_samples, "DCGAN Üretimi (DCGAN Generated)")
    
    fig.suptitle("Model Karşılaştırması (Model Comparison)", fontsize=18, fontweight='bold', y=0.98)
    fig.tight_layout()
    
    comparison_path = output_dir / "model_comparison.png"
    fig.savefig(comparison_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved comparison: {comparison_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"VAE samples saved to: {output_dir / 'vae_samples.png'}")
    print(f"GAN samples saved to: {output_dir / 'gan_samples.png'}")
    print(f"Comparison saved to: {comparison_path}")
    
    print("\n📊 Karşılaştırma Kriterleri (Comparison Criteria):")
    print("-" * 50)
    print("1. Görsel Kalite (Visual Quality):")
    print("   - VAE: Daha bulanık ama tutarlı görüntüler")
    print("   - GAN: Daha keskin ama bazen tutarsız görüntüler")
    print()
    print("2. Çeşitlilik (Diversity):")
    print("   - VAE: Latent space'ten örnekleme sayesinde iyi çeşitlilik")
    print("   - GAN: Mode collapse riski, sınırlı çeşitlilik olabilir")
    print()
    print("3. Eğitim Kararlılığı (Training Stability):")
    print("   - VAE: Kararlı eğitim, loss sürekli azalır")
    print("   - GAN: Dengesiz eğitim, G ve D arasında denge gerekir")
    

if __name__ == "__main__":
    main()

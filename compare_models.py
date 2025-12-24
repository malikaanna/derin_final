"""
Model Comparison Script

This script provides comprehensive comparison between VAE and DCGAN models
including quantitative metrics and qualitative analysis.

Usage:
    python compare_models.py --vae_checkpoint checkpoints/vae_final.pt --gan_checkpoint checkpoints/dcgan_final.pt
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.vae import VAE
from models.dcgan import Generator
from utils.dataloader import get_dataloader, denormalize
from utils.visualization import save_generated_images


def parse_args():
    parser = argparse.ArgumentParser(description="Compare VAE and DCGAN models")
    
    parser.add_argument("--vae_checkpoint", type=str, default="checkpoints/vae_final.pt",
                        help="Path to VAE checkpoint")
    parser.add_argument("--gan_checkpoint", type=str, default="checkpoints/dcgan_final.pt",
                        help="Path to DCGAN checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/comparison",
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples for analysis")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="VAE latent dimension")
    parser.add_argument("--noise_dim", type=int, default=100,
                        help="GAN noise dimension")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to dataset")
    
    return parser.parse_args()


def load_model(checkpoint_path, model_class, device, **kwargs):
    """Load a model from checkpoint."""
    model = model_class(**kwargs).to(device)
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'generator_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['generator_state_dict'])
        return model, checkpoint
    
    return model, None


def compute_pixel_statistics(samples):
    """Compute pixel-level statistics of generated samples."""
    # Convert to numpy and denormalize
    if samples.min() < 0:
        samples = denormalize(samples)
    
    samples_np = samples.cpu().numpy()
    
    return {
        'mean': float(np.mean(samples_np)),
        'std': float(np.std(samples_np)),
        'min': float(np.min(samples_np)),
        'max': float(np.max(samples_np)),
    }


def compute_diversity_score(samples, method='pairwise_distance'):
    """
    Compute diversity score of generated samples.
    Higher score = more diverse samples.
    """
    # Flatten samples
    samples_flat = samples.view(samples.size(0), -1).cpu().numpy()
    
    if method == 'pairwise_distance':
        # Compute average pairwise distance
        from scipy.spatial.distance import pdist
        distances = pdist(samples_flat, metric='euclidean')
        return float(np.mean(distances))
    
    elif method == 'variance':
        # Compute variance across samples
        return float(np.mean(np.var(samples_flat, axis=0)))
    
    return 0.0


def measure_generation_time(model, num_samples, device, model_type, num_runs=10):
    """Measure average generation time."""
    import time
    
    times = []
    
    for _ in range(num_runs):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        
        with torch.no_grad():
            if model_type == 'vae':
                _ = model.sample(num_samples, device)
            else:
                _ = model.generate(num_samples, device)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end = time.time()
        
        times.append(end - start)
    
    return {
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'samples_per_second': num_samples / np.mean(times)
    }


def create_comparison_report(vae_metrics, gan_metrics, output_path):
    """Create a text report comparing the models."""
    report = []
    report.append("=" * 60)
    report.append("GAN vs VAE KARŞILAŞTIRMA RAPORU")
    report.append("Comparative Analysis Report")
    report.append("=" * 60)
    report.append(f"\nTarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n" + "-" * 60)
    report.append("1. PİKSEL İSTATİSTİKLERİ (Pixel Statistics)")
    report.append("-" * 60)
    report.append(f"\nVAE:")
    report.append(f"  Mean: {vae_metrics['pixel_stats']['mean']:.4f}")
    report.append(f"  Std:  {vae_metrics['pixel_stats']['std']:.4f}")
    report.append(f"\nDCGAN:")
    report.append(f"  Mean: {gan_metrics['pixel_stats']['mean']:.4f}")
    report.append(f"  Std:  {gan_metrics['pixel_stats']['std']:.4f}")
    
    report.append("\n" + "-" * 60)
    report.append("2. ÇEŞİTLİLİK SKORU (Diversity Score)")
    report.append("-" * 60)
    report.append(f"\nVAE Diversity:   {vae_metrics['diversity']:.4f}")
    report.append(f"DCGAN Diversity: {gan_metrics['diversity']:.4f}")
    
    if vae_metrics['diversity'] > gan_metrics['diversity']:
        report.append("\n→ VAE daha çeşitli görüntüler üretiyor")
    else:
        report.append("\n→ DCGAN daha çeşitli görüntüler üretiyor")
    
    report.append("\n" + "-" * 60)
    report.append("3. ÜRETİM HIZİ (Generation Speed)")
    report.append("-" * 60)
    report.append(f"\nVAE:")
    report.append(f"  Ortalama süre: {vae_metrics['timing']['mean_time']*1000:.2f} ms")
    report.append(f"  Görüntü/saniye: {vae_metrics['timing']['samples_per_second']:.1f}")
    report.append(f"\nDCGAN:")
    report.append(f"  Ortalama süre: {gan_metrics['timing']['mean_time']*1000:.2f} ms")
    report.append(f"  Görüntü/saniye: {gan_metrics['timing']['samples_per_second']:.1f}")
    
    report.append("\n" + "-" * 60)
    report.append("4. MODEL PARAMETRELERİ (Model Parameters)")
    report.append("-" * 60)
    report.append(f"\nVAE Parameters:   {vae_metrics.get('num_params', 'N/A'):,}")
    report.append(f"DCGAN Parameters: {gan_metrics.get('num_params', 'N/A'):,}")
    
    report.append("\n" + "-" * 60)
    report.append("5. GENEL DEĞERLENDİRME (Overall Assessment)")
    report.append("-" * 60)
    report.append("""
VAE Güçlü Yönleri:
  ✓ Kararlı eğitim süreci
  ✓ Latent space üzerinde interpolasyon yapılabilir
  ✓ Reconstruction + Generation yapabilir
  ✓ Mode collapse problemi yok

VAE Zayıf Yönleri:
  ✗ Üretilen görüntüler bulanık olabilir
  ✗ MSE loss detayları kaybedebilir

DCGAN Güçlü Yönleri:
  ✓ Keskin ve detaylı görüntüler
  ✓ Gerçekçi dokular üretebilir
  ✓ Yüksek çözünürlüklü üretim potansiyeli

DCGAN Zayıf Yönleri:
  ✗ Eğitim dengesizlikleri (mode collapse)
  ✗ Hiperparametre hassasiyeti
  ✗ Değerlendirme zorluğu
""")
    
    report.append("=" * 60)
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    return '\n'.join(report)


def create_visual_comparison(vae_samples, gan_samples, real_samples, output_path):
    """Create a side-by-side visual comparison."""
    fig, axes = plt.subplots(1, 3 if real_samples is not None else 2, figsize=(18, 6))
    
    from torchvision.utils import make_grid
    
    def show_grid(ax, images, title):
        if images.min() < 0:
            images = denormalize(images)
        grid = make_grid(images[:16].cpu(), nrow=4, padding=2)
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    idx = 0
    if real_samples is not None:
        show_grid(axes[idx], real_samples, "Gerçek (Real)")
        idx += 1
    
    show_grid(axes[idx], vae_samples, "VAE")
    show_grid(axes[idx + 1], gan_samples, "DCGAN")
    
    plt.suptitle("Görsel Karşılaştırma (Visual Comparison)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def main():
    args = parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("\nLoading models...")
    vae, vae_ckpt = load_model(args.vae_checkpoint, VAE, device, latent_dim=args.latent_dim)
    generator, gan_ckpt = load_model(args.gan_checkpoint, Generator, device, noise_dim=args.noise_dim)
    
    vae.eval()
    generator.eval()
    
    # Count parameters
    vae_params = sum(p.numel() for p in vae.parameters())
    gan_params = sum(p.numel() for p in generator.parameters())
    
    # Generate samples
    print("\nGenerating samples...")
    with torch.no_grad():
        vae_samples = vae.sample(args.num_samples, device)
        gan_samples = generator.generate(args.num_samples, device)
    
    # Load real samples
    real_samples = None
    try:
        train_loader, _ = get_dataloader(
            data_dir=args.data_dir,
            batch_size=args.num_samples,
            num_workers=0
        )
        for images, _ in train_loader:
            real_samples = images
            break
    except:
        print("Could not load real samples")
    
    # Compute metrics
    print("\nComputing metrics...")
    
    vae_metrics = {
        'pixel_stats': compute_pixel_statistics(vae_samples),
        'diversity': compute_diversity_score(vae_samples),
        'timing': measure_generation_time(vae, 64, device, 'vae'),
        'num_params': vae_params
    }
    
    gan_metrics = {
        'pixel_stats': compute_pixel_statistics(gan_samples),
        'diversity': compute_diversity_score(gan_samples),
        'timing': measure_generation_time(generator, 64, device, 'gan'),
        'num_params': gan_params
    }
    
    # Save metrics as JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({'vae': vae_metrics, 'gan': gan_metrics}, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Create report
    report_path = output_dir / "comparison_report.txt"
    report = create_comparison_report(vae_metrics, gan_metrics, report_path)
    print(f"\nSaved report to {report_path}")
    print("\n" + report)
    
    # Create visual comparison
    visual_path = output_dir / "visual_comparison.png"
    create_visual_comparison(vae_samples, gan_samples, real_samples, visual_path)
    print(f"\nSaved visual comparison to {visual_path}")
    
    # Save individual samples
    save_generated_images(vae_samples[:64], output_dir / "vae_samples.png", nrow=8)
    save_generated_images(gan_samples[:64], output_dir / "gan_samples.png", nrow=8)
    
    print("\n✅ Comparison completed!")


if __name__ == "__main__":
    main()

"""
Variational Autoencoder (VAE) for Fruit Freshness Image Generation

This module implements a VAE architecture for generating fruit images.
The model uses convolutional layers for encoding and transposed convolutions for decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Variational Autoencoder for 64x64 RGB images.
    
    Architecture:
        Encoder: Conv2d layers → Flatten → FC → (mu, logvar)
        Decoder: FC → Reshape → ConvTranspose2d layers → Image
    
    Args:
        latent_dim (int): Dimension of the latent space. Default: 128
        image_channels (int): Number of image channels. Default: 3 (RGB)
    """
    
    def __init__(self, latent_dim: int = 128, image_channels: int = 3):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        
        # Encoder: 3x64x64 → 256x4x4 → latent_dim
        self.encoder = nn.Sequential(
            # Input: 3x64x64 → 32x32x32
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32x32 → 64x16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x16x16 → 128x8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x8x8 → 256x4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Flatten size: 256 * 4 * 4 = 4096
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder: latent_dim → 256x4x4 → 3x64x64
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            # 256x4x4 → 128x8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 128x8x8 → 64x16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64x16x16 → 32x32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 32x32x32 → 3x64x64
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Output: [-1, 1]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input image to latent space parameters.
        
        Args:
            x: Input tensor of shape (batch, channels, 64, 64)
            
        Returns:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon
        
        Args:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to image.
        
        Args:
            z: Latent vector of shape (batch, latent_dim)
            
        Returns:
            Reconstructed image of shape (batch, channels, 64, 64)
        """
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.decoder(h)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input tensor of shape (batch, channels, 64, 64)
            
        Returns:
            recon_x: Reconstructed image
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate new images by sampling from the latent space.
        
        Args:
            num_samples: Number of images to generate
            device: Device to generate on
            
        Returns:
            Generated images of shape (num_samples, channels, 64, 64)
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, 
             mu: torch.Tensor, logvar: torch.Tensor,
             kl_weight: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss = Reconstruction Loss + KL Divergence.
    
    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Mean of the latent distribution
        logvar: Log variance of the latent distribution
        kl_weight: Weight for KL divergence term (beta-VAE)
        
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss (MSE)
        kl_loss: KL divergence loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    
    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing VAE on {device}")
    
    model = VAE(latent_dim=128).to(device)
    
    # Test forward pass
    x = torch.randn(4, 3, 64, 64, device=device)
    recon_x, mu, logvar = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {recon_x.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Test loss
    loss, recon_loss, kl_loss = vae_loss(recon_x, x, mu, logvar)
    print(f"Total loss: {loss.item():.4f}")
    print(f"Recon loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    
    # Test sampling
    samples = model.sample(8, device)
    print(f"Sampled images shape: {samples.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\n✅ VAE test passed!")

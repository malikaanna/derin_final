"""
Deep Convolutional GAN (DCGAN) for Fruit Freshness Image Generation

This module implements the DCGAN architecture with Generator and Discriminator
networks following the guidelines from the original DCGAN paper.
"""

import torch
import torch.nn as nn


def weights_init(m):
    """
    Custom weight initialization for DCGAN.
    All weights are initialized from a Normal distribution with mean=0, std=0.02.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    DCGAN Generator network.
    
    Takes a latent vector (noise) and generates a 64x64 RGB image.
    Uses transposed convolutions for upsampling.
    
    Args:
        noise_dim (int): Dimension of the input noise vector. Default: 100
        ngf (int): Number of generator feature maps. Default: 64
        image_channels (int): Number of output image channels. Default: 3
    """
    
    def __init__(self, noise_dim: int = 100, ngf: int = 64, image_channels: int = 3):
        super(Generator, self).__init__()
        
        self.noise_dim = noise_dim
        self.ngf = ngf
        
        self.main = nn.Sequential(
            # Input: noise_dim x 1 x 1 → ngf*8 x 4 x 4
            nn.ConvTranspose2d(noise_dim, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # ngf*8 x 4 x 4 → ngf*4 x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # ngf*4 x 8 x 8 → ngf*2 x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # ngf*2 x 16 x 16 → ngf x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # ngf x 32 x 32 → image_channels x 64 x 64
            nn.ConvTranspose2d(ngf, image_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output: [-1, 1]
        )
        
        # Apply weight initialization
        self.apply(weights_init)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: Noise vector of shape (batch, noise_dim, 1, 1)
            
        Returns:
            Generated image of shape (batch, 3, 64, 64)
        """
        return self.main(z)
    
    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate random samples.
        
        Args:
            num_samples: Number of images to generate
            device: Device to generate on
            
        Returns:
            Generated images of shape (num_samples, 3, 64, 64)
        """
        z = torch.randn(num_samples, self.noise_dim, 1, 1, device=device)
        return self.forward(z)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator network.
    
    Takes a 64x64 RGB image and outputs a probability of it being real.
    Uses strided convolutions for downsampling.
    
    Args:
        ndf (int): Number of discriminator feature maps. Default: 64
        image_channels (int): Number of input image channels. Default: 3
    """
    
    def __init__(self, ndf: int = 64, image_channels: int = 3):
        super(Discriminator, self).__init__()
        
        self.ndf = ndf
        
        self.main = nn.Sequential(
            # Input: image_channels x 64 x 64 → ndf x 32 x 32
            nn.Conv2d(image_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # ndf x 32 x 32 → ndf*2 x 16 x 16
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # ndf*2 x 16 x 16 → ndf*4 x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # ndf*4 x 8 x 8 → ndf*8 x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # ndf*8 x 4 x 4 → 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()  # Output: probability [0, 1]
        )
        
        # Apply weight initialization
        self.apply(weights_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image of shape (batch, 3, 64, 64)
            
        Returns:
            Probability of being real, shape (batch, 1, 1, 1)
        """
        return self.main(x)


class DCGAN:
    """
    DCGAN wrapper class combining Generator and Discriminator.
    
    This is a convenience class for training and inference.
    """
    
    def __init__(
        self,
        noise_dim: int = 100,
        ngf: int = 64,
        ndf: int = 64,
        image_channels: int = 3,
        device: torch.device = None
    ):
        self.noise_dim = noise_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.generator = Generator(noise_dim, ngf, image_channels).to(self.device)
        self.discriminator = Discriminator(ndf, image_channels).to(self.device)
    
    def generate(self, num_samples: int) -> torch.Tensor:
        """Generate random samples."""
        self.generator.eval()
        with torch.no_grad():
            return self.generator.generate(num_samples, self.device)
    
    def save(self, path: str):
        """Save both models."""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load both models."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])


if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing DCGAN on {device}")
    
    # Test Generator
    generator = Generator(noise_dim=100).to(device)
    z = torch.randn(4, 100, 1, 1, device=device)
    fake_images = generator(z)
    print(f"Generator input: {z.shape}")
    print(f"Generator output: {fake_images.shape}")
    
    # Test Discriminator
    discriminator = Discriminator().to(device)
    output = discriminator(fake_images)
    print(f"Discriminator input: {fake_images.shape}")
    print(f"Discriminator output: {output.shape}")
    
    # Test generate method
    samples = generator.generate(8, device)
    print(f"Generated samples: {samples.shape}")
    
    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    print(f"Total parameters: {g_params + d_params:,}")
    
    # Test DCGAN wrapper
    dcgan = DCGAN(device=device)
    samples = dcgan.generate(4)
    print(f"DCGAN generated: {samples.shape}")
    
    print("\n✅ DCGAN test passed!")

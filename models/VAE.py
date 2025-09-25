import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


class ResNetBlock(nn.Module):
    """ResNet-style block for VAE"""
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        
        self.norm1 = nn.GroupNorm(ModelConfig.group_size, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(ModelConfig.group_size, out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip_connection(x)


class AttentionBlock(nn.Module):
    """Self-attention block for VAE"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(ModelConfig.group_size, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # (b, hw, c)
        k = k.reshape(b, c, h * w)  # (b, c, hw)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)  # (b, hw, c)
        
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(attn, v)  # (b, hw, c)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        out = self.proj_out(out)
        
        return x + out


class DownsampleBlock(nn.Module):
    """Downsampling block that reduces spatial resolution"""
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResNetBlock(in_channels, out_channels),
            ResNetBlock(out_channels, out_channels)
        ])
        self.attention = AttentionBlock(out_channels) if use_attention else None
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        for res_block in self.res_blocks:
            x = res_block(x)
        if self.attention is not None:
            x = self.attention(x)
        return self.downsample(x)


class UpsampleBlock(nn.Module):
    """Upsampling block that increases spatial resolution"""
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.res_blocks = nn.ModuleList([
            ResNetBlock(in_channels, out_channels),
            ResNetBlock(out_channels, out_channels)
        ])
        self.attention = AttentionBlock(out_channels) if use_attention else None

    def forward(self, x):
        x = self.upsample(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        if self.attention is not None:
            x = self.attention(x)
        return x


class VAEEncoder(nn.Module):
    """VAE Encoder that reduces image resolution and encodes to latent space"""
    def __init__(self, in_channels=3, latent_dim=4, base_channels=128):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling blocks - each reduces resolution by 2x
        # 64x64 -> 32x32 -> 16x16 -> 8x8
        self.down_blocks = nn.ModuleList([
            DownsampleBlock(base_channels, base_channels * 2),  # 128 -> 256
            DownsampleBlock(base_channels * 2, base_channels * 4),  # 256 -> 512
            DownsampleBlock(base_channels * 4, base_channels * 4, use_attention=True),  # 512 -> 512
        ])
        
        # Middle blocks
        self.mid_block1 = ResNetBlock(base_channels * 4)
        self.mid_attn = AttentionBlock(base_channels * 4)
        self.mid_block2 = ResNetBlock(base_channels * 4)
        
        # Convert to latent distribution parameters
        self.norm_out = nn.GroupNorm(ModelConfig.group_size, base_channels * 4)
        self.conv_out = nn.Conv2d(base_channels * 4, latent_dim * 2, 3, padding=1)  # *2 for mean and logvar

    def forward(self, x):
        # Initial conv
        h = self.conv_in(x)
        
        # Downsample
        for down_block in self.down_blocks:
            h = down_block(h)
        
        # Middle
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        # To latent
        h = self.conv_out(F.silu(self.norm_out(h)))
        
        # Split into mean and logvar
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar


class VAEDecoder(nn.Module):
    """VAE Decoder that reconstructs images from latent space"""
    def __init__(self, out_channels=3, latent_dim=4, base_channels=128):
        super().__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        
        # Convert from latent
        self.conv_in = nn.Conv2d(latent_dim, base_channels * 4, 3, padding=1)
        
        # Middle blocks
        self.mid_block1 = ResNetBlock(base_channels * 4)
        self.mid_attn = AttentionBlock(base_channels * 4)
        self.mid_block2 = ResNetBlock(base_channels * 4)
        
        # Upsampling blocks - each increases resolution by 2x
        # 8x8 -> 16x16 -> 32x32 -> 64x64
        self.up_blocks = nn.ModuleList([
            UpsampleBlock(base_channels * 4, base_channels * 4, use_attention=True),  # 512 -> 512
            UpsampleBlock(base_channels * 4, base_channels * 2),  # 512 -> 256
            UpsampleBlock(base_channels * 2, base_channels),  # 256 -> 128
        ])
        
        # Final output
        self.norm_out = nn.GroupNorm(ModelConfig.group_size, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, z):
        # From latent
        h = self.conv_in(z)
        
        # Middle
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        # Upsample
        for up_block in self.up_blocks:
            h = up_block(h)
        
        # Final conv
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


class VAE(nn.Module):
    """Complete VAE model with encoder and decoder"""
    def __init__(self, in_channels=3, latent_dim=4, base_channels=128, kl_weight=1e-6):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim, base_channels)
        self.decoder = VAEDecoder(in_channels, latent_dim, base_channels)
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

    def encode(self, x):
        """Encode images to latent space"""
        mean, logvar = self.encoder(x)
        return mean, logvar

    def decode(self, z):
        """Decode latent vectors to images"""
        return self.decoder(z)

    def reparameterize(self, mean, logvar):
        """Reparameterization trick for VAE"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean

    def forward(self, x):
        """Full forward pass: encode -> reparameterize -> decode"""
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar, z

    def loss_function(self, recon, x, mean, logvar):
        """VAE loss: reconstruction + KL divergence"""
        recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)
        
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / x.size(0)
        
        return recon_loss + self.kl_weight * kl_loss, recon_loss, kl_loss

    @torch.no_grad()
    def encode_to_latent(self, x):
        """Encode images to latent space (for inference)"""
        mean, logvar = self.encode(x)
        return mean

    @torch.no_grad()
    def decode_from_latent(self, z):
        """Decode from latent space (for inference)"""
        return self.decode(z)

    def get_latent_size(self, input_size):
        """Calculate latent space size given input image size"""
        return input_size // 8
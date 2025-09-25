import math
import torch
import torch.nn as nn
from config import ModelConfig


def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999)

class LatentDDPM(nn.Module):
    """DDPM modified to work with VAE latent space"""
    def __init__(self, model, vae=None, timesteps=ModelConfig.timesteps, beta_schedule=ModelConfig.beta_schedule, device=ModelConfig.device):
        super().__init__()
        self.model = model
        self.vae = vae
        self.T = timesteps
        self.latent_scaling_factor = 1.0
        
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = torch.linspace(1e-4, 0.02, timesteps)
        self.register_buffer("betas", betas.float().to(device))
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def encode_to_latent(self, x):
        """Encode images to latent space using VAE"""
        if self.vae is None:
            return x
        with torch.no_grad():
            latents = self.vae.encode_to_latent(x)
            return latents * ModelConfig.scale_latent

    def decode_from_latent(self, z):
        """Decode latents to images using VAE"""
        if self.vae is None:
            return z
        with torch.no_grad():
            z = z / ModelConfig.scale_latent
            return self.vae.decode_from_latent(z)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x0 + sqrt_om * noise, noise

    def p_mean_variance(self, x_t, t, context, guidance_scale=None, uncond_context=None):
        if guidance_scale is None or uncond_context is None:
            eps = self.model(x_t, t, context)
        else:
            eps_cond = self.model(x_t, t, context)
            eps_un = self.model(x_t, t, uncond_context)
            eps = eps_un + guidance_scale * (eps_cond - eps_un)

        beta_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        ac_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        one_minus_ac_t = (1 - ac_t)
        x0_pred = (x_t - eps * self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)) / (self.sqrt_alphas_cumprod[t].view(-1,1,1,1) + 1e-8)
        mean = (1.0 / torch.sqrt(self.alphas[t]).view(-1,1,1,1)) * (x_t - beta_t / torch.sqrt(one_minus_ac_t + 1e-8) * eps)
        var = self.posterior_variance[t].view(-1, 1, 1, 1)
        return mean, var.clamp(min=1e-20), x0_pred, eps

    def p_sample(self, x_t, t, context, guidance_scale=None, uncond_context=None):
        mean, var, _, _ = self.p_mean_variance(x_t, t, context, guidance_scale, uncond_context)
        if (t > 0).any():
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, shape, text_context, device, guidance_scale=ModelConfig.guidance_scale, text_encoder=None, return_latents=False):
        """Sample images, optionally returning intermediate latents"""
        B = shape[0]
        
        # If using VAE, adjust shape for latent space
        if self.vae is not None:
            # Calculate latent dimensions
            latent_h = shape[2] // 8  # VAE downsamples by factor of 8
            latent_w = shape[3] // 8
            latent_shape = (B, ModelConfig.vae_latent_dim, latent_h, latent_w)
            x = torch.randn(latent_shape, device=device)
        else:
            x = torch.randn(shape, device=device)
        
        un_ctx = text_encoder.unconditional(B, device=device) if (guidance_scale is not None and text_encoder is not None) else None
        
        # Denoising loop
        for i in reversed(range(self.T)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, text_context, guidance_scale, un_ctx)
        
        if return_latents:
            return x
        
        # Decode to image space if using VAE
        if self.vae is not None:
            x = self.decode_from_latent(x)
        
        return x

    @torch.no_grad()
    def sample_progressive(self, shape, text_context, device, guidance_scale=ModelConfig.guidance_scale, text_encoder=None, save_every=100):
        """Sample with progressive outputs for visualization"""
        B = shape[0]
        
        if self.vae is not None:
            latent_h = shape[2] // 8
            latent_w = shape[3] // 8
            latent_shape = (B, ModelConfig.vae_latent_dim, latent_h, latent_w)
            x = torch.randn(latent_shape, device=device)
        else:
            x = torch.randn(shape, device=device)
        
        un_ctx = text_encoder.unconditional(B, device=device) if (guidance_scale is not None and text_encoder is not None) else None
        
        intermediates = []
        for i in reversed(range(self.T)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, text_context, guidance_scale, un_ctx)
            
            # Save intermediate results
            if i % save_every == 0 or i == 0:
                if self.vae is not None:
                    img = self.decode_from_latent(x)
                else:
                    img = x
                intermediates.append((i, img.clone()))
        
        return x, intermediates

    def loss(self, x0, t, context):
        """Compute diffusion loss (works in latent or pixel space)"""
        x_t, noise = self.q_sample(x0, t)
        eps_pred = self.model(x_t, t, context)
        return torch.nn.functional.mse_loss(eps_pred, noise)



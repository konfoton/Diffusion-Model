from training import train_vae_only
from training import get_device
from models.VAE import VAE
from config import ModelConfig, TrainConfig
import torch
import os


checkpoint_vae = None

def main():
    os.makedirs("checkpoints", exist_ok=True)
    device = get_device()
    vae = VAE(
            in_channels=3,
            latent_dim=ModelConfig.vae_latent_dim,
            base_channels=ModelConfig.vae_base_channels,
            kl_weight=ModelConfig.vae_kl_weight
        ).to(device)
    
    if checkpoint_vae:
        vae.load_state_dict(torch.load(checkpoint_vae, map_location=device)["model"])
        print(f"Loaded VAE checkpoint from {checkpoint_vae}")

    train_vae_only(vae, dl=None, device=device, epochs=TrainConfig.vae_epochs)

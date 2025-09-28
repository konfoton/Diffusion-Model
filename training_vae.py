from training.training import train_vae_only
from training.training import get_device
from models.VAE import VAE
from config import ModelConfig, TrainConfig
import torch
import os
from data.coco_dataloader import get_coco_train_dataloader

checkpoint_vae = None

def main():
    os.makedirs("checkpoints", exist_ok=True)
    device = get_device()
    dl = get_coco_train_dataloader(batch_size=TrainConfig.vae_batch_size, image_size=ModelConfig.image_size)
    vae = VAE(
            in_channels=3,
            latent_dim=ModelConfig.vae_latent_dim,
            base_channels=ModelConfig.vae_base_channels,
            kl_weight=ModelConfig.vae_kl_weight
        ).to(device)
    
    if checkpoint_vae:
        vae.load_state_dict(torch.load(checkpoint_vae, map_location=device)["model"])
        print(f"Loaded VAE checkpoint from {checkpoint_vae}")
    vae.kl_weight = 0.6
    vae.train()

    train_vae_only(vae, dataloader=dl, device=device, epochs=TrainConfig.vae_epochs, lr=TrainConfig.vae_lr)


if __name__ == "__main__":
    main()

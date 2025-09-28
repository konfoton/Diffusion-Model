from training.training import train_vae_only
from training.training import train_diffusion_only
from training.training import get_device
from models.VAE import VAE
from config import ModelConfig, TrainConfig
from models.CLIP import CLIPTextEncoder
from models.UNET import ConditionalUNet
from models.LatentDDPM import LatentDDPM
import torch
import os
from data.coco_dataloader import get_coco_train_dataloader

checkpoint_vae = None


def main_first():
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

def main_second():
    checkpoint_vae = None
    checkpoint_unet = None
    device = get_device()
    dl = get_coco_train_dataloader(batch_size=TrainConfig.unet_batch_size, image_size=ModelConfig.image_size)
    vae = VAE(
            in_channels=3,
            latent_dim=ModelConfig.vae_latent_dim,
            base_channels=ModelConfig.vae_base_channels,
            kl_weight=ModelConfig.vae_kl_weight
        ).to(device)
    

    if checkpoint_vae:
        vae.load_state_dict(torch.load(checkpoint_vae, map_location=device)['model'])
        print(f"Loaded VAE checkpoint from {checkpoint_vae}")

    vae.eval()

    text_enc = CLIPTextEncoder().to(device)
    text_enc.text_model.eval()
    for p in text_enc.parameters(): 
        p.requires_grad = False


    model = ConditionalUNet(
        img_channels=ModelConfig.vae_latent_dim,
        base_ch=ModelConfig.unet_base_channels,
        ch_mults=ModelConfig.chan_mults,
        ctx_dim=ModelConfig.embed_dim
    ).to(device)

    if checkpoint_unet:
        model.load_state_dict(torch.load(checkpoint_unet, map_location=device)["model"])
        print(f"Loaded UNet checkpoint from {checkpoint_unet}")

    ddpm = LatentDDPM(model, timesteps=ModelConfig.timesteps, device=device)


    train_diffusion_only(model, vae, ddpm, text_enc, dl, device, epochs=TrainConfig.unet_epochs, lr=TrainConfig.unet_lr)



if __name__ == "__main__":
    main_first()
    main_second()

import os, random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import wandb
from models.UNET import ConditionalUNet
from models.CLIP import CLIPTextEncoder
from models.VAE import VAE
from config import ModelConfig, TrainConfig


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_vae_only(vae, dataloader, device, epochs=50, lr=1e-4):
    print("Training VAE...")
    vae_optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)
    vae.train()

    wandb.init(project="vae-training")

    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_loss_last_100 = 0
        total_recon_loss_last_100 = 0
        total_kl_loss_last_100 = 0
        counter = 0

        pbar = tqdm(dataloader, desc=f"VAE Epoch {epoch+1}/{epochs}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            
            vae_optimizer.zero_grad()
            recon, mean, logvar, _ = vae(imgs)
            loss, recon_loss, kl_loss = vae.loss_function(recon, imgs, mean, logvar)
            
            loss.backward()
            vae_optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

            total_loss_last_100 += loss.item()
            total_recon_loss_last_100 += recon_loss.item()
            total_kl_loss_last_100 += kl_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })
            counter += 1
            if counter % 100 == 0:
                wandb.log({
                    "step": counter // 100,
                    "loss": total_loss_last_100 / 100,
                    "recon": total_recon_loss_last_100 / 100,
                    "kl": total_kl_loss_last_100 / 100
                })
                total_loss_last_100 = 0
                total_recon_loss_last_100 = 0
                total_kl_loss_last_100 = 0

        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "model": vae.state_dict(),
            "epoch": epoch,
            "loss": total_loss / len(dataloader)
        }, f"checkpoints/vae_epoch_{epoch+1}.pt")
        
        print(f"VAE Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, "
              f"Recon={total_recon_loss/len(dataloader):.4f}, "
              f"KL={total_kl_loss/len(dataloader):.4f}")

def train_diffusion_only(model, vae, ddpm, text_enc, dataloader, device, epochs=100, lr=1e-5):
    print("Training Diffusion Model (VAE frozen)...")
    wandb.init(project="diffusion-training")
    diffusion_optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=(device.type in ["cuda"]))
    
    model.train()
    vae.eval()
    for epoch in range(epochs):
        total_loss = 0
        loss_last_100 = 0
        counter = 0
        pbar = tqdm(dataloader, desc=f"Diffusion Epoch {epoch+1}/{epochs}")
        
        for imgs, captions in pbar:
            imgs = imgs.to(device)
            
            with torch.no_grad():
                context = text_enc.encode(list(captions), device=device)
                latents = vae.encode_to_latent(imgs) * ModelConfig.scale_latent
            

            t = torch.randint(0, ddpm.T, (latents.size(0),), device=device, dtype=torch.long)

            diffusion_optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=device.type if device.type != "mps" else "cpu", 
                               dtype=torch.float16, enabled=(device.type in ["cuda"])):
                loss = ddpm.loss(latents, t, context)
            
            scaler.scale(loss).backward()
            scaler.step(diffusion_optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            loss_last_100 += loss.item()
            counter += 1
            if counter % 100 == 0:
                wandb.log({
                    "step": counter // 100,
                    "loss": loss_last_100 / 100
                })
                loss_last_100 = 0
        
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "loss": total_loss / len(dataloader)
        }, f"checkpoints/diffusion_epoch_{epoch+1}.pt")
        
        print(f"Diffusion Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}")

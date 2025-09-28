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
import torch.nn.utils as nn_utils

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(vae_optimizer, T_max=epochs, eta_min=lr * 0.1)
    wandb.init(project="vae-training-final-really")

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
            nn_utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
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
            if counter % 250 == 0:
                wandb.log({
                    "step": counter // 250,
                    "loss": total_loss_last_100 / 250,
                    "recon": total_recon_loss_last_100 / 250,
                    "kl": total_kl_loss_last_100 / 250
                })
                total_loss_last_100 = 0
                total_recon_loss_last_100 = 0
                total_kl_loss_last_100 = 0


        if (epoch + 1) % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "model": vae.state_dict(),
                "epoch": epoch,
                "loss": total_loss / len(dataloader)
            }, f"checkpoints/vae_epoch_{epoch+1}.pt")
        scheduler.step()
            
    wandb.finish()

def train_diffusion_only(model, vae, ddpm, text_enc, dataloader, device, epochs=100, lr=1e-5, cfg_dropout_prob=0.0):
    print("Training Diffusion Model (VAE frozen)...")
    wandb.init(project="final-diffusion-training-really-CFG")
    diffusion_optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(diffusion_optimizer, T_max=epochs, eta_min=lr * 0.1)
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

                # Build text context, with optional CFG dropout
                if cfg_dropout_prob > 0.0 and isinstance(captions, (list, tuple)):
                    cfg_captions = []
                    for caption in captions:
                        if random.random() < cfg_dropout_prob:
                            cfg_captions.append("")  # Use empty string for unconditional
                        else:
                            cfg_captions.append(caption)
                    captions = cfg_captions

                # Build text context and optional key padding mask
                if isinstance(captions, (list, tuple)) and len(captions) > 0 and isinstance(captions[0], str):
                    if counter == 0:
                        print("True")
                    context, kpm = text_enc.encode(list(captions), device=device, return_mask=True)
                else:
                    context = captions.to(device)
                    kpm = None
                latents = vae.encode_to_latent(imgs) * ModelConfig.scale_latent
            

            t = torch.randint(0, ddpm.T, (latents.size(0),), device=device, dtype=torch.long)

            diffusion_optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=device.type if device.type != "mps" else "cpu", 
                               dtype=torch.float16, enabled=(device.type in ["cuda"])):
                loss = ddpm.loss(latents, t, context, key_padding_mask=kpm)
            
            scaler.scale(loss).backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(diffusion_optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            loss_last_100 += loss.item()
            counter += 1
            if counter % 250 == 0:
                wandb.log({
                    "step": counter // 250,
                    "loss": loss_last_100 / 250
                })
                loss_last_100 = 0


        scheduler.step()
        if (epoch + 1) % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "loss": total_loss / len(dataloader)
            }, f"checkpoints/diffusion_epoch_CFG{epoch+1}.pt")
        
    wandb.finish()

    
def train_diffusion_with_cfg_loss(model, vae, ddpm, text_enc, dataloader, device, epochs=100, lr=1e-5, cfg_weight=0.5):
    """
    Alternative CFG training using explicit conditional/unconditional loss.
    cfg_weight: weight for unconditional loss component
    """
    print("Training Diffusion Model with explicit CFG loss (VAE frozen)...")
    wandb.init(project="final-diffusion-cfg-training")
    diffusion_optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(diffusion_optimizer, T_max=epochs, eta_min=lr * 0.1)
    scaler = torch.amp.GradScaler(enabled=(device.type in ["cuda"]))
    
    model.train()
    vae.eval()
    
    for epoch in range(epochs):
        total_loss = 0
        loss_last_100 = 0
        counter = 0
        pbar = tqdm(dataloader, desc=f"CFG Epoch {epoch+1}/{epochs}")
        
        for imgs, captions in pbar:
            imgs = imgs.to(device)
            with torch.no_grad():
                # Build conditional context
                if isinstance(captions, (list, tuple)) and len(captions) > 0 and isinstance(captions[0], str):
                    context, kpm = text_enc.encode(list(captions), device=device, return_mask=True)
                    # Build unconditional context (empty strings)
                    uncond_context, _ = text_enc.encode([""] * len(captions), device=device, return_mask=True)
                else:
                    context = captions.to(device)
                    uncond_context = torch.zeros_like(context)
                    kpm = None
                
                latents = vae.encode_to_latent(imgs) * ModelConfig.scale_latent

            t = torch.randint(0, ddpm.T, (latents.size(0),), device=device, dtype=torch.long)

            diffusion_optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=device.type if device.type != "mps" else "cpu", 
                               dtype=torch.float16, enabled=(device.type in ["cuda"])):
                loss = ddpm.cfg_loss(latents, t, context, uncond_context, cfg_weight=cfg_weight, key_padding_mask=kpm)
            
            scaler.scale(loss).backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(diffusion_optimizer)
            scaler.update()
            
            total_loss += loss.item()
            loss_last_100 += loss.item()
            counter += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if counter % 250 == 0:
                wandb.log({
                    "step": counter // 250,
                    "loss": loss_last_100 / 250
                })
                loss_last_100 = 0

        scheduler.step()
        if (epoch + 1) % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "loss": total_loss / len(dataloader)
            }, f"checkpoints/diffusion_cfg_epoch_{epoch+1}.pt")
        
    wandb.finish()


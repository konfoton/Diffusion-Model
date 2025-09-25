import os, random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models.UNET import ConditionalUNet
from models.CLIP import CLIPTextEncoder
from models.DDPM import DDPM
from models.VAE import VAE
from config import ModelConfig, TrainConfig

class ImageTextFolder(Dataset):
    def __init__(self, root, captions_file, image_size=64, cfg_dropout=0.1):
        self.root = root
        self.items = []
        with open(captions_file, "r") as f:
            for line in f:
                p, c = line.rstrip("\n").split("\t", 1)
                self.items.append((p, c))
        self.tfm = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        self.cfg_dropout = cfg_dropout

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        p, caption = self.items[idx]
        img = Image.open(os.path.join(self.root, p)).convert("RGB")
        x = self.tfm(img)
        if random.random() < self.cfg_dropout:
            caption = ""
        return x, caption

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_vae_only(vae, dataloader, device, epochs=50, lr=1e-4):
    """Train VAE separately before diffusion training"""
    print("Training VAE...")
    vae_optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)
    vae.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
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
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })
        
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
    """Train diffusion model only with frozen VAE (Sequential Training Phase 2)"""
    print("Phase 2: Training Diffusion Model (VAE frozen)...")
    
    diffusion_optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type in ["cuda"]))
    
    model.train()
    vae.eval()
    for epoch in range(epochs):
        total_loss = 0
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
        
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "loss": total_loss / len(dataloader)
        }, f"checkpoints/diffusion_epoch_{epoch+1}.pt")
        
        print(f"Diffusion Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}")

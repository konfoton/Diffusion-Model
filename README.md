# Diffusion-Model

Latent text-conditional diffusion model trained on COCO captions. Images are encoded with a VAE, denoised in latent space by a CLIP-conditioned U-Net under a DDPM schedule, and logged with Weights & Biases. Text conditioning uses Classifier-Free Guidance (CFG).

## Key components
- Config: `config.ModelConfig`, `config.TrainConfig`
- Training: `training.training.train_vae_only`, `training.training.train_diffusion_only`, `training.training.get_device`
- Models:
  - VAE: `models.VAE.VAE`
  - Text encoder (CLIP): `models.CLIP.CLIPTextEncoder`
  - Conditional U-Net: `models.UNET.ConditionalUNet`
  - Diffusion wrapper: `models.LatentDDPM.LatentDDPM`
- Data: `data.coco_dataloader.get_coco_train_dataloader`

## Project structure
- Entry points: `training_vae.py`, `training_diffusion.py`
- Models: `models/`
- Training: `training/training.py`
- Data: `data/coco_dataloader.py`
- Config: `config.py`
- W&B logs: `wandb/`

## Setup
```bash
pip install torch torchvision transformers pillow tqdm wandb
```
Prepare MS COCO images and captions; adjust paths in `data/coco_dataloader.py` if needed.

## Training
- Train VAE:
```bash
python training_vae.py
```
- Train diffusion (requires a VAE checkpoint, default: `checkpoints/vae_epoch_10.pt`):
```bash
python training_diffusion.py
```

Device selection (CUDA/MPS/CPU) is automatic via `training.training.get_device`.

## Configuration
Edit `config.py`:
- Model: latent_dim, channels, timesteps, U-Net widths/depths
- Guidance: guidance_scale (CFG; set to 1.0 to effectively disable guidance)
- Training: batch sizes, learning rates, epochs, image size

## Logging
Metrics are logged to Weights & Biases (offline/online). Artifacts are stored under `wandb/`.
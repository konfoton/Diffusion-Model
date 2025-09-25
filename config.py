from dataclasses import dataclass
@dataclass
class ModelConfig:
    img_channels: int = 3
    timesteps: int = 500 
    model_name: str = "openai/clip-vit-base-patch32"
    embed_dim: int = 512
    guidance_scale: float = 5.0
    base_ch: int = 32  
    chan_mults: tuple = (1, 2, 4, 8)
    img_channels: int = 3
    beta_schedule: str = "cosine"
    device: str = "gpu"
    group_size: int = 8
    vae_latent_dim: int = 4
    vae_base_channels: int = 128
    vae_kl_weight: float = 1e-2
    scale_latent: float = 1.0
    use_vae: bool = True

@dataclass 
class TrainConfig:
    image_size: int = 64
    batch_size: int = 16
    vae_lr: float = 1e-4
    uner_lr: float = 1e-4
    timesteps: int = 500  
    vae_epochs: int = 10
    unet_epoch: int = 10

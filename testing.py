import os
import torch
from torchvision.utils import save_image
from models.unet import ConditionalUNet
from models.text_encoder import CLIPTextEncoder
from diffusion.ddpm import DDPM

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

@torch.no_grad()
def main():
    device = get_device()
    image_size = int(os.environ.get("IMAGE_SIZE", "64"))
    timesteps = int(os.environ.get("TIMESTEPS", "1000"))
    guidance_scale = float(os.environ.get("GUIDANCE", "5.0"))
    prompt = os.environ.get("PROMPT", "a cute corgi playing guitar")
    ckpt = os.environ.get("CKPT", "checkpoints/unet_epoch_1.pt")
    outdir = os.environ.get("OUTDIR", "samples")
    os.makedirs(outdir, exist_ok=True)

    text_enc = CLIPTextEncoder().to(device)
    text_enc.text_model.eval()
    for p in text_enc.parameters(): p.requires_grad = False

    ctx_dim = text_enc.text_model.config.hidden_size
    model = ConditionalUNet(img_channels=3, base_ch=64, ch_mults=(1,2,4,8), ctx_dim=ctx_dim).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    ddpm = DDPM(model, timesteps=timesteps, device=device)

    context = text_enc.encode([prompt], device=device)  # (1, L, C)
    B = 4
    context = context.expand(B, -1, -1).contiguous()

    imgs = ddpm.sample(shape=(B, 3, image_size, image_size), text_context=context, device=device, guidance_scale=guidance_scale, text_encoder=text_enc)
    imgs = (imgs.clamp(-1, 1) + 1) * 0.5
    for i in range(B):
        save_image(imgs[i], os.path.join(outdir, f"sample_{i}.png"))

if __name__ == "__main__":
    main()
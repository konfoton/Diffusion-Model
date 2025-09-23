import os, random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models.unet import ConditionalUNet
from models.text_encoder import CLIPTextEncoder
from diffusion.ddpm import DDPM

class ImageTextFolder(Dataset):
    def __init__(self, root, captions_file, image_size=64, cfg_dropout=0.1):
        self.root = root
        self.items = []  # list of (path, caption)
        with open(captions_file, "r") as f:
            # Expect lines: "relative_image_path\tcaption"
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
        # classifier-free: randomly drop 10% captions
        if random.random() < self.cfg_dropout:
            caption = ""
        return x, caption

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    data_root = os.environ.get("DATA_ROOT", "./data/images")
    captions_file = os.environ.get("CAPTIONS_FILE", "./data/captions.tsv")
    image_size = int(os.environ.get("IMAGE_SIZE", "64"))
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    lr = float(os.environ.get("LR", "1e-4"))
    timesteps = int(os.environ.get("TIMESTEPS", "1000"))
    epochs = int(os.environ.get("EPOCHS", "10"))

    device = get_device()
    print(f"Using device: {device}")

    ds = ImageTextFolder(data_root, captions_file, image_size=image_size, cfg_dropout=0.1)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True if device.type != "mps" else False)

    text_enc = CLIPTextEncoder().to(device)
    text_enc.text_model.eval()  # freeze CLIP
    for p in text_enc.parameters(): p.requires_grad = False

    model = ConditionalUNet(img_channels=3, base_ch=64, ch_mults=(1,2,4,8), ctx_dim=512 if "RN50" in text_enc.text_model.config.model_type else text_enc.text_model.config.hidden_size).to(device)
    ddpm = DDPM(model, timesteps=timesteps, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type in ["cuda"]))

    global_step = 0
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, captions in pbar:
            imgs = imgs.to(device)
            with torch.no_grad():
                context = text_enc.encode(list(captions), device=device)  # (B, L, C)
            t = torch.randint(0, timesteps, (imgs.size(0),), device=device, dtype=torch.long)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type if device.type != "mps" else "cpu", dtype=torch.float16, enabled=(device.type in ["cuda"])):
                loss = ddpm.loss(imgs, t, context)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            global_step += 1
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        # save checkpoint each epoch
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({"model": model.state_dict()}, f"checkpoints/unet_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(timesteps, dim, max_period=10000):
    # timesteps: (B,)
    half = dim // 2
    exps = torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    freqs = torch.exp(-math.log(max_period) * exps)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb  # (B, dim)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim, dropout=0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.t_proj(self.act(t_emb)).unsqueeze(-1).unsqueeze(-2)
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.skip(x)

class CrossAttention2D(nn.Module):
    def __init__(self, channels, n_heads=4, context_dim=None):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=n_heads, batch_first=True
        )
        self.to_q = nn.Linear(channels, channels)
        cdim = channels if context_dim is None else context_dim
        self.to_k = nn.Linear(cdim, channels)
        self.to_v = nn.Linear(cdim, channels)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.GroupNorm(32, channels)

    def forward(self, x, context):
        # x: (B,C,H,W), context: (B,L,Cc)
        B, C, H, W = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, N, C)
        q = self.to_q(h)
        k = self.to_k(context)  # (B, L, C)
        v = self.to_v(context)
        out, _ = self.attn(q, k, v, need_weights=False)
        out = self.proj(out)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return self.norm(x + out)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim, use_attn=False, ctx_dim=None):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, t_emb_dim)
        self.res2 = ResBlock(out_ch, out_ch, t_emb_dim)
        self.attn = CrossAttention2D(out_ch, n_heads=4, context_dim=ctx_dim) if use_attn else None
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb, context=None):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        if self.attn is not None and context is not None:
            x = self.attn(x, context)
        skip = x
        x = self.down(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim, use_attn=False, ctx_dim=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res1 = ResBlock(out_ch * 2, out_ch, t_emb_dim)
        self.res2 = ResBlock(out_ch, out_ch, t_emb_dim)
        self.attn = CrossAttention2D(out_ch, n_heads=4, context_dim=ctx_dim) if use_attn else None

    def forward(self, x, skip, t_emb, context=None):
        x = self.up(x)
        # match spatial dims in case of off-by-one
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        if self.attn is not None and context is not None:
            x = self.attn(x, context)
        return x

class ConditionalUNet(nn.Module):
    def __init__(self, img_channels=3, base_ch=64, ch_mults=(1,2,4,8), ctx_dim=768, out_channels=None):
        super().__init__()
        self.img_channels = img_channels
        self.ctx_dim = ctx_dim
        out_channels = out_channels or img_channels
        self.t_emb_dim = base_ch * 4

        self.t_mlp = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
        )

        self.in_conv = nn.Conv2d(img_channels, base_ch, 3, padding=1)

        chs = [base_ch * m for m in ch_mults]
        self.downs = nn.ModuleList()
        in_ch = base_ch
        for i, ch in enumerate(chs):
            use_attn = (i >= 1)  # attention at lower resolutions
            self.downs.append(DownBlock(in_ch, ch, self.t_emb_dim, use_attn=use_attn, ctx_dim=ctx_dim))
            in_ch = ch

        self.mid_res1 = ResBlock(in_ch, in_ch, self.t_emb_dim)
        self.mid_attn = CrossAttention2D(in_ch, n_heads=8, context_dim=ctx_dim)
        self.mid_res2 = ResBlock(in_ch, in_ch, self.t_emb_dim)

        self.ups = nn.ModuleList()
        for i, ch in reversed(list(enumerate(chs))):
            use_attn = (i >= 1)
            self.ups.append(UpBlock(in_ch, ch, self.t_emb_dim, use_attn=use_attn, ctx_dim=ctx_dim))
            in_ch = ch

        self.out_norm = nn.GroupNorm(32, in_ch)
        self.out_conv = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, x, t, context):
        # context: (B, L, ctx_dim)
        t_emb = timestep_embedding(t, self.t_emb_dim)
        t_emb = self.t_mlp(t_emb)

        x = self.in_conv(x)

        skips = []
        for down in self.downs:
            x, s = down(x, t_emb, context)
            skips.append(s)

        x = self.mid_res1(x, t_emb)
        x = self.mid_attn(x, context)
        x = self.mid_res2(x, t_emb)

        for up in self.ups:
            s = skips.pop()
            x = up(x, s, t_emb, context)

        x = self.out_conv(F.silu(self.out_norm(x)))
        return x  # predicts noise ε
"""ノイズ予測ネットワーク ε_θ(x_t, t)。ToyMLP は2次元点群用、UNet は画像用。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """時刻 t (B,) を sin/cos の dim 次元ベクトルに変換する。"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ToyMLP(nn.Module):
    def __init__(self, hidden=512, n_blocks=4, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.inp = nn.Linear(2, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(n_blocks)
        ])
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2))

    def forward(self, x, t):
        temb = self.time_mlp(timestep_embedding(t, self.time_dim))
        h = self.inp(x)
        for block in self.blocks:
            h = h + block(h + temb)
        return self.out(h)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)

        def to_tokens(z):
            return z.reshape(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)

        h = F.scaled_dot_product_attention(to_tokens(q), to_tokens(k), to_tokens(v))
        h = h.transpose(2, 3).reshape(B, C, H, W)
        return x + self.proj(h)


class UNet(nn.Module):
    """64x64 入力で 64 -> 32 -> 16 (Attention) -> 8 の4段 U-Net。"""

    def __init__(self, in_ch=3, ch=64, ch_mults=(1, 2, 2, 4), attn_levels=(2,), dropout=0.1):
        super().__init__()
        time_dim = ch * 4
        self.ch = ch

        self.time_mlp = nn.Sequential(
            nn.Linear(ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_ch, ch, 3, padding=1)

        chs = [ch * m for m in ch_mults]
        self.down = nn.ModuleList()
        cur = ch
        for i, c in enumerate(chs):
            self.down.append(nn.ModuleDict({
                "res1": ResBlock(cur, c, time_dim, dropout),
                "res2": ResBlock(c, c, time_dim, dropout),
                "attn": AttnBlock(c) if i in attn_levels else nn.Identity(),
                "down": nn.Conv2d(c, c, 3, stride=2, padding=1) if i < len(chs) - 1 else nn.Identity(),
            }))
            cur = c

        self.mid1 = ResBlock(cur, cur, time_dim, dropout)
        self.mid_attn = AttnBlock(cur)
        self.mid2 = ResBlock(cur, cur, time_dim, dropout)

        self.up = nn.ModuleList()
        for i in reversed(range(len(chs))):
            c = chs[i]
            self.up.append(nn.ModuleDict({
                "res1": ResBlock(cur + c, c, time_dim, dropout),
                "res2": ResBlock(c, c, time_dim, dropout),
                "attn": AttnBlock(c) if i in attn_levels else nn.Identity(),
                "up": nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(c, c, 3, padding=1),
                ) if i > 0 else nn.Identity(),
            }))
            cur = c

        self.out = nn.Sequential(
            nn.GroupNorm(32, cur),
            nn.SiLU(),
            nn.Conv2d(cur, in_ch, 3, padding=1),
        )

    def forward(self, x, t):
        temb = self.time_mlp(timestep_embedding(t, self.ch))

        h = self.init_conv(x)
        skips = []
        for level in self.down:
            h = level["res1"](h, temb)
            h = level["res2"](h, temb)
            h = level["attn"](h)
            skips.append(h)
            h = level["down"](h)

        h = self.mid1(h, temb)
        h = self.mid_attn(h)
        h = self.mid2(h, temb)

        for level in self.up:
            h = torch.cat([h, skips.pop()], dim=1)
            h = level["res1"](h, temb)
            h = level["res2"](h, temb)
            h = level["attn"](h)
            h = level["up"](h)

        return self.out(h)

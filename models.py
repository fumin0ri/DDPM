"""
ノイズ予測ネットワーク ε_θ(x_t, t)。

どちらのモデルも「x_t と時刻 t を受け取り、x_t と同じ形のノイズ予測を返す」。
  - ToyMLP : 2次元の点 (x, y) 用。数十秒で学習できる。
  - UNet   : 画像用。論文と同じく、ResBlock + Self-Attention + 時刻埋め込みの U-Net。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# 時刻埋め込み (共通)
# ======================================================================
def timestep_embedding(t, dim):
    """
    整数の時刻 t (形 (B,)) を dim 次元のベクトルに変換する。
    Transformer の位置エンコーディングと同じ sin / cos の組み合わせ。

    なぜ必要か:
      ネットワークは「今どれくらいノイズが乗っているか」を知らないと、
      どの程度ノイズを取り除けばよいか判断できない。
      t をそのまま1つの数として入れるより、いろいろな周波数の sin/cos に
      展開した方がネットワークが扱いやすい。
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None, :]  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)


# ======================================================================
# 2次元点群用: シンプルな MLP
# ======================================================================
class ToyMLP(nn.Module):
    """
    入力: 点の座標 x (B, 2) と時刻 t (B,)
    出力: その点に乗っているノイズの予測 (B, 2)

    構造は「残差ブロックを積んだ MLP」。各ブロックで時刻埋め込みを足し込む
    (画像用 U-Net の ResBlock と同じ考え方)。
    """

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
            h = h + block(h + temb)  # 時刻情報を足してから変換し、残差接続
        return self.out(h)


# ======================================================================
# 画像用: U-Net
# ======================================================================
class ResBlock(nn.Module):
    """
    残差ブロック。ここで時刻埋め込みを特徴マップに足し込む。
        x ─ GN ─ SiLU ─ Conv ─(+ 時刻埋め込み)─ GN ─ SiLU ─ Dropout ─ Conv ─(+)─> out
        └──────────────────────(チャネル数を合わせる 1x1 Conv)──────────────────┘
    """

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
        # 時刻埋め込み (B, time_dim) -> (B, out_ch, 1, 1) にして全画素に同じ値を足す
        h = h + self.time_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """
    画素どうしの Self-Attention。離れた場所の情報をやり取りできる
    (例: 蝶の左右の羽の模様をそろえる)。計算が重いので低解像度 (16x16 など) でだけ使う。
    """

    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)

        # (B, C, H, W) -> (B, heads, H*W, C/heads): 各画素を1トークンとみなす
        def to_tokens(z):
            return z.reshape(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)

        h = F.scaled_dot_product_attention(to_tokens(q), to_tokens(k), to_tokens(v))
        h = h.transpose(2, 3).reshape(B, C, H, W)
        return x + self.proj(h)


class UNet(nn.Module):
    """
    64x64 画像用の U-Net (ch_mults=(1, 2, 2, 4) のとき)。

      解像度  チャネル
      64x64   64   ──────────────────────────────────── skip ──┐
       ↓ down                                                    │
      32x32   128  ─────────────────────────── skip ──┐          │
       ↓ down                                           │          │
      16x16   128 (+Attention) ──────── skip ──┐       │          │
       ↓ down                                    │       │          │
       8x8    256  ── skip ──┐                  │       │          │
                     middle (Res + Attn + Res)   │       │          │
       8x8    256  <─ concat ┘                  │       │          │
       ↑ up                                      │       │          │
      16x16   128 (+Attention) <──── concat ────┘       │          │
       ↑ up                                              │          │
      32x32   128  <─────────────────────── concat ─────┘          │
       ↑ up                                                         │
      64x64   64   <──────────────────────────────────── concat ────┘
        → 出力 (3チャネル = RGB 各画素のノイズ予測)

    エンコーダで解像度を下げて大域的な情報をつかみ、デコーダで元の解像度に戻す。
    skip 接続で細かい情報をデコーダに直接渡す。
    """

    def __init__(self, in_ch=3, ch=64, ch_mults=(1, 2, 2, 4), attn_levels=(2,), dropout=0.1):
        super().__init__()
        time_dim = ch * 4
        self.ch = ch

        # 時刻埋め込み: sin/cos -> MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_ch, ch, 3, padding=1)

        # ---- エンコーダ (down) ----
        chs = [ch * m for m in ch_mults]
        self.down = nn.ModuleList()
        cur = ch
        for i, c in enumerate(chs):
            self.down.append(nn.ModuleDict({
                "res1": ResBlock(cur, c, time_dim, dropout),
                "res2": ResBlock(c, c, time_dim, dropout),
                "attn": AttnBlock(c) if i in attn_levels else nn.Identity(),
                # 最後の段以外は stride 2 の畳み込みで解像度を半分に
                "down": nn.Conv2d(c, c, 3, stride=2, padding=1) if i < len(chs) - 1 else nn.Identity(),
            }))
            cur = c

        # ---- 中間 (最も低い解像度) ----
        self.mid1 = ResBlock(cur, cur, time_dim, dropout)
        self.mid_attn = AttnBlock(cur)
        self.mid2 = ResBlock(cur, cur, time_dim, dropout)

        # ---- デコーダ (up) ----
        self.up = nn.ModuleList()
        for i in reversed(range(len(chs))):
            c = chs[i]
            self.up.append(nn.ModuleDict({
                # skip 接続を concat するので入力チャネルは cur + c
                "res1": ResBlock(cur + c, c, time_dim, dropout),
                "res2": ResBlock(c, c, time_dim, dropout),
                "attn": AttnBlock(c) if i in attn_levels else nn.Identity(),
                # 最初の段 (最高解像度) 以外は2倍に拡大
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
            skips.append(h)  # 解像度を下げる前の特徴を保存
            h = level["down"](h)

        h = self.mid1(h, temb)
        h = self.mid_attn(h)
        h = self.mid2(h, temb)

        for level in self.up:
            h = torch.cat([h, skips.pop()], dim=1)  # 同じ解像度の特徴をくっつける
            h = level["res1"](h, temb)
            h = level["res2"](h, temb)
            h = level["attn"](h)
            h = level["up"](h)

        return self.out(h)

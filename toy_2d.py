"""
2次元の点群で DDPM を体験するデモ (CPU でも1分程度)。

  uv run python toy_2d.py --shape swissroll
  uv run python toy_2d.py --shape heart
  uv run python toy_2d.py --shape text --text DDPM

出力 (outputs/toy_<shape>/):
  forward.png  : 拡散過程。データにノイズが足され、ただのガウス分布になっていく様子
  reverse.png  : 逆過程。ノイズから少しずつ元の形が浮かび上がる様子
  reverse.gif  : 上をアニメーションにしたもの (勉強会で見せる用)
  loss.png     : 学習曲線
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image, ImageDraw, ImageFont

from ddpm import GaussianDiffusion
from models import ToyMLP


# ----------------------------------------------------------------------
# データ: いろいろな形の点群 (平均0・標準偏差1 に正規化)
# ----------------------------------------------------------------------
def make_swissroll(n):
    u = 1.5 * math.pi * (1 + 2 * np.random.rand(n))
    x = np.stack([u * np.cos(u), u * np.sin(u)], axis=1)
    return x + 0.3 * np.random.randn(n, 2)


def make_heart(n):
    u = 2 * math.pi * np.random.rand(n)
    x = 16 * np.sin(u) ** 3
    y = 13 * np.cos(u) - 5 * np.cos(2 * u) - 2 * np.cos(3 * u) - np.cos(4 * u)
    return np.stack([x, y], axis=1) + 0.4 * np.random.randn(n, 2)


def make_text(n, text):
    # 文字を画像に描き、黒い画素の位置から点をサンプルする
    img = Image.new("L", (40 * len(text), 60), 255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 50)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 50)
        except OSError:
            font = ImageFont.load_default()
    draw.text((5, 0), text, fill=0, font=font)
    ys, xs = np.nonzero(np.array(img) < 128)
    idx = np.random.randint(0, len(xs), n)
    pts = np.stack([xs[idx], -ys[idx]], axis=1).astype(np.float64)
    return pts + np.random.rand(n, 2)  # 画素の中で少しばらつかせる


def make_data(shape, n, text):
    if shape == "swissroll":
        x = make_swissroll(n)
    elif shape == "heart":
        x = make_heart(n)
    elif shape == "text":
        x = make_text(n, text)
    else:
        raise ValueError(shape)
    x = x - x.mean(axis=0)
    x = x / x.std()  # 平均0・標準偏差1 にそろえる (x_T ~ N(0, I) と同じくらいの広がり)
    return torch.tensor(x, dtype=torch.float32)


# ----------------------------------------------------------------------
# 描画
# ----------------------------------------------------------------------
LIM = 3.0


def scatter_row(snapshots, titles, path, suptitle):
    fig, axes = plt.subplots(1, len(snapshots), figsize=(3 * len(snapshots), 3.2))
    for ax, pts, title in zip(axes, snapshots, titles):
        ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.5)
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_gif(trajectory, steps, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    sc = ax.scatter(trajectory[0][:, 0], trajectory[0][:, 1], s=1, alpha=0.5)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("")

    # 最後のフレームを少し長めに見せる
    frames = list(range(len(trajectory))) + [len(trajectory) - 1] * 15

    def update(i):
        sc.set_offsets(trajectory[i].numpy())
        title.set_text(f"t = {steps[i]}")
        return sc, title

    anim = FuncAnimation(fig, update, frames=frames, interval=50)
    anim.save(path, writer=PillowWriter(fps=20))
    plt.close(fig)


# ----------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="swissroll", choices=["swissroll", "heart", "text"])
    p.add_argument("--text", default="DDPM")
    p.add_argument("--n-data", type=int, default=20000)
    p.add_argument("--n-samples", type=int, default=3000)
    p.add_argument("--steps", type=int, default=20000, help="学習の反復回数")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or f"outputs/toy_{args.shape}"
    os.makedirs(out, exist_ok=True)

    data = make_data(args.shape, args.n_data, args.text).to(device)
    diffusion = GaussianDiffusion(T=args.T).to(device)
    model = ToyMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 学習率を cosine カーブで 0 まで下げる。終盤に細かい形まで合わせ込むため
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    # ---- 1. 拡散過程を可視化 (学習は不要。q_sample だけで描ける) ----
    show_t = [0, args.T // 10, args.T // 4, args.T // 2, args.T - 1]
    x0 = data[: args.n_samples]
    snaps = []
    for t in show_t:
        tt = torch.full((x0.shape[0],), t, device=device, dtype=torch.long)
        snaps.append(diffusion.q_sample(x0, tt, torch.randn_like(x0)).cpu())
    scatter_row(snaps, [f"t = {t}" for t in show_t], f"{out}/forward.png",
                "Forward process q(x_t | x_0): data -> noise")

    # ---- 2. 学習 (Algorithm 1) ----
    losses = []
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, data.shape[0], (args.batch_size,), device=device)
        loss = diffusion.loss(model, data[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 1000 == 0 or step == 1:
            print(f"step {step:6d}  loss {np.mean(losses[-1000:]):.4f}")

    fig, ax = plt.subplots(figsize=(5, 3))
    # 生の loss はばらつきが大きいので移動平均を描く
    w = min(100, len(losses))
    ax.plot(np.convolve(losses, np.ones(w) / w, mode="valid"))
    ax.set_xlabel("step")
    ax.set_ylabel("loss (moving avg)")
    fig.tight_layout()
    fig.savefig(f"{out}/loss.png", dpi=120)
    plt.close(fig)

    # ---- 3. 生成 (Algorithm 2) ----
    model.eval()
    every = max(1, args.T // 100)  # 100コマ程度記録する
    _, traj = diffusion.sample(model, (args.n_samples, 2), device, return_every=every)
    # traj[0] は x_T (純粋なノイズ)、traj[k] は t_index = T - k*every 付近
    traj_steps = [args.T] + [t for t in reversed(range(args.T)) if t % every == 0]

    pick = [0, len(traj) // 2, int(len(traj) * 0.8), int(len(traj) * 0.95), len(traj) - 1]
    scatter_row([traj[i] for i in pick], [f"t = {traj_steps[i]}" for i in pick],
                f"{out}/reverse.png", "Reverse process p_θ: noise -> data")
    save_gif(traj, traj_steps, f"{out}/reverse.gif")

    torch.save(model.state_dict(), f"{out}/model.pt")
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()

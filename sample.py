"""
学習済みチェックポイントから画像を生成し、勉強会で見せる図を作る。

  uv run python sample.py --ckpt runs/butterflies/ckpt.pt

出力 (チェックポイントと同じフォルダ):
  generated.png : 生成画像 (64枚)
  forward.png   : 学習画像にノイズを足していく様子 (q(x_t|x_0))
  process.png   : ノイズから画像が浮かび上がる様子 (行 = サンプル, 列 = 時刻)
  process.gif   : 上をアニメーションにしたもの
"""

import argparse
import os

import torch

from ddpm import GaussianDiffusion
from image_utils import load_butterflies, load_folder, save_grid, to_pil_grid
from models import UNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/butterflies/ckpt.pt")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-ema", action="store_true", help="EMA ではなく生の重みで生成")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    targs = ckpt["args"]
    out = os.path.dirname(args.ckpt)
    size = targs["image_size"]

    diffusion = GaussianDiffusion(T=targs["T"]).to(device)
    model = UNet().to(device).eval()
    model.load_state_dict(ckpt["model" if args.no_ema else "ema"])
    print(f"loaded step {ckpt['step']} from {args.ckpt}")

    torch.manual_seed(args.seed)
    use_amp = device == "cuda"
    T = targs["T"]

    # ---- 1. 拡散過程: 学習データにノイズを足す (学習不要, 式(4)) ----
    if not targs.get("dry_run"):
        if targs.get("data_dir"):
            data = load_folder(targs["data_dir"], size)
        else:
            data = load_butterflies(size)
        x0 = data[:6].to(device)
        cols = [int(T * r) for r in (0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.999)]
        rows = []
        for t in cols:
            tt = torch.full((x0.shape[0],), t, device=device, dtype=torch.long)
            rows.append(diffusion.q_sample(x0, tt, torch.randn_like(x0)))
        # (時刻, サンプル, C, H, W) -> サンプルごとに横並び
        grid = torch.stack(rows, dim=1).flatten(0, 1)
        save_grid(grid, f"{out}/forward.png", nrow=len(cols))
        print(f"forward t = {cols}")

    # ---- 2. 生成画像 ----
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        x = diffusion.sample(model, (args.n, 3, size, size), device)
    save_grid(x.float(), f"{out}/generated.png", nrow=8)

    # ---- 3. 生成の過程 ----
    every = max(1, T // 50)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        _, traj = diffusion.sample(model, (6, 3, size, size), device, return_every=every)
    traj = [z.float() for z in traj]  # traj[0] = x_T, traj[-1] = x_0

    pick = torch.linspace(0, len(traj) - 1, 10).round().long().tolist()
    grid = torch.stack([traj[i] for i in pick], dim=1).flatten(0, 1)
    save_grid(grid, f"{out}/process.png", nrow=len(pick))

    frames = [to_pil_grid(z, nrow=6) for z in traj] + [to_pil_grid(traj[-1], nrow=6)] * 20
    frames = [f.resize((f.width * 2, f.height * 2)) for f in frames]
    frames[0].save(f"{out}/process.gif", save_all=True, append_images=frames[1:],
                   duration=60, loop=0)
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()

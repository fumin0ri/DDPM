"""
蝶の画像 (64x64) で DDPM を学習する。40,000 step で RTX 4090 なら1時間前後、
RTX 3090 なら1.5〜2時間程度の想定 (VRAM はどちらも十分足りる)。

  uv run python train_butterflies.py                       # 蝶の画像で学習
  uv run python train_butterflies.py --data-dir ./my_imgs  # 自分のフォルダの画像で学習
  uv run python train_butterflies.py --dry-run             # 動作確認 (乱数画像・数 step)

出力 (runs/butterflies/):
  samples_XXXXXX.png : 学習途中の生成結果 (EMA モデル)
  data.png           : 学習データの例
  ckpt.pt            : チェックポイント (sample.py で読み込む)
  loss.csv           : 100 step ごとの平均 loss (plot_loss.py で学習曲線にする)
"""

import argparse
import copy
import os
import time

import torch

from ddpm import GaussianDiffusion
from image_utils import load_butterflies, load_folder, save_grid
from models import UNet


@torch.no_grad()
def ema_update(ema_model, model, decay):
    """
    EMA (指数移動平均): ema = decay * ema + (1 - decay) * 現在の重み
    学習中の重みは step ごとに揺れるので、平均をとった重みで生成すると画質が安定する。
    論文でも decay = 0.9999 の EMA を使っている。DDPM の理論とは独立した「実用上の工夫」。
    """
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.lerp_(p, 1.0 - decay)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None, help="指定するとこのフォルダの画像で学習")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema-decay", type=float, default=0.9995)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--sample-every", type=int, default=5000)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--out", default="runs/butterflies")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="乱数データで数 step だけ回す")
    args = p.parse_args()

    if args.dry_run:
        args.steps, args.sample_every, args.n_samples, args.T = 4, 2, 4, 10
        args.image_size = 32

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    # ---- データ: 全画像を最初に読み込んで GPU に載せておく (1000枚程度なら数十MB) ----
    if args.dry_run:
        data = torch.rand(32, 3, args.image_size, args.image_size) * 2 - 1
    elif args.data_dir:
        data = load_folder(args.data_dir, args.image_size)
    else:
        data = load_butterflies(args.image_size)
    data = data.to(device)
    print(f"data: {tuple(data.shape)}  (range [{data.min():.1f}, {data.max():.1f}])")
    save_grid(data[:64], f"{args.out}/data.png")

    # ---- モデル ----
    diffusion = GaussianDiffusion(T=args.T).to(device)
    model = UNet().to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet parameters: {n_params / 1e6:.1f}M")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 最初の warmup step は学習率を 0 から徐々に上げる (論文と同じ)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))

    # Ampere 以降 (RTX 3090 / 4090 など) は bfloat16 が速い。CPU のときは使わない。
    use_amp = device == "cuda"

    # 学習曲線用に 100 step ごとの平均 loss を CSV に残す (1行ずつ flush するので途中で止めても残る)
    loss_file = open(f"{args.out}/loss.csv", "w", encoding="utf-8")
    loss_file.write("step,loss,elapsed_sec\n")

    t0 = time.time()
    running = 0.0
    for step in range(1, args.steps + 1):
        model.train()
        idx = torch.randint(0, data.shape[0], (args.batch_size,), device=device)
        x0 = data[idx]
        # データ拡張: ランダムに左右反転
        flip = torch.rand(x0.shape[0], device=device) < 0.5
        x0 = torch.where(flip[:, None, None, None], x0.flip(3), x0)

        # ---- ここが DDPM の学習 (Algorithm 1) の全て ----
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = diffusion.loss(model, x0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 論文と同じ勾配クリップ
        opt.step()
        sched.step()
        ema_update(ema_model, model, args.ema_decay)

        running += loss.item()
        if step % 100 == 0 or args.dry_run:
            n = 1 if args.dry_run else 100
            elapsed = time.time() - t0
            print(f"step {step:6d}  loss {running / n:.4f}  {elapsed:.0f}s")
            loss_file.write(f"{step},{running / n:.6f},{elapsed:.1f}\n")
            loss_file.flush()
            running = 0.0

        if step % args.sample_every == 0 or step == args.steps:
            # ---- 生成 (Algorithm 2) で途中経過を確認 ----
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                x = diffusion.sample(
                    ema_model, (args.n_samples, 3, args.image_size, args.image_size), device)
            save_grid(x.float(), f"{args.out}/samples_{step:06d}.png", nrow=4)
            torch.save({
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "step": step,
                "args": vars(args),
            }, f"{args.out}/ckpt.pt")
            print(f"  -> saved samples / ckpt at step {step}")

    loss_file.close()


if __name__ == "__main__":
    main()

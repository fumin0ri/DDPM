"""
train_butterflies.py の学習曲線を描く。

  uv run python plot_loss.py                                # runs/butterflies/loss.csv から
  uv run python plot_loss.py train.log                      # nohup で残したログから
  uv run python plot_loss.py train.log --logy --marks 5000  # 縦軸を対数に + samples_*.png の step に線

入力は2種類どちらでもよい (拡張子で判別):
  *.csv : train_butterflies.py が書き出す loss.csv (step,loss,elapsed_sec)
  その他: 標準出力のログ。"step    100  loss 0.1234  12s" の行だけを拾う

出力: 入力と同じフォルダの loss.png (--out で変更可)
"""

import argparse
import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_LINE = re.compile(r"^step\s+(\d+)\s+loss\s+([0-9.eE+-]+)")


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [int(r["step"]) for r in rows], [float(r["loss"]) for r in rows]


def read_log(path):
    steps, losses = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LOG_LINE.match(line.strip())
            if not m:
                continue
            step = int(m.group(1))
            # 同じログに複数回の学習が追記されていたら、最後の回だけを使う
            if steps and step <= steps[-1]:
                steps, losses = [], []
            steps.append(step)
            losses.append(float(m.group(2)))
    return steps, losses


def moving_average(x, w):
    if len(x) < w:
        return None
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default="runs/butterflies/loss.csv",
                   help="loss.csv または学習ログ (train.log など)")
    p.add_argument("--out", default=None, help="出力先 (省略時は入力と同じフォルダの loss.png)")
    p.add_argument("--logy", action="store_true", help="縦軸を対数にする")
    p.add_argument("--smooth", type=int, default=10, help="移動平均の点数 (1点 = 100 step)")
    p.add_argument("--marks", type=int, default=0,
                   help="この step ごとに縦線を引く (samples_*.png と対応させる用。例: 5000)")
    args = p.parse_args()

    read = read_csv if args.path.endswith(".csv") else read_log
    steps, losses = read(args.path)
    if not steps:
        raise SystemExit(f"{args.path} から loss が読み取れませんでした")
    steps, losses = np.array(steps), np.array(losses)
    print(f"{len(steps)} points  (step {steps[0]} - {steps[-1]}, last loss {losses[-1]:.4f})")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, losses, color="tab:blue", alpha=0.3, lw=1, label="100 step average")
    ma = moving_average(losses, args.smooth) if args.smooth > 1 else None
    if ma is not None:
        ax.plot(steps[args.smooth - 1:], ma, color="tab:blue", lw=2,
                label=f"moving average ({args.smooth * 100} step)")
    if args.marks > 0:
        for s in range(args.marks, steps[-1] + 1, args.marks):
            ax.axvline(s, color="gray", ls=":", lw=1)
    if args.logy:
        ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss  ||ε - ε_θ(x_t, t)||²")
    ax.set_title("DDPM training loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = args.out or os.path.join(os.path.dirname(args.path) or ".", "loss.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

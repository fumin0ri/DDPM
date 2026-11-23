"""画像の読み込み・保存。"""

import os

import numpy as np
import torch
from PIL import Image

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def pil_to_tensor(img, size):
    """PIL 画像 -> (3, size, size), 値域 [-1, 1]"""
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    return x.permute(2, 0, 1) * 2.0 - 1.0


def load_butterflies(size):
    from datasets import load_dataset

    ds = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
    return torch.stack([pil_to_tensor(ex["image"], size) for ex in ds])


def load_folder(path, size):
    files = []
    for root, _, names in os.walk(path):
        files += [os.path.join(root, n) for n in names if n.lower().endswith(IMG_EXTS)]
    if not files:
        raise FileNotFoundError(f"no images in {path}")
    return torch.stack([pil_to_tensor(Image.open(f), size) for f in sorted(files)])


def to_pil_grid(x, nrow):
    x = ((x.clamp(-1, 1) + 1) / 2 * 255).round().byte().cpu()
    n, c, h, w = x.shape
    ncol = (n + nrow - 1) // nrow
    grid = torch.full((c, ncol * (h + 2) + 2, nrow * (w + 2) + 2), 255, dtype=torch.uint8)
    for i in range(n):
        r, col = divmod(i, nrow)
        grid[:, 2 + r * (h + 2): 2 + r * (h + 2) + h, 2 + col * (w + 2): 2 + col * (w + 2) + w] = x[i]
    return Image.fromarray(grid.permute(1, 2, 0).numpy())


def save_grid(x, path, nrow=8):
    to_pil_grid(x, nrow).save(path)

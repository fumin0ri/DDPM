# DDPM 最小実装

Ho et al., *Denoising Diffusion Probabilistic Models* (2020) を、論文の数式・アルゴリズムと1対1で対応するように書いた PyTorch 実装。

解説: [docs/explanation_ja.md](docs/explanation_ja.md)

## セットアップ

環境は [uv](https://docs.astral.sh/uv/) で管理する。`pyproject.toml` と `uv.lock` があるので、次の1行で `.venv/` に同じ環境ができる (torch は CUDA 12.8 版。RTX 3090 / 4090 のどちらでも動く)。

```bash
uv sync
```

以降のコマンドは `uv run python ...` で実行する (または `.venv` を activate して `python ...`)。

pip を使う場合は `pip install -r requirements.txt`。ただし Windows では PyPI の torch が CPU 版になるので、先に [PyTorch 公式](https://pytorch.org/get-started/locally/) の手順で CUDA 版を入れておく。

## 実行

```bash
# 2次元の点群 (CPU でも数分, GPU なら1分程度)
uv run python toy_2d.py --shape swissroll     # swissroll / heart / text
uv run python toy_2d.py --shape text --text DDPM

# 蝶の画像 64x64 (RTX 4090 で1時間前後, RTX 3090 で1.5〜2時間程度)
uv run python train_butterflies.py
uv run python train_butterflies.py --dry-run          # 動作確認だけ
uv run python train_butterflies.py --data-dir ./imgs  # 自分の画像フォルダで学習
uv run python train_butterflies.py --steps 20000      # 時間を半分にしたいとき

# 学習済みモデルから生成と可視化
uv run python sample.py --ckpt runs/butterflies/ckpt.pt

# 学習曲線 (runs/butterflies/loss.csv → loss.png)
uv run python plot_loss.py
uv run python plot_loss.py train.log --logy --marks 5000  # nohup のログからも描ける
```

SSH 先で長時間学習する場合は、接続が切れても止まらないように `tmux` や `nohup` で実行する:

```bash
nohup uv run python train_butterflies.py > train.log 2>&1 &
```

## ファイル

| ファイル | 内容 |
|---|---|
| `ddpm.py` | 拡散の本体 (スケジュール, q_sample, loss, sampling)。ネットワークに依存しない |
| `models.py` | ε_θ: 2次元用 MLP と画像用 U-Net |
| `toy_2d.py` | 2次元デモ → `outputs/toy_*/` |
| `train_butterflies.py` | 画像の学習 → `runs/butterflies/` |
| `sample.py` | 生成と可視化 |
| `plot_loss.py` | 学習曲線 (loss.csv または学習ログから) |
| `image_utils.py` | 画像の読み込み・保存 |

データ: [huggan/smithsonian_butterflies_subset](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) (初回実行時に自動ダウンロード)

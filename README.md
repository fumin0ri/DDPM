# DDPM 最小実装 (勉強会用)

Ho et al., *Denoising Diffusion Probabilistic Models* (2020) を、論文の数式・アルゴリズムと1対1で対応するように書いた PyTorch 実装。

解説: [docs/explanation_ja.md](docs/explanation_ja.md)

## セットアップ

```bash
pip install -r requirements.txt
```

## 実行

```bash
# 2次元の点群 (CPU でも数分, GPU なら1分程度)
python toy_2d.py --shape swissroll     # swissroll / heart / text
python toy_2d.py --shape text --text DDPM

# 蝶の画像 64x64 (RTX 4090 で1時間前後)
python train_butterflies.py
python train_butterflies.py --dry-run          # 動作確認だけ
python train_butterflies.py --data-dir ./imgs  # 自分の画像フォルダで学習

# 学習済みモデルから生成 + 発表用の図
python sample.py --ckpt runs/butterflies/ckpt.pt
```

SSH 先で長時間学習する場合は、接続が切れても止まらないように `tmux` や `nohup` で実行する:

```bash
nohup python train_butterflies.py > train.log 2>&1 &
```

## ファイル

| ファイル | 内容 |
|---|---|
| `ddpm.py` | 拡散の本体 (スケジュール, q_sample, loss, sampling)。ネットワークに依存しない |
| `models.py` | ε_θ: 2次元用 MLP と画像用 U-Net |
| `toy_2d.py` | 2次元デモ → `outputs/toy_*/` |
| `train_butterflies.py` | 画像の学習 → `runs/butterflies/` |
| `sample.py` | 生成と可視化 |
| `image_utils.py` | 画像の読み込み・保存 |

データ: [huggan/smithsonian_butterflies_subset](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) (初回実行時に自動ダウンロード)

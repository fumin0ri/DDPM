# DDPM 実装解説

対象: 深層学習 (PyTorch, CNN) は分かるが、拡散モデルは初めての人
論文: Ho, Jain, Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS 2020 ([arXiv:2006.11239](https://arxiv.org/abs/2006.11239))

---

## 0. 全体像

DDPM がやっていることは、次の2行に集約できる。

1. **学習**: データにノイズを足し、「足したノイズは何か」をネットワークに当てさせる
2. **生成**: 純粋なノイズから始め、ネットワークが予測したノイズを少しずつ取り除いていく

```
拡散過程 q (固定, 学習しない)   x_0 ──ノイズ──> x_1 ──> ... ──> x_T ≈ N(0, I)
逆過程 p_θ (これを学習する)     x_0 <──除去── x_1 <── ... <── x_T
```

### ファイル構成

| ファイル | 中身 | 論文との対応 |
|---|---|---|
| [ddpm.py](../ddpm.py) | ノイズスケジュール、拡散過程、損失、生成ループ | 式(4), (11), (14), Algorithm 1, 2 |
| [models.py](../models.py) | ノイズ予測ネットワーク ε_θ (2次元用 MLP / 画像用 U-Net) | 付録B |
| [toy_2d.py](../toy_2d.py) | 2次元点群でのデモ (直感をつかむ用) | — |
| [train_butterflies.py](../train_butterflies.py) | 蝶の画像 64×64 で学習 | — |
| [sample.py](../sample.py) | 学習済みモデルからの生成と可視化 | — |
| [plot_loss.py](../plot_loss.py) | 蝶の学習の学習曲線を描く | — |

**大事な設計**: `ddpm.py` はネットワークの中身を知らない。「`model(x_t, t)` がノイズを返す」ことだけを約束にしているので、2次元の MLP でも画像の U-Net でも**同じ拡散コードがそのまま動く**。

---

## 1. 記号の整理

| 論文 | コード | 意味 |
|---|---|---|
| $T$ | `self.T` | ステップ数 (1000) |
| $\beta_t$ | `self.betas` | ステップ $t$ で足すノイズの分散 |
| $\alpha_t = 1-\beta_t$ | `self.alphas` | |
| $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$ | `self.alpha_bars` | 元のデータがどれだけ残っているか |
| $\epsilon \sim \mathcal N(0, I)$ | `noise` | |
| $\epsilon_\theta(x_t, t)$ | `model(x_t, t)` | ノイズを予測するネットワーク |

> **注意**: 論文の $t$ は 1 始まり ($1..T$)、コードの `t` は 0 始まり (`0..T-1`)。コードの `t=0` が論文の $t=1$ に当たる。

---

## 2. 拡散過程 (forward process) — ノイズを足す

### ノイズスケジュール

[ddpm.py](../ddpm.py) の `__init__`:

```python
betas = torch.linspace(beta_start, beta_end, T)   # 1e-4 → 0.02
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
```

1ステップでは、少し縮めて少しノイズを足す:

$$q(x_t \mid x_{t-1}) = \mathcal N\left(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\right)$$

$\sqrt{1-\beta_t}$ で縮めるのは、分散が発散せず最終的に $\mathcal N(0, I)$ に落ち着くようにするため。

この線形スケジュールだと $\bar\alpha_T \approx 4\times10^{-5}$。つまり $x_T$ には元データが $\sqrt{\bar\alpha_T}\approx 0.006$ 倍しか残っておらず、ほぼ純粋なノイズになる。

### 一気に $x_t$ を作る (式4) — 実装上いちばん重要な式

ガウス分布を重ねてもガウス分布なので、$t$ 回足したノイズは1回にまとめられる:

$$q(x_t \mid x_0) = \mathcal N\left(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t) I\right)$$

$$\Longleftrightarrow\quad x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I)$$

```python
def q_sample(self, x0, t, noise):
    return (extract(self.sqrt_alpha_bars, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alpha_bars, t, x0.shape) * noise)
```

- **これがあるおかげで、学習時に $t$ をランダムに選んで1回の計算で $x_t$ を作れる。** 1000回ループを回す必要がない。
- `extract` は、バッチ内でサンプルごとに異なる $t$ の係数を取り出し、`(B, 1, 1, 1)` に整形してブロードキャストするための小道具。

`toy_2d.py` の `forward.png` / `sample.py` の `forward.png` が、この式だけで描いた図 (学習不要)。

---

## 3. 学習 — 足したノイズを当てる

### Algorithm 1 とコードの対応

```
1: repeat
2:   x_0 ~ q(x_0)
3:   t ~ Uniform({1, ..., T})
4:   ε ~ N(0, I)
5:   ∇_θ || ε − ε_θ( √ᾱ_t x_0 + √(1−ᾱ_t) ε, t ) ||²  で勾配降下
6: until converged
```

```python
def loss(self, model, x0):                                    # 2: x0 はミニバッチ
    t = torch.randint(0, self.T, (B,), device=x0.device)      # 3
    noise = torch.randn_like(x0)                              # 4
    x_t = self.q_sample(x0, t, noise)                         # 5: ε_θ の第1引数
    pred_noise = model(x_t, t)
    return F.mse_loss(pred_noise, noise)                      # 5: || ε − ε_θ ||²
```

**学習のコードはこれで全部。** 普通の回帰問題 (入力: ノイズの乗ったデータと時刻、正解: ノイズ) と同じ形をしている。

### なぜ「ノイズ」を予測するのか (式14 の背景)

論文の流れをかいつまむと:

1. 本来の目的は対数尤度の最大化。変分下界 (ELBO) を最大化する (VAE と同じ発想)。
2. ELBO は各ステップの KL ダイバージェンス $D_{KL}(q(x_{t-1}\mid x_t, x_0)\,\|\,p_\theta(x_{t-1}\mid x_t))$ の和に分解できる。
3. $q(x_{t-1}\mid x_t, x_0)$ (「$x_0$ を知っているときの1ステップ前」) はガウス分布で、平均が閉じた式で書ける (式7)。$p_\theta$ もガウス分布にすれば、KL は**平均どうしの二乗誤差**になる。
4. その平均を、式4 の $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\epsilon)/\sqrt{\bar\alpha_t}$ を使って書き換えると、
   $$\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\right)$$
   となり、未知なのは $\epsilon$ だけ。だから「平均を当てる」問題は「$\epsilon$ を当てる」問題に置き換えられる (式11, 12)。
5. さらに時刻ごとの重み係数を捨てて単純な二乗誤差にしたもの (式14, $L_{simple}$) の方が、実験では画質がよかった。

### loss の値について

- loss は学習初期に急に下がり、そのあとはほぼ横ばいに見える。$t$ が大きいとき (ほぼノイズ) は当てやすく、$t$ が小さいときは当てにくい。ランダムな $t$ の平均を見ているため、下がり方が分かりにくい。
- **loss が横ばいでも生成画像はまだ良くなっていく**ことが多い。途中の `samples_*.png` で確認するのが確実。

---

## 4. 生成 — ノイズを少しずつ取り除く

### Algorithm 2 とコードの対応

```
1: x_T ~ N(0, I)
2: for t = T, ..., 1 do
3:   z ~ N(0, I) if t > 1, else z = 0
4:   x_{t−1} = 1/√α_t ( x_t − (1−α_t)/√(1−ᾱ_t) · ε_θ(x_t, t) ) + σ_t z
5: end for
6: return x_0
```

```python
def sample(self, model, shape, device, ...):
    x = torch.randn(shape, device=device)              # 1
    for t_index in reversed(range(self.T)):            # 2
        x = self.p_sample(model, x, t_index)           # 3-4
    return x                                           # 6

def p_sample(self, model, x_t, t_index):
    pred_noise = model(x_t, t)
    mean = (x_t - beta_t / sqrt_one_minus_alpha_bar_t * pred_noise) / alpha_t.sqrt()
    if t_index == 0:
        return mean                                    # 最後はノイズを足さない
    return mean + beta_t.sqrt() * torch.randn_like(x_t)
```

- `mean` は 3節4. の $\tilde\mu_t$ の $\epsilon$ を $\epsilon_\theta$ に置き換えたもの。**学習で使った式と生成で使う式は同じところから来ている。**
- $\sigma_t^2$ は学習せず固定。論文では $\sigma_t^2=\beta_t$ と $\sigma_t^2=\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ の2通りを試し、結果は同程度だった。ここでは簡単な $\beta_t$ を使っている。
- **毎ステップでノイズ $z$ を足し直すのがポイント。** これを足さないと、どのサンプルも似た平均的な画像になりやすい。$p_\theta(x_{t-1}\mid x_t)$ は分布なので、そこからサンプリングしている。

### 生成が遅い理由

1枚生成するのにネットワークを $T=1000$ 回呼ぶ。学習は1回の呼び出しで済むのと対照的。これを減らすのが後続研究の DDIM などの動機 (8節)。

---

## 5. ネットワーク ε_θ

### 時刻埋め込み

[models.py](../models.py) の `timestep_embedding`。整数 $t$ を Transformer の位置エンコーディングと同じ sin/cos のベクトルにする。

- ネットワークは「今どのくらいノイズが乗っているか」を知らないと、どれだけ取り除けばよいか決められない。そのため $t$ を入力する必要がある。
- **1つのネットワークを全ての $t$ で共有している**のがポイント。$t$ ごとに別のネットワークを用意するわけではない。

### ToyMLP (2次元用)

座標 $(x, y)$ を全結合層に通し、各残差ブロックで時刻埋め込みを足すだけ。出力は2次元のノイズ予測。

### UNet (画像用)

```
64x64 (64ch) ─────────────────────────── skip ──┐
  32x32 (128ch) ─────────────── skip ──┐        │
    16x16 (128ch, Attention) ─ skip ┐  │        │
      8x8 (256ch) ─ middle ─ 8x8    │  │        │
    16x16 (Attention) <── concat ───┘  │        │
  32x32 <────────────────── concat ────┘        │
64x64 <──────────────────────────────── concat ─┘ → 3ch (ノイズ予測)
```

- **入出力の形が同じ** (画像 → 画像)。セグメンテーションと同じく U-Net が自然に使える。
- **ResBlock**: 畳み込みの途中で、時刻埋め込みを線形変換したものをチャネルごとに足し込む (`h + time_proj(temb)[:, :, None, None]`)。
- **AttnBlock**: 16×16 などの低解像度でだけ Self-Attention を使う。画像の離れた場所 (例: 蝶の左右の羽) の整合性をとる役割。
- GroupNorm, SiLU, Dropout は論文 (PixelCNN++ ベースの U-Net) と同じ部品。
- パラメータ数は約 13M (論文の CIFAR-10 用は 35.7M)。

---

## 6. 学習スクリプトの実用上の工夫

[train_butterflies.py](../train_butterflies.py) の中で、DDPM の理論とは独立した部分:

| 工夫 | 理由 |
|---|---|
| 画素値を $[-1, 1]$ に正規化 | $x_T\sim\mathcal N(0,I)$ とスケールを合わせる。論文と同じ |
| 左右反転 | 1000枚と少ないのでデータ拡張 |
| 学習率 warmup | 序盤の不安定さを避ける (論文: 5000 step) |
| 勾配クリップ (1.0) | 論文と同じ |
| **EMA** (指数移動平均) | 重みの移動平均で生成すると画質が大きく安定する。論文でも decay 0.9999。データが少なく step 数も少ないので、ここでは 0.9995 |
| bfloat16 (autocast) | RTX 3090 / 4090 (Ampere 以降) で高速化 |

---

## 7. 実行手順

```bash
uv sync

# (1) 2次元デモ: GPU なら数分
uv run python toy_2d.py --shape swissroll
uv run python toy_2d.py --shape text --text DDPM

# (2) 蝶の画像: 40,000 step で 4090 なら1時間前後, 3090 なら1.5〜2時間程度
uv run python train_butterflies.py

# (3) 生成と可視化
uv run python sample.py --ckpt runs/butterflies/ckpt.pt
uv run python plot_loss.py --logy --marks 5000   # 学習曲線 → runs/butterflies/loss.png
```

---

## 8. この先 (発展)

| 手法 | 何を改善したか |
|---|---|
| Improved DDPM (Nichol & Dhariwal, 2021) | cosine スケジュール、分散 σ_t の学習 |
| DDIM (Song et al., 2021) | 同じ学習済みモデルのまま、50 ステップ程度で生成 |
| Classifier-Free Guidance (Ho & Salimans, 2022) | 条件付き生成 (クラスやテキストの指定) を強める |
| Latent Diffusion / Stable Diffusion (Rombach et al., 2022) | VAE の潜在空間で拡散させ、高解像度でも軽くする |
| Score-based SDE (Song et al., 2021) | DDPM をスコアマッチング・確率微分方程式の枠組みで統一的に説明 |

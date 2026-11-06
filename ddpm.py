"""
DDPM (Denoising Diffusion Probabilistic Models) の「拡散」部分。

論文: Ho, Jain, Abbeel, "Denoising Diffusion Probabilistic Models", NeurIPS 2020
      https://arxiv.org/abs/2006.11239

このファイルはモデル(ニューラルネット)の中身には依存しない。
  - model(x_t, t) が「x_t に含まれるノイズ ε」を予測して返す
という約束さえ守れば、2次元点群用の MLP でも画像用の U-Net でも同じコードで動く。

記号と変数名の対応 (論文の記号 -> コード):
  T            -> self.T              拡散のステップ数 (論文では 1000)
  β_t          -> self.betas[t]       各ステップで足すノイズの分散
  α_t = 1-β_t  -> self.alphas[t]
  ᾱ_t = Π α_s  -> self.alpha_bars[t]  「元の画像がどれだけ残っているか」
  ε            -> noise               標準正規分布 N(0, I) のノイズ
  ε_θ(x_t, t)  -> model(x_t, t)       ノイズを予測するネットワーク

※ コード上の t は 0 始まり (0, 1, ..., T-1)。論文の t は 1 始まり (1, ..., T)。
   つまりコードの t=0 が論文の t=1 に対応する。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(a, t, x_shape):
    """
    1次元テンソル a (長さ T) から、バッチ内の各サンプルの時刻 t に対応する値を取り出し、
    x と掛け算できる形 (B, 1, 1, ...) に変形する。

    例: a = sqrt(ᾱ), t = [3, 500], x が画像 (B, C, H, W)
        -> [sqrt(ᾱ_3), sqrt(ᾱ_500)] を形 (2, 1, 1, 1) にして返す
    """
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.T = T

        # ---- ノイズスケジュール (論文 4節) ----
        # β_t を 1e-4 から 0.02 まで線形に増やす。
        # 最初は少しずつ、後半ほど大きなノイズを足していく。
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)  # ᾱ_t = α_1 α_2 ... α_t

        # register_buffer にしておくと model.to(device) で一緒に GPU に移動し、
        # state_dict にも保存される (学習するパラメータではない)。
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    # ------------------------------------------------------------------
    # 拡散過程 (forward process): データにノイズを足す
    # ------------------------------------------------------------------
    def q_sample(self, x0, t, noise):
        """
        x_0 から一気に x_t を作る。論文 式(4):

            q(x_t | x_0) = N(x_t; sqrt(ᾱ_t) x_0, (1 - ᾱ_t) I)

        再パラメータ化すると
            x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε,   ε ~ N(0, I)

        ポイント: t ステップ分ノイズを1回ずつ足す必要はなく、1回の計算で x_t が得られる。
                  だから学習時に t をランダムに選べる。
        """
        return (
            extract(self.sqrt_alpha_bars, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alpha_bars, t, x0.shape) * noise
        )

    # ------------------------------------------------------------------
    # 学習: 論文 Algorithm 1
    # ------------------------------------------------------------------
    def loss(self, model, x0):
        """
        Algorithm 1 (Training) をそのまま書いたもの:

          1: repeat
          2:   x_0 ~ q(x_0)                       ← 引数 x0 (データのミニバッチ)
          3:   t ~ Uniform({1, ..., T})           ← t をサンプルごとにランダムに選ぶ
          4:   ε ~ N(0, I)                        ← ノイズを作る
          5:   ∇_θ || ε - ε_θ(sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε, t) ||^2  で勾配降下
          6: until converged

        損失は論文 式(14) の L_simple。「足したノイズを当てる」二乗誤差になる。
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)  # 3行目
        noise = torch.randn_like(x0)                           # 4行目
        x_t = self.q_sample(x0, t, noise)                      # 5行目の ε_θ の引数
        pred_noise = model(x_t, t)                             # ε_θ(x_t, t)
        return F.mse_loss(pred_noise, noise)                   # || ε - ε_θ ||^2

    # ------------------------------------------------------------------
    # 生成: 論文 Algorithm 2
    # ------------------------------------------------------------------
    @torch.no_grad()
    def p_sample(self, model, x_t, t_index):
        """
        逆過程 (reverse process) を1ステップ進める: x_t -> x_{t-1}

        論文 式(11) より、ノイズ予測 ε_θ を使うと平均は
            μ_θ(x_t, t) = 1/sqrt(α_t) * ( x_t - β_t / sqrt(1 - ᾱ_t) * ε_θ(x_t, t) )

        Algorithm 2 の4行目:
            x_{t-1} = μ_θ(x_t, t) + σ_t z,   z ~ N(0, I)

        分散 σ_t^2 は学習せず固定。論文では σ_t^2 = β_t と
        σ_t^2 = β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) β_t のどちらでも同程度の結果だったとあり、
        ここでは簡単な σ_t^2 = β_t を使う。
        """
        B = x_t.shape[0]
        t = torch.full((B,), t_index, device=x_t.device, dtype=torch.long)

        beta_t = extract(self.betas, t, x_t.shape)
        alpha_t = extract(self.alphas, t, x_t.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape)

        pred_noise = model(x_t, t)
        mean = (x_t - beta_t / sqrt_one_minus_alpha_bar_t * pred_noise) / alpha_t.sqrt()

        if t_index == 0:
            # 最後のステップ (論文の t=1) ではノイズを足さず平均をそのまま出力する
            # (Algorithm 2 の3行目: t > 1 のときだけ z ~ N(0, I), それ以外は z = 0)
            return mean
        z = torch.randn_like(x_t)
        return mean + beta_t.sqrt() * z

    @torch.no_grad()
    def sample(self, model, shape, device, return_every=None):
        """
        Algorithm 2 (Sampling):

          1: x_T ~ N(0, I)                         ← 純粋なノイズから始める
          2: for t = T, ..., 1 do
          3:   z ~ N(0, I) if t > 1, else z = 0
          4:   x_{t-1} = 1/sqrt(α_t) (x_t - (1-α_t)/sqrt(1-ᾱ_t) ε_θ(x_t, t)) + σ_t z
          5: end for
          6: return x_0

        return_every を指定すると、その間隔で途中の x_t も記録して返す
        (勉強会で「ノイズから徐々に絵が浮かび上がる様子」を見せるため)。
        """
        x = torch.randn(shape, device=device)  # 1行目
        trajectory = [x.cpu()] if return_every else None

        for t_index in reversed(range(self.T)):  # 2行目: T-1, ..., 0
            x = self.p_sample(model, x, t_index)  # 3-4行目
            if return_every and (t_index % return_every == 0):
                trajectory.append(x.cpu())

        if return_every:
            return x, trajectory
        return x

"""
DDPM の拡散過程・学習・生成。
Ho et al., "Denoising Diffusion Probabilistic Models" (2020) https://arxiv.org/abs/2006.11239

model(x_t, t) はノイズ ε の予測を返すネットワークなら何でもよい。
コードの t は 0 始まり (論文の t=1 がコードの t=0)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(a, t, x_shape):
    """a[t] を x と掛け算できる形 (B, 1, 1, ...) にして返す。"""
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.T = T

        # 線形スケジュール (論文 4節)
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    def q_sample(self, x0, t, noise):
        """式(4): x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε"""
        return (
            extract(self.sqrt_alpha_bars, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alpha_bars, t, x0.shape) * noise
        )

    def loss(self, model, x0):
        """Algorithm 1 / 式(14) L_simple: || ε - ε_θ(x_t, t) ||^2"""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        return F.mse_loss(model(x_t, t), noise)

    @torch.no_grad()
    def p_sample(self, model, x_t, t_index):
        """
        x_t -> x_{t-1} (式(11), Algorithm 2 の4行目)。
        分散は σ_t^2 = β_t で固定。
        """
        B = x_t.shape[0]
        t = torch.full((B,), t_index, device=x_t.device, dtype=torch.long)

        beta_t = extract(self.betas, t, x_t.shape)
        alpha_t = extract(self.alphas, t, x_t.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape)

        pred_noise = model(x_t, t)
        mean = (x_t - beta_t / sqrt_one_minus_alpha_bar_t * pred_noise) / alpha_t.sqrt()

        if t_index == 0:
            return mean
        return mean + beta_t.sqrt() * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(self, model, shape, device, return_every=None):
        """Algorithm 2。return_every を指定すると途中の x_t も返す。"""
        x = torch.randn(shape, device=device)
        trajectory = [x.cpu()] if return_every else None

        for t_index in reversed(range(self.T)):
            x = self.p_sample(model, x, t_index)
            if return_every and (t_index % return_every == 0):
                trajectory.append(x.cpu())

        if return_every:
            return x, trajectory
        return x

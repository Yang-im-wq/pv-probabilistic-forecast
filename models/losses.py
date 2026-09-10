# -*- coding: utf-8 -*-
"""
losses.py —— 损失函数
- pinball_loss：分位数损失（Quantile Loss），概率预测的核心
- gaussian_nll：高斯负对数似然（DeepAR 风格，可选进阶，默认不使用）
"""
import torch


def pinball_loss(y_true, y_pred, quantiles):
    """
    分位数损失（Pinball / Quantile Loss）。

    参数
    ----
    y_true    : (batch, horizon) 或 (batch, horizon, 1)
    y_pred    : (batch, horizon, n_quantiles)
    quantiles : 形状 (n_quantiles,) 的张量，如 [0.1, 0.5, 0.9]

    原理（关键）
    ----------
    对分位数 q、真实值 y、预测值 ŷ：
        L_q(y, ŷ) = max( q * (y - ŷ), (q - 1) * (y - ŷ) )

    - 当预测偏低（y > ŷ）时，损失 ≈ q * (y - ŷ)；
    - 当预测偏高（y < ŷ）时，损失 ≈ (1 - q) * (ŷ - y)。

    因此 q=0.9 这条线"宁可偏高"——它要保证约 90% 的真实值落在自己下方；
    q=0.1 这条线"宁可偏低"。三条线合起来就框出一个预测区间。
    """
    if y_true.dim() == 2:
        y_true = y_true.unsqueeze(-1)                  # (batch, horizon, 1)
    q = quantiles.to(y_pred.device).view(1, 1, -1)     # (1, 1, n_quantiles)
    error = y_true - y_pred                            # (batch, horizon, n_quantiles)
    loss = torch.maximum(q * error, (q - 1.0) * error)
    return loss.mean()


def gaussian_nll(y_true, mu, sigma):
    """
    高斯负对数似然（DeepAR 风格，可选进阶，默认关闭）。

    模型输出高斯分布的参数 (μ, σ)，用负对数似然训练，σ 经 softplus 保证 >0。
    """
    sigma = torch.clamp(sigma, min=1e-6)
    return (torch.log(sigma) + 0.5 * ((y_true - mu) / sigma) ** 2).mean()

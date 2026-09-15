# -*- coding: utf-8 -*-
"""
calibrate.py —— 分位数事后校准（Conformal Quantile Regression）

问题：分位数回归模型输出的分位数不一定校准——比如 q50 的实际覆盖率可能是 72% 而不是 50%。
方法：在验证集上，对每个分位数 τ 算一个平移量 δ_τ = τ分位数(y - q̂_τ)，
      校准后 q̂_τ^cal = q̂_τ + δ_τ，使实际覆盖率对齐标称 τ。

用法：python calibrate.py
"""
import os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from data import preprocess
from models.lstm_quantile import LSTMQuantileForecaster

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_predictions():
    """在验证集和测试集上预测 11 个分位数，返回反归一化后的 (y_val, pred_val, y_test, pred_test)。"""
    data = preprocess.get_dataset()
    mean, scale = data["target_mean"], data["target_scale"]

    model = LSTMQuantileForecaster(
        data["X_test"].shape[-1], config.HIDDEN_SIZE, config.NUM_LAYERS,
        config.DROPOUT, config.HORIZON, config.NUM_QUANTILES, config.MODEL_TYPE)
    model.load_state_dict(torch.load(
        os.path.join(config.CHECKPOINT_DIR, "best_quantile.pt"), map_location="cpu"))
    model.eval()

    def predict(X):
        with torch.no_grad():
            p = model(torch.from_numpy(X).float()).numpy()   # (n, horizon, nq)
        return p[:, 0, :] * scale + mean                      # (n, nq) 反归一化

    pred_val = predict(data["X_val"])
    pred_test = predict(data["X_test"])
    y_val = data["y_val"][:, 0] * scale + mean
    y_test = data["y_test"][:, 0] * scale + mean
    return y_val, pred_val, y_test, pred_test


def coverage(y, pred, q_levels):
    """每个分位数的实际覆盖率 = P(y <= 预测分位数)。"""
    return np.array([(y <= pred[:, j]).mean() for j in range(len(q_levels))])


def fit_deltas(y_cal, pred_cal, q_levels):
    """在标定集上算每个分位数的平移量 δ_τ，并用 max-accumulate 保证单调。"""
    deltas = np.array([
        np.quantile(y_cal - pred_cal[:, j], q_levels[j])
        for j in range(len(q_levels))
    ])
    return np.maximum.accumulate(deltas)   # 单调不减，保证校准后分位数仍单调


def main():
    q_levels = np.asarray(config.QUANTILES)
    y_val, pred_val, y_test, pred_test = load_predictions()

    # 在验证集上拟合校准，应用到测试集
    deltas = fit_deltas(y_val, pred_val, q_levels)
    pred_test_cal = pred_test + deltas[None, :]

    cov_before = coverage(y_test, pred_test, q_levels)
    cov_after = coverage(y_test, pred_test_cal, q_levels)

    print("=" * 52)
    print("分位数  标称  校准前  校准后")
    print("=" * 52)
    for j, q in enumerate(q_levels):
        print("  tau={:.2f}  {:.3f}   {:.3f}".format(q, cov_before[j], cov_after[j]))
    err_before = float(np.mean(np.abs(cov_before - q_levels)))
    err_after = float(np.mean(np.abs(cov_after - q_levels)))
    print("=" * 52)
    print("平均校准误差(MAE)：校准前 {:.3f} -> 校准后 {:.3f}".format(err_before, err_after))
    print("平移量 deltas(kW)：{}".format(np.round(deltas, 0).tolist()))

    # 画校准曲线
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="完美校准(对角线)")
    ax.plot(q_levels, cov_before, "o-", color="red", lw=1.5, label="校准前")
    ax.plot(q_levels, cov_after, "s-", color="#1f77b4", lw=1.5, label="校准后")
    ax.set_xlabel("标称分位数 tau")
    ax.set_ylabel("实际覆盖率")
    ax.set_title("分位数校准前后对比")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(config.FIGURE_DIR, "calibration.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("图已保存 -> " + out)

    # 保存校准参数，供后续（比如 dro_demo / evaluate）复用
    np.savez(os.path.join(config.RESULT_DIR, "calibration_deltas.npz"),
             deltas=deltas, quantiles=q_levels)
    print("校准参数已保存 -> results/calibration_deltas.npz")


if __name__ == "__main__":
    main()

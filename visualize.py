# -*- coding: utf-8 -*-
"""
visualize.py —— 出图（全部保存为 300 dpi 高分辨率 PNG）

1) 预测区间图（核心）       prediction_interval.png
2) 典型波动日放大图          volatile_day.png
3) 训练损失曲线             training_loss.png
4) 预测值 vs 真实值散点图   scatter.png

用法：python visualize.py --mode quantile   （或 --mode point）
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")   # 无界面环境下也能出图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def setup_font():
    """尽量使用中文字体，避免中文标签显示为方框；找不到时回退英文。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_prediction_interval(y_true, median, lower, upper, time, save_path):
    """核心图：真实功率曲线 + q50 中位线 + q10–q90 半透明区间带。"""
    t = pd.to_datetime(time)
    n_show = min(len(y_true), 500)   # 画太多点会糊，最多取 500 个
    t, y_true = t[:n_show], y_true[:n_show]
    median, lower, upper = median[:n_show], lower[:n_show], upper[:n_show]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, y_true, label="真实功率", color="black", lw=1.2, zorder=3)
    ax.plot(t, median, label="预测中位线 (q50)", color="#1f77b4", lw=1.2, zorder=2)
    ax.fill_between(t, lower, upper, alpha=0.25, color="#1f77b4",
                    label="90% 预测区间 (q10–q90)")
    ax.set_xlabel("时间")
    ax.set_ylabel("光伏出力 (kW)")
    ax.set_title("光伏出力超短期概率预测 · 预测区间")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print("已保存 → {}".format(save_path))


def plot_volatile_day(y_true, median, lower, upper, time, save_path, n_days=2):
    """挑波动最剧烈（有云团遮挡）的日子放大展示，突出"不确定大时区间自动变宽"。"""
    t = pd.to_datetime(time)
    df = pd.DataFrame({"t": t, "y": y_true})
    df["date"] = df["t"].dt.date
    # 按天计算方差，挑方差最大的几天（波动最剧烈）
    day_var = df.groupby("date")["y"].var().sort_values(ascending=False)
    top_days = list(day_var.index[:n_days])

    fig, axes = plt.subplots(n_days, 1, figsize=(12, 3.5 * n_days))
    if n_days == 1:
        axes = [axes]
    for ax, day in zip(axes, top_days):
        mask = (df["date"] == day).to_numpy()
        tt, yy = t[mask], y_true[mask]
        mm, ll, uu = median[mask], lower[mask], upper[mask]
        ax.plot(tt, yy, label="真实功率", color="black", lw=1.2)
        ax.plot(tt, mm, label="q50 预测", color="#1f77b4", lw=1.2)
        ax.fill_between(tt, ll, uu, alpha=0.25, color="#1f77b4", label="90% 区间")
        ax.set_title("典型波动日：{}".format(day))
        ax.set_ylabel("功率 (kW)")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("时间")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print("已保存 → {}".format(save_path))


def plot_loss_curve(train, val, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train, label="训练损失", lw=1.5)
    ax.plot(val, label="验证损失", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("训练 / 验证损失曲线")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print("已保存 → {}".format(save_path))


def plot_scatter(y_true, pred, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, pred, s=8, alpha=0.4)
    lo = min(y_true.min(), pred.min())
    hi = max(y_true.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
    ax.set_xlabel("真实功率 (kW)")
    ax.set_ylabel("预测功率 (kW)")
    ax.set_title("预测值 vs 真实值")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print("已保存 → {}".format(save_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["point", "quantile"], default="quantile")
    args = parser.parse_args()

    setup_font()
    os.makedirs(config.FIGURE_DIR, exist_ok=True)

    pred_path = config.pred_path(args.mode)
    if not os.path.exists(pred_path):
        raise SystemExit("未找到预测结果：{}\n请先运行 python evaluate.py --mode {}".format(
            pred_path, args.mode))

    pred = np.load(pred_path, allow_pickle=True)
    y_true = pred["y_true"]
    time = pred["time"]

    if args.mode == "quantile":
        median, lower, upper = pred["median"], pred["lower"], pred["upper"]
        plot_prediction_interval(y_true, median, lower, upper, time,
                                 os.path.join(config.FIGURE_DIR, "prediction_interval.png"))
        plot_volatile_day(y_true, median, lower, upper, time,
                          os.path.join(config.FIGURE_DIR, "volatile_day.png"))
        plot_scatter(y_true, median,
                     os.path.join(config.FIGURE_DIR, "scatter_{}.png".format(args.mode)))
    else:
        plot_scatter(y_true, pred["pred"],
                     os.path.join(config.FIGURE_DIR, "scatter_{}.png".format(args.mode)))

    losses_path = config.losses_path(args.mode)
    if os.path.exists(losses_path):
        losses = np.load(losses_path, allow_pickle=True)
        plot_loss_curve(losses["train"], losses["val"],
                        os.path.join(config.FIGURE_DIR, "training_loss_{}.png".format(args.mode)))
    else:
        print("未找到损失曲线 {}，请先运行 train.py".format(losses_path))


if __name__ == "__main__":
    main()

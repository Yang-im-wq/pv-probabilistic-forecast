# -*- coding: utf-8 -*-
"""
vision/train_seg.py —— 训练 U-Net 云分割模型（自带合成数据 demo）

用法：python vision/train_seg.py

用「合成天空图」快速验证模型能学到云分割（CPU 即可，几分钟出结果）；
真实数据（SWIMSEG 天空图 / FY-4A 云图）下来后，替换 make_data 的数据源即可，
模型结构、训练、评估、可视化全部复用。
"""
import os
import json
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unet import UNet

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
SIZE = 128
N_TRAIN, N_VAL = 160, 40
EPOCHS = 15
BATCH = 8
SEED = 42


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def synthetic_sky(size=SIZE, rng=None):
    """生成一张合成天空图 + 云掩膜（用于验证模型；真实数据下来后替换此数据源即可）。"""
    if rng is None:
        rng = np.random.default_rng()
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    # 天空渐变：顶部偏蓝 → 底部偏白（地平线）
    t = yy / size
    img = np.stack([30 + 80 * t, 110 + 90 * t, 220 - 60 * t], axis=-1)  # H,W,3

    mask = np.zeros((size, size), dtype=np.float32)
    n_clusters = int(rng.integers(0, 4))  # 0~3 团云
    for _ in range(n_clusters):
        cx, cy = rng.integers(10, size - 10, 2)
        density = np.zeros((size, size), dtype=np.float32)
        for _ in range(int(rng.integers(2, 6))):  # 每团云由几个椭圆高斯斑拼成
            bx = cx + int(rng.integers(-20, 20))
            by = cy + int(rng.integers(-12, 12))
            rx = float(rng.integers(12, 30))
            ry = float(rng.integers(8, 20))
            d = ((xx - bx) / rx) ** 2 + ((yy - by) / ry) ** 2
            density += np.exp(-d / 2)
        density = np.clip(density, 0, 1.2) / 1.2
        bright = float(rng.uniform(0.5, 1.0))
        for ch in range(3):
            img[..., ch] = img[..., ch] * (1 - density * bright) + 255 * density * bright
        mask = np.maximum(mask, (density > 0.25).astype(np.float32))

    img = np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
    return img, mask


def make_data(n, rng):
    xs, ys = [], []
    for _ in range(n):
        img, mask = synthetic_sky(rng=rng)
        xs.append(img.transpose(2, 0, 1))   # C,H,W
        ys.append(mask[None])               # 1,H,W
    X = torch.from_numpy(np.stack(xs)).float() / 255.0
    Y = torch.from_numpy(np.stack(ys)).float()
    return X, Y


def batch_iou(pred, gt):
    """预测掩膜与真值的 IoU（按图平均）。pred/gt: (B,1,H,W) 0/1。"""
    inter = (pred * gt).sum(dim=(1, 2, 3))
    union = ((pred + gt) > 0).float().sum(dim=(1, 2, 3))
    return (inter / (union + 1e-6)).mean().item()


def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备：{}".format(device))

    rng = np.random.default_rng(SEED)
    print("生成合成数据 ...")
    Xtr, Ytr = make_data(N_TRAIN, rng)
    Xva, Yva = make_data(N_VAL, rng)
    print("训练 {} 张 / 验证 {} 张，尺寸 {}x{}".format(N_TRAIN, N_VAL, SIZE, SIZE))

    model = UNet(in_channels=3, out_channels=1, base=16).to(device)
    print("U-Net 参数量：{:,}".format(sum(p.numel() for p in model.parameters())))

    tr_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva, Yva), batch_size=BATCH, shuffle=False)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_iou, best_acc = 0.0, 0.0
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * xb.size(0)

        model.eval()
        ious, accs = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pm = (torch.sigmoid(model(xb)) > 0.5).float()
                ious.append(batch_iou(pm, yb) * xb.size(0))
                accs.append((pm == yb).float().mean().item() * xb.size(0))
        miou = sum(ious) / len(Yva)
        acc = sum(accs) / len(Yva)
        best_iou = max(best_iou, miou)
        best_acc = max(best_acc, acc)
        print("Epoch {:2d}/{} | loss {:.4f} | val IoU {:.3f} | acc {:.3f} | {:.1f}s".format(
            ep, EPOCHS, loss_sum / len(Ytr), miou, acc, time.time() - t0))

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "unet_cloud.pt"))
    metrics = {"val_iou": round(best_iou, 4), "val_acc": round(best_acc, 4), "epochs": EPOCHS}
    with open(os.path.join(OUT_DIR, "unet_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("\n最佳 IoU：{:.3f}，最佳 acc：{:.3f}".format(best_iou, best_acc))

    # 可视化：4 个验证样本的 原图 / 真值 / 预测
    model.eval()
    idx = np.random.default_rng(0).choice(len(Yva), 4, replace=False)
    fig, axes = plt.subplots(4, 3, figsize=(9, 12))
    with torch.no_grad():
        for r, i in enumerate(idx):
            img = Xva[i].numpy().transpose(1, 2, 0)
            gt = Yva[i, 0].numpy()
            pm = (torch.sigmoid(model(Xva[i:i + 1].to(device)))[0, 0].cpu().numpy() > 0.5).astype(np.float32)
            axes[r, 0].imshow(img); axes[r, 0].set_title("原图"); axes[r, 0].axis("off")
            axes[r, 1].imshow(gt, cmap="gray"); axes[r, 1].set_title("真值云掩膜"); axes[r, 1].axis("off")
            axes[r, 2].imshow(pm, cmap="gray"); axes[r, 2].set_title("预测云掩膜"); axes[r, 2].axis("off")
    fig.suptitle("U-Net 云分割结果（合成数据验证）")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "unet_seg_result.png"), dpi=200)
    plt.close(fig)
    print("结果图已保存 → {} / unet_seg_result.png".format(OUT_DIR))


if __name__ == "__main__":
    main()

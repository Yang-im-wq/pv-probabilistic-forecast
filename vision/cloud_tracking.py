# -*- coding: utf-8 -*-
"""
vision/cloud_tracking.py —— 云团运动追踪（视觉×能源 的核心增值）

核心思想：
  本地辐照度只能反映"此刻头顶的云"；卫星云图能提前看到"云团往哪飘"。
  本脚本用相邻两张卫星云图估计云团运动场（光流），
  再把"此刻云量"外推到未来时刻，预测"云会不会遮住电站"。

方法：
  1. 加载相邻两张 FY-4B 云图 (t0, t1)，裁剪目标区域（广东沿海）
  2. 云掩膜：亮度阈值（CLOD 云光学厚度图里云比晴空亮）
  3. 光流：Farneback 算法估计 t0→t1 的运动矢量场
  4. 外推：用运动场把 t0 的云掩膜"推"到 t1，得到预测掩膜
  5. 对比：预测掩膜 vs t1 实际掩膜（覆盖率 + IoU）

说明：这里用"光流"做追踪（无监督、CPU 可跑）；
      YOLOv5 检测+追踪是有 GPU/标注数据后的升级版，检测环节可无缝替换。
"""
import os
import glob

import numpy as np
import cv2
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fy_cloud

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = fy_cloud.DATA_DIR
OUT_DIR = fy_cloud.OUT_DIR


def load_region(path, bbox=None):
    """加载全圆盘云图，裁剪目标区域，返回 (区域RGB, 区域灰度, 使用的bbox)。

    bbox 传 None 时自动检测（只在第一张图用），后续图共用同一个 bbox，
    避免逐图检测的微小差异导致裁剪尺寸不一致（光流要求两张图同尺寸）。
    """
    arr = np.array(Image.open(path).convert("RGB"))
    if bbox is None:
        cx, cy, r = fy_cloud.detect_disk(arr)
        scale = np.arcsin(fy_cloud.RE / fy_cloud.RS) / r
        bbox = fy_cloud.region_bbox(cx, cy, scale)
    x0, x1, y0, y1 = bbox
    crop = arr[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return crop, gray, bbox


def optical_flow(prev_gray, next_gray):
    """Farneback 光流：估计 prev→next 的运动矢量场 (H, W, 2)。"""
    return cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0)


def advect(mask, flow):
    """用光流把掩膜从 t0 外推到 t1（cv2.remap 重映射）。"""
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "FY*.jpg")))
    if len(paths) < 2:
        print("至少需要两张连续云图")
        return

    # 取相邻两张（15 分钟间隔）
    p0, p1 = paths[0], paths[1]
    t0 = fy_cloud.parse_time(os.path.basename(p0))
    t1 = fy_cloud.parse_time(os.path.basename(p1))

    crop0, gray0, bbox = load_region(p0)
    crop1, gray1, _ = load_region(p1, bbox)

    mask0 = (gray0 > fy_cloud.CLOUD_GRAY_MIN).astype(np.float32)
    mask1 = (gray1 > fy_cloud.CLOUD_GRAY_MIN).astype(np.float32)

    flow = optical_flow(gray0, gray1)
    pred1 = advect(mask0, flow)          # 用 t0 + 光流 外推预测 t1
    pred1_bin = (pred1 > 0.5).astype(np.float32)

    # 量化
    frac0 = float(mask0.mean())
    frac_pred = float(pred1_bin.mean())
    frac1 = float(mask1.mean())
    inter = float((pred1_bin * mask1).sum())
    union = float(((pred1_bin + mask1) > 0).sum())
    iou = inter / union if union > 0 else 0.0
    flow_mag = np.hypot(flow[..., 0], flow[..., 1])

    print("=" * 56)
    print("云团运动追踪（{} → {}，15 分钟间隔）".format(t0.strftime("%H:%M"), t1.strftime("%H:%M")))
    print("=" * 56)
    print("云量：t0 {:.1%} → 预测 t1 {:.1%} vs 实际 t1 {:.1%}".format(frac0, frac_pred, frac1))
    print("预测掩膜与实际的 IoU = {:.3f}".format(iou))
    print("光流位移：平均 {:.2f} px，最大 {:.2f} px（本区域像素尺寸约 4km）".format(
        float(flow_mag.mean()), float(flow_mag.max())))
    print("  → 云团平均移动约 {:.1f} km / 15 分钟".format(float(flow_mag.mean()) * 4.0))

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].imshow(crop0); axes[0, 0].set_title("t0 原图"); axes[0, 0].axis("off")
    axes[0, 1].imshow(mask0, cmap="gray"); axes[0, 1].set_title("t0 云掩膜"); axes[0, 1].axis("off")
    axes[0, 2].imshow(crop0); axes[0, 2].set_title("云团运动场(光流)"); axes[0, 2].axis("off")
    step = 12
    h, w = flow.shape[:2]
    yy, xx = np.mgrid[step // 2:h:step, step // 2:w:step]
    axes[0, 2].quiver(xx, yy, flow[yy, xx, 0], flow[yy, xx, 1],
                      color="yellow", angles="xy", scale_units="xy",
                      scale=0.5, width=0.003, headwidth=3)

    axes[1, 0].imshow(pred1_bin, cmap="gray"); axes[1, 0].set_title("预测 t1 云掩膜(外推)"); axes[1, 0].axis("off")
    axes[1, 1].imshow(mask1, cmap="gray"); axes[1, 1].set_title("实际 t1 云掩膜"); axes[1, 1].axis("off")
    axes[1, 2].imshow(crop1); axes[1, 2].set_title("t1 原图"); axes[1, 2].axis("off")

    fig.suptitle("云团运动追踪：t0 光流外推预测 t1", fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "cloud_tracking.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("图已保存 -> " + out)


if __name__ == "__main__":
    main()

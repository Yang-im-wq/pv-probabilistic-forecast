# -*- coding: utf-8 -*-
"""
vision/cloud_detect.py —— 云检测 / 云量估计（视觉模块核心）

对天空图 / 卫星云图做"云 vs 天空"的像素级分割，输出：
  - 云掩膜（cloud mask，白=云）
  - 云量 cloud_fraction（天空区域内云像素占比，0~1）
  - 可视化（原图 + 掩膜 + 叠加）

方法说明：先用经典「颜色阈值法」跑通流水线——
  云 = 偏白（低饱和 + 高亮度，红蓝差≈0）
  天空 = 偏蓝（红蓝差明显 >0）
这是无监督、CPU 可跑、可解释的基线；后续可无缝替换为 YOLOv5 检测
或 U-Net 分割等深度模型（只需改 segment_sky_cloud 一个函数）。

用法：
    python vision/cloud_detect.py
"""
import os
import json
import glob

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体（与主项目 visualize.py 一致），避免标签显示为方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "output")

# 云/天空判别的颜色阈值（可按数据类型微调）
WHITE_SAT_MAX = 0.18     # 云：饱和度低于此值（白/灰）
CLOUD_GRAY_MIN = 120.0   # 云：亮度高于此值
SKY_RB_MIN = 0.05        # 天空：归一化红蓝差 (B-R)/(B+R) 高于此值（偏蓝）
SKY_GRAY_MIN = 40.0      # 天空：亮度高于此值（排除暗部地面）


def segment_sky_cloud(img_rgb):
    """把 RGB 图分成 云 / 天空 / 其它 三类，返回 (cloud, sky) 布尔掩膜。

    原理：蓝天空像素 B 明显大于 R（红蓝差大）；白云像素 R≈G≈B（低饱和、高亮度）。
    """
    R = img_rgb[..., 0].astype(np.float32)
    G = img_rgb[..., 1].astype(np.float32)
    B = img_rgb[..., 2].astype(np.float32)

    # 归一化红蓝差：>0 偏蓝，≈0 中性（白/灰）
    rb = (B - R) / (B + R + 1e-6)
    # 亮度
    gray = (R + G + B) / 3.0
    # 饱和度（HSV 中 S 的近似 = (max-min)/max）
    mx = np.maximum(np.maximum(R, G), B)
    mn = np.minimum(np.minimum(R, G), B)
    sat = (mx - mn) / (mx + 1e-6)

    cloud = (sat < WHITE_SAT_MAX) & (gray > CLOUD_GRAY_MIN)
    sky = (rb > SKY_RB_MIN) & (gray > SKY_GRAY_MIN)
    return cloud, sky


def cloud_fraction(cloud, sky):
    """云量 = 云像素 / (云 + 天空) 像素，只在天区域内统计（排除地面等）。"""
    denom = int(cloud.sum() + sky.sum())
    if denom == 0:
        return 0.0
    return float(cloud.sum() / denom)


def visualize(img_rgb, cloud, sky, out_path, title=""):
    """三面板可视化：原图 / 分割掩膜 / 叠加。"""
    H, W = img_rgb.shape[:2]
    mask = np.zeros((H, W, 3), dtype=np.uint8)
    mask[sky] = [90, 160, 220]    # 天空 → 蓝
    mask[cloud] = [255, 255, 255]  # 云 → 白
    overlay = img_rgb.copy()
    overlay[cloud] = (overlay[cloud] * 0.3 + np.array([255, 60, 60]) * 0.7).astype(np.uint8)  # 云 → 红

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb); axes[0].set_title("原图"); axes[0].axis("off")
    axes[1].imshow(mask); axes[1].set_title("分割结果 (蓝=天空 白=云)"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title("云叠加 (红=云)"); axes[2].axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")) + glob.glob(os.path.join(DATA_DIR, "*.png")))
    if not paths:
        print("未在 {} 找到图片".format(DATA_DIR))
        return

    results = {}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        img = np.asarray(Image.open(p).convert("RGB"))
        cloud, sky = segment_sky_cloud(img)
        frac = cloud_fraction(cloud, sky)
        results[name] = round(frac, 4)
        print("{:<16s} 云量 = {:.2%}  (云像素 {} / 天空像素 {})".format(
            name, frac, int(cloud.sum()), int(sky.sum())))
        visualize(img, cloud, sky,
                  os.path.join(OUT_DIR, "{}_seg.png".format(name)),
                  title="{}  云量 {:.1%}".format(name, frac))

    with open(os.path.join(OUT_DIR, "cloud_fraction.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n结果已保存 → {} / cloud_fraction.json".format(OUT_DIR))


if __name__ == "__main__":
    main()

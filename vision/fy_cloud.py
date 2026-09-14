# -*- coding: utf-8 -*-
"""
vision/fy_cloud.py —— 处理风云四号全圆盘云图（批量 → 云量时间序列）

流程：加载多张 JPG → 定位地球圆盘 → 地理投影(经纬度→像素) → 裁剪目标区域
      → 云检测 → 每张算云量 → 按文件名里的时间戳排序 → 输出时间序列

输出：
  - vision/output/cloud_cover.csv    云量时间序列（time, cloud_fraction），可直接接 merge_cloud
  - vision/output/cloud_series.png   云量变化曲线
  - vision/output/<name>_region.png  每张图的区域裁剪 + 云掩膜（可视化）

用法：把 FY-4 云图 JPG 放进 vision/data/（文件名以 fy 开头），然后
      python vision/fy_cloud.py
"""
import os
import re
import glob
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "output")

# ---- 风云四号 B 星（FY-4B）投影参数 ----
SAT_LON = 105.0            # 卫星星下点经度（FY-4B 在 105°E）
RE = 6378.137              # 地球赤道半径 (km)
H = 35786.0                # 卫星高度 (km)
RS = RE + H                # 卫星到地心距离 (km)

# 目标区域（默认广东沿海，可改）
REGION = {"name": "广东沿海", "lon": (108.0, 118.0), "lat": (20.0, 26.0)}

CLOUD_GRAY_MIN = 90.0      # 云检测：亮度阈值（CLOD 图里云比晴空亮）


def detect_disk(arr):
    """定位地球圆盘：返回 (中心x, 中心y, 半径)。"""
    gray = arr.mean(axis=2)
    ys, xs = np.where(gray > 20)
    xmin, xmax = xs.min(), xs.max()
    cx = (xmin + xmax) / 2.0
    r = (xmax - xmin) / 2.0
    cy = ys.min() + r
    return cx, cy, r


def lonlat_to_pixel(lon, lat, cx, cy, scale):
    """地理投影：经纬度 → 像素坐标（静止卫星视角，北向上、东向右）。"""
    lon = np.radians(lon) - np.radians(SAT_LON)
    lat = np.radians(lat)
    c_lat, s_lat = np.cos(lat), np.sin(lat)
    c_lon, s_lon = np.cos(lon), np.sin(lon)
    vx = RE * c_lat * c_lon - RS
    vy = RE * c_lat * s_lon
    vz = RE * s_lat
    x_ang = np.arctan2(vy, -vx)
    y_ang = np.arctan2(vz, np.sqrt(vx * vx + vy * vy))
    return cx + x_ang / scale, cy - y_ang / scale


def parse_time(filename):
    """从文件名里提取观测时间（第一个 14 位时间戳，如 20260914031500）。"""
    m = re.findall(r"(\d{14})", filename)
    return pd.to_datetime(m[0], format="%Y%m%d%H%M%S") if m else None


def region_bbox(cx, cy, scale):
    """目标区域四个角 → 像素包围盒。"""
    lon0, lon1 = REGION["lon"]
    lat0, lat1 = REGION["lat"]
    px, py = lonlat_to_pixel(np.array([lon0, lon1, lon1, lon0]),
                             np.array([lat0, lat1, lat0, lat1]), cx, cy, scale)
    return int(px.min()), int(px.max()), int(py.min()), int(py.max())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(set(glob.glob(os.path.join(DATA_DIR, "fy*.jpg")) + glob.glob(os.path.join(DATA_DIR, "fy*.JPG"))))
    if not paths:
        print("未在 {} 找到风云云图（文件名以 fy 开头）".format(DATA_DIR))
        return

    records = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        arr = np.array(Image.open(p).convert("RGB"))
        cx, cy, r = detect_disk(arr)
        scale = np.arcsin(RE / RS) / r
        x0, x1, y0, y1 = region_bbox(cx, cy, scale)
        crop = arr[y0:y1, x0:x1]
        cloud = crop.mean(axis=2) > CLOUD_GRAY_MIN
        frac = cloud.mean()
        t = parse_time(name)
        records.append((t, frac))
        print("{:<20s} | 时间 {} | 云量 {:.1%}".format(
            name, t.strftime("%m-%d %H:%M") if t is not None else "?", frac))

        # 每张图的可视化（可选，图多了会占空间；想省时间可注释掉）
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        from matplotlib.patches import Rectangle
        axes[0].imshow(arr)
        axes[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", lw=2))
        axes[0].set_title("全圆盘 + 区域框"); axes[0].axis("off")
        axes[1].imshow(crop); axes[1].set_title(REGION["name"]); axes[1].axis("off")
        axes[2].imshow(cloud, cmap="gray"); axes[2].set_title("云掩膜 {:.1%}".format(frac)); axes[2].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "{}_region.png".format(name)), dpi=150)
        plt.close(fig)

    # 按时间排序，输出时间序列
    records = [r for r in records if r[0] is not None]
    records.sort(key=lambda r: r[0])
    if len(records) >= 2:
        df = pd.DataFrame(records, columns=["time", "cloud_fraction"])
        csv_path = os.path.join(OUT_DIR, "cloud_cover.csv")
        df.to_csv(csv_path, index=False)
        print("\n已保存云量时间序列 → {}（共 {} 个时次）".format(csv_path, len(df)))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["time"], df["cloud_fraction"], "o-", lw=1.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel("时间"); ax.set_ylabel("云量")
        ax.set_title("{} 云量时间序列".format(REGION["name"]))
        ax.grid(alpha=0.3)
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "cloud_series.png"), dpi=200)
        plt.close(fig)
        print("已保存云量曲线 → {} / cloud_series.png".format(OUT_DIR))
    else:
        print("\n注意：图少于 2 张，无法画时间序列。等更多图下载后再跑一次。")


if __name__ == "__main__":
    main()

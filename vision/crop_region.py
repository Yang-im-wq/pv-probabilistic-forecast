# -*- coding: utf-8 -*-
"""vision/crop_region.py —— 区域裁剪：全圆盘 L1+CLM → 广东沿海局部 → YOLO 数据集。

解决全圆盘 2748×2748 压到训练尺寸后云团 <3px 的小目标问题：
裁剪广东沿海 256×256 窗口，云团 ~13px，训练时放大到 ~30px，可被 YOLO 检测。

流程：
  1. 地理投影定位广东沿海中心 (113E, 23N) 的像素坐标
  2. 按时间戳配对 L1 HDF 与 CLM NC
  3. 裁 256×256 窗口：L1 Ch02 灰度图（输入）+ CLM 云掩膜（标注）
  4. CLM 掩膜 → 连通域 → YOLO 框 → 组织成 images/labels + data.yaml

用法：
  python vision/crop_region.py --l1-dir data/l1 --clm-dir data/clm --out-dir vision/cloud_dataset_crop
"""
import os
import re
import glob
import argparse

import numpy as np
import h5py
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_yolo_dataset import mask_to_yolo_boxes, write_yolo_label

# ---- FY-4B 4km 全圆盘 NOM 投影参数 ----
RE = 6378.137
H = 35786.0
RS = RE + H
SAT_LON = 105.0
R = 1374                 # 视盘半径（像素）
CX = CY = 1373.5         # 网格中心
SCALE = np.arcsin(RE / RS) / R

REGION_LON, REGION_LAT = 113.0, 23.0   # 广东沿海中心
WIN = 256                              # 裁剪窗口边长（像素）


def ll2px(lon, lat):
    """经纬度 → 全圆盘 NOM 网格像素坐标。"""
    lon = np.radians(lon) - np.radians(SAT_LON)
    lat = np.radians(lat)
    c_lat, s_lat = np.cos(lat), np.sin(lat)
    c_lon, s_lon = np.cos(lon), np.sin(lon)
    vx = RE * c_lat * c_lon - RS
    vy = RE * c_lat * s_lon
    vz = RE * s_lat
    x_ang = np.arctan2(vy, -vx)
    y_ang = np.arctan2(vz, np.sqrt(vx * vx + vy * vy))
    return CX + x_ang / SCALE, CY - y_ang / SCALE


def time_of(filename):
    m = re.findall(r"(\d{14})", filename)
    return m[0] if m else None


def l1_ch12(hdf_path):
    """读 L1 Ch12(12μm 长波红外)，填充值→0，反转映射：冷云→亮、暖晴空→暗。

    红外亮温 DN 约 1500(冷云)~4000(暖晴空)，反转后云显示为亮斑，全天候可见。
    """
    with h5py.File(hdf_path, "r") as f:
        dn = f["Data/NOMChannel12"][:].astype(np.float32)
    dn[dn == 65535] = 0.0
    # 云 DN~2100-2500(冷)，晴空 DN~2800-3800(暖)；反转聚焦 [1600,3200] 使云亮、晴空暗
    return np.clip((3200.0 - dn) / 1600.0 * 255.0, 0, 255).astype(np.uint8)


def read_clm_mask(path):
    """读 CLM，云(1/2/3)=1，其它=0。"""
    with h5py.File(path, "r") as f:
        clm = f["CLM"][:]
    return ((clm >= 1) & (clm <= 3)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="区域裁剪 L1+CLM -> YOLO 数据集")
    ap.add_argument("--l1-dir", default="data/l1")
    ap.add_argument("--clm-dir", default="data/clm")
    ap.add_argument("--out-dir", default="vision/cloud_dataset_crop")
    ap.add_argument("--win", type=int, default=WIN, help="裁剪窗口边长")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--min-area", type=int, default=16, help="云团最小面积(像素)")
    args = ap.parse_args()

    # 窗口像素范围
    cx_px, cy_px = ll2px(REGION_LON, REGION_LAT)
    half = args.win // 2
    x0, x1 = int(round(cx_px)) - half, int(round(cx_px)) + half
    y0, y1 = int(round(cy_px)) - half, int(round(cy_px)) + half
    print("窗口中心 (%.1fE, %.1fN) -> 像素 (%.1f, %.1f)" % (REGION_LON, REGION_LAT, cx_px, cy_px))
    print("裁剪范围 x[%d:%d] y[%d:%d] (%dx%d)" % (x0, x1, y0, y1, args.win, args.win))

    # L1 时间戳映射
    l1_map = {}
    for p in glob.glob(os.path.join(args.l1_dir, "*L1*FDI*.HDF")):
        t = time_of(os.path.basename(p))
        if t:
            l1_map[t] = p

    clm_paths = sorted(glob.glob(os.path.join(args.clm_dir, "*.NC")))
    if not clm_paths:
        raise SystemExit("没找到 CLM：" + args.clm_dir)

    n_val = max(1, int(len(clm_paths) * args.val_ratio))
    splits = [("train", clm_paths[:-n_val]), ("val", clm_paths[-n_val:])]

    paired = empty = 0
    for split, paths in splits:
        img_out = os.path.join(args.out_dir, "images", split)
        lbl_out = os.path.join(args.out_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for cp in paths:
            name = os.path.splitext(os.path.basename(cp))[0]
            t = time_of(name)
            if t not in l1_map:
                print("  [跳过] 无 L1：", name)
                continue

            try:
                img = l1_ch12(l1_map[t])
                mask = read_clm_mask(cp)
            except Exception as e:
                print("  [跳过] 文件损坏:", name, str(e)[:60])
                continue
            c_img = img[y0:y1, x0:x1]
            c_mask = mask[y0:y1, x0:x1]
            boxes = mask_to_yolo_boxes(c_mask, min_area=args.min_area)

            Image.fromarray(c_img).save(os.path.join(img_out, name + ".png"))
            write_yolo_label(boxes, os.path.join(lbl_out, name + ".txt"))
            paired += 1
            if not boxes:
                empty += 1
            print("  [{}] {} -> {} 个云框".format(split, name, len(boxes)))

    yaml_path = os.path.join(args.out_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# YOLOv5 云检测数据集（广东沿海 256x256 裁剪 + CLM 官方标注）\n")
        f.write("path: {}\n".format(os.path.abspath(args.out_dir)))
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['cloud']\n")

    print("\n数据集生成 -> " + args.out_dir)
    print("成功配对 {} 个时次，其中 {} 个无云(空标注)".format(paired, empty))


if __name__ == "__main__":
    main()

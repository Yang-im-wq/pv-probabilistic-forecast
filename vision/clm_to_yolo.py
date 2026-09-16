# -*- coding: utf-8 -*-
"""
vision/clm_to_yolo.py —— 用 CLM 云掩膜做标注，生成 YOLOv5 数据集

CLM 云检测产品（.NC）里的 CLM 变量编码：
  0 = 晴空，1/2/3 = 云，126/127 = 填充值（空间外/无效）

流程：
  1. 读 CLM .NC → 云掩膜（CLM ∈ {1,2,3}）
  2. 掩膜 → 连通域 → YOLO 框
  3. 按时间戳配对 L1 可见光图像（输入）→ 输出 YOLOv5 数据集

用法：
  python vision/clm_to_yolo.py --clm-dir data/clm --image-dir <L1图像目录> --out-dir <输出>
"""
import os
import re
import argparse
import glob
import shutil

import numpy as np
import cv2
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_yolo_dataset import mask_to_yolo_boxes, write_yolo_label

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def read_clm_mask(path):
    """读 CLM .NC，返回云掩膜（1=云，0=非云）。"""
    import h5py
    with h5py.File(path, "r") as f:
        clm = f["CLM"][:]
    # 云 = 1/2/3；晴空 = 0；填充 126/127 也归为非云（后续会被排除在框外）
    return ((clm >= 1) & (clm <= 3)).astype(np.uint8)


def time_of(filename):
    """从文件名提取 14 位时间戳（起始时间），返回字符串。"""
    m = re.findall(r"(\d{14})", filename)
    return m[0] if m else None


def main():
    ap = argparse.ArgumentParser(description="CLM 云掩膜 -> YOLOv5 数据集")
    ap.add_argument("--clm-dir", required=True, help="CLM .NC 所在目录")
    ap.add_argument("--image-dir", required=True, help="对应 L1 图像目录（同时间戳）")
    ap.add_argument("--out-dir", required=True, help="输出 YOLOv5 数据集目录")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    args = ap.parse_args()

    clm_paths = sorted(glob.glob(os.path.join(args.clm_dir, "*.NC")))
    if not clm_paths:
        raise SystemExit("没找到 CLM .NC 文件：" + args.clm_dir)

    # 时间戳 -> L1 图像文件 的映射
    img_map = {}
    for ip in glob.glob(os.path.join(args.image_dir, "*")):
        t = time_of(os.path.basename(ip))
        if t:
            img_map[t] = ip

    n_val = max(1, int(len(clm_paths) * args.val_ratio))
    splits = [("train", clm_paths[:-n_val]), ("val", clm_paths[-n_val:])]

    paired = 0
    for split, paths in splits:
        img_out = os.path.join(args.out_dir, "images", split)
        lbl_out = os.path.join(args.out_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for cp in paths:
            name = os.path.splitext(os.path.basename(cp))[0]
            t = time_of(name)
            mask = read_clm_mask(cp)
            boxes = mask_to_yolo_boxes(mask)
            write_yolo_label(boxes, os.path.join(lbl_out, name + ".txt"))

            # 找同时间戳的 L1 图像，拷贝过去
            if t and t in img_map:
                src = img_map[t]
                dst = os.path.join(img_out, name + os.path.splitext(src)[1])
                shutil.copy(src, dst)
                paired += 1
            else:
                print("  [警告] 没找到时间戳 {} 的 L1 图像，跳过图像拷贝".format(t))

            print("  [{}] {} -> {} 个云框".format(split, name, len(boxes)))

    yaml_path = os.path.join(args.out_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# YOLOv5 云检测数据集（CLM 官方云掩膜标注）\n")
        f.write("path: {}\n".format(os.path.abspath(args.out_dir)))
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['cloud']\n")

    print("\n数据集已生成 -> " + args.out_dir)
    print("成功配对 {} 个时次（有 L1 图像 + CLM 标注）".format(paired))
    print("data.yaml -> " + yaml_path)


if __name__ == "__main__":
    main()

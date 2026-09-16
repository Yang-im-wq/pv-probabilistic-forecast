# -*- coding: utf-8 -*-
"""
vision/l1_to_yolo.py —— 把 FY-4B L1 HDF 转成 YOLOv5 训练数据

流程（全自动）：
  1. 读 L1 HDF，提取可见光波段（默认 Ch02 0.65μm）→ 归一化成 8bit 灰度图
  2. 阈值分割生成"伪云掩膜"（可见光波段里亮=云）→ 轮廓 → YOLO 框
  3. 组织成 YOLOv5 数据集结构（images/labels + data.yaml）

注：伪标注是"弱标注"（阈值法），用于快速跑通全流程；
     想要权威标注，之后用 CLM 云检测产品替换 mask 来源即可。

用法：
  python vision/l1_to_yolo.py --hdf-dir "C:/Users/Lenovo/Downloads" --out-dir <输出目录>
"""
import os
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

CLASS_NAME = "cloud"


def l1_to_image(hdf_path, channel=2):
    """读 L1 HDF，提取某个波段，用固定 DN 范围(0~2000)归一化到 0~255 灰度图。"""
    import h5py
    with h5py.File(hdf_path, "r") as f:
        dn = f["Data/NOMChannel%02d" % channel][:].astype(np.float32)
    # 固定范围归一化（不是百分位，百分位会被太空背景带偏）；DN>2000 饱和到 255
    return np.clip(dn / 2000.0 * 255.0, 0, 255).astype(np.uint8)


def cloud_mask_from_visible(img, thresh=190):
    """可见光波段里"亮=云" → 云掩膜（伪标注，弱标注）。thresh 对应 DN≈1500。"""
    return (img > thresh).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="L1 HDF -> YOLOv5 数据集")
    ap.add_argument("--hdf-dir", required=True, help="L1 HDF 所在目录")
    ap.add_argument("--out-dir", required=True, help="输出 YOLOv5 数据集目录")
    ap.add_argument("--channel", type=int, default=2, help="用哪个波段(默认2=可见光0.65μm)")
    ap.add_argument("--thresh", type=int, default=180, help="云检测亮度阈值")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    args = ap.parse_args()

    hdf_paths = sorted(glob.glob(os.path.join(args.hdf_dir, "*L1*FDI*.HDF")))
    if not hdf_paths:
        raise SystemExit("没找到 L1 FDI 的 HDF 文件：" + args.hdf_dir)

    n_val = max(1, int(len(hdf_paths) * args.val_ratio))
    val_paths = hdf_paths[-n_val:]
    train_paths = hdf_paths[:-n_val]

    for split, paths in [("train", train_paths), ("val", val_paths)]:
        img_out = os.path.join(args.out_dir, "images", split)
        lbl_out = os.path.join(args.out_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for hp in paths:
            name = os.path.splitext(os.path.basename(hp))[0]
            img = l1_to_image(hp, args.channel)
            mask = cloud_mask_from_visible(img, args.thresh)
            boxes = mask_to_yolo_boxes(mask)

            Image.fromarray(img).save(os.path.join(img_out, name + ".png"))
            write_yolo_label(boxes, os.path.join(lbl_out, name + ".txt"))

            # 顺便存一张"云掩膜"预览（方便看效果）
            mask_png = os.path.join(args.out_dir, "mask_preview", name + "_mask.png")
            os.makedirs(os.path.dirname(mask_png), exist_ok=True)
            Image.fromarray(mask * 255).save(mask_png)

            print("  [{}] {} -> {} 个云框".format(split, name, len(boxes)))

    # data.yaml
    yaml_path = os.path.join(args.out_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# YOLOv5 云检测数据集（FY-4B L1 可见光波段 + 阈值伪标注）\n")
        f.write("path: {}\n".format(os.path.abspath(args.out_dir)))
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['{}']\n".format(CLASS_NAME))

    print("\n数据集已生成 -> " + args.out_dir)
    print("data.yaml -> " + yaml_path)
    print("\n训练命令（在你的 YOLOv5 目录，如 D:\\yolov-test）：")
    print("  python train.py --data {} --weights yolov5s.pt --epochs 100 --img 640 --batch 16".format(yaml_path))


if __name__ == "__main__":
    main()

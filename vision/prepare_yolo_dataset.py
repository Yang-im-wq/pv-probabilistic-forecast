# -*- coding: utf-8 -*-
"""
vision/prepare_yolo_dataset.py —— 把云掩膜(CLM)转成 YOLOv5 训练数据

用途：YOLOv5 训练需要"图像 + 云的 bounding box 标注"。
     FY-4A 的 CLM（云检测）产品已经标好"哪里是云"，本脚本把云掩膜转成
     YOLO 格式的框（class_id cx cy w h，归一化 0~1），并组织成 YOLOv5 数据集结构。

流程：
  1. 读云掩膜（支持 PNG/TIF 图像，或原始 HDF 用 h5py 读）
  2. 找连通域（云团）→ 每个云团一个 bounding box
  3. 写 YOLO txt 标注
  4. 组织成 YOLOv5 的 images/labels + data.yaml

用法：
  python vision/prepare_yolo_dataset.py \
      --mask-dir /path/to/masks --image-dir /path/to/images --out-dir /path/to/dataset
"""
import os
import argparse
import glob

import numpy as np
import cv2
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CLASS_NAME = "cloud"          # 单一类别：云


def read_mask(path):
    """读云掩膜。支持 PNG/JPG/TIF（0=晴空，>0=云）；HDF 用 h5py（可选）。"""
    if path.lower().endswith((".hdf", ".h5", ".nc", ".hdf5")):
        try:
            import h5py
            with h5py.File(path, "r") as f:
                # CLM 数据的具体字段名因版本而异，这里打印并取第一个 2D 数组
                def find_2d(name, obj):
                    return name if (hasattr(obj, "shape") and len(obj.shape) == 2) else None
                keys = []
                f.visititems(lambda n, o: keys.append(n) if find_2d(n, o) else None)
                if not keys:
                    raise ValueError("HDF 里没有 2D 数组，请先确认 CLM 字段名")
                arr = f[keys[0]][:]
                print("  读 HDF 字段 {}，形状 {}".format(keys[0], arr.shape))
                return arr
        except ImportError:
            raise SystemExit("读 HDF 需要 h5py：pip install h5py")
    # 普通图像：转灰度，>0 视为云
    img = np.array(Image.open(path).convert("L"))
    return (img > 10).astype(np.uint8)   # 灰度阈值，云掩膜里非零即云


def mask_to_yolo_boxes(mask, min_area=100):
    """云掩膜 -> YOLO 框列表 [(cx, cy, w, h), ...]，归一化 0~1。"""
    mask_bin = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = mask_bin.shape
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:     # 过滤太小的噪点
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append(((x + w / 2.0) / W, (y + h / 2.0) / H, w / W, h / H))
    return boxes


def write_yolo_label(boxes, out_path, class_id=0):
    with open(out_path, "w") as f:
        for cx, cy, bw, bh in boxes:
            f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(class_id, cx, cy, bw, bh))


def main():
    ap = argparse.ArgumentParser(description="云掩膜 -> YOLOv5 数据集")
    ap.add_argument("--mask-dir", required=True, help="云掩膜所在目录")
    ap.add_argument("--image-dir", required=True, help="对应云图所在目录（文件名需一致）")
    ap.add_argument("--out-dir", required=True, help="输出 YOLOv5 数据集目录")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    args = ap.parse_args()

    mask_paths = sorted(glob.glob(os.path.join(args.mask_dir, "*")))
    if not mask_paths:
        raise SystemExit("mask 目录为空：" + args.mask_dir)

    # 划分 train / val
    n_val = max(1, int(len(mask_paths) * args.val_ratio))
    val_paths = mask_paths[-n_val:]
    train_paths = mask_paths[:-n_val]

    # 输出目录结构
    for split, paths in [("train", train_paths), ("val", val_paths)]:
        img_out = os.path.join(args.out_dir, "images", split)
        lbl_out = os.path.join(args.out_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for mp in paths:
            name = os.path.splitext(os.path.basename(mp))[0]
            mask = read_mask(mp)
            boxes = mask_to_yolo_boxes(mask)
            write_yolo_label(boxes, os.path.join(lbl_out, name + ".txt"))

            # 拷贝对应图像（文件名一致）
            img_src = None
            for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                cand = os.path.join(args.image_dir, name + ext)
                if os.path.exists(cand):
                    img_src = cand
                    break
            if img_src:
                import shutil
                shutil.copy(img_src, os.path.join(img_out, os.path.basename(img_src)))
            else:
                print("  [警告] 没找到对应图像：{}".format(name))

        print("  [{}] {} 张".format(split, len(paths)))

    # 写 data.yaml
    yaml_path = os.path.join(args.out_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# YOLOv5 云检测数据集\n")
        f.write("path: {}\n".format(os.path.abspath(args.out_dir)))
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['{}']\n".format(CLASS_NAME))
    print("\n数据集已生成 -> " + args.out_dir)
    print("data.yaml -> " + yaml_path)
    print("\n接下来在你的 YOLOv5 目录训练：")
    print("  python train.py --data {} --weights yolov5s.pt --epochs 100 --img 640 --batch 16".format(
        yaml_path))


if __name__ == "__main__":
    main()

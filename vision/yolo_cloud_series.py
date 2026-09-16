# -*- coding: utf-8 -*-
"""vision/yolo_cloud_series.py —— 用训练好的 YOLOv5 批量检测云团 → 云量时间序列。

流程：
  1. 加载 exp7 best.pt（红外 Ch12 云团检测模型）
  2. 遍历广东沿海裁剪图（train+val 全部 346 时次）
  3. 每张：YOLO 检测云团 → 云量 = 检测框并集面积 / 图面积
     同时读 CLM 真值框算「真值云量」做对比
  4. 按时间排序，输出 CSV（time, cloud_fraction_yolo, cloud_fraction_true, n_clouds）
     可直接被 data/merge_cloud.py 的 load_cloud_series 读入

用法（用 yolov5 环境）：
  D:\\anaconda\\envs\\yolov5\\python.exe vision/yolo_cloud_series.py
"""
import os
import re
import glob
import sys

import numpy as np
import cv2
import torch
import pandas as pd

sys.path.insert(0, "D:/yolov-test")
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from utils.torch_utils import select_device

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

WEIGHTS = "D:/yolov-test/runs/cloud_test/exp7/weights/best.pt"
IMG_DIR = "D:/pv-probabilistic-forecast/vision/cloud_dataset_crop_ir/images"
LBL_DIR = "D:/pv-probabilistic-forecast/vision/cloud_dataset_crop_ir/labels"
OUT_CSV = "D:/pv-probabilistic-forecast/vision/output/cloud_cover_yolo.csv"
OUT_PNG = "D:/pv-probabilistic-forecast/vision/output/cloud_series_yolo.png"


def predict(model, device, stride, img0_bgr, conf=0.25):
    img = letterbox(img0_bgr, 640, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    im = torch.from_numpy(img).to(device).float() / 255.0
    im = im[None]
    with torch.no_grad():
        pred = model(im)
    pred = non_max_suppression(pred, conf, 0.45, None, False, max_det=500)
    det = pred[0]
    if det is not None and len(det):
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0_bgr.shape).round()
        return det.cpu().numpy()
    return np.zeros((0, 6))


def boxes_to_fraction(boxes, H, W):
    """框并集面积 / 图面积 = 云量近似。"""
    mask = np.zeros((H, W), dtype=np.uint8)
    for (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        mask[y1:y2, x1:x2] = 1
    return float(mask.mean())


def read_true_boxes(txt, W, H):
    boxes = []
    with open(txt) as f:
        for line in f:
            _, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            boxes.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
    return boxes


def main():
    device = select_device("")
    model = DetectMultiBackend(WEIGHTS, device=device)
    model.eval()
    stride = model.stride

    paths = sorted(glob.glob(os.path.join(IMG_DIR, "*", "*.png")))
    print("共 %d 张裁剪图" % len(paths))

    records = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        split = os.path.basename(os.path.dirname(p))
        ts = re.findall(r"(\d{14})", name)[0]
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g.ndim == 3:
            g = g[:, :, 0]
        H, W = g.shape
        det = predict(model, device, stride, cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
        pred_boxes = [(d[0], d[1], d[2], d[3]) for d in det]
        frac_yolo = boxes_to_fraction(pred_boxes, H, W)
        true_boxes = read_true_boxes(os.path.join(LBL_DIR, split, name + ".txt"), W, H)
        frac_true = boxes_to_fraction(true_boxes, H, W)
        records.append((ts, frac_yolo, frac_true, len(pred_boxes), len(true_boxes)))

    records.sort()
    df = pd.DataFrame(records, columns=["time", "cloud_fraction", "cloud_fraction_true",
                                        "n_clouds", "n_clouds_true"])
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S")
    df.to_csv(OUT_CSV, index=False)
    print("云量序列已保存 ->", OUT_CSV)

    # 相关系数（YOLO 云量 vs 真值云量）
    corr = df["cloud_fraction"].corr(df["cloud_fraction_true"])
    mae = (df["cloud_fraction"] - df["cloud_fraction_true"]).abs().mean()
    print("YOLO vs CLM真值：相关系数 %.3f，平均绝对误差 %.3f" % (corr, mae))
    print("云量均值：YOLO %.1f%% vs 真值 %.1f%%" % (df['cloud_fraction'].mean()*100, df['cloud_fraction_true'].mean()*100))

    # 曲线
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["time"], df["cloud_fraction_true"], "o-", ms=3, lw=1, alpha=0.7, label="CLM 真值云量")
    ax.plot(df["time"], df["cloud_fraction"], "o-", ms=3, lw=1, alpha=0.7, label="YOLO 检测云量")
    ax.set_ylim(0, 1)
    ax.set_ylabel("云量")
    ax.set_xlabel("时间")
    ax.set_title("广东沿海云量时间序列（09-01~09-16，YOLO 检测 vs CLM 真值，r=%.3f）" % corr)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print("曲线已保存 ->", OUT_PNG)


if __name__ == "__main__":
    main()

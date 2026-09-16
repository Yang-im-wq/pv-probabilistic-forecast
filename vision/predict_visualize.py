# -*- coding: utf-8 -*-
"""用训练好的 best.pt 检测云团，画框，与 CLM 官方真值对比可视化。"""
import os
import glob
import sys

import numpy as np
import cv2
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "D:/yolov-test")
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from utils.torch_utils import select_device

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

WEIGHTS = "D:/yolov-test/runs/cloud_test/exp5/weights/best.pt"
IMG_DIR = "D:/pv-probabilistic-forecast/vision/cloud_dataset_crop/images/val"
LBL_DIR = "D:/pv-probabilistic-forecast/vision/cloud_dataset_crop/labels/val"
OUT = "D:/pv-probabilistic-forecast/vision/output/predict_compare.png"


def load_model():
    device = select_device("")
    model = DetectMultiBackend(WEIGHTS, device=device)
    model.eval()
    return model, device, model.stride


def predict(model, device, stride, img0_bgr, imgsz=640, conf=0.25):
    img = letterbox(img0_bgr, imgsz, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
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


def read_true_boxes(txt_path, W, H):
    boxes = []
    with open(txt_path) as f:
        for line in f:
            _, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            boxes.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
    return boxes


def draw(ax, img, boxes, color):
    ax.imshow(img, cmap="gray")
    for (x1, y1, x2, y2) in boxes:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=color, lw=1.6))
    ax.axis("off")


def main():
    model, device, stride = load_model()
    paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))[:4]
    fig, axes = plt.subplots(len(paths), 2, figsize=(9, 3.2 * len(paths)))
    for i, p in enumerate(paths):
        name = os.path.splitext(os.path.basename(p))[0]
        ts = name[-18:-4]
        img_gray = np.array(Image.open(p).convert("L"))
        H, W = img_gray.shape
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        det = predict(model, device, stride, img_bgr)
        pred_boxes = [(d[0], d[1], d[2], d[3]) for d in det]
        true_boxes = read_true_boxes(os.path.join(LBL_DIR, name + ".txt"), W, H)
        draw(axes[i, 0], img_gray, true_boxes, "lime")
        axes[i, 0].set_title("CLM 官方真值  云团 {} 个\n{}".format(len(true_boxes), ts), fontsize=10)
        draw(axes[i, 1], img_gray, pred_boxes, "red")
        axes[i, 1].set_title("模型预测  云团 {} 个".format(len(pred_boxes)), fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT, dpi=140)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()

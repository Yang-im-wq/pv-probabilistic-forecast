# -*- coding: utf-8 -*-
"""cv2 版：best.pt 检测云团，绿框=CLM真值，红框=模型预测，拼接对比。"""
import os
import glob
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "D:/yolov-test")
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from utils.torch_utils import select_device

WEIGHTS = "D:/yolov-test/runs/cloud_test/exp7/weights/best.pt"
IMG_DIR = "D:/pv-probabilistic-forecast/vision/cloud_dataset_crop_ir/images/val"
LBL_DIR = "D:/pv-probabilistic-forecast/vision/cloud_dataset_crop_ir/labels/val"
OUT = "D:/pv-probabilistic-forecast/vision/output/predict_compare_cv.png"


def main():
    device = select_device("")
    model = DetectMultiBackend(WEIGHTS, device=device)
    model.eval()
    stride = model.stride

    def predict(img0_bgr):
        img = letterbox(img0_bgr, 640, stride=stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(device).float() / 255.0
        im = im[None]
        pred = model(im)
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=500)
        det = pred[0]
        if det is not None and len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0_bgr.shape).round()
            return det.cpu().numpy()
        return np.zeros((0, 6))

    def read_true(txt, W, H):
        boxes = []
        for line in open(txt):
            _, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            boxes.append((int((cx - w / 2) * W), int((cy - h / 2) * H),
                          int((cx + w / 2) * W), int((cy + h / 2) * H)))
        return boxes

    paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))[:4]
    rows = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g.ndim == 3:
            g = g[:, :, 0]
        H, W = g.shape
        bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

        det = predict(bgr)
        predb = [(int(d[0]), int(d[1]), int(d[2]), int(d[3])) for d in det]
        trueb = read_true(os.path.join(LBL_DIR, name + ".txt"), W, H)

        t = bgr.copy()
        for (x1, y1, x2, y2) in trueb:
            cv2.rectangle(t, (x1, y1), (x2, y2), (0, 255, 0), 2)
        pr = bgr.copy()
        for (x1, y1, x2, y2) in predb:
            cv2.rectangle(pr, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.putText(t, "CLM truth: %d" % len(trueb), (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(pr, "predict: %d" % len(predb), (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        t = cv2.resize(t, (512, 512), interpolation=cv2.INTER_NEAREST)
        pr = cv2.resize(pr, (512, 512), interpolation=cv2.INTER_NEAREST)
        rows.append(np.hstack([t, pr]))

    out = np.vstack(rows)
    cv2.imwrite(OUT, out)
    print("saved", OUT, "shape", out.shape)


if __name__ == "__main__":
    main()

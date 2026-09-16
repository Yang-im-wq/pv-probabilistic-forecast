# -*- coding: utf-8 -*-
"""把 L1 FDI HDF 转成可见光灰度 PNG，供 clm_to_yolo.py 做输入图像。

与 l1_to_yolo.py 里的 l1_to_image 区别：修掉 65535 填充值（圆盘外太空/无效）
被误当成亮像素变白的 bug —— 先把填充值置 0（黑），再做固定 DN 范围归一化。
"""
import os
import glob
import argparse

import numpy as np
import h5py
from PIL import Image


def l1_to_image(hdf_path, channel=2, dn_max=3000):
    """读 L1 HDF 的某个波段，填充值(65535)→0，归一化到 0~255 灰度图。"""
    with h5py.File(hdf_path, "r") as f:
        dn = f["Data/NOMChannel%02d" % channel][:].astype(np.float32)
    dn[dn == 65535] = 0.0  # 填充值（圆盘外太空/无效像元）→ 黑
    return np.clip(dn / dn_max * 255.0, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="L1 HDF -> 可见光灰度 PNG")
    ap.add_argument("--hdf-dir", default="data/l1", help="L1 HDF 目录")
    ap.add_argument("--out-dir", default="vision/data/l1_png", help="PNG 输出目录")
    ap.add_argument("--channel", type=int, default=2, help="波段(2=0.65μm 红光)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.hdf_dir, "*L1*FDI*.HDF")))
    if not paths:
        raise SystemExit("没找到 L1 HDF：" + args.hdf_dir)

    done = skip = 0
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(args.out_dir, name + ".png")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            skip += 1
            continue
        Image.fromarray(l1_to_image(p, args.channel)).save(out)
        done += 1

    print("转换完成：新生成 {} 跳过 {} -> {}".format(done, skip, args.out_dir))


if __name__ == "__main__":
    main()

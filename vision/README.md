# vision/ —— 视觉模块（卫星云图 → 云团检测 → 云量）

> 这是「视觉 × 能源」的视觉半边：从风云四号卫星云图里检测出云团，算出**云量时间序列**，作为光伏爬坡/骤降的**前兆信号**，喂给数值预测模块（LSTM 概率预测）。数值半边在项目根目录，两边通过 `data/merge_cloud.py` 打通。

## 成果（2026-09）

| 云团检测（YOLOv5s）| 值 |
|---|---|
| mAP50 / mAP50-95 | 0.196 / 0.084 |
| 训练数据 | 346 时次（FY-4B，09-01 ~ 09-16）|
| 云量序列 vs CLM 官方真值 | 相关系数 0.699 |

## 数据：风云四号 B 星（FY-4B）

- **L1 FDI**：全圆盘辐射数据（2748×2748），取长波红外 **Ch12（12μm）** 通道
- **L2 CLM**：官方云检测产品，作为 YOLO 训练的「标准答案」
- 下载：`download_batch.py`（NSMC 批量下载，含 429 限流重试）

> **为什么用红外而不是可见光**：可见光 Ch02 夜间全黑（近一半时次无信号，却仍被 CLM 标了云，会污染训练）；红外 Ch12 全天候可见云（云顶温度低）。这是云检测全天候的关键修正。

## 完整流程

```
NSMC 下载 L1 + CLM → 红外 Ch12 裁剪广东沿海 256×256 → CLM 掩膜转 YOLO 标注
    → 训练 YOLOv5 → 检测云团 → 云量时间序列 CSV → merge_cloud 接光伏
```

| 步骤 | 脚本 | 说明 |
|---|---|---|
| 下载 | `download_batch.py` | 批量下载 L1 FDI + L2 CLM（限流自动重试）|
| 裁剪 | `crop_region.py` | 红外 Ch12 广东沿海 256×256（地理投影定位）|
| 标注 | `clm_to_yolo.py` | CLM 云掩膜 → 连通域 → YOLO 框 |
| 训练 | （`D:\yolov-test` 目录）| YOLOv5s，best.pt 在 `runs/cloud_test/exp7/` |
| 云量 | `yolo_cloud_series.py` | YOLO 检测 → 云量时间序列 CSV |
| 可视化 | `predict_all_val.py` | 真值（绿框）vs 预测（红框）对比图 |

## 云量 → 光伏对接

云量 CSV（`output/cloud_cover_yolo.csv`）由 `data/merge_cloud.py` 的 `load_cloud_series` 读入，对齐到光伏 15 分钟时间轴，成为 `CLOUD_COVER` 特征。

> ⚠️ **当前卡点**：FY-4B 云量是 2026 年的，光伏数据是 2019–2020 的，时间对不上，直接对齐会是垃圾特征。要真正接入训练，需 **FY-4A 同期云量 + 电站真实位置**（待老师提供）。代码链路已全通，数据一到即可无缝替换（目前训练走 `clear_sky_cloud_proxy` 晴空指数占位）。

## 方法演进（可选基线）

- `cloud_detect.py`：颜色阈值法云检测（无监督基线）
- `unet.py` / `train_seg.py`：U-Net 云分割（可选）
- `cloud_tracking.py`：光流云团追踪（YOLO 检测可替换其云掩膜环节）
- `fy_cloud.py`：FY-4B 云图地理投影 / 云量（早期版本）

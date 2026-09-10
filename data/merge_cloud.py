# -*- coding: utf-8 -*-
"""
data/merge_cloud.py —— 视觉模块 → 数值预测 的「对接脚本」

作用：把视觉模块（云检测）输出的「云量时间序列」合并进光伏预测的特征里，
作为新增特征 CLOUD_COVER，喂给 LSTM 概率预测模型。

两个入口：
  1. load_cloud_series(csv)      —— 读取真实云量序列（来自 vision 模块 / FY-4A 云检测）
  2. clear_sky_cloud_proxy(df)   —— 无真实云量时，用「晴空指数」从辐照度反推云量代理，
                                    先把整条链路跑通；真实云量下来后直接替换

为什么云量有价值（讲给老师用）：
  本地辐照度只能反映"已经压在电站头顶的云"；卫星云量能提前看到"还没飘到电站上空
  的云"，是光伏爬坡/骤降的「前兆」信号——这正是超短期概率预测最需要、而纯本地气象
  给不了的信息。
"""
import os

import numpy as np
import pandas as pd

import config


def clear_sky_cloud_proxy(df):
    """用「晴空指数」反推云量代理（无真实云量时的演示/占位）。

    思路：对一天中的每个 15 分钟档，取该档所有天的辐照度高值（90 分位）作为
    「晴空辐照度」参考包络；云量 ≈ 1 - 实测辐照度 / 晴空辐照度。
    这样得到的云量已分离太阳高度角的影响，只反映云层遮挡程度（0=晴，1=全阴）。
    """
    if "IRRADIATION" not in df.columns:
        return df
    irr = df["IRRADIATION"].values.astype(np.float32)
    # 一天内的档位（0~95），按档取 90 分位作为晴空参考
    slot = (df.index.hour * 4 + df.index.minute // 15).to_numpy()
    clear_ref = pd.Series(irr).groupby(slot).quantile(0.90)
    clear = clear_ref.reindex(slot).to_numpy()

    # 晴空指数 CSI = 实测/晴空；夜间（晴空≈0）令 CSI=1 → 云量 0
    csi = np.divide(irr, clear, out=np.ones_like(irr), where=clear > 1.0)
    df["CLOUD_COVER"] = np.clip(1.0 - csi, 0.0, 1.0)
    return df


def load_cloud_series(path, index):
    """读取真实云量 CSV，对齐到光伏数据的时间轴（15 分钟）。

    云量 CSV 由 vision 模块输出，格式：第一列为时间，另有一列云量(0~1)。
    列名含 cloud / fraction / 云量 之一即可被自动识别。
    """
    raw = pd.read_csv(path)
    time_col = raw.columns[0]
    val_col = [c for c in raw.columns
               if ("cloud" in c.lower() or "fraction" in c.lower() or "云量" in c)]
    if not val_col:
        val_col = [raw.columns[1]]  # 兜底取第二列

    s = pd.Series(pd.to_numeric(raw[val_col[0]], errors="coerce").to_numpy(),
                  index=pd.to_datetime(raw[time_col]))
    s = s[~s.index.duplicated()].sort_index()
    # 对齐到光伏 15 分钟时间轴：先最近邻再插值，补掉任何空档
    aligned = s.reindex(index, method="nearest").interpolate().ffill().bfill()
    return np.clip(aligned.to_numpy(), 0.0, 1.0)


if __name__ == "__main__":
    # 自检：打印一个合成云量序列示例
    idx = pd.date_range("2020-06-01", periods=96 * 3, freq="15min")
    demo = pd.DataFrame({"time": idx,
                         "cloud_fraction": np.clip(np.random.default_rng(0).random(96 * 3), 0, 1)})
    demo.to_csv(config.CLOUD_FILE, index=False)
    print("已生成示例云量序列 → {}".format(config.CLOUD_FILE))
    print(demo.head().to_string())

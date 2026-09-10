# -*- coding: utf-8 -*-
"""
download_data.py —— 数据集获取指引 + 合成数据生成（容错兜底）

优先使用 Kaggle 公开数据；数据缺失时打印清晰下载指引，
并可自动生成结构一致的合成数据，保证全流程在无网 / 无数据时仍可跑通。
"""
import os
import sys

# 允许直接运行本脚本（python data/download_data.py）时也能找到项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config

KAGGLE_URL = "https://www.kaggle.com/datasets/anikannal/solarpowergeneration"
KAGGLE_SLUG = "anikannal/solarpowergeneration"

USAGE_HINT = """
====================================================================
未找到数据集：
    {gen}
    {wea}
请任选一种方式获取：

【方式一】下载真实数据（推荐）
    1) 安装 Kaggle CLI：      pip install kaggle
    2) 配置 API token：在 {url} 或
       https://www.kaggle.com/settings/account 创建 token，
       把 kaggle.json 放到 ~/.kaggle/ 目录
       （Windows 即 C:/Users/<你的用户名>/.kaggle/kaggle.json）
    3) 下载并解压到 data/raw/：
       kaggle datasets download -d {slug} -p data/raw --unzip

【方式二】手动下载
    打开 {url} 点 Download，把 4 个 CSV 解压到 data/raw/ 目录。

【方式三】先跑通全流程（无需联网）
    python train.py --synthetic      # 自动生成合成数据再训练
====================================================================
""".format(gen=config.GENERATION_FILE, wea=config.WEATHER_FILE,
           url=KAGGLE_URL, slug=KAGGLE_SLUG)


# 国家电网新能源发电预测竞赛数据集（CC BY 4.0、可引用）
# figshare 官方出处（部分网络被拦）；GitHub 镜像（国内可直连，实测可用）
STATE_GRID_URL = ("https://figshare.com/articles/dataset/"
                  "Solar_and_wind_power_data_from_the_Chinese_State_Grid_"
                  "Renewable_Energy_Generation_Forecasting_Competition/17304221")
STATE_GRID_GITHUB = ("https://github.com/Bob05757/"
                     "Renewable-energy-generation-input-feature-variables-analysis")

USAGE_HINT_STATE_GRID = """
====================================================================
未找到国家电网单站数据：
    {file}
请下载「国家电网新能源发电预测竞赛」数据集（光伏站部分）：

【推荐】GitHub 镜像（国内可直连）
    {github}
    站点数据在 data_original/solar_stations/ 下（8 个 .xlsx）。
    下载站点 1：
    curl -L -o data/raw/solar_station_1.xlsx \\
      "https://raw.githubusercontent.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis/main/data_original/solar_stations/Solar%20station%20site%201%20(Nominal%20capacity-50MW).xlsx"

【备选】figshare 官方出处
    {url}

【说明】该数据来自中国国家电网 2021 年新能源预测竞赛，发表于 Nature 旗下
《Scientific Data》期刊。8 个光伏电站，2019–2020 两年，15 分钟采样，CC BY 4.0。
列含：Power(MW)、总辐照度、GHI、DNI、环境温度、相对湿度、大气压力。

【兜底】也可先跑合成数据：python train.py --synthetic
====================================================================
""".format(file=config.STATION_FILE, github=STATE_GRID_GITHUB, url=STATE_GRID_URL)


def generate_synthetic(days=20, step_min=None):
    """生成一份结构同 Kaggle 数据的合成光伏数据，用于无数据时演示全流程。

    原理：用"钟形辐照度曲线 + 云团随机遮挡 + 噪声"近似真实一天的光伏出力，
    其中云团的慢变化正是概率预测要捕捉的不确定性来源。
    """
    step_min = step_min or config.TIME_STEP_MIN
    n_steps = int(days * 24 * 60 / step_min)
    t = pd.date_range("2020-05-01", periods=n_steps, freq="{}min".format(step_min))

    rng = np.random.default_rng(config.SEED)

    # 一天内太阳高度角近似：白天用 sin 曲线，夜晚为 0（6:00–18:00 为正）
    hour = t.hour + t.minute / 60.0
    solar = np.sin(np.pi * (hour - 6.0) / 12.0)
    solar = np.clip(solar, 0.0, None)

    # 云团随机遮挡：低频平滑随机游走，模拟"整段被云遮住"的阴晴变化
    cloud = _smooth_noise(n_steps, rng, scale=0.5, smooth=40)
    irradiation = 1000.0 * solar * np.clip(1.0 - cloud, 0.05, 1.0)
    irradiation = np.clip(irradiation + rng.normal(0, 10, n_steps), 0.0, None)

    # 温度随辐照度变化
    ambient = 18.0 + 0.012 * irradiation + rng.normal(0, 0.5, n_steps)
    module = ambient + 0.02 * irradiation + rng.normal(0, 0.5, n_steps)

    # 功率 ≈ 辐照度 × 面板效率（加噪声）
    dc_power = np.clip(0.8 * irradiation * (1.0 + rng.normal(0, 0.02, n_steps)), 0.0, None)
    ac_power = np.clip(0.95 * dc_power + rng.normal(0, 1.0, n_steps), 0.0, None)

    # 写生成数据 CSV（单逆变器 + 单气象传感器，列名与真实数据一致）
    gen = pd.DataFrame({
        "DATE_TIME": t, "PLANT_ID": 1, "SOURCE_KEY": "SYNTHETIC_1",
        "DC_POWER": dc_power, "AC_POWER": ac_power,
        "DAILY_YIELD": np.cumsum(dc_power) * step_min / 60.0,
        "TOTAL_YIELD": np.cumsum(dc_power) * step_min / 60.0,
    })
    weather = pd.DataFrame({
        "DATE_TIME": t, "PLANT_ID": 1, "SOURCE_KEY": "SYNTHETIC_W1",
        "AMBIENT_TEMPERATURE": ambient,
        "MODULE_TEMPERATURE": module,
        "IRRADIATION": irradiation,
    })
    os.makedirs(config.DATA_DIR, exist_ok=True)
    gen.to_csv(config.GENERATION_FILE, index=False)
    weather.to_csv(config.WEATHER_FILE, index=False)
    print("[合成数据] 已生成 {} 天（{} 步，{} 分钟间隔）数据：".format(days, n_steps, step_min))
    print("  " + config.GENERATION_FILE)
    print("  " + config.WEATHER_FILE)


def _smooth_noise(n, rng, scale=1.0, smooth=10):
    """低频平滑随机游走：先生成白噪声再滑动平均，模拟云团的慢变化。"""
    raw = rng.normal(0, scale, n + smooth)
    kernel = np.ones(smooth) / smooth
    return np.convolve(raw, kernel, mode="same")[:n]


if __name__ == "__main__":
    if os.path.exists(config.GENERATION_FILE) and os.path.exists(config.WEATHER_FILE):
        print("数据已存在，无需生成。")
    else:
        generate_synthetic()

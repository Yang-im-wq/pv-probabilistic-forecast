# -*- coding: utf-8 -*-
"""
preprocess.py —— 数据清洗、特征工程、滑动窗口数据集构建

完整流程：
    读 CSV → 打印列名/前几行 → 聚合到电站级 → 补全等间隔时间轴
    → 特征工程（时间周期编码）→ 时间顺序切分 → 归一化（仅在训练集 fit）
    → 滑动窗口构造 → 保存为 npz
"""
import os
import sys

# 允许直接运行本脚本（python data/preprocess.py）时也能找到项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from data import download_data
from data import merge_cloud


def find_col(df, *keywords):
    """大小写不敏感地查找包含所有关键词的第一列，找不到返回 None。

    用于自适应处理不同版本数据的列名差异（需求书要求：第一步必须打印列名并自适应）。
    """
    for c in df.columns:
        if all(k.lower() in str(c).lower() for k in keywords):
            return c
    return None


def load_and_merge(gen_path, wea_path):
    """读取 CSV，打印列名与前几行，聚合到电站级别（跨逆变器求和 / 取均值）。"""
    gen = pd.read_csv(gen_path)
    wea = pd.read_csv(wea_path)

    print("=" * 60)
    print("[1/6] 生成数据列名：", list(gen.columns))
    print(gen.head(3).to_string())
    print("[2/6] 气象数据列名：", list(wea.columns))
    print(wea.head(3).to_string())

    # ---- 自适应匹配列名 ----
    time_col = find_col(gen, "date", "time") or find_col(gen, "time")
    ac_col = find_col(gen, "ac", "power")
    dc_col = find_col(gen, "dc", "power")
    if not (time_col and ac_col and dc_col):
        raise ValueError("无法在生成数据中识别 时间/AC/DC 列，实际列名：{}".format(list(gen.columns)))

    w_time_col = find_col(wea, "date", "time") or find_col(wea, "time")
    irr_col = find_col(wea, "irradiation")
    mod_col = find_col(wea, "module", "temp") or find_col(wea, "module")
    amb_col = find_col(wea, "ambient", "temp") or find_col(wea, "ambient")
    if not (w_time_col and irr_col):
        raise ValueError("无法在气象数据中识别 时间/辐照度 列，实际列名：{}".format(list(wea.columns)))

    # ---- 解析时间并聚合 ----
    gen[time_col] = pd.to_datetime(gen[time_col])
    wea[w_time_col] = pd.to_datetime(wea[w_time_col])

    # 生成数据：同一时刻多个逆变器 → 求和得到电站总功率
    gen_agg = gen.groupby(time_col).agg({ac_col: "sum", dc_col: "sum"})
    gen_agg = gen_agg.rename(columns={ac_col: "AC_POWER", dc_col: "DC_POWER"})

    # 气象数据：同一时刻多个传感器 → 取均值
    agg_map = {irr_col: "mean"}
    rename_map = {irr_col: "IRRADIATION"}
    if mod_col:
        agg_map[mod_col] = "mean"
        rename_map[mod_col] = "MODULE_TEMPERATURE"
    if amb_col:
        agg_map[amb_col] = "mean"
        rename_map[amb_col] = "AMBIENT_TEMPERATURE"
    wea_agg = wea.groupby(w_time_col).agg(agg_map).rename(columns=rename_map)

    df = gen_agg.join(wea_agg, how="inner")
    print("[3/6] 聚合到电站级：{} 个时间点（跨度 {} ~ {}）".format(
        len(df), df.index.min(), df.index.max()))
    return df


def find_col_any(df, patterns):
    """按多组关键词依次匹配列名，返回第一个命中的列（找不到返回 None）。

    patterns 是一组关键词元组，例如 [("actual", "power"), ("active", "power"), ("功率",)]。
    用于国家电网等列名不固定的数据，按"最精确 → 最宽"的优先级依次尝试。
    """
    for pattern in patterns:
        col = find_col(df, *pattern)
        if col:
            return col
    return None


def _to_numeric(series):
    """把列安全转为数值：先把常见缺失标记替换成 NaN，再强转并剔除 -99 等异常负值。"""
    series = series.replace(["--", "NA", "null", "NaN", ""], np.nan)
    out = pd.to_numeric(series, errors="coerce")
    out = out.where(out > -99.0)   # -99 是数据里常见的缺失标记（功率/气象都不会真实为 -99）
    return out


def load_state_grid(path):
    """加载国家电网新能源预测竞赛的单个光伏电站数据（功率 + 气象同一文件，支持 csv/xlsx）。

    列名按关键词自适应匹配（兼容英文与中文），统一映射到标准列名：
      有功功率 → AC_POWER（目标，MW → kW）、总辐照度 → IRRADIATION（主特征）、
      GHI → GHI、DNI → DNI、环境温度 → AMBIENT_TEMPERATURE、湿度 → HUMIDITY、气压 → PRESSURE
    """
    if str(path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    print("=" * 60)
    print("[1/6] 国家电网光伏站数据列名：", list(df.columns))
    print(df.head(3).to_string())

    time_col = find_col_any(df, [("time",), ("date",), ("时间",), ("日期",)])
    # 目标功率：优先"实际/有功"，避免误匹配"预测功率"列
    power_col = find_col_any(df, [("actual", "power"), ("active", "power"),
                                  ("实际",), ("有功",), ("power",)])
    # 三种辐照度：总辐照度（与功率相关性最高，作主特征）、GHI、DNI 分别映射
    irr_col = find_col_any(df, [("total", "irradiance"), ("total", "solar"),
                                ("总辐",), ("irradiance",), ("辐照",)])
    ghi_col = find_col_any(df, [("global", "horizontal"), ("ghi",), ("水平辐照",)])
    dni_col = find_col_any(df, [("direct", "normal"), ("dni",), ("direct",), ("法向",)])
    temp_col = find_col_any(df, [("temperature",), ("temp",), ("温度",)])
    hum_col = find_col_any(df, [("humidity",), ("湿度",)])
    pres_col = find_col_any(df, [("atmosphere",), ("pressure",), ("hpa",), ("气压",), ("大气",)])

    if not (time_col and power_col and irr_col):
        raise ValueError(
            "无法识别 时间/功率/辐照度 列，实际列名：{}\n"
            "请把这一行列名发给我，我帮你改一行匹配规则即可。".format(list(df.columns)))

    # MW → kW，保证与其它数据源（及全部图表/指标单位）一致
    data = {
        "AC_POWER": (_to_numeric(df[power_col]).to_numpy() * 1000.0),
        "IRRADIATION": _to_numeric(df[irr_col]).to_numpy(),
    }
    if ghi_col:
        data["GHI"] = _to_numeric(df[ghi_col]).to_numpy()
    if dni_col:
        data["DNI"] = _to_numeric(df[dni_col]).to_numpy()
    if temp_col:
        data["AMBIENT_TEMPERATURE"] = _to_numeric(df[temp_col]).to_numpy()
    if hum_col:
        data["HUMIDITY"] = _to_numeric(df[hum_col]).to_numpy()
    if pres_col:
        data["PRESSURE"] = _to_numeric(df[pres_col]).to_numpy()

    out = pd.DataFrame(data, index=pd.to_datetime(df[time_col])).sort_index()
    print("[2/6] 列映射：{}→AC_POWER，{}→IRRADIATION，共 {} 个时间点（{} ~ {}）".format(
        power_col, irr_col, len(out), out.index.min(), out.index.max()))
    return out


def add_time_features(df):
    """周期性时间编码：用 sin/cos 表示小时与分钟，避免 23:59 → 00:00 的跳变。

    直接喂整数小时（0–23）会让模型把 23 点和 0 点当成"相距最远"，
    而 sin/cos 编码能表达"它们其实只差 1 小时"这个周期事实。
    """
    hour = df.index.hour + df.index.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    minute = df.index.minute
    df["minute_sin"] = np.sin(2 * np.pi * minute / 60.0)
    df["minute_cos"] = np.cos(2 * np.pi * minute / 60.0)
    return df


def clean(df):
    """清洗：去重排序、补全等间隔时间轴、处理负值与突刺。"""
    df = df[~df.index.duplicated(keep="first")].sort_index()
    # 建立等间隔时间轴，缺失的时间步会变成 NaN
    df = df.resample("{}min".format(config.TIME_STEP_MIN)).asfreq()
    # 线性插值补缺失，首尾用前后向填充兜底
    df = df.interpolate(method="linear", limit_direction="both").ffill().bfill()

    # 负值 → 0（光伏不可能输出负功率，负值视为传感器误差）
    for col in ["AC_POWER", "DC_POWER", "IRRADIATION"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)
    # 突刺 → 截断到上限（避免个别异常值主导归一化）。上限默认自动取 99.9 分位数，
    # 单位自适应（kW / MW），避免固定阈值对不同数据源（功率量级不同）失效。
    for col in ["AC_POWER", "DC_POWER"]:
        if col in df.columns:
            cap = config.MAX_POWER if config.MAX_POWER > 0 else float(df[col].quantile(0.999))
            df[col] = df[col].clip(upper=cap)
    return df


def make_windows(feat, target, day_flag, time_str, seq_len, horizon):
    """滑动窗口构造：用过去 seq_len 步预测未来 horizon 步。

    feat     : (n, n_features) 已归一化特征
    target   : (n,)            已归一化目标
    day_flag : (n,)            每个时间步是否白天（用于 DAYTIME_ONLY 过滤）
    time_str : (n,)            每个时间步的时间字符串（用于画图）
    """
    xs, ys, ds, ts = [], [], [], []
    for i in range(len(feat) - seq_len - horizon + 1):
        xs.append(feat[i:i + seq_len])
        ys.append(target[i + seq_len:i + seq_len + horizon])
        ds.append(day_flag[i + seq_len])           # 目标首个时间步是否白天
        ts.append(time_str[i + seq_len])           # 目标首个时间步的时间
    return (np.asarray(xs, dtype=np.float32),
            np.asarray(ys, dtype=np.float32),
            np.asarray(ds, dtype=bool),
            np.asarray(ts))


def build_dataset(use_synthetic=False):
    """构建完整数据集，返回 dict。数据缺失时按 use_synthetic 决定报错还是生成合成数据。"""
    # ---- 按数据源加载原始数据（各数据源都映射到统一的标准列名）----
    if use_synthetic:
        if not (os.path.exists(config.GENERATION_FILE) and os.path.exists(config.WEATHER_FILE)):
            download_data.generate_synthetic()
        df = load_and_merge(config.GENERATION_FILE, config.WEATHER_FILE)
    elif config.DATA_SOURCE == "state_grid":
        if not os.path.exists(config.STATION_FILE):
            raise SystemExit(download_data.USAGE_HINT_STATE_GRID)
        df = load_state_grid(config.STATION_FILE)
    else:  # "kaggle"（默认）
        if not (os.path.exists(config.GENERATION_FILE) and os.path.exists(config.WEATHER_FILE)):
            raise SystemExit(download_data.USAGE_HINT)
        df = load_and_merge(config.GENERATION_FILE, config.WEATHER_FILE)

    df = add_time_features(df)
    df = clean(df)
    print("[4/6] 清洗后：{} 个时间点".format(len(df)))

    # ---- 云量特征（视觉模块 → 数值预测 的对接点）----
    # 有真实云量 CSV（来自 vision 模块 / FY-4A 云检测）就合并；否则用晴空指数反推代理占位
    if os.path.exists(config.CLOUD_FILE):
        df["CLOUD_COVER"] = merge_cloud.load_cloud_series(config.CLOUD_FILE, df.index)
        print("[4.5/6] 已合并真实云量 → {}".format(config.CLOUD_FILE))
    else:
        df = merge_cloud.clear_sky_cloud_proxy(df)
        print("[4.5/6] 无真实云量，使用晴空指数代理云量（占位，可替换）")

    # ---- 白天标记（基于原始辐照度，用于过滤夜间样本）----
    day_flag = (df["IRRADIATION"].values >= config.IRRADIATION_THRESHOLD)
    print("[5/6] 白天样本 {}/{}（阈值 {} W/m^2），DAYTIME_ONLY={}".format(
        int(day_flag.sum()), len(df), config.IRRADIATION_THRESHOLD, config.DAYTIME_ONLY))

    # 特征列 = 数据中实际存在的候选列 + 时间编码（不同数据源列不同，动态选取）
    feat_cols = [c for c in config.PREFERRED_FEATURES if c in df.columns]
    if config.TARGET_COL not in feat_cols:
        raise ValueError("数据缺少目标列 {}，现有列：{}".format(config.TARGET_COL, list(df.columns)))
    feat_cols = feat_cols + ["hour_sin", "hour_cos", "minute_sin", "minute_cos"]

    time_str = df.index.strftime("%Y-%m-%d %H:%M").to_numpy()
    target = df[config.TARGET_COL].values.astype(np.float32)
    feat = df[feat_cols].values.astype(np.float32)

    # ---- 时间顺序切分（严禁随机打乱）----
    n = len(df)
    tr_end = int(n * config.TRAIN_RATIO)
    va_end = tr_end + int(n * config.VAL_RATIO)
    slices = {"train": slice(0, tr_end),
              "val": slice(tr_end, va_end),
              "test": slice(va_end, n)}
    print("[6/6] 时间顺序切分 → 训练 {} / 验证 {} / 测试 {} 个时间点".format(
        tr_end, va_end - tr_end, n - va_end))

    # ---- 归一化：只在训练集上 fit（严禁数据泄漏）----
    # 关键点：scaler 的均值/方差只能由训练集统计得到，
    # 再对验证/测试集做 transform，绝不能用全体数据 fit。
    scaler = StandardScaler()
    scaler.fit(feat[slices["train"]])
    feat_scaled = scaler.transform(feat)
    target_idx = feat_cols.index(config.TARGET_COL)
    target_scaled = feat_scaled[:, target_idx]

    # ---- 滑窗（每个切分独立构造，避免窗口跨过切分边界泄漏）----
    out = {}
    for split, sl in slices.items():
        X, y, d, t = make_windows(feat_scaled[sl], target_scaled[sl],
                                  day_flag[sl], time_str[sl],
                                  config.SEQ_LEN, config.HORIZON)
        if config.DAYTIME_ONLY:
            keep = d
            X, y, t = X[keep], y[keep], t[keep]
            print("  [{}] 滑窗 {} 条（已过滤夜间目标）".format(split, len(X)))
        else:
            print("  [{}] 滑窗 {} 条".format(split, len(X)))
        out["X_{}".format(split)] = X
        out["y_{}".format(split)] = y
        out["t_{}".format(split)] = t

    # ---- 反归一化所需参数（评估 / 画图时回到原始单位）----
    out["target_mean"] = float(scaler.mean_[target_idx])
    out["target_scale"] = float(scaler.scale_[target_idx])
    out["seq_len"] = config.SEQ_LEN
    out["horizon"] = config.HORIZON
    out["source"] = "synthetic" if use_synthetic else config.DATA_SOURCE

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    np.savez_compressed(config.PROCESSED_PATH, **out)
    print("已保存预处理结果 → {}".format(config.PROCESSED_PATH))
    return out


def get_dataset(use_synthetic=False, force=False):
    """优先加载缓存；缓存不存在、参数/数据源不匹配或原始数据比缓存新时重新构建。"""
    source = "synthetic" if use_synthetic else config.DATA_SOURCE
    if not force and os.path.exists(config.PROCESSED_PATH):
        # 若原始数据比缓存新（例如刚下载真实数据覆盖了合成数据），自动重建
        raw_files = [config.GENERATION_FILE, config.WEATHER_FILE, config.STATION_FILE]
        raw_mtimes = [os.path.getmtime(f) for f in raw_files if os.path.exists(f)]
        raw_newer = bool(raw_mtimes) and max(raw_mtimes) > os.path.getmtime(config.PROCESSED_PATH)
        if not raw_newer:
            d = np.load(config.PROCESSED_PATH, allow_pickle=True)
            if (int(d["seq_len"]) == config.SEQ_LEN
                    and int(d["horizon"]) == config.HORIZON
                    and str(d["source"]) == source):
                print("加载已缓存的预处理结果 → {}".format(config.PROCESSED_PATH))
                return {k: d[k] for k in d.files}
    return build_dataset(use_synthetic)


if __name__ == "__main__":
    build_dataset(use_synthetic=True)

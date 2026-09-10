# -*- coding: utf-8 -*-
"""
config.py —— 全项目超参数集中管理
============================================================
项目：光伏出力超短期概率预测（展示型 MVP）

所有可调参数都集中在这个文件里，训练 / 评估 / 可视化脚本统一从这里读取，
避免参数散落各处、改一处漏一处。
"""
import os

# ============================================================
# 路径
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")             # 原始 CSV 目录
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")  # 预处理结果目录
RESULT_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR = os.path.join(RESULT_DIR, "figures")
CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")

GENERATION_FILE = os.path.join(DATA_DIR, "Plant_1_Generation_Data.csv")
WEATHER_FILE = os.path.join(DATA_DIR, "Plant_1_Weather_Sensor_Data.csv")
STATION_FILE = os.path.join(DATA_DIR, "solar_station_1.xlsx")  # 国家电网单站数据
CLOUD_FILE = os.path.join(DATA_DIR, "cloud_cover.csv")   # 云量时间序列（视觉模块输出，可选）
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "dataset.npz")   # 预处理后的数据


def metrics_path(mode):
    """指标存档（按模式区分：point / quantile 各一份，互不覆盖）。"""
    return os.path.join(RESULT_DIR, "metrics_{}.json".format(mode))


def pred_path(mode):
    """测试集预测结果（按模式区分）。"""
    return os.path.join(RESULT_DIR, "predictions_{}.npz".format(mode))


def losses_path(mode):
    """训练损失曲线数据（按模式区分）。"""
    return os.path.join(RESULT_DIR, "train_losses_{}.npz".format(mode))

# ============================================================
# 随机种子（保证结果可复现）
# ============================================================
SEED = 42

# ============================================================
# 数据
# ============================================================
# 数据源："state_grid"（国家电网单站，推荐）| "kaggle"（两个 CSV）
# 注：加 --synthetic 时一律用合成数据，与 DATA_SOURCE 无关
DATA_SOURCE = "state_grid"
TARGET_COL = "AC_POWER"      # 预测目标（标准内部列名，各数据源都映射到它）
TIME_STEP_MIN = 15           # 采样间隔（分钟）
SEQ_LEN = 96                 # 输入窗口：过去 N 步（96 × 15 分钟 = 24 小时）
HORIZON = 1                  # 输出步数 H（默认 1 = 未来 15 分钟）

# 数据按【时间顺序】切分，严禁随机打乱（打乱会引入未来信息 → 数据泄漏）
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 夜间样本处理策略（二选一，见 README / preprocess.py 注释）：
#   True  —— 只对"白天时段的目标时刻"计算损失与评估（夜间功率≈0，无预测价值，
#            且会虚高区间覆盖率 PICP，掩盖真实预测能力）
#   False —— 保留全部样本
DAYTIME_ONLY = True
IRRADIATION_THRESHOLD = 10.0  # W/m²，低于此值视为夜间

# 异常值处理：功率突刺截断上限。0 = 自动（取该列 99.9 分位数，单位自适应 kW/MW）
MAX_POWER = 0.0

# 候选特征（按优先级）：加载器把各数据源映射到这些标准列名后，
# 实际特征 = 数据中存在的候选列 + 时间编码（hour/minute 的 sin/cos）。
# 不同数据源列不同（如国家电网无 DC_POWER，但有 DNI/湿度/气压），故"存在即用"动态选取。
PREFERRED_FEATURES = [
    "AC_POWER", "DC_POWER", "IRRADIATION", "GHI", "DNI",
    "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE", "HUMIDITY", "PRESSURE",
    "CLOUD_COVER",
]

# ============================================================
# 模型
# ============================================================
MODEL_TYPE = "lstm"           # 骨干网络："lstm" 或 "gru"
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.1
QUANTILES = [0.1, 0.5, 0.9]   # 分位数回归输出的分位点（可配置）
NUM_QUANTILES = len(QUANTILES)

# ============================================================
# 训练
# ============================================================
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 10                 # 早停：验证损失连续 N 轮不下降则提前结束（0 = 关闭）

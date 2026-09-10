# -*- coding: utf-8 -*-
"""
evaluate.py —— 在测试集上评估并计算指标

点预测指标：RMSE / MAE / nRMSE
概率预测指标：PICP（区间覆盖率）/ PINAW（归一化区间宽度）/（可选）CRPS

用法：python evaluate.py --mode quantile   （或 --mode point）
"""
import argparse
import json
import os

import numpy as np
import torch

import config
from data import preprocess
from models.lstm_point import LSTMPointForecaster
from models.lstm_quantile import LSTMQuantileForecaster


def inverse_scale(values, mean, scale):
    """反归一化：把标准化后的值还原到原始单位（kW）。"""
    return values * scale + mean


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def nrmse(y_true, y_pred):
    """归一化 RMSE：RMSE 除以目标值极差，便于跨数据集比较。"""
    r = y_true.max() - y_true.min()
    return float(rmse(y_true, y_pred) / r) if r > 0 else float("nan")


def picp(y_true, lower, upper):
    """PICP —— 区间覆盖率：真实值落在 [lower, upper] 内的比例，理想值≈标称覆盖率。"""
    inside = ((y_true >= lower) & (y_true <= upper)).astype(float)
    return float(inside.mean())


def pinaw(y_true, lower, upper):
    """PINAW —— 归一化区间宽度：区间平均宽度除以目标极差，越小越好（区间越"紧"）。"""
    r = y_true.max() - y_true.min()
    return float(np.mean(upper - lower) / r) if r > 0 else float("nan")


def crps_from_quantiles(y_true, y_pred, quantiles):
    """CRPS 近似：CRPS = 2 * ∫ pinball_q dq，用有限分位数 + 梯形权重近似积分。

    分位数越多越准，此处仅有 3 个分位数，属于粗近似，仅作参考。
    y_true: (n, horizon)；y_pred: (n, horizon, n_quantiles)
    """
    q = np.asarray(quantiles, dtype=np.float64).reshape(1, 1, -1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(y_true.shape[0], y_true.shape[1], 1)
    error = y_true - np.asarray(y_pred, dtype=np.float64)
    pinball = np.maximum(q * error, (q - 1.0) * error)
    weights = np.empty_like(q)
    weights[..., 0] = (quantiles[1] - quantiles[0]) / 2.0
    weights[..., -1] = (quantiles[-1] - quantiles[-2]) / 2.0
    for i in range(1, len(quantiles) - 1):
        weights[..., i] = (quantiles[i + 1] - quantiles[i - 1]) / 2.0
    return float(2.0 * np.mean(pinball * weights))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["point", "quantile"], default="quantile")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = preprocess.get_dataset(use_synthetic=args.synthetic)
    X_test = torch.from_numpy(data["X_test"]).float().to(device)
    y_test_scaled = data["y_test"]
    mean, scale = data["target_mean"], data["target_scale"]
    input_size = X_test.shape[-1]

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_{}.pt".format(args.mode))
    if not os.path.exists(ckpt_path):
        raise SystemExit("未找到模型权重：{}\n请先运行 python train.py --mode {}".format(
            ckpt_path, args.mode))

    if args.mode == "point":
        model = LSTMPointForecaster(input_size, config.HIDDEN_SIZE, config.NUM_LAYERS,
                                    config.DROPOUT, config.HORIZON, config.MODEL_TYPE)
    else:
        model = LSTMQuantileForecaster(input_size, config.HIDDEN_SIZE, config.NUM_LAYERS,
                                       config.DROPOUT, config.HORIZON,
                                       config.NUM_QUANTILES, config.MODEL_TYPE)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()

    # ---- 预测并反归一化 ----
    with torch.no_grad():
        pred_scaled = model(X_test).cpu().numpy()   # point:(n,horizon) / quantile:(n,horizon,nq)
    y_true = inverse_scale(y_test_scaled, mean, scale)      # (n, horizon)
    pred = inverse_scale(pred_scaled, mean, scale)

    os.makedirs(config.RESULT_DIR, exist_ok=True)

    if args.mode == "point":
        y_true_0 = y_true[:, 0]
        pred_0 = pred[:, 0]
        metrics = {
            "RMSE(kW)": round(rmse(y_true_0, pred_0), 3),
            "MAE(kW)": round(mae(y_true_0, pred_0), 3),
            "nRMSE": round(nrmse(y_true_0, pred_0), 4),
        }
        save_pred = {"y_true": y_true_0, "pred": pred_0,
                     "time": data["t_test"], "mode": "point"}
    else:
        q = config.QUANTILES
        pred_0 = pred[:, 0, :]                          # (n, n_quantiles)
        i_mid = q.index(0.5)
        lower, median, upper = pred_0[:, 0], pred_0[:, i_mid], pred_0[:, -1]
        y_true_0 = y_true[:, 0]
        metrics = {
            "RMSE_q50(kW)": round(rmse(y_true_0, median), 3),
            "MAE_q50(kW)": round(mae(y_true_0, median), 3),
            "nRMSE_q50": round(nrmse(y_true_0, median), 4),
            "PICP_90%": round(picp(y_true_0, lower, upper), 4),
            "PINAW": round(pinaw(y_true_0, lower, upper), 4),
            "CRPS_approx": round(crps_from_quantiles(y_true[:, 0:1], pred[:, 0:1, :], q), 3),
        }
        save_pred = {"y_true": y_true_0, "median": median, "lower": lower, "upper": upper,
                     "time": data["t_test"], "mode": "quantile"}

    np.savez(config.pred_path(args.mode), **save_pred)
    with open(config.metrics_path(args.mode), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("\n预测结果已保存 → {}".format(config.pred_path(args.mode)))
    print("指标已保存 → {}".format(config.metrics_path(args.mode)))


if __name__ == "__main__":
    main()

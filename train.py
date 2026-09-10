# -*- coding: utf-8 -*-
"""
train.py —— 训练入口（支持 --mode point / quantile）

用法：
    python train.py --mode point               # 点预测（阶段 1）
    python train.py --mode quantile            # 分位数概率预测（阶段 2，默认）
    python train.py --mode quantile --synthetic --epochs 30   # 无数据时用合成数据
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from data import preprocess
from models.lstm_point import LSTMPointForecaster
from models.lstm_quantile import LSTMQuantileForecaster
from models.losses import pinball_loss


def set_seed(seed):
    """固定随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(input_size, mode, device):
    """按模式构建模型（点预测 / 分位数回归共用同一骨干网络）。"""
    if mode == "point":
        model = LSTMPointForecaster(
            input_size, hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT, horizon=config.HORIZON, rnn_type=config.MODEL_TYPE)
    else:
        model = LSTMQuantileForecaster(
            input_size, hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT, horizon=config.HORIZON,
            num_quantiles=config.NUM_QUANTILES, rnn_type=config.MODEL_TYPE)
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="光伏出力超短期概率预测训练")
    parser.add_argument("--mode", choices=["point", "quantile"], default="quantile")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--synthetic", action="store_true", help="无真实数据时生成合成数据")
    args = parser.parse_args()

    set_seed(config.SEED)
    device = get_device()
    print("设备：{}（{}），模式：{}".format(device, "GPU" if device.type == "cuda" else "CPU", args.mode))

    # ---- 数据 ----
    data = preprocess.get_dataset(use_synthetic=args.synthetic)
    X_train = torch.from_numpy(data["X_train"])
    y_train = torch.from_numpy(data["y_train"])
    X_val = torch.from_numpy(data["X_val"])
    y_val = torch.from_numpy(data["y_val"])
    input_size = X_train.shape[-1]

    # ---- 模型与损失 ----
    model = build_model(input_size, args.mode, device)
    n_params = sum(p.numel() for p in model.parameters())
    print("模型参数量：{:,}".format(n_params))

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=args.batch_size, shuffle=False)

    if args.mode == "point":
        criterion = nn.MSELoss()
    else:
        quantiles = torch.tensor(config.QUANTILES, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_{}.pt".format(args.mode))

    best_val = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        # ---- 训练 ----
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            if args.mode == "point":
                loss = criterion(pred, yb)
            else:
                loss = pinball_loss(yb, pred, quantiles)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # ---- 验证 ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                if args.mode == "point":
                    val_loss += criterion(pred, yb).item() * xb.size(0)
                else:
                    val_loss += pinball_loss(yb, pred, quantiles).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print("Epoch {:3d}/{} | train_loss {:.5f} | val_loss {:.5f}".format(
            epoch, args.epochs, train_loss, val_loss))

        # ---- 保存最佳 checkpoint + 早停 ----
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print("          ↑ 保存最佳模型 → {}".format(ckpt_path))
        else:
            patience_counter += 1
            if config.PATIENCE > 0 and patience_counter >= config.PATIENCE:
                print("早停：验证损失连续 {} 轮无改善，提前结束。".format(config.PATIENCE))
                break

    # ---- 存档训练损失曲线（供 visualize.py 画图）----
    np.savez(config.losses_path(args.mode),
             train=np.asarray(train_losses, dtype=np.float32),
             val=np.asarray(val_losses, dtype=np.float32),
             mode=args.mode)
    print("训练完成。最佳验证损失：{:.5f}".format(best_val))
    print("损失曲线已保存 → {}".format(config.losses_path(args.mode)))


if __name__ == "__main__":
    main()

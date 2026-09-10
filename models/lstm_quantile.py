# -*- coding: utf-8 -*-
"""
lstm_quantile.py —— 分位数回归 LSTM（阶段 2，本项目核心）

与点预测共用同一个骨干网络，仅把输出层改为同时输出多个分位数，
从而用一次前向得到 q10 / q50 / q90 三条曲线，构成预测区间。
"""
import torch
import torch.nn as nn


class LSTMQuantileForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 dropout=0.1, horizon=1, num_quantiles=3, rnn_type="lstm"):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_size, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, horizon * num_quantiles)
        self.horizon = horizon
        self.num_quantiles = num_quantiles

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.rnn(x)
        last = out[:, -1, :]                        # (batch, hidden_size)
        out = self.fc(last)                         # (batch, horizon * n_quantiles)
        # 重排为 (batch, horizon, n_quantiles)：每个时间步对应一组分位数预测
        return out.view(-1, self.horizon, self.num_quantiles)

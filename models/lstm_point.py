# -*- coding: utf-8 -*-
"""
lstm_point.py —— 点预测模型（阶段 1 基线）
输出未来 H 步的单一预测值。
"""
import torch
import torch.nn as nn


class LSTMPointForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 dropout=0.1, horizon=1, rnn_type="lstm"):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        # dropout 作用于层与层之间（num_layers >= 2 时才生效）
        self.rnn = rnn_cls(input_size, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.rnn(x)         # (batch, seq_len, hidden_size)
        last = out[:, -1, :]         # 取最后一步的隐状态作为整个序列的表示
        return self.fc(last)         # (batch, horizon)

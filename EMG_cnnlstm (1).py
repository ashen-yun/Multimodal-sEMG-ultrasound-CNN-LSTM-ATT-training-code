#!/usr/bin/env python3
import os
import copy
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt  # 导入matplotlib
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

# ---------- 1. 读取数据（含动作标签），并按 40213 格式转换 ----------
# 这部分在脚本开始时执行一次
with h5py.File('train_val_data_subject73set_seq6_withlabel_new.mat', 'r') as f:
    train_inputs_emg = np.array(f['all_train_inputs'])
    train_labels_emg = np.array(f['all_train_labels'])
    train_actions = np.array(f['all_train_actions'])
    val_inputs_emg = np.array(f['all_valid_inputs'])
    val_labels_emg = np.array(f['all_valid_labels'])
    val_actions = np.array(f['all_valid_actions'])

train_inputs_emg = np.transpose(train_inputs_emg, (4, 0, 2, 1, 3))
val_inputs_emg = np.transpose(val_inputs_emg, (4, 0, 2, 1, 3))
train_labels_emg = train_labels_emg.T
val_labels_emg = val_labels_emg.T
train_actions = train_actions.reshape(-1)
val_actions = val_actions.reshape(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_inputs_emg = torch.tensor(train_inputs_emg, dtype=torch.float32).to(device)
train_labels_emg = torch.tensor(train_labels_emg, dtype=torch.float32).to(device)
train_actions = torch.tensor(train_actions, dtype=torch.long).to(device)
val_inputs_emg = torch.tensor(val_inputs_emg, dtype=torch.float32).to(device)
val_labels_emg = torch.tensor(val_labels_emg, dtype=torch.float32).to(device)
val_actions = torch.tensor(val_actions, dtype=torch.long).to(device)

BATCH = 64
train_loader = DataLoader(
    TensorDataset(train_inputs_emg, train_actions, train_labels_emg),
    batch_size=BATCH, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(val_inputs_emg, val_actions, val_labels_emg),
    batch_size=BATCH, shuffle=False
)


# ---------- 2. 定义【消融后】的纯 CNN+LSTM 模型 ----------
class CNN_LSTM_ActionSpatialAttn(nn.Module):
    def __init__(self, time_steps, channels, height, width,
                 num_actions, num_outputs,
                 d_emb=64, num_heads=8, drop=0.0, lstm_hidden=100):
        super().__init__()
        self.T, self.C, self.H, self.W = time_steps, channels, height, width
        self.transpose_conv = nn.Sequential(
            nn.ConvTranspose2d(channels, 8, 2, 2), nn.BatchNorm2d(8), nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 3, 2, 2), nn.BatchNorm2d(3), nn.LeakyReLU(),
            nn.ConvTranspose2d(3, 1, 2, 2), nn.BatchNorm2d(1), nn.LeakyReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=2), nn.LeakyReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(3, 8, kernel_size=2), nn.LeakyReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(8, 16, kernel_size=2, padding=1), nn.LeakyReLU(), nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(16, 32, kernel_size=2, padding=1), nn.LeakyReLU(), nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(32, 64, kernel_size=2), nn.LeakyReLU(), nn.MaxPool2d((2, 2), (2, 2)),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            x = self.transpose_conv(dummy);
            x = self.conv(x)
            _, Cc, Hc, Wc = x.shape
        feat_dim = Cc * Hc * Wc
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(feat_dim, lstm_hidden, batch_first=True)
        self.fc_reg = nn.Linear(lstm_hidden, num_outputs)

    def forward(self, x, action_id):
        B, T, C, H, W = x.shape
        x_bt = x.view(B * T, C, H, W)
        x_bt = self.transpose_conv(x_bt)
        x_bt = self.conv(x_bt)
        feat = self.flatten(x_bt)
        seq = feat.view(B, T, -1)
        out_seq, _ = self.lstm(seq)
        h_last = out_seq[:, -1, :]
        return self.fc_reg(h_last)


### --- 新的训练函数 --- ###
def train_single_run(run_index):
    """
    为给定的运行索引（run_index）执行一次完整的训练。
    """
    print(f"--- 开始执行训练，当前索引: {run_index} ---")

    # ---------- 3. 初始化 & 训练配置 ----------
    num_actions = int(train_actions.max().item()) + 1
    num_outputs = train_labels_emg.shape[1]
    model = CNN_LSTM_ActionSpatialAttn(
        time_steps=train_inputs_emg.shape[1],
        channels=train_inputs_emg.shape[2],
        height=train_inputs_emg.shape[3],
        width=train_inputs_emg.shape[4],
        num_actions=num_actions,
        num_outputs=num_outputs,
        lstm_hidden=100
    ).to(device)

    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_val, wait, patience = float('inf'), 0, 40
    best_wts = None
    MAX_EPOCHS = 5000

    # 用于绘图的损失记录列表
    train_losses, val_losses = [], []

    # ---------- 4. 训练循环 ----------
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_reg, n = 0., 0
        for X, a, Y in train_loader:
            optimizer.zero_grad()
            pr = model(X, a)
            loss_r = criterion_reg(pr, Y)
            loss_r.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_reg += loss_r.item() * X.size(0)
            n += X.size(0)
        train_mse = total_reg / n
        train_losses.append(train_mse)  # 记录训练损失

        model.eval()
        val_reg, cnt = 0., 0
        with torch.no_grad():
            for Xv, av, Yv in val_loader:
                pr = model(Xv, av)
                val_reg += criterion_reg(pr, Yv).item() * Xv.size(0)
                cnt += Xv.size(0)
        val_mse = val_reg / cnt
        val_losses.append(val_mse)  # 记录验证损失

        print(f"  轮次 {epoch:4d} — 训练 MSE: {train_mse:.6f}, 验证 MSE: {val_mse:.6f}")

        if val_mse < best_val:
            best_val, wait, best_wts = val_mse, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                print(f"  在轮次 {epoch} 提前停止")
                break

    # ---------- 5. 保存模型与绘图 ----------
    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    model.load_state_dict(best_wts)

    # --- 已更改 --- 动态的模型文件名
    model_filename = f'models/emg_model_cnnlstm_subject7{run_index}.pth'
    torch.save(model.state_dict(), model_filename)

    # --- 新增 --- 绘制并保存损失曲线图
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train MSE')
    plt.plot(val_losses, label='Validation MSE', linestyle='--')
    plt.title(f'Training and Validation Loss for Run {run_index}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    figure_filename = f'figures/emg_loss_yunchen_run{run_index}.png'
    plt.savefig(figure_filename, dpi=300)
    plt.close()

    print(f"--- 训练 {run_index} 已完成。模型保存在 {model_filename}，图片保存在 {figure_filename} ---")


### --- 主执行循环 --- ###
if __name__ == '__main__':
    # --- 在这里配置 ---
    # 定义你想运行多少次训练
    NUM_RUNS = 5
    # 定义文件名的起始数字
    START_RUN_INDEX = 1

    for i in range(NUM_RUNS):
        current_run_index = START_RUN_INDEX + i
        train_single_run(run_index=current_run_index)

    print("\n所有训练任务已全部完成！")
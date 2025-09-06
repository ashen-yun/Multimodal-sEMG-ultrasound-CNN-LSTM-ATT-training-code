#!/usr/bin/env python3
import os
import copy
import math
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ---------- 0. 随机种子初始化 ----------
torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# ---------- 1. 数据加载 (Ultrasound + labels) ----------
# 这部分代码在脚本开始时执行一次
with h5py.File('us_train_val_data_zhitaonewfiltered.mat', 'r') as f:
    train_inputs_np  = f['all_train_inputs'][()]
    train_labels_np  = f['all_train_labels'][()]
    val_inputs_np    = f['all_valid_inputs'][()]
    val_labels_np    = f['all_valid_labels'][()]

with h5py.File('us_test_data_zhitaonewfiltered.mat', 'r') as f:
    test_inputs_np   = f['all_test_inputs'][()]
    test_labels_np   = f['all_test_labels'][()]

train_labels = torch.tensor(train_labels_np.T, dtype=torch.float32)
val_labels   = torch.tensor(val_labels_np.T,   dtype=torch.float32)
test_labels  = torch.tensor(test_labels_np.T,  dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_inputs, train_labels = (
    torch.tensor(train_inputs_np, dtype=torch.float32).to(device),
    train_labels.to(device)
)
val_inputs,   val_labels   = (
    torch.tensor(val_inputs_np,   dtype=torch.float32).to(device),
    val_labels.to(device)
)
test_inputs,  test_labels  = (
    torch.tensor(test_inputs_np,  dtype=torch.float32).to(device),
    test_labels.to(device)
)

# ---------- 2. DataLoader 定义 ----------
BATCH = 64
train_loader = DataLoader(TensorDataset(train_inputs, train_labels),
                          batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(TensorDataset(val_inputs,   val_labels),
                          batch_size=BATCH, shuffle=False)
test_loader  = DataLoader(TensorDataset(test_inputs,  test_labels),
                          batch_size=BATCH, shuffle=False)

# ---------- 3. 纯 CNN+LSTM 回归模型 ----------
# 模型定义保持不变
class CNN_LSTM_Regression_US(nn.Module):
    def __init__(self, T, C, H, W, out_dim, lstm_hidden=64):
        super().__init__()
        self.T = T
        self.transpose_conv = nn.Sequential(
            nn.ConvTranspose2d(C,  8, 2, 2), nn.BatchNorm2d(8), nn.LeakyReLU(),
            nn.ConvTranspose2d(8,  3, 2, 2), nn.BatchNorm2d(3), nn.LeakyReLU(),
            nn.ConvTranspose2d(3,  1, 2, 2), nn.BatchNorm2d(1), nn.LeakyReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1,  3, 2, 1), nn.BatchNorm2d(3),  nn.LeakyReLU(),
            nn.Conv2d(3,  8, 2, 1), nn.BatchNorm2d(8),  nn.LeakyReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(8, 16, 2, 1), nn.BatchNorm2d(16), nn.LeakyReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,2,1,padding=2),  nn.BatchNorm2d(32), nn.LeakyReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,2,1),                   nn.LeakyReLU(),             nn.MaxPool2d(2,2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            dummy = self.transpose_conv(dummy)
            dummy = self.conv(dummy)
            conv_flat_dim = dummy.numel()
        self.flatten = nn.Flatten()
        self.lstm    = nn.LSTM(conv_flat_dim, lstm_hidden, batch_first=True)
        self.fc_reg = nn.Linear(lstm_hidden, out_dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_bt = x.view(B*T, C, H, W)
        x_bt = self.transpose_conv(x_bt)
        x_bt = self.conv(x_bt)
        feat = self.flatten(x_bt)
        seq  = feat.view(B, T, -1)
        out_seq, _ = self.lstm(seq)
        h_last     = out_seq[:, -1, :]
        return self.fc_reg(h_last)

### --- 新的训练函数 --- ###
def train_single_run(run_index):
    """
    为给定的运行索引（run_index）执行一次完整的训练、测试和保存。
    返回测试集上的MSE。
    """
    print(f"\n===== [ 开始执行第 {run_index} 次训练 ] =====")

    # ---------- 4. 初始化 & 训练配置 ----------
    out_dim = train_labels.shape[1]
    model = CNN_LSTM_Regression_US(
        T           = train_inputs.shape[1],
        C           = train_inputs.shape[2],
        H           = train_inputs.shape[3],
        W           = train_inputs.shape[4],
        out_dim     = out_dim,
        lstm_hidden = 64
    ).to(device)

    criterion_reg = nn.MSELoss()
    optimizer     = optim.Adam(model.parameters(), lr=1e-3)
    train_losses, val_losses = [], []
    best_val, wait, patience = float('inf'), 0, 25

    # ---------- 5. 训练循环 ----------
    for epoch in range(1, 201):
        model.train()
        tr_reg = 0.0
        for X, Y in train_loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion_reg(pred, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tr_reg += loss.item() * X.size(0)
        tr_mse = tr_reg / len(train_loader.dataset)
        train_losses.append(tr_mse)

        model.eval()
        v_reg = 0.0
        with torch.no_grad():
            for Xv, Yv in val_loader:
                pv = model(Xv)
                v_reg += criterion_reg(pv, Yv).item() * Xv.size(0)
        val_mse = v_reg / len(val_loader.dataset)
        val_losses.append(val_mse)

        print(f"  轮次 {epoch:3d} — 训练 MSE: {tr_mse:.6f}, 验证 MSE: {val_mse:.6f}")
        if val_mse < best_val:
            best_val, wait, best_wts = val_mse, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                print(f"  在轮次 {epoch} 提前停止。")
                break

    # ---------- 6. 测试评估 & 保存 ----------
    print("\n  开始在测试集上评估最佳模型...")
    model.load_state_dict(best_wts)
    model.eval()
    test_reg = 0.0
    with torch.no_grad():
        for Xt, Yt in test_loader:
            pr = model(Xt)
            test_reg += criterion_reg(pr, Yt).item() * Xt.size(0)
    test_mse = test_reg / len(test_loader.dataset)
    print(f"  -> 第 {run_index} 次训练的测试 MSE: {test_mse:.6f}")

    os.makedirs('models', exist_ok=True)
    model_filename = f'models/us_cnnlstm_zhitaonew{run_index}.pth'
    torch.save(best_wts, model_filename)

    plt.figure()
    plt.plot(train_losses, label='Train MSE')
    plt.plot(val_losses,   '--', label='Val MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    figure_filename = f'figures/us_regression_only_loss_run{run_index}.png'
    plt.savefig(figure_filename, dpi=300)
    plt.close()
    
    print(f"  模型和图片已保存: {model_filename}, {figure_filename}")
    print(f"===== [ 第 {run_index} 次训练执行完毕 ] =====")
    
    return test_mse

### --- 主执行循环 --- ###
if __name__ == '__main__':
    # --- 在这里配置 ---
    # 定义你想运行多少次训练
    NUM_RUNS = 5
    # 定义文件名的起始数字
    START_RUN_INDEX = 1

    all_test_results = []
    for i in range(NUM_RUNS):
        run_idx = START_RUN_INDEX + i
        test_mse = train_single_run(run_index=run_idx)
        all_test_results.append(test_mse)

    # --- 最终结果总结 ---
    print("\n\n=================================================")
    print("              所有训练任务已完成")
    print("=================================================")
    print(f"在 {NUM_RUNS} 次独立训练中的测试集 MSE 结果:")
    for i, mse in enumerate(all_test_results):
        print(f"  - 第 {START_RUN_INDEX + i} 次: {mse:.6f}")
    
    mean_mse = np.mean(all_test_results)
    std_mse = np.std(all_test_results)
    print("\n--- 统计结果 ---")
    print(f"平均 MSE: {mean_mse:.6f}")
    print(f"标准差:   {std_mse:.6f}")
    print("=================================================")

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
from datetime import datetime

# ---------- 0. 随机种子初始化 ----------
# 这部分执行一次
torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# ---------- 1. 数据加载 (EMG + Ultrasound + labels + actions) ----------
# 这部分执行一次
with h5py.File('train_val_data_subject73set_seq6_withlabel_new.mat', 'r') as f:
    emg_train_in_np  = f['all_train_inputs'][()]
    emg_train_lb_np  = f['all_train_labels'][()]
    emg_train_act_np = f['all_train_actions'][()]
    emg_val_in_np    = f['all_valid_inputs'][()]
    emg_val_lb_np    = f['all_valid_labels'][()]
    emg_val_act_np   = f['all_valid_actions'][()]

with h5py.File('us_train_val_data_subject7filtered.mat', 'r') as f:
    us_train_in_np  = f['all_train_inputs'][()]
    us_train_lb_np  = f['all_train_labels'][()]
    us_train_act_np = f['all_train_actions'][()]
    us_val_in_np    = f['all_valid_inputs'][()]
    us_val_lb_np    = f['all_valid_labels'][()]
    us_val_act_np   = f['all_valid_actions'][()]

# EMG: (T, W, C, H, N) -> (N, T, C, H, W)
train_emg_np = np.transpose(emg_train_in_np, (4,0,2,1,3))
val_emg_np   = np.transpose(emg_val_in_np,   (4,0,2,1,3))
train_lb_np  = emg_train_lb_np.T
val_lb_np    = emg_val_lb_np.T
train_act_np = emg_train_act_np.reshape(-1)
val_act_np   = emg_val_act_np.reshape(-1)

# US: already (N, T, C, H, W)
train_us_np  = us_train_in_np
val_us_np    = us_val_in_np

assert us_train_lb_np.T.shape == train_lb_np.shape
assert us_val_lb_np.T.shape   == val_lb_np.shape

# ---------- 2. 转为 Tensor & 单个 DataLoader 构建 ----------
# 这部分执行一次
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_us    = torch.tensor(train_us_np,    dtype=torch.float32).to(device)
val_us      = torch.tensor(val_us_np,      dtype=torch.float32).to(device)
train_emg   = torch.tensor(train_emg_np,   dtype=torch.float32).to(device)
val_emg     = torch.tensor(val_emg_np,     dtype=torch.float32).to(device)
train_lbl   = torch.tensor(train_lb_np,    dtype=torch.float32).to(device)
val_lbl     = torch.tensor(val_lb_np,      dtype=torch.float32).to(device)
train_act   = torch.tensor(train_act_np,   dtype=torch.long).to(device)
val_act     = torch.tensor(val_act_np,     dtype=torch.long).to(device)

train_ds = TensorDataset(train_us, train_emg, train_act, train_lbl)
val_ds   = TensorDataset(val_us,   val_emg,   val_act,   val_lbl)

BATCH = 64
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

# ---------- 3. US 分支: 【消融后】的纯 CNN 结构 ----------
class USBranch(nn.Module):
    def __init__(self, C, H, W, num_actions, d_emb=64, num_heads=4, drop=0.0):
        super().__init__()
        self.transpose_conv = nn.Sequential(
            nn.ConvTranspose2d(C, 8, 2,2), nn.BatchNorm2d(8), nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 3, 2,2), nn.BatchNorm2d(3), nn.LeakyReLU(),
            nn.ConvTranspose2d(3, 1, 2,2), nn.BatchNorm2d(1), nn.LeakyReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1,  3, 2,1), nn.BatchNorm2d(3),  nn.LeakyReLU(),
            nn.Conv2d(3,  8, 2,1), nn.BatchNorm2d(8),  nn.LeakyReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(8, 16, 2,1), nn.BatchNorm2d(16), nn.LeakyReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(16,32, 2,1,padding=2), nn.BatchNorm2d(32), nn.LeakyReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64, 2,1),                     nn.LeakyReLU(),             nn.MaxPool2d(2,2),
        )
        self.flatten = nn.Flatten()
    def forward(self, x, action_id):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.transpose_conv(x)
        x = self.conv(x)
        feat = self.flatten(x)
        return feat.view(B, T, -1)

# ---------- 4. EMG 分支: 【消融后】的纯 CNN 结构 ----------
class EMGBranch(nn.Module):
    def __init__(self, C, H, W, num_actions, d_emb=64, num_heads=8, drop=0.0):
        super().__init__()
        self.transpose_conv = nn.Sequential(
            nn.ConvTranspose2d(C, 8, 2,2), nn.BatchNorm2d(8), nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 3, 2,2), nn.BatchNorm2d(3), nn.LeakyReLU(),
            nn.ConvTranspose2d(3, 1, 2,2), nn.BatchNorm2d(1), nn.LeakyReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1,3,2),       nn.LeakyReLU(), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(3,8,2),       nn.LeakyReLU(), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(8,16,2,padding=1), nn.LeakyReLU(), nn.MaxPool2d((2,2),(2,2)),
            nn.Conv2d(16,32,2,padding=1),nn.LeakyReLU(), nn.MaxPool2d((2,2),(2,2)),
            nn.Conv2d(32,64,2),     nn.LeakyReLU(), nn.MaxPool2d((2,2),(2,2)),
        )
        self.flatten = nn.Flatten()
    def forward(self, x, action_id):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.transpose_conv(x)
        x = self.conv(x)
        feat = self.flatten(x)
        return feat.view(B, T, -1)

# ---------- 5. 融合主干 & 头部 (仅回归) ----------
class SensorFusion(nn.Module):
    def __init__(self, T, us_C,us_H,us_W, emg_C,emg_H,emg_W,
                 num_actions, out_dim):
        super().__init__()
        self.us_branch  = USBranch(us_C, us_H, us_W, num_actions)
        self.emg_branch = EMGBranch(emg_C, emg_H, emg_W, num_actions)
        
        with torch.no_grad():
            du = torch.zeros(1,T,us_C,us_H,us_W)
            de = torch.zeros(1,T,emg_C,emg_H,emg_W)
            act_dummy = torch.zeros(1,dtype=torch.long)
            
            fu = self.us_branch(du, act_dummy)
            fe = self.emg_branch(de, act_dummy)
            
        fuse_dim = fu.size(2) + fe.size(2)
        self.lstm   = nn.LSTM(fuse_dim, 64, batch_first=True)
        self.fc_reg = nn.Linear(64, out_dim)

    def forward(self, us, emg, act):
        seq_us = self.us_branch(us, act)
        seq_em = self.emg_branch(emg, act)
        seq    = torch.cat([seq_us, seq_em], dim=2)
        out, _ = self.lstm(seq)
        h_last = out[:, -1, :]
        return self.fc_reg(h_last)


### --- 新的训练函数 --- ###
# 将完整的单次训练流程封装起来
def train_single_run(run_index):
    """
    为给定的运行索引（run_index）执行一次完整的训练。
    run_index 将被用于命名输出文件。
    """
    print(f"--- 开始执行训练，当前索引: {run_index} ---")

    # ---------- 6. 初始化 & 训练配置 ----------
    # 这部分在每次函数调用时都会重新执行，以确保模型是全新的
    _, T, us_C, us_H, us_W    = train_us.shape
    _, _, emg_C,emg_H,emg_W = train_emg.shape
    num_actions = int(train_act.max().item()) + 1
    out_dim     = train_lbl.size(1)

    model       = SensorFusion(T, us_C,us_H,us_W, emg_C,emg_H,emg_W,
                               num_actions, out_dim).to(device)
    crit_r      = nn.MSELoss()
    opt         = optim.Adam(model.parameters(), lr=1e-3)
    best_val, wait, patience = float('inf'), 0, 20
    best_wts = None
    train_losses, val_losses = [], []

    # ---------- 7. 训练循环 ----------
    for epoch in range(1, 501):
        model.train()
        tr_sum = 0.0
        for us_b, emg_b, a_b, y_b in train_loader:
            opt.zero_grad()
            pr = model(us_b, emg_b, a_b)
            loss_r = crit_r(pr, y_b)
            loss_r.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_sum += loss_r.item() * us_b.size(0)
        tr_mse = tr_sum / len(train_loader.dataset)
        train_losses.append(tr_mse)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for us_v, emg_v, a_v, y_v in val_loader:
                pr = model(us_v, emg_v, a_v)
                val_sum += crit_r(pr, y_v).item() * us_v.size(0)
        val_mse = val_sum / len(val_loader.dataset)
        val_losses.append(val_mse)

        print(f"  轮次 {epoch:3d} — 训练 MSE: {tr_mse:.6f}, 验证 MSE: {val_mse:.6f}")
        if val_mse < best_val:
            best_val, wait, best_wts = val_mse, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                print(f"  在轮次 {epoch} 提前停止")
                break

    # ---------- 8. 保存 & 绘图 ----------
    # 文件名现在使用 run_index 来确保唯一性
    model.load_state_dict(best_wts)
    os.makedirs('models', exist_ok=True)
    
    # --- 已更改 --- 动态的模型文件名
    model_filename = f'models/sensorfusion_cnnlstm_subject7{run_index}.pth'
    torch.save(model.state_dict(), model_filename)

    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='Train MSE')
    plt.plot(val_losses,   '--', label='Val MSE')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.tight_layout()
    os.makedirs('figures', exist_ok=True)

    # --- 已更改 --- 动态的图片文件名，防止被覆盖
    figure_filename = f'figures/sensorfusion_loss_run{run_index}.png'
    plt.savefig(figure_filename, dpi=300)
    plt.close()

    print(f"--- 训练 {run_index} 已完成。模型保存在 {model_filename}，图片保存在 {figure_filename} ---")


### --- 主执行循环 --- ###
if __name__ == '__main__':
    # --- 在这里配置 ---
    # 定义你想运行多少次训练
    NUM_RUNS = 3
    # 定义文件名的起始数字
    START_RUN_INDEX = 1

    for i in range(NUM_RUNS):
        current_run_index = START_RUN_INDEX + i
        train_single_run(run_index=current_run_index)

    print("\n所有训练任务已全部完成！")
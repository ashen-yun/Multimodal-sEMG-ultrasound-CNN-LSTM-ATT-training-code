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
torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# ---------- 1. 固定配置 ----------
PRETRAINED_MODEL_PATH = 'models/sensorfusion_cnnlstmatt_ouyang3.pth'
BATCH                 = 96
TRANSFER_LEARNING_LR  = 1e-3
EPOCHS                = 500
PATIENCE              = 15
SUBJECTS              = ['subject7', 'qige', 'jinyang', 'ray', 'yunchen', 'zhitaonew']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- 2. 权重初始化函数 ----------
def weights_init(m):
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# ---------- 3. 网络定义：无分类头版本 ----------
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
            nn.Conv2d(32,64, 2,1),                   nn.LeakyReLU(),             nn.MaxPool2d(2,2),
        )
        self.act_emb = nn.Embedding(num_actions, d_emb)
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            x = self.transpose_conv(dummy); x = self.conv(x)
            _, Cc, Hc, Wc = x.shape
        self.Cc, self.Hc, self.Wc = Cc, Hc, Wc
        self.proj = nn.Linear(Cc, d_emb, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim=d_emb, num_heads=num_heads,
                                          dropout=drop, batch_first=True)
        self.flatten = nn.Flatten()

    def forward(self, x, action_id):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.transpose_conv(x); x = self.conv(x)
        emb = x.view(B*T, self.Cc, self.Hc*self.Wc).permute(0,2,1)
        emb = self.proj(emb)
        a = self.act_emb(action_id).unsqueeze(1).expand(-1, T, -1).reshape(B*T,1,-1)
        attn_out, attn_w = self.attn(a, emb, emb)
        att_map = attn_w.squeeze(1).view(B*T,1,self.Hc,self.Wc)
        x_att   = x * att_map
        feat_c  = self.flatten(x_att)
        context = attn_out.squeeze(1)
        feat = torch.cat([feat_c, context], dim=1)
        return feat.view(B, T, -1)

class EMGBranch(nn.Module):
    def __init__(self, C, H, W, num_actions, d_emb=64, num_heads=8, drop=0.0):
        super().__init__()
        self.transpose_conv = nn.Sequential(
            nn.ConvTranspose2d(C, 8, 2,2), nn.BatchNorm2d(8), nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 3, 2,2), nn.BatchNorm2d(3), nn.LeakyReLU(),
            nn.ConvTranspose2d(3, 1, 2,2), nn.BatchNorm2d(1), nn.LeakyReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1,3,2),            nn.LeakyReLU(), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(3,8,2),            nn.LeakyReLU(), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(8,16,2,padding=1), nn.LeakyReLU(), nn.MaxPool2d((2,2),(2,2)),
            nn.Conv2d(16,32,2,padding=1),nn.LeakyReLU(), nn.MaxPool2d((2,2),(2,2)),
            nn.Conv2d(32,64,2),          nn.LeakyReLU(), nn.MaxPool2d((2,2),(2,2)),
        )
        self.act_emb = nn.Embedding(num_actions, d_emb)
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            x = self.transpose_conv(dummy); x = self.conv(x)
            _, Cc, Hc, Wc = x.shape
        self.Cc, self.Hc, self.Wc = Cc, Hc, Wc
        self.proj = nn.Linear(Cc, d_emb, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim=d_emb, num_heads=num_heads,
                                          dropout=drop, batch_first=True)
        self.flatten = nn.Flatten()

    def forward(self, x, action_id):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.transpose_conv(x); x = self.conv(x)
        emb = x.view(B*T, self.Cc, self.Hc*self.Wc).permute(0,2,1)
        emb = self.proj(emb)
        a = self.act_emb(action_id).unsqueeze(1).expand(-1, T, -1).reshape(B*T,1,-1)
        attn_out, attn_w = self.attn(a, emb, emb)
        att_map = attn_w.squeeze(1).view(B*T,1,self.Hc,self.Wc)
        x_att = x * att_map
        feat_c = self.flatten(x_att)
        context = attn_out.squeeze(1)
        feat = torch.cat([feat_c, context], dim=1)
        return feat.view(B, T, -1)

class SensorFusion(nn.Module):
    def __init__(self, T, us_C, us_H, us_W, emg_C, emg_H, emg_W, num_actions, out_dim):
        super().__init__()
        self.us_branch  = USBranch(us_C, us_H, us_W, num_actions)
        self.emg_branch = EMGBranch(emg_C, emg_H, emg_W, num_actions)
        with torch.no_grad():
            du = torch.zeros(1, T, us_C, us_H, us_W)
            de = torch.zeros(1, T, emg_C, emg_H, emg_W)
            fu = self.us_branch(du, torch.zeros(1, dtype=torch.long))
            fe = self.emg_branch(de, torch.zeros(1, dtype=torch.long))
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

# ======================================================================
# ---------- 4. 转移学习循环 --------------------------------------------
# ======================================================================
for subj in SUBJECTS:
    print(f"\n=== Transfer Learning to subject: {subj} ===")

    # ----- 加载数据 -----
    emg_file = f'train_val_data_{subj}3set_seq6_withlabel_new_transfer.mat'
    us_file  = f'us_train_val_data_{subj}transfer.mat'
    with h5py.File(emg_file, 'r') as f:
        emg_train_in_np  = f['all_train_inputs'][()]
        emg_train_lb_np  = f['all_train_labels'][()]
        emg_train_act_np = f['all_train_actions'][()]
        emg_val_in_np    = f['all_valid_inputs'][()]
        emg_val_lb_np    = f['all_valid_labels'][()]
        emg_val_act_np   = f['all_valid_actions'][()]
    with h5py.File(us_file, 'r') as f:
        us_train_in_np  = f['all_train_inputs'][()]
        us_train_lb_np  = f['all_train_labels'][()]
        us_train_act_np = f['all_train_actions'][()]
        us_val_in_np    = f['all_valid_inputs'][()]
        us_val_lb_np    = f['all_valid_labels'][()]
        us_val_act_np   = f['all_valid_actions'][()]

    # ----- 预处理维度 -----
    train_emg_np = np.transpose(emg_train_in_np, (4,0,2,1,3))
    val_emg_np   = np.transpose(emg_val_in_np,   (4,0,2,1,3))
    train_us_np  = us_train_in_np
    val_us_np    = us_val_in_np
    train_lbl_np = emg_train_lb_np.T
    val_lbl_np   = emg_val_lb_np.T
    train_act_np = emg_train_act_np.reshape(-1)
    val_act_np   = emg_val_act_np.reshape(-1)

    # ----- 转为 Tensor & DataLoader -----
    train_us    = torch.tensor(train_us_np,   dtype=torch.float32).to(device)
    val_us      = torch.tensor(val_us_np,     dtype=torch.float32).to(device)
    train_emg   = torch.tensor(train_emg_np,  dtype=torch.float32).to(device)
    val_emg     = torch.tensor(val_emg_np,    dtype=torch.float32).to(device)
    train_lbl   = torch.tensor(train_lbl_np,  dtype=torch.float32).to(device)
    val_lbl     = torch.tensor(val_lbl_np,    dtype=torch.float32).to(device)
    train_act   = torch.tensor(train_act_np,  dtype=torch.long).to(device)
    val_act     = torch.tensor(val_act_np,    dtype=torch.long).to(device)

    train_ds = TensorDataset(train_us, train_emg, train_act, train_lbl)
    val_ds   = TensorDataset(val_us,   val_emg,   val_act,   val_lbl)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

    # ----- 初始化模型 & 加载预训练 -----
    _, T, us_C, us_H, us_W   = train_us.shape
    _, _, emg_C, emg_H, emg_W = train_emg.shape
    num_actions = int(train_act.max().item()) + 1
    out_dim     = train_lbl.size(1)

    model = SensorFusion(T, us_C, us_H, us_W, emg_C, emg_H, emg_W, num_actions, out_dim).to(device)
    print(f"Loading pretrained model from: {PRETRAINED_MODEL_PATH}")
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device), strict=False)
    print("Pretrained model loaded (strict=False).")

    # ----- 冻结 conv 层 & 重初始化 transpose_conv -----
    for param in model.us_branch.conv.parameters():  param.requires_grad = False
    for param in model.emg_branch.conv.parameters(): param.requires_grad = False
    model.us_branch.transpose_conv.apply(weights_init)
    model.emg_branch.transpose_conv.apply(weights_init)
    print("Frozen 'conv' layers and reinitialized 'transpose_conv'.")

    # ----- 优化器 & 损失 -----
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer        = optim.Adam(trainable_params, lr=TRANSFER_LEARNING_LR)
    crit             = nn.MSELoss()

    best_val, wait = float('inf'), 0
    best_wts       = None
    train_losses, val_losses = [], []

    # ----- 训练循环 -----
    for epoch in range(1, EPOCHS+1):
        model.train()
        tr_sum = 0.0
        for us_b, emg_b, a_b, y_b in train_loader:
            optimizer.zero_grad()
            pr = model(us_b, emg_b, a_b)
            loss = crit(pr, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
            optimizer.step()
            tr_sum += loss.item() * us_b.size(0)
        tr_mse = tr_sum / len(train_loader.dataset)
        train_losses.append(tr_mse)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for us_v, emg_v, a_v, y_v in val_loader:
                pr = model(us_v, emg_v, a_v)
                val_sum += crit(pr, y_v).item() * us_v.size(0)
        val_mse = val_sum / len(val_loader.dataset)
        val_losses.append(val_mse)

        print(f"  Epoch {epoch:3d}/{EPOCHS} — Train MSE: {tr_mse:.6f}, Val MSE: {val_mse:.6f}")

        if val_mse < best_val:
            best_val, wait, best_wts = val_mse, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ----- 保存最优模型 & 绘制损失曲线 -----
    model.load_state_dict(best_wts)
    out_dir = f'models/transfer_learned_sensorfusion_fromouyangto_{subj}'
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f'model_transfer_ouyangto_{subj}.pth')
    torch.save(model.state_dict(), model_path)

    fig_dir = f'figures/transfer_learned_retrain_{subj}'
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f'loss_transfer_ouyangto_{subj}.png')
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='Train MSE')
    plt.plot(val_losses,   '--', label='Val MSE')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Saved fine‑tuned model to: {model_path}")
    print(f"Saved loss curve to:     {fig_path}")

print("\nAll subjects processed.")

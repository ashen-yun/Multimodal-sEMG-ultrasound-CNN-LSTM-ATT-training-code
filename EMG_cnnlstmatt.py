#!/usr/bin/env python3
import os
import h5py
import torch
import torch.nn as nn
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# ---------- 1. Initialize Random Seeds -------------------------------------
# =============================================================================
torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# =============================================================================
# ---------- 2. Data Loading and Preprocessing ------------------------------
# =============================================================================
# 这部分代码在脚本开始时执行一次
with h5py.File('train_val_data_subject73set_seq6_withlabel_new.mat', 'r') as f:
    train_inputs_np = np.transpose(f['all_train_inputs'][()], (4,0,2,1,3))
    val_inputs_np   = np.transpose(f['all_valid_inputs'][()], (4,0,2,1,3))
    train_labels_np = f['all_train_labels'][()].T
    val_labels_np   = f['all_valid_labels'][()].T
    train_actions_np = f['all_train_actions'][()].reshape(-1)
    val_actions_np   = f['all_valid_actions'][()].reshape(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def create_loader(inputs_np, actions_np, labels_np, shuffle=False, batch_size=64):
    if inputs_np.size == 0: return None
    dataset = TensorDataset(
        torch.tensor(inputs_np,  dtype=torch.float32).to(device),
        torch.tensor(actions_np, dtype=torch.long).to(device),
        torch.tensor(labels_np, dtype=torch.float32).to(device)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = create_loader(train_inputs_np, train_actions_np, train_labels_np, shuffle=True)
val_loader   = create_loader(val_inputs_np,   val_actions_np,   val_labels_np,   shuffle=False)


# =============================================================================
# ---------- 3. Model Definition --------------------------------------------
# =============================================================================
# 模型定义保持不变
class CNN_LSTM_ActionSpatialAttn(nn.Module):
    def __init__(self, time_steps, channels, height, width,
                 num_actions, num_outputs,
                 d_emb=64, num_heads=8, drop=0.1, lstm_hidden=100):
        super().__init__()
        self.T, self.C, self.H, self.W = time_steps, channels, height, width
        self.d_emb, self.num_heads = d_emb, num_heads
        self.act_emb = nn.Embedding(num_actions, d_emb)
        self.transpose_conv = nn.Sequential(
            nn.ConvTranspose2d(channels, 8, 2, 2), nn.BatchNorm2d(8), nn.LeakyReLU(),
            nn.ConvTranspose2d(8,       3, 2, 2), nn.BatchNorm2d(3), nn.LeakyReLU(),
            nn.ConvTranspose2d(3,       1, 2, 2), nn.BatchNorm2d(1), nn.LeakyReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=2),    nn.LeakyReLU(), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(3, 8, kernel_size=2),    nn.LeakyReLU(), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(8,16, kernel_size=2, padding=1), nn.LeakyReLU(), nn.MaxPool2d((2,2),(2,2)),
            nn.Conv2d(16,32, kernel_size=2, padding=1), nn.LeakyReLU(), nn.MaxPool2d((2,2),(2,2)),
            nn.Conv2d(32,64, kernel_size=2),    nn.LeakyReLU(), nn.MaxPool2d((2,2),(2,2)),
        )
        with torch.no_grad():
            x = self.conv(self.transpose_conv(torch.zeros(1, channels, height, width)))
        _, self.Cc, self.Hc, self.Wc = x.shape
        self.proj = nn.Linear(self.Cc, d_emb, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim=d_emb, num_heads=num_heads, dropout=drop, batch_first=True)
        self.flatten = nn.Flatten()
        feat_dim = self.Cc * self.Hc * self.Wc + d_emb
        self.lstm   = nn.LSTM(feat_dim, lstm_hidden, batch_first=True)
        self.fc_reg = nn.Linear(lstm_hidden, num_outputs)
        self.fc_cls = nn.Linear(lstm_hidden, num_actions)

    def forward(self, x, action_id):
        B, T, C, H, W = x.shape
        x_bt = self.conv(self.transpose_conv(x.view(B*T, C, H, W)))
        emb = self.proj(x_bt.view(B*T, self.Cc, self.Hc*self.Wc).permute(0,2,1))
        a = self.act_emb(action_id).unsqueeze(1).expand(-1, T, -1).reshape(B*T,1,self.d_emb)
        attn_out, attn_w = self.attn(a, emb, emb)
        att_map = attn_w.squeeze(1).view(B*T,1,self.Hc,self.Wc)
        feat_c  = self.flatten(x_bt * att_map)
        context = attn_out.squeeze(1)
        feat = torch.cat([feat_c, context], dim=1)
        seq  = feat.view(B, T, -1)
        out_seq, _ = self.lstm(seq)
        h_last     = out_seq[:, -1, :]
        return self.fc_reg(h_last), self.fc_cls(h_last)

### --- 新的训练函数 --- ###
def train_single_run(run_index):
    """
    为给定的运行索引（run_index）执行一次完整的训练。
    """
    print(f"\n===== [ 开始执行第 {run_index} 次训练 ] =====")
    
    # =============================================================================
    # ---------- 4. Model & Training Configuration ------------------------------
    # =============================================================================
    # 这部分在每次运行时都会重新初始化
    num_actions = int(train_actions_np.max().item()) + 1
    out_dim = train_labels_np.shape[1]

    model = CNN_LSTM_ActionSpatialAttn(
        time_steps  = train_inputs_np.shape[1],
        channels    = train_inputs_np.shape[2],
        height      = train_inputs_np.shape[3],
        width       = train_inputs_np.shape[4],
        num_actions = num_actions,
        num_outputs = out_dim
    ).to(device)

    crit_r = nn.MSELoss()
    crit_c = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    alpha = 0.2
    patience = 20
    max_epochs = 500

    best_val_loss, wait = float('inf'), 0
    best_wts = None
    train_losses, val_losses = [], []
    
    print("模型已初始化。开始训练...")

    # =============================================================================
    # ---------- 5. Training Loop -----------------------------------------------
    # =============================================================================
    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_loss_sum = 0.0
        for X, a, Y in train_loader:
            optimizer.zero_grad()
            pr, pc = model(X, a)
            loss_r = crit_r(pr, Y)
            loss_c = crit_c(pc, a)
            loss = loss_r + alpha * loss_c
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tr_loss_sum += loss_r.item() * X.size(0)
        
        train_mse = tr_loss_sum / len(train_loader.dataset)
        train_losses.append(train_mse)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for X_v, a_v, Y_v in val_loader:
                pr_v, _ = model(X_v, a_v)
                val_loss_sum += crit_r(pr_v, Y_v).item() * X_v.size(0)
        
        val_mse = val_loss_sum / len(val_loader.dataset)
        val_losses.append(val_mse)

        print(f"  轮次 {epoch:3d}/{max_epochs} — 训练 MSE: {train_mse:.6f}, 验证 MSE: {val_mse:.6f}")

        if val_mse < best_val_loss:
            best_val_loss, wait = val_mse, 0
            best_wts = copy.deepcopy(model.state_dict())
            print(f"    -> 新的最佳验证损失: {best_val_loss:.6f}。正在保存模型权重。")
        else:
            wait += 1
            if wait >= patience:
                print(f"验证损失在 {patience} 个轮次内没有改善。提前停止。")
                break
    
    print("\n训练完成。")

    # =============================================================================
    # ---------- 6. Save Best Model and Loss Curve ------------------------------
    # =============================================================================
    if best_wts:
        os.makedirs('models', exist_ok=True)
        # --- 已更改 --- 动态的模型文件名
        save_path = f'models/emg_model_cnnlstmatt_subject7{run_index}.pth'
        torch.save(best_wts, save_path)
        print(f"最佳模型权重已保存到: {save_path}")
    else:
        print("\n警告: 训练未完成或未找到最佳权重，模型未保存。")

    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train MSE Loss')
    plt.plot(val_losses, label='Validation MSE Loss')
    plt.title(f'Training & Validation Loss - Run {run_index}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    # --- 已更改 --- 动态的图片文件名
    figure_path = f'figures/training_loss_curve_run{run_index}.png'
    plt.savefig(figure_path, dpi=300)
    plt.close() # 关闭图形，防止在循环中显示多个窗口
    
    print(f"损失曲线图已保存到: {figure_path}")
    print(f"===== [ 第 {run_index} 次训练执行完毕 ] =====")

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

    print("\n\n所有训练任务已全部完成！")

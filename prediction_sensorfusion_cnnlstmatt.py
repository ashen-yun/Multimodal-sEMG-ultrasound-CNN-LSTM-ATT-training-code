#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate: SensorFusion (US+EMG) branches with CNN+MHA and LSTM head (your last script),
no classification output used at eval time. 
Computes the 8 metrics (Val/Test, Global/Local RMSE & r), aggregates across N checkpoints,
and appends ONE row to the shared CSV (keeping its column order).

Edit the USER CONFIG block only.
"""

import os
import csv
import h5py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict

# ===========================
# USER CONFIG
# ===========================
SUBJECT_NAME     = 'subject5'                         # 第一列
MODEL_NAME       = 'sensorfusion_cnnlstmatt'            # 第二列（自定义）
NUM_MODELS       = 3                                   # 从 subject71 开始
CKPT_TEMPLATE    = 'models/sensorfusion_cnnlstmatt_subject7{}.pth'

CSV_PATH         = 'model_evaluation_subject4567.csv'
SAVE_CSV         = True

verbose          = False
print_per_model  = False
# ===========================

def vprint(*a, **kw):
    if verbose: print(*a, **kw)

# ---------- Metrics ----------
def rmse_per_dim(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))

def corrcoef_per_dim(y_pred, y_true):
    corr = []
    for i in range(y_true.shape[1]):
        if np.std(y_true[:, i]) < 1e-9 or np.std(y_pred[:, i]) < 1e-9:
            corr.append(0.0)
        else:
            corr.append(np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1])
    return np.array(corr)

def compute_local_metrics(y_true, y_pred, actions, action_label_map, num_actions_local):
    local_metrics = {}
    for act_idx in range(num_actions_local):
        if act_idx not in action_label_map:
            continue
        idx = np.where(actions == act_idx)[0]
        if len(idx) < 2:
            continue
        ts, ps = y_true[idx], y_pred[idx]
        local_metrics[act_idx] = {}
        for dim_idx in action_label_map[act_idx]:
            td, pd = ts[:, dim_idx], ps[:, dim_idx]
            ss_tot = np.sum((td - np.mean(td))**2)
            rmse = np.sqrt(np.mean((pd - td)**2))
            r = 0.0 if (np.std(td) < 1e-9 or np.std(pd) < 1e-9) else np.corrcoef(td, pd)[0, 1]
            r2 = 1 - (np.sum((td - pd)**2) / (ss_tot + 1e-9))
            local_metrics[act_idx][dim_idx] = {'rmse': rmse, 'r': r, 'R2': r2}
    return local_metrics

# ---------- 1. Load Data ----------
with h5py.File('train_val_data_subject73set_seq6_withlabel_new.mat', 'r') as f:
    emg_tr = np.transpose(f['all_train_inputs'][()], (4, 0, 2, 1, 3))
    emg_va = np.transpose(f['all_valid_inputs'][()], (4, 0, 2, 1, 3))
    lbl_tr = f['all_train_labels'][()].T
    lbl_va = f['all_valid_labels'][()].T
    act_tr = f['all_train_actions'][()].reshape(-1)
    act_va = f['all_valid_actions'][()].reshape(-1)

with h5py.File('us_train_val_data_subject7filtered.mat', 'r') as f:
    us_tr = f['all_train_inputs'][()]
    us_va = f['all_valid_inputs'][()]
    assert f['all_train_labels'][()].T.shape == lbl_tr.shape
    assert f['all_valid_labels'][()].T.shape == lbl_va.shape

# test
emg_test_file = 'test_data_subject73set_seq6_withlabel_new.mat'
us_test_file  = 'us_test_data_subject7filtered.mat'
if os.path.exists(emg_test_file) and os.path.exists(us_test_file):
    with h5py.File(emg_test_file, 'r') as f:
        emg_te = np.transpose(f['all_test_inputs'][()], (4, 0, 2, 1, 3))
        lbl_te = f['all_test_labels'][()].T
        act_te = f['all_test_actions'][()].reshape(-1)
    with h5py.File(us_test_file, 'r') as f:
        us_te = f['all_test_inputs'][()]
else:
    emg_te, us_te, lbl_te, act_te = (np.array([]),) * 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_loader(us_np, emg_np, act_np, lbl_np, shuffle=False, batch_size=64):
    if us_np.size == 0:
        return None
    ds = TensorDataset(
        torch.tensor(us_np,  dtype=torch.float32).to(device),
        torch.tensor(emg_np, dtype=torch.float32).to(device),
        torch.tensor(act_np, dtype=torch.long).to(device),
        torch.tensor(lbl_np, dtype=torch.float32).to(device)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = create_loader(us_tr, emg_tr, act_tr, lbl_tr, shuffle=False)
val_loader   = create_loader(us_va, emg_va, act_va, lbl_va, shuffle=False)
test_loader  = create_loader(us_te, emg_te, act_te, lbl_te, shuffle=False)

# ---------- 2. Model ----------
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
        self.attn = nn.MultiheadAttention(d_emb, num_heads, dropout=drop, batch_first=True)
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
        feat    = torch.cat([feat_c, context], dim=1)
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
            dummy = torch.zeros(1,C,H,W)
            x = self.transpose_conv(dummy); x = self.conv(x)
            _, Cc, Hc, Wc = x.shape
        self.Cc, self.Hc, self.Wc = Cc, Hc, Wc
        self.proj = nn.Linear(Cc, d_emb, bias=False)
        self.attn = nn.MultiheadAttention(d_emb, num_heads, dropout=drop, batch_first=True)
        self.flatten = nn.Flatten()

    def forward(self, x, action_id):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.transpose_conv(x); x = self.conv(x)
        emb = x.view(B*T, self.Cc, self.Hc*self.Wc).permute(0,2,1)
        emb = self.proj(emb)
        a = self.act_emb(action_id).unsqueeze(1).expand(-1,T,-1).reshape(B*T,1,-1)
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
            du = torch.zeros(1,T,us_C,us_H,us_W)
            de = torch.zeros(1,T,emg_C,emg_H,emg_W)
            fu = self.us_branch(du, torch.zeros(1,dtype=torch.long))
            fe = self.emg_branch(de, torch.zeros(1,dtype=torch.long))
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

# ---------- Common params ----------
num_actions = int(act_tr.max().item()) + 1
out_dim     = lbl_tr.shape[1]
_, T, us_C, us_H, us_W    = us_tr.shape
_, _, emg_C, emg_H, emg_W = emg_tr.shape

num_actions_local = 11
action_label_map = {
    1: [0,1], 2: [2,3], 3: [2,3], 4: [4,5], 5: [6,7],
    6: [6,7], 7: [0,1,2,3], 8: [0,1,2,3], 9: list(range(8)), 10: list(range(8))
}

def infer(model, loader):
    if loader is None: return (np.array([]),)*4
    regs, trues, acts = [], [], []
    with torch.no_grad():
        for us_b, emg_b, a_b, y_b in loader:
            pr = model(us_b, emg_b, a_b)
            regs.append(pr.cpu().numpy())
            trues.append(y_b.cpu().numpy())
            acts.append(a_b.cpu().numpy())
    if not regs: return (np.array([]),)*4
    return np.vstack(regs), np.array([]), np.vstack(trues), np.concatenate(acts)

def evaluate_one_ckpt(path):
    model = SensorFusion(T, us_C, us_H, us_W, emg_C, emg_H, emg_W, num_actions, out_dim).to(device)
    try:
        state = torch.load(path, map_location=device)
    except FileNotFoundError:
        vprint(f"[Skip] not found: {path}")
        return None
    try:
        model.load_state_dict(state)
    except Exception as e:
        vprint(f"[Skip] load fail: {path} -> {e}")
        return None
    model.eval()

    va_pr, _, va_tr, va_ac = infer(model, val_loader)
    te_pr, _, te_tr, te_ac = infer(model, test_loader)

    def set_metrics(pr, tr, ac):
        if pr.size == 0:
            return {'G_RMSE': np.nan, 'G_Corr': np.nan, 'L_RMSE': np.nan, 'L_Corr': np.nan}
        rmse = rmse_per_dim(pr, tr)
        corr = corrcoef_per_dim(pr, tr)
        G_RMSE = float(np.mean(rmse))
        G_Corr = float(np.mean(corr))
        L_RMSE = L_Corr = np.nan
        if ac is not None and ac.size > 0:
            local = compute_local_metrics(tr, pr, ac, action_label_map, num_actions_local)
            all_lr = [m['rmse'] for mets in local.values() for m in mets.values() if mets]
            all_lc = [m['r']    for mets in local.values() for m in mets.values() if mets]
            if all_lr: L_RMSE = float(np.mean(all_lr))
            if all_lc: L_Corr = float(np.mean(all_lc))
        return {'G_RMSE': G_RMSE, 'G_Corr': G_Corr, 'L_RMSE': L_RMSE, 'L_Corr': L_Corr}

    val_m  = set_metrics(va_pr, va_tr, va_ac)
    test_m = set_metrics(te_pr, te_tr, te_ac)

    return OrderedDict([
        ('Val_Global_Avg_RMSE',  val_m['G_RMSE']),
        ('Val_Global_Avg_Corr',  val_m['G_Corr']),
        ('Val_Local_Avg_RMSE',   val_m['L_RMSE']),
        ('Val_Local_Avg_Corr',   val_m['L_Corr']),
        ('Test_Global_Avg_RMSE', test_m['G_RMSE']),
        ('Test_Global_Avg_Corr', test_m['G_Corr']),
        ('Test_Local_Avg_RMSE',  test_m['L_RMSE']),
        ('Test_Local_Avg_Corr',  test_m['L_Corr']),
    ])

# ---------- Loop ----------
all_results = []
model_files = []

for i in range(1, NUM_MODELS + 1):
    ckpt = CKPT_TEMPLATE.format(i)
    res  = evaluate_one_ckpt(ckpt)
    if res is None:
        continue
    all_results.append(list(res.values()))
    model_files.append(os.path.basename(ckpt))
    if print_per_model:
        print(f"\n[{model_files[-1]}]")
        for k, v in res.items():
            print(f"  {k}: {v:.4f}" if not np.isnan(v) else f"  {k}: NaN")

if len(all_results) == 0:
    print("无有效模型，退出。")
    raise SystemExit

all_results = np.array(all_results, dtype=float)

metric_names = [
    'Val_Global_Avg_RMSE',
    'Val_Global_Avg_Corr',
    'Val_Local_Avg_RMSE',
    'Val_Local_Avg_Corr',
    'Test_Global_Avg_RMSE',
    'Test_Global_Avg_Corr',
    'Test_Local_Avg_RMSE',
    'Test_Local_Avg_Corr'
]

means = np.nanmean(all_results, axis=0)
stds  = np.nanstd(all_results,  axis=0, ddof=1)

# ---------- Append CSV ----------
if SAVE_CSV:
    default_cols = [
        'Subject',
        'Model',
        'Mean_Global_Val_RMSE', 'Std_Global_Val_RMSE',
        'Mean_Global_Val_r',    'Std_Global_Val_r',
        'Mean_Global_Test_RMSE','Std_Global_Test_RMSE',
        'Mean_Global_Test_r',   'Std_Global_Test_r',
        'Mean_Local_Val_RMSE',  'Std_Local_Val_RMSE',
        'Mean_Local_Val_r',     'Std_Local_Val_r',
        'Mean_Local_Test_RMSE', 'Std_Local_Test_RMSE',
        'Mean_Local_Test_r',    'Std_Local_Test_r'
    ]
    mean_map = {
        'Val_Global_Avg_RMSE': 'Mean_Global_Val_RMSE',
        'Val_Global_Avg_Corr': 'Mean_Global_Val_r',
        'Val_Local_Avg_RMSE':  'Mean_Local_Val_RMSE',
        'Val_Local_Avg_Corr':  'Mean_Local_Val_r',
        'Test_Global_Avg_RMSE':'Mean_Global_Test_RMSE',
        'Test_Global_Avg_Corr':'Mean_Global_Test_r',
        'Test_Local_Avg_RMSE': 'Mean_Local_Test_RMSE',
        'Test_Local_Avg_Corr': 'Mean_Local_Test_r'
    }
    std_map = {
        'Val_Global_Avg_RMSE': 'Std_Global_Val_RMSE',
        'Val_Global_Avg_Corr': 'Std_Global_Val_r',
        'Val_Local_Avg_RMSE':  'Std_Local_Val_RMSE',
        'Val_Local_Avg_Corr':  'Std_Local_Val_r',
        'Test_Global_Avg_RMSE':'Std_Global_Test_RMSE',
        'Test_Global_Avg_Corr':'Std_Global_Test_r',
        'Test_Local_Avg_RMSE': 'Std_Local_Test_RMSE',
        'Test_Local_Avg_Corr': 'Std_Local_Test_r'
    }

    if os.path.exists(CSV_PATH):
        df_exist = pd.read_csv(CSV_PATH)
        cols = list(df_exist.columns)
    else:
        cols = default_cols
        pd.DataFrame(columns=cols).to_csv(CSV_PATH, index=False)

    row = {c: np.nan for c in cols}
    row['Subject'] = SUBJECT_NAME
    row['Model']   = MODEL_NAME

    for key, m in zip(metric_names, means):
        col = mean_map.get(key)
        if col in row: row[col] = m
    for key, s in zip(metric_names, stds):
        col = std_map.get(key)
        if col in row: row[col] = s

    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writerow(row)

# ---------- Final Prints ----------
print("\nModels evaluated:")
for nm in model_files:
    print("  -", nm)

print("\n" + "="*60)
print(f"=== METRICS AGGREGATED OVER {len(model_files)} MODELS ===")
print("="*60)
for name, m, s in zip(metric_names, means, stds):
    if np.isnan(m):
        print(f"{name:>26}:  mean=NaN   std=NaN")
    else:
        print(f"{name:>26}:  mean={m:.4f}   std={s:.4f}")

print(f"\nRow appended to: {CSV_PATH}" if SAVE_CSV else "\n(No CSV save)")
print("Done.")

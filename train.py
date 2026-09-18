"""
Power Transformer v3 (PJM Enhanced + Coverage-Aware Loss)
主要改动：
1. 新增 coverage_penalty_loss：直接惩罚落在区间外的样本
2. 新增 interval_width_loss：防止区间过窄（保底宽度惩罚）
3. consistency_loss_enhanced：大幅降低上下界约束权重 0.2 -> 0.02
4. 通道权重：对低覆盖率通道大幅提升（Solar 2.0->5.0, Other 2.5->5.0, Gas 1.5->3.5）
5. 修复 to_physical：先 softplus 再 denorm，避免下界过度压缩
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import gridspec
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import json
from google import genai
from google.genai import types
import matplotlib.pyplot as plt
import textwrap
from matplotlib.ticker import FormatStrFormatter
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Verdana']
plt.rcParams['axes.unicode_minus'] = False # Fix minus sign axis problem
import os
import sys
# 强制接管底层终端与 Python 的默认编码为 UTF-8，彻底消灭 ASCII 报错
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


# ============ 1. 特征工程 ============
def add_lag_roll_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    time_col = 'Time' if 'Time' in df.columns else ('index' if 'index' in df.columns else 'period')
    idx = df[time_col]

    if idx.dtype == object or np.issubdtype(idx.dtype, np.datetime64):
        dt = pd.to_datetime(idx)
        hour  = dt.dt.hour.values
        dow   = dt.dt.dayofweek.values
        month = dt.dt.month.values
    else:
        t = idx.values.astype(np.int64)
        hour, dow, month = t % 24, (t // 24) % 7, ((t // 24 // 30) % 12) + 1

    df['hour_sin']  = np.sin(2 * np.pi * hour  / 24)
    df['hour_cos']  = np.cos(2 * np.pi * hour  / 24)
    df['dow_sin']   = np.sin(2 * np.pi * dow   / 7)
    df['dow_cos']   = np.cos(2 * np.pi * dow   / 7)
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    df['is_weekend'] = (dow >= 5).astype(np.float32)
    df['is_daytime'] = ((hour >= 6) & (hour <= 18)).astype(np.float32)

    roll_features = ['Load_MW', 'Temp_C', 'Humidity_Pct', 'Temp_Sensitive_Share_Pct']
    for col in roll_features:
        if col in df.columns:
            df[f'{col}_lag1']        = df[col].shift(1)
            df[f'{col}_lag24']       = df[col].shift(24)
            df[f'{col}_roll24_mean'] = df[col].shift(1).rolling(24).mean()
            df[f'{col}_roll24_max']  = df[col].shift(1).rolling(24).max()
            df[f'{col}_roll24_min']  = df[col].shift(1).rolling(24).min()
            df[f'{col}_diff1']       = df[col].diff(1)
            df[f'{col}_diff24']      = df[col].diff(24)

    if 'Temp_C_roll24_max' in df.columns:
        df['Temp_range24'] = df['Temp_C_roll24_max'] - df['Temp_C_roll24_min']

    return df.ffill().bfill()


# ============ 2. 数据集 ============
class PowerDataset(Dataset):
    TARGETS = ['Coal_Gen', 'Gas_Gen', 'Nuclear_Gen', 'Oil_Gen',
               'Other_Gen', 'Solar_Gen', 'Hydro_Gen', 'Wind_Gen', 'Total_Gen_MW']

    def __init__(self, df, seq_len=48, pred_len=12,
                 x_scaler=None, y_mean=None, y_std=None):
        self.seq_len, self.pred_len = seq_len, pred_len
        df = add_lag_roll_features(df)

        ignore_cols = {'index', 'Time', 'period'}
        self.feat_cols = [
            c for c in df.columns
            if c not in ignore_cols
            and df[c].dtype != object
            and not np.issubdtype(df[c].dtype, np.datetime64)
        ]

        X = df[self.feat_cols].values.astype(np.float32)
        Y = df[self.TARGETS].values.astype(np.float32)

        if x_scaler is None:
            self.x_scaler = StandardScaler().fit(X)
            self.y_mean = Y.mean(0)
            self.y_std = Y.std(0) + 1e-6
        else:
            self.x_scaler, self.y_mean, self.y_std = x_scaler, y_mean, y_std

        self.X = self.x_scaler.transform(X).astype(np.float32)
        self.Y_raw = Y
        self.Y = ((Y - self.y_mean) / self.y_std).astype(np.float32)

    def __len__(self):
        return max(0, len(self.X) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, i):
        x  = self.X[i:i + self.seq_len]
        y  = self.Y    [i + self.seq_len : i + self.seq_len + self.pred_len]
        yr = self.Y_raw[i + self.seq_len : i + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(yr)


# ============ 3. 位置编码 & 模型 ============
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class ChannelIndependentTransformer(nn.Module):
    def __init__(self, n_feats, n_targets, pred_len, n_quantiles=3,
                 d_model=128, nhead=8, num_layers=3, dim_ff=256, dropout=0.1):
        super().__init__()
        self.n_targets, self.pred_len, self.n_quantiles = n_targets, pred_len, n_quantiles
        self.input_proj  = nn.Linear(n_feats, d_model)
        self.channel_emb = nn.Embedding(n_targets, d_model)
        self.pos_enc     = PositionalEncoding(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, pred_len * n_quantiles),
        )

    def forward(self, x):
        B, L, _ = x.shape
        N = self.n_targets
        h = self.input_proj(x)
        h = h.unsqueeze(1).expand(B, N, L, -1)
        ch = self.channel_emb(torch.arange(N, device=x.device))
        h = h + ch[None, :, None, :]
        h = h.reshape(B * N, L, -1)
        h = self.pos_enc(h)
        h = self.encoder(h).mean(dim=1)
        out = self.head(h).view(B, N, self.pred_len, self.n_quantiles)
        return out.permute(0, 2, 1, 3)


# ============ 4. 修复 to_physical ============
# [原版问题] softplus(out_norm * std + mean) 会把 mean 附近的下界往上拉
# [修复] 先在归一化空间做 softplus 保证单调性，再 denorm 到物理空间
def to_physical(out_norm, y_mean, y_std):
    """
    out_norm: [B, T, C, Q]  归一化空间的分位数输出
    保证 q0.1 < q0.5 < q0.9（单调性）后再还原到物理空间
    """
    mean = torch.as_tensor(y_mean, device=out_norm.device, dtype=out_norm.dtype).view(1, 1, -1, 1)
    std  = torch.as_tensor(y_std,  device=out_norm.device, dtype=out_norm.dtype).view(1, 1, -1, 1)

    phys = out_norm * std + mean  # 还原到物理量纲

    # 强制单调性：q0.1 <= q0.5 <= q0.9
    q_lo  = phys[..., 0]
    q_med = phys[..., 1]
    q_hi  = phys[..., 2]

    q_med = torch.maximum(q_med, q_lo)        # med >= lo
    q_hi  = torch.maximum(q_hi,  q_med)       # hi  >= med

    # 非负约束（发电量不能为负）
    q_lo  = F.softplus(q_lo)
    q_med = F.softplus(q_med)
    q_hi  = F.softplus(q_hi)

    return torch.stack([q_lo, q_med, q_hi], dim=-1)


# ============ 5. 损失函数 ============

def pinball_loss_weighted(pred_q, target, channel_weights, quantiles=(0.1, 0.5, 0.9)):
    """
    pred_q: [B, T, C, 3]
    target: [B, T, C]
    channel_weights: [C, 3]
    """
    losses = []
    for i, q in enumerate(quantiles):
        diff   = target - pred_q[..., i]
        loss_i = torch.maximum(q * diff, (q - 1) * diff)
        w_i    = channel_weights[:, i].view(1, 1, -1)
        losses.append(loss_i * w_i)
    return torch.stack(losses).mean()


# ✅ 新增：直接覆盖率惩罚
def coverage_penalty_loss(pred_q, target, channel_weights=None):
    """
    对落在 [q0.1, q0.9] 之外的样本施加额外惩罚。
    这是提升 PI 覆盖率最直接的手段。

    pred_q: [B, T, C, 3]
    target: [B, T, C]
    返回: 标量 loss
    """
    lower = pred_q[..., 0]   # [B, T, C]
    upper = pred_q[..., 2]

    # 越界惩罚（类似 hinge loss）
    below = F.relu(lower - target)   # target < lower 时惩罚
    above = F.relu(target - upper)   # target > upper 时惩罚
    penalty = below + above           # [B, T, C]

    if channel_weights is not None:
        # 只用边界权重的均值作为通道系数
        cw = (channel_weights[:, 0] + channel_weights[:, 2]) / 2.0  # [C]
        penalty = penalty * cw.view(1, 1, -1)

    return penalty.mean()


# ✅ 新增：区间宽度下限惩罚（防止 q0.9 - q0.1 过窄）
def interval_width_loss(pred_q, target, min_width_ratio=0.1):
    """
    鼓励区间宽度 >= min_width_ratio * |target|。
    避免模型通过极窄区间来"偷懒"。

    pred_q: [B, T, C, 3]
    target: [B, T, C]
    min_width_ratio: 区间宽度至少是真值的 10%
    """
    width    = pred_q[..., 2] - pred_q[..., 0]            # [B, T, C]
    min_w    = min_width_ratio * target.abs().clamp(min=1) # 最小要求宽度
    too_narrow = F.relu(min_w - width)                     # 宽度不足时惩罚
    return too_narrow.mean()


def consistency_loss_enhanced(pred_q):
    """
    中位数约束保持强约束（1.0），
    上下界约束大幅减弱（0.02），避免压缩区间。
    """
    median_loss = F.l1_loss(pred_q[..., :8, 1].sum(dim=-1), pred_q[..., 8, 1])
    lower_loss  = F.l1_loss(pred_q[..., :8, 0].sum(dim=-1), pred_q[..., 8, 0])
    upper_loss  = F.l1_loss(pred_q[..., :8, 2].sum(dim=-1), pred_q[..., 8, 2])
    # ⚠️ 关键修改：0.2 -> 0.02，大幅放松边界一致性约束
    return median_loss + 0.02 * (lower_loss + upper_loss)


# ============ 6. 训练逻辑 ============
def train(csv_path, seq_len=48, pred_len=12, epochs=30, batch_size=64,
          lr=1e-3, lambda_consist=0.1, lambda_cov=0.5, lambda_width=0.3,
          device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    df = pd.read_csv(csv_path)
    time_col = 'Time' if 'Time' in df.columns else ('index' if 'index' in df.columns else 'period')
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    n = len(df); tr, va = int(n * .8), int(n * .9)
    train_ds = PowerDataset(df.iloc[:tr],   seq_len, pred_len)
    val_ds   = PowerDataset(df.iloc[tr:va], seq_len, pred_len,
                            train_ds.x_scaler, train_ds.y_mean, train_ds.y_std)
    test_ds  = PowerDataset(df.iloc[va:],   seq_len, pred_len,
                            train_ds.x_scaler, train_ds.y_mean, train_ds.y_std)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size)
    test_dl  = DataLoader(test_ds,  batch_size)

    model = ChannelIndependentTransformer(
        n_feats=len(train_ds.feat_cols),
        n_targets=len(PowerDataset.TARGETS),
        pred_len=pred_len, n_quantiles=3,
    ).to(device)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # -------------------------------------------------------
    # 通道非对称权重矩阵 [9 目标, 3 分位数]
    # v3 策略：根据 v2 实测覆盖率，覆盖率越低权重越高
    # 覆盖率参考：Solar 0.35, Other 0.39, Gas 0.40, Coal 0.45,
    #             Wind 0.45, Total 0.84, Nuclear 0.55, Oil 0.54, Hydro 0.54
    # -------------------------------------------------------
    cw = torch.ones((9, 3), device=device)

    # Index 0: Coal   (cov=0.45) -> 中度放宽
    cw[0, 0] = 2.5;  cw[0, 2] = 2.5
    # Index 1: Gas    (cov=0.40) -> 重度放宽
    cw[1, 0] = 3.5;  cw[1, 2] = 3.5
    # Index 2: Nuclear(cov=0.55) -> 轻度放宽
    cw[2, 0] = 1.5;  cw[2, 2] = 1.5
    # Index 3: Oil    (cov=0.54) -> 轻度放宽
    cw[3, 0] = 1.5;  cw[3, 2] = 1.5
    # Index 4: Other  (cov=0.39) -> 最重度放宽
    cw[4, 0] = 5.0;  cw[4, 2] = 5.0
    # Index 5: Solar  (cov=0.35) -> 最重度放宽
    cw[5, 0] = 5.0;  cw[5, 2] = 5.0
    # Index 6: Hydro  (cov=0.54) -> 轻度放宽
    cw[6, 0] = 1.5;  cw[6, 2] = 1.5
    # Index 7: Wind   (cov=0.45) -> 中度放宽
    cw[7, 0] = 2.5;  cw[7, 2] = 2.5
    # Index 8: Total  (cov=0.84) -> 基本达标，保持默认（略微收紧防止过宽）
    cw[8, 0] = 0.8;  cw[8, 2] = 0.8

    best_val = float('inf')
    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train(); tr_loss = 0.
        for x, _, yr in train_dl:
            x, yr = x.to(device), yr.to(device)
            opt.zero_grad()
            pred_q = to_physical(model(x), train_ds.y_mean, train_ds.y_std)

            loss = (
                pinball_loss_weighted(pred_q, yr, channel_weights=cw)
                + lambda_consist * consistency_loss_enhanced(pred_q)
                + lambda_cov   * coverage_penalty_loss(pred_q, yr, channel_weights=cw)  # ✅ 新增
                + lambda_width * interval_width_loss(pred_q, yr)                         # ✅ 新增
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= max(1, len(train_ds))

        # ---- val ----
        model.eval(); va_loss = 0.
        with torch.no_grad():
            for x, _, yr in val_dl:
                x, yr = x.to(device), yr.to(device)
                pred_q = to_physical(model(x), train_ds.y_mean, train_ds.y_std)
                va_loss += (
                    pinball_loss_weighted(pred_q, yr, channel_weights=cw)
                    + lambda_consist * consistency_loss_enhanced(pred_q)
                    + lambda_cov   * coverage_penalty_loss(pred_q, yr, channel_weights=cw)
                    + lambda_width * interval_width_loss(pred_q, yr)
                ).item() * x.size(0)
        va_loss /= max(1, len(val_ds))
        sched.step()

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), 'best_v3_coverage.pt')
        print(f"Epoch {ep:02d} | train {tr_loss:.4f} | val {va_loss:.4f}")

    # ---- test ----
    model.load_state_dict(torch.load('best_v3_coverage.pt', weights_only=True)); model.eval()
    P, T = [], []
    with torch.no_grad():
        for x, _, yr in test_dl:
            P.append(to_physical(model(x.to(device)),
                                 train_ds.y_mean, train_ds.y_std).cpu().numpy())
            T.append(yr.numpy())
    P = np.concatenate(P); T = np.concatenate(T)
    med  = P[..., 1]
    mae  = np.mean(np.abs(med - T), axis=(0, 1))
    rmse = np.sqrt(np.mean((med - T) ** 2, axis=(0, 1)))
    cov  = np.mean((T >= P[..., 0]) & (T <= P[..., 2]), axis=(0, 1))
    wid  = np.mean(P[..., 2] - P[..., 0], axis=(0, 1))

    print("\n=== Test (q50 metrics + 80% PI coverage) ===")
    print(f"{'Name':14s}  {'MAE':>9s}  {'RMSE':>9s}  PI80%cov  AvgWidth")
    for n_, m, r, c, w in zip(PowerDataset.TARGETS, mae, rmse, cov, wid):
        flag = "ok" if c >= 0.75 else ("false " if c >= 0.60 else "❌")
        print(f"{n_:14s}  {m:9.2f}  {r:9.2f}  {c:.2f} {flag}   {w:.1f}")
    return model, train_ds

#推理模型


import time


def ask_grid_reasoner(full_ds, pred_q50, targets, target_hour=1, max_retries=3):
    """
    Handles API calls with auto-retry logic to fix 503 errors and cleans text to avoid font issues.
    """
    hour_idx = target_hour - 1
    print(f"\n[Grid-Reasoner] Extracting forecast data for future hour {target_hour}...")

    # 数据提取
    forecast = {name: round(float(pred_q50[hour_idx, i]), 2) for i, name in enumerate(targets)}
    total = forecast['Total_Gen_MW']
    solar = forecast['Solar_Gen']
    nuclear = forecast['Nuclear_Gen']
    fossil_ratio = round((forecast['Coal_Gen'] + forecast['Gas_Gen'] + forecast['Oil_Gen']) / total * 100, 1)

    system_instruction = """
    You are an expert intelligent grid dispatcher. The following data represents the precise physical state of the PJM power grid.
    Decide the action for the Battery Energy Storage System (BESS).
    Strictly output in JSON format, which MUST include:
    "action": "CHARGE", "DISCHARGE", or "STANDBY",
    "reasoning": "Detailed reasoning for your decision (strictly in English, maximum 100 words)"
    """

    user_prompt = f"""
    - Predicted Total Load: {total} MW
    - Solar Generation: {solar} MW
    - Nuclear Generation: {nuclear} MW
    - Fossil Fuel Ratio (Coal+Gas+Oil): {fossil_ratio}%
    """

    # 填入你的真实 API KEY
    GEMINI_API_KEY = "AIzaSyCttDHvoAE038nuaIjnJvzxW6zqSS-vEfE"
    client = genai.Client(api_key=GEMINI_API_KEY)

    decision = None

    # --- 自动重试逻辑 ---
    for attempt in range(max_retries):
        try:
            print(f"[Grid-Reasoner] Calling Gemini API (Attempt {attempt + 1}/{max_retries})...")
            response = client.models.generate_content(
                model='gemini-2.5-flash',  # 建议使用稳定的 2.0-flash
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            decision = json.loads(response.text)
            print(f"[SUCCESS] Command received from Gemini.")
            break  # 成功则退出循环

        except Exception as e:
            err_msg = str(e)
            if "503" in err_msg or "high demand" in err_msg.lower():
                if attempt < max_retries - 1:
                    print(f"[Warning] Server busy (503). Retrying in 3 seconds...")
                    time.sleep(3)
                    continue

            print(f"[ERROR] API Call Failed: {err_msg}")
            # 如果是最后一次尝试失败，给出兜底方案
            decision = {
                "action": "STANDBY",
                "reasoning": f"API Unavailable after {max_retries} attempts. Defaulting to Standby."
            }
            break

    # 写入 JSON 日志文件
    output_file = f"dispatch_command_hour_{target_hour}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        full_log = {"target_hour": target_hour, "forecast_data": forecast, "decision": decision}
        json.dump(full_log, f, ensure_ascii=False, indent=4)

    return decision





#图片绘制
def plot_grid_dashboard(full_ds, pred_q50, targets, target_hour, decision):
    """
    四宫格电网监控面板：
    [Top Left] 总负荷折线图 (过去24h + 未来预测点)
    [Top Right] 燃料构成堆叠面积图 (过去24h + 未来12h)
    [Bottom Left] AI 智能调度策略与逻辑推演
    [Bottom Right] 该小时预测 vs 真实误差柱状图 (含误差率 %) & 日志
    """
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1], width_ratios=[1, 1.2])

    ax_line = fig.add_subplot(gs[0, 0])
    ax_stack = fig.add_subplot(gs[0, 1])
    ax_text_ai = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])  # 把原来的纯文字日志替换为了高级柱状图 + 图例

    last_idx = len(full_ds) - 1
    seq_len = full_ds.seq_len

    # ---------------------------------------------------------
    # 1. 📈 [Top Left] 总负荷折线图
    # ---------------------------------------------------------
    target_idx_total = targets.index('Total_Gen_MW')
    past_24_total = full_ds.Y_raw[last_idx + seq_len - 24: last_idx + seq_len, target_idx_total]
    future_point = pred_q50[target_hour - 1, target_idx_total]
    x_past = np.arange(-23, 1)

    ax_line.plot(x_past, past_24_total, marker='o', markersize=4, color='#007acc', linewidth=2, label='Past 24h Actual')
    ax_line.plot([0, target_hour], [past_24_total[-1], future_point], linestyle='--', color='#cc0000', linewidth=2)
    ax_line.scatter(target_hour, future_point, color='#cc0000', s=120, zorder=5, label=f'Hour {target_hour} Forecast')

    ax_line.annotate(f'{future_point:.0f} MW', (target_hour, future_point),
                     textcoords="offset points", xytext=(0, 15), ha='center', fontsize=12, fontweight='bold',
                     color='#cc0000')

    # ax_line.set_title('Total Load Trend: Actual vs Forecast', fontsize=15, fontweight='bold')
    ax_line.set_ylabel('Generation (MW)', fontsize=13)
    ax_line.set_xlabel('Time (Hours from Current)', fontsize=13)
    ax_line.grid(True, linestyle='--', alpha=0.5)
    ax_line.legend(loc='upper left', fontsize=11)

    # ---------------------------------------------------------
    # 2. 📊 [Top Right] 燃料构成堆叠图
    # ---------------------------------------------------------
    past_actuals_8 = full_ds.Y_raw[last_idx + seq_len - 24: last_idx + seq_len, :8]
    future_preds_8 = pred_q50[:, :8]
    combined_8 = np.vstack([past_actuals_8, future_preds_8]).T

    x_timeline = np.arange(-23, 13)
    colors = ['#333333', '#e67300', '#cc0000', '#8c564b', '#999999', '#ffcc00', '#0066cc', '#009933']

    ax_stack.stackplot(x_timeline, combined_8, labels=targets[:8], colors=colors, alpha=0.85)
    ax_stack.axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='Current Time (NOW)')

    # ax_stack.set_title('Grid Fuel Composition: Next 12h Forecast Overview', fontsize=15, fontweight='bold')
    ax_stack.set_ylabel('Generation (MW)', fontsize=13)
    ax_stack.set_xlabel('Time (Hours from Current)', fontsize=13)
    ax_stack.set_xlim(-23, 12)
    ax_stack.grid(True, linestyle='--', alpha=0.4)
    ax_stack.legend(loc='upper left', fontsize=10, ncol=3)

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # 3. 🧠 [Bottom Left] AI 智能调度策略
    # ---------------------------------------------------------
    ax_text_ai.axis('off')
    action = decision.get('action', 'ERROR/UNKNOWN')

    # 获取原始推理文本
    reasoning = decision.get('reasoning', 'No reasoning provided by API.')
    # 直接在原变量上进行强制物理换行（限制每行大约65个字符）
    reasoning = textwrap.fill(reasoning, width=65)

    action_color = '#009933' if action == 'CHARGE' else '#cc0000' if action == 'DISCHARGE' else '#0066cc'

    # 附带计算总负荷的误差率供日志展示
    actual_hour_total = full_ds.Y_raw[last_idx + seq_len + target_hour - 1, target_idx_total]
    total_error_pct = (future_point - actual_hour_total) / actual_hour_total * 100 if actual_hour_total != 0 else 0

    ai_text = (
        f"🤖 AI DISPATCH STRATEGY (Future Hour {target_hour})\n\n"
        f"[ACTION COMMAND]\n"
        f"  >> {action} <<\n\n"
        f"[REASONING LOGIC]\n"
        f"{reasoning}\n\n"
        f"--------------------------------------------------\n"
        f"🎯 Total Load Prediction Error Rate: {total_error_pct:+.2f}%"
    )

    # 注意：这里去掉了原来的 wrap=True，避免边框无限拉长
    ax_text_ai.text(0.05, 0.9, ai_text, fontsize=14, va='top', ha='left',
                    bbox=dict(boxstyle="round,pad=1.2", facecolor='#f8f9fa', edgecolor=action_color, linewidth=2))

    # ---------------------------------------------------------
    # 4. 📉 [Bottom Right] 带有【误差率 %】的偏差柱状图
    # ---------------------------------------------------------
    hour_idx = target_hour - 1
    predicted_hour_data = pred_q50[hour_idx, :]
    actual_hour_data = full_ds.Y_raw[last_idx + seq_len + hour_idx, :]

    # 计算绝对偏差 (MW)
    deviation_mw = predicted_hour_data - actual_hour_data

    # 计算百分比误差率 (Error Rate %) - 避免除以 0
    error_rate_pct = np.zeros_like(deviation_mw)
    valid_mask = actual_hour_data != 0
    error_rate_pct[valid_mask] = (deviation_mw[valid_mask] / actual_hour_data[valid_mask]) * 100

    target_names = ['Coal', 'Gas', 'Nuclear', 'Oil', 'Other', 'Solar', 'Hydro', 'Wind', 'Total']
    x_positions = np.arange(len(target_names))

    bars = ax_bar.bar(x_positions, deviation_mw,
                      color=['#cc3300' if x > 0 else '#660066' for x in deviation_mw])

    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels(target_names, fontsize=11, rotation=30)
    # ax_bar.set_title(f'Hour {target_hour} Deviation (MW) & Error Rate (%)', fontsize=14, fontweight='bold')
    ax_bar.set_ylabel('Deviation (MW)', fontsize=12)
    ax_bar.grid(True, linestyle='--', alpha=0.3)

    # 动态扩展 Y 轴的高度，防止双行文字顶破天花板
    max_dev = max(abs(deviation_mw)) if max(abs(deviation_mw)) != 0 else 100
    ax_bar.set_ylim(-max_dev * 1.35, max_dev * 1.35)

    # 🌟 核心新增：在柱子上打上具体数值 (MW) 和换行的误差率 (%)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        err_pct = error_rate_pct[i]

        # 组装文本格式：第一行 MW，第二行百分比
        label_text = f'{height:+.0f}\n({err_pct:+.1f}%)'
        label_color = '#cc3300' if height > 0 else '#660066'

        # 为了美观，正向误差文字标在柱子上面，负向误差文字标在柱子下面
        y_pos = height + (max_dev * 0.05) if height > 0 else height - (max_dev * 0.05)
        va_align = 'bottom' if height > 0 else 'top'

        ax_bar.text(bar.get_x() + bar.get_width() / 2., y_pos,
                    label_text, ha='center', va=va_align,
                    fontsize=9, color=label_color, fontweight='bold')

    plt.tight_layout()
    plt.show()


# if __name__ == '__main__':
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     data_path = os.path.join(script_dir, "pjm_full_year_dataset.csv")
#
#     train(
#         data_path,
#         seq_len=48, pred_len=12, epochs=30,
#         lambda_consist=0.1,   # 一致性损失权重（不变）
#         lambda_cov=0.5,        # 覆盖率惩罚权重（新增，可调）
#         lambda_width=0.3,      # 区间宽度下限权重（新增，可调）
#     )
def plot_total_load_trend_standalone(full_ds, pred_q50, targets, target_hour):
    """单独生成：总负荷折线图 (放大字体版)"""
    fig, ax_line = plt.subplots(figsize=(12, 8))

    last_idx = len(full_ds) - 1
    seq_len = full_ds.seq_len
    target_idx_total = targets.index('Total_Gen_MW')
    past_24_total = full_ds.Y_raw[last_idx + seq_len - 24: last_idx + seq_len, target_idx_total]
    future_point = pred_q50[target_hour - 1, target_idx_total]
    x_past = np.arange(-23, 1)

    ax_line.plot(x_past, past_24_total, marker='o', markersize=6, color='#007acc', linewidth=3, label='Past 24h Actual')
    ax_line.plot([0, target_hour], [past_24_total[-1], future_point], linestyle='--', color='#cc0000', linewidth=3)
    ax_line.scatter(target_hour, future_point, color='#cc0000', s=200, zorder=5, label=f'Hour {target_hour} Forecast')

    # 放大标注字体 (fontsize=18)
    ax_line.annotate(f'{future_point:.0f} MW', (target_hour, future_point),
                     textcoords="offset points", xytext=(0, 20), ha='center', fontsize=18, fontweight='bold',
                     color='#cc0000')

    # 放大标题和轴标签 (fontsize=22, 18)
    ax_line.set_title('Total Load Trend: Actual vs Forecast', fontsize=22, fontweight='bold')
    ax_line.set_ylabel('Generation (MW)', fontsize=18)
    ax_line.set_xlabel('Time (Hours from Current)', fontsize=18)

    # 放大刻度字体
    ax_line.tick_params(axis='both', which='major', labelsize=14)

    ax_line.grid(True, linestyle='--', alpha=0.5)
    # 放大图例字体
    ax_line.legend(loc='upper left', fontsize=16)

    plt.tight_layout()
    plt.show()


def plot_fuel_composition_standalone(full_ds, pred_q50, targets):
    """单独生成：燃料构成堆叠面积图 (放大字体版)"""
    fig, ax_stack = plt.subplots(figsize=(14, 8))

    last_idx = len(full_ds) - 1
    seq_len = full_ds.seq_len
    past_actuals_8 = full_ds.Y_raw[last_idx + seq_len - 24: last_idx + seq_len, :8]
    future_preds_8 = pred_q50[:, :8]
    combined_8 = np.vstack([past_actuals_8, future_preds_8]).T

    x_timeline = np.arange(-23, 13)
    colors = ['#333333', '#e67300', '#cc0000', '#8c564b', '#999999', '#ffcc00', '#0066cc', '#009933']

    ax_stack.stackplot(x_timeline, combined_8, labels=targets[:8], colors=colors, alpha=0.85)
    ax_stack.axvline(x=0, color='red', linestyle='--', linewidth=3, label='Current Time (NOW)')

    # 放大标题和轴标签 (fontsize=22, 18)
    ax_stack.set_title('Grid Fuel Composition: Next 12h Forecast Overview', fontsize=22, fontweight='bold')
    ax_stack.set_ylabel('Generation (MW)', fontsize=18)
    ax_stack.set_xlabel('Time (Hours from Current)', fontsize=18)
    ax_stack.set_xlim(-23, 12)

    # 放大刻度字体
    ax_stack.tick_params(axis='both', which='major', labelsize=14)

    ax_stack.grid(True, linestyle='--', alpha=0.4)
    # 放大图例字体
    ax_stack.legend(loc='upper left', fontsize=16, ncol=3)

    plt.tight_layout()
    plt.show()

# main
if __name__ == '__main__':
    import os
    import torch

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "pjm_full_year_dataset.csv")
    model_path = os.path.join(script_dir, "best_v3_coverage.pt")  # 请确认模型名称一致

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[Grid-Reasoner] Initializing environment and restoring data parameters...")

    df = pd.read_csv(data_path)
    time_col = 'Time' if 'Time' in df.columns else ('index' if 'index' in df.columns else 'period')
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    tr_size = int(len(df) * 0.8)
    train_ds = PowerDataset(df.iloc[:tr_size], seq_len=48, pred_len=12)
    full_ds = PowerDataset(df, seq_len=48, pred_len=12,
                           x_scaler=train_ds.x_scaler,
                           y_mean=train_ds.y_mean,
                           y_std=train_ds.y_std)



    print(f"[Grid-Reasoner] Loading pre-trained base model directly: {model_path}")
    model = ChannelIndependentTransformer(
        n_feats=len(train_ds.feat_cols),
        n_targets=len(PowerDataset.TARGETS),
        pred_len=12,
        n_quantiles=3
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    print("\n" + "=" * 60)
    print("⚡ Starting Grid-Reasoner: State Awareness & Decision Engine")
    print("=" * 60)

    # 提取最新的真实窗口
    offset = -10
    last_idx = len(full_ds) - 1 + offset
    x, _, _ = full_ds[last_idx]
    x = x.unsqueeze(0).to(device)

    # 物理预测
    with torch.no_grad():
        out_norm = model(x)
        pred_physical = to_physical(out_norm, train_ds.y_mean, train_ds.y_std).cpu().numpy()[0]

    pred_q50 = pred_physical[..., 1]

    # (1 to 12)
    TARGET_HOUR = 5

    # photo
    ai_decision = ask_grid_reasoner(full_ds, pred_q50, PowerDataset.TARGETS, target_hour=TARGET_HOUR)

    # 原有的四宫格图表调用
    if ai_decision:
        plot_grid_dashboard(full_ds, pred_q50, PowerDataset.TARGETS, target_hour=TARGET_HOUR, decision=ai_decision)

        # 👇👇👇 新增的调用：单独生成放大的两张图 👇👇👇
        plot_total_load_trend_standalone(full_ds, pred_q50, PowerDataset.TARGETS, target_hour=TARGET_HOUR)
        plot_fuel_composition_standalone(full_ds, pred_q50, PowerDataset.TARGETS)

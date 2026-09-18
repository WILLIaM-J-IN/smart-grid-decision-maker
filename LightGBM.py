"""
LightGBM 9 目标多步预测 (带模型保存/加载)
保存位置: ./lgbm_models/<target>_h<horizon>_q<quantile>.txt
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

TARGETS = ['Coal_Gen', 'Gas_Gen', 'Nuclear_Gen', 'Oil_Gen',
           'Other_Gen', 'Solar_Gen', 'Hydro_Gen', 'Wind_Gen', 'Total_Gen_MW']

MODEL_DIR = 'lgbm_models'   # 模型保存目录


# ============ 特征工程 ============
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['index'] = pd.to_datetime(df['index'])
    df = df.sort_values('index').reset_index(drop=True)

    dt = df['index']
    df['hour']        = dt.dt.hour
    df['dow']         = dt.dt.dayofweek
    df['month']       = dt.dt.month
    df['day']         = dt.dt.day
    df['is_weekend']  = (df['dow'] >= 5).astype(int)
    df['is_daytime']  = ((df['hour'] >= 6) & (df['hour'] <= 19)).astype(int)
    df['solar_angle'] = np.maximum(0, np.sin(np.pi * (df['hour'] - 6) / 13))

    for col in ['Load_MW', 'Temp_C', 'Humidity_Pct', 'Temp_Sensitive_Share_Pct']:
        df[f'{col}_lag1']        = df[col].shift(1)
        df[f'{col}_lag24']       = df[col].shift(24)
        df[f'{col}_lag168']      = df[col].shift(168)
        df[f'{col}_roll24_mean'] = df[col].shift(1).rolling(24).mean()
        df[f'{col}_roll24_std']  = df[col].shift(1).rolling(24).std()
        df[f'{col}_diff1']       = df[col].diff(1)

    for col in TARGETS:
        df[f'{col}_lag1']   = df[col].shift(1)
        df[f'{col}_lag24']  = df[col].shift(24)
        df[f'{col}_lag168'] = df[col].shift(168)
        df[f'{col}_roll24_mean'] = df[col].shift(1).rolling(24).mean()

    return df.bfill().ffill()


def fit_one(X_tr, y_tr, X_va, y_va, quantile):
    params = dict(
        n_estimators=1500, learning_rate=0.03, num_leaves=63,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
        objective='quantile', alpha=quantile,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    return model


# ============ 训练 + 保存 ============
def train_and_save(csv_path, horizons=(1, 6, 12)):
    df = pd.read_csv(csv_path)
    df = build_features(df)

    feat_cols = [c for c in df.columns if c != 'index' and c not in TARGETS]
    print(f"Loaded {len(df)} rows | features: {len(feat_cols)}")

    n = len(df); tr, va = int(n * .8), int(n * .9)
    df_tr, df_va, df_te = df.iloc[:tr], df.iloc[tr:va], df.iloc[va:]

    os.makedirs(MODEL_DIR, exist_ok=True)
    # 保存特征列名,推理时要用
    with open(os.path.join(MODEL_DIR, 'feat_cols.json'), 'w') as f:
        json.dump(feat_cols, f)

    results = {t: {} for t in TARGETS}

    for target in TARGETS:
        print(f"\n========== {target} ==========")
        for h in horizons:
            ytr_full = df_tr[target].shift(-h)
            yva_full = df_va[target].shift(-h)
            yte_full = df_te[target].shift(-h)

            m_tr, m_va, m_te = ytr_full.notna(), yva_full.notna(), yte_full.notna()
            Xtr = df_tr.loc[m_tr, feat_cols].values
            Xva = df_va.loc[m_va, feat_cols].values
            Xte = df_te.loc[m_te, feat_cols].values
            ytr = ytr_full[m_tr].values
            yva = yva_full[m_va].values
            yte = yte_full[m_te].values

            preds = {}
            for q in [0.1, 0.5, 0.9]:
                model = fit_one(Xtr, ytr, Xva, yva, quantile=q)
                # 保存
                fname = f"{target}_h{h}_q{int(q*100)}.txt"
                model.booster_.save_model(os.path.join(MODEL_DIR, fname))
                preds[q] = np.clip(model.predict(Xte), 0, None)

            p10, p50, p90 = preds[0.1], preds[0.5], preds[0.9]
            p10 = np.minimum(p10, p50); p90 = np.maximum(p90, p50)

            mae   = mean_absolute_error(yte, p50)
            rmse  = np.sqrt(mean_squared_error(yte, p50))
            cov80 = np.mean((yte >= p10) & (yte <= p90))
            results[target][h] = dict(mae=mae, rmse=rmse, cov80=cov80,
                                      mean=float(yte.mean()))
            print(f"  h={h:2d}h  MAE={mae:9.2f}  RMSE={rmse:9.2f}  PI80%cov={cov80:.2f}")

    # 保存结果指标
    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # 汇总
    print("\n" + "=" * 70)
    print(f"SUMMARY  (models saved to ./{MODEL_DIR}/)")
    print("=" * 70)
    print(f"{'Target':15s} {'h=1':>10s} {'h=6':>10s} {'h=12':>10s} {'Acc(h=12)':>12s}")
    for t in TARGETS:
        r = results[t]
        acc = (1 - r[12]['mae'] / max(abs(r[12]['mean']), 1.0)) * 100
        print(f"{t:15s} {r[1]['mae']:10.1f} {r[6]['mae']:10.1f} {r[12]['mae']:10.1f} "
              f"{acc:11.1f}%")

    total_models = len(TARGETS) * len(horizons) * 3
    print(f"\n✓ Saved {total_models} models to '{MODEL_DIR}/' directory")
    return results


# ============ 加载 + 预测 (推理用) ============
def load_and_predict(csv_path, horizon=12):
    """加载已保存的模型, 对 csv_path 数据做预测"""
    with open(os.path.join(MODEL_DIR, 'feat_cols.json')) as f:
        feat_cols = json.load(f)

    df = pd.read_csv(csv_path)
    df = build_features(df)
    X = df[feat_cols].values

    print(f"\nLoading models and predicting h={horizon}h ahead...")
    predictions = {}
    for target in TARGETS:
        preds_q = {}
        for q in [10, 50, 90]:
            fname = f"{target}_h{horizon}_q{q}.txt"
            booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, fname))
            preds_q[q] = np.clip(booster.predict(X), 0, None)
        # 修复 quantile crossing
        p10 = np.minimum(preds_q[10], preds_q[50])
        p90 = np.maximum(preds_q[90], preds_q[50])
        predictions[target] = {'q10': p10, 'q50': preds_q[50], 'q90': p90}

    # 保存为 CSV 方便查看
    out = pd.DataFrame({'index': df['index']})
    for target in TARGETS:
        out[f'{target}_pred']     = predictions[target]['q50']
        out[f'{target}_pred_low'] = predictions[target]['q10']
        out[f'{target}_pred_high']= predictions[target]['q90']
    out_path = f'predictions_h{horizon}.csv'
    out.to_csv(out_path, index=False)
    print(f"✓ Predictions saved to {out_path}")
    return predictions


if __name__ == '__main__':
    csv = r'E:\PythonProject\smart-grid decision maker\pjm_full_year_dataset.csv'

    # # 第一次: 训练 + 保存
    # train_and_save(csv, horizons=(1, 6, 12))

    #直接加载推理 (注释掉上面那行, 解开下面这行)
    load_and_predict(csv, horizon=12)
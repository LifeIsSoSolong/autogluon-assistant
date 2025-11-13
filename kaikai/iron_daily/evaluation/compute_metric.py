import argparse
import json
import numpy as np
import pandas as pd

def load_series(path, value_col="value"):
    df = pd.read_csv(path)
    # 统一时间列名
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} 必须包含列 'timestamp'")
    if value_col not in df.columns:
        raise ValueError(f"{path} 必须包含列 '{value_col}'")

    # 解析日期，容忍 2025/9/1 与 2025/09/01
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        bad = df[df["timestamp"].isna()]
        raise ValueError(f"{path} 中存在无法解析的时间：\n{bad}")

    df = df[["timestamp", value_col]].copy()
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return df

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def mape(y_true, y_pred):
    # 排除真实值为0的样本，避免除0
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100.0)

def mda(y_true, y_pred):
    """
    Mean Directional Accuracy:
    比较相邻时刻的变化方向是否一致：sign(Δŷ_t) == sign(Δy_t)
    """
    dy_true = np.diff(y_true)
    dy_pred = np.diff(y_pred)
    # 方向一致记为1，否则0（含一方为0时，只有两者都为0才算一致）
    sign_true = np.sign(dy_true)
    sign_pred = np.sign(dy_pred)
    return float(np.mean(sign_true == sign_pred))

def main(results_path, labels_path, out_path=None):
    pred_df = load_series(results_path, value_col="value")
    true_df = load_series(labels_path,   value_col="value")

    # 按时间戳内连接对齐（只评估两边都存在的日期）
    df = pd.merge(true_df.rename(columns={"value": "y_true"}),
                  pred_df.rename(columns={"value": "y_pred"}),
                  on="timestamp",
                  how="inner").sort_values("timestamp")

    if len(df) == 0:
        raise ValueError("两份文件在时间戳上没有重叠，无法计算指标。")
    if len(pred_df) != len(true_df):
        print(f"[提示] 时间戳未完全一致：pred={len(pred_df)}, true={len(true_df)}, 使用对齐后的 {len(df)} 条记录评估。")

    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)

    metrics = {
        "RMSE": rmse(y_true, y_pred),
        "MAPE_percent": mape(y_true, y_pred),
        "MDA": mda(y_true, y_pred),
        "n_aligned_points": int(len(df)),
        "first_timestamp": df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
        "last_timestamp": df["timestamp"].iloc[-1].strftime("%Y-%m-%d"),
    }

    print("== Metrics ==")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"已保存到 {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute RMSE, MAPE, MDA for forecast results.")
    parser.add_argument("--results", default="results.csv", help="预测结果CSV路径，含列 timestamp,value")
    parser.add_argument("--labels",  default="label.csv",   help="真实标签CSV路径，含列 timestamp,value")
    parser.add_argument("--out",     default=None,          help="可选：输出JSON路径")
    args = parser.parse_args()
    main(args.results, args.labels, args.out)

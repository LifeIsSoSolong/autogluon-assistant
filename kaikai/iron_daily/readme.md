## 1. Objective

Run the `kaikai/iron_daily` project inside MLZero (Autogluon-assistant) to **forecast the next 12 trading-day closes** of iron ore futures (contract `FU00002776`).  
Dataset specifics for this revision:

- Only **one raw file** (`data/train.csv`) is provided.
- The timestamp column is named **`timestamp`**.
- All non-target columns (`ID01002312`, …, `CM0000013263`) are **already shifted by one trading day** (lag = 1). Do **not** lag them again.
- There is **no `test.csv`**. Forecast timestamps must be extrapolated from the last row of `train.csv`.
- You should add  item_id column to the dataframe if you want to use item_id column.
- You should use freq="B" in the training code.
- You must use "RMSE" as the metric: `eval_metric="RMSE"`
- You should try to find the best model: `presets="high_quality"`
- You should split many validation sets from `train.csv` to avoid overfitting.

## 2. Environment & Dependencies

All five previous coding attempts failed because the generated scripts created bare conda envs without installing `pandas` or `autogluon.timeseries`, and torch downloads timed out (see `runs/iron_daily/detail_log.txt:101-211`). To avoid this, every MLZero workflow must:

1. Create/activate a environment using `conda create -n ****** python=3.10 -y`
2. Install the required libraries **before** running any project code:
   - `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `lightgbm`
   - `autogluon` (the latest stable release)
   - `matplotlib` / `seaborn` if you generate plots.
3. when install torch, you can use mirror or using this command: 
`pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 -i https://pypi.tuna.tsinghua.edu.cn/simple`

4. Verify availability with a quick import test.

5. if you can not create a right environment, you can directly use the existing environment:
`conda activate kaikai_temp`

Example shell snippet for create MLZero’s environment:

```bash
#!/bin/bash
set -euo pipefail
ENV_PATH="runs/iron_daily/conda_env"
conda create -p "${ENV_PATH}" python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

# Improve reliability of big downloads (torch, autogluon wheels)
export UV_HTTP_TIMEOUT=300
pip install --upgrade pip
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pandas numpy pyarrow scikit-learn lightgbm matplotlib seaborn
pip install autogluon

python - <<'PY'
import pandas, autogluon
print("Environment ready.")
PY
```

Only after the environment check succeeds should the script proceed to preprocessing/training commands described below.

## 3. Directory Layout

```
kaikai/iron_daily/
├── data/
│   └── train.csv              # 2015-12-28 – 2025-08-29, lagged features
└── readme.md                  # THIS FILE (agent instructions)
```

## 4. Data Schema & Column Handling

Current columns:

| Column          | Role                                   |
|-----------------|----------------------------------------|
| `timestamp`     | Trading day (string `YYYY/MM/DD`)      |
| `value`         | Target close price (FU00002776)        |
| `ID01002312`    | Lagged feature (already t-1)           |
| `ID00186575`    | Lagged feature (already t-1)           |
| `ID00186100`    | Lagged feature (already t-1)           |
| `ID00183109`    | Lagged feature (already t-1)           |
| `GM0000033031`  | Lagged feature (already t-1)           |
| `CM0000013263`  | Lagged feature (already t-1)           |

**Dynamic column discovery is mandatory** because extra predictors may appear:

```python
import pandas as pd

df = pd.read_csv("kaikai/iron_daily/data/train.csv", parse_dates=["timestamp"])
time_col = "timestamp"
target_col = "value"
feature_cols = [c for c in df.columns if c not in {time_col, target_col}]
```

All `feature_cols` already represent information from `t-1`, so they can be aligned directly with `value` at `t`. Further feature lags are optional but **not required**.


## 7. Forecast Generation (No test.csv)

1. **Determine forecast timestamps**
   - Let `t_last = df['timestamp'].max()` (currently 2025-08-29).
   - Generate the next 12 trading days using `pandas.tseries.offsets.BDay` or a custom trading-calendar util.
2. **Prepare input features**
   - Use the **latest available row** (or multiple recent rows if the model expects sequences).
   - If the model needs rolling windows, ensure the final window aligns with `t_last`.
3. **Predict horizons**
   - Model outputs a length-12 vector.
   - Combine with generated timestamps to build a dataframe:
     ```python
     forecast = pd.DataFrame({
         "timestamp": future_dates,
         "value": predictions
     })
     ```

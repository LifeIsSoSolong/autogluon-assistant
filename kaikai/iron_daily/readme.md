## 1. Objective

Run the `kaikai/iron_daily` project inside MLZero (Autogluon-assistant) to **forecast the next 12 trading-day closes** of iron ore futures (contract `FU00002776`).  

- Only **one raw file** (`data/train.csv`) is provided.  
- The timestamp column is named **`timestamp`**.  
- All non-target columns (`ID01002312`, …, `CM0000013263`) are features you should use and **already shifted by one trading day** (lag = 1). Do **not** lag them again.  
- There is **no `test.csv`**. You must extrapolate future timestamps based on the last row of `train.csv`.

## 2. Directory Layout

```
kaikai/iron_daily/
├── data/
│   └── train.csv              # 2015-12-28 – 2025-08-29, lagged features
└── readme.md                  # THIS FILE (agent instructions)
```

## 3. Data Schema & Column Handling

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

**Dynamic column discovery is mandatory** because extra features may appear:

```python
import pandas as pd

df = pd.read_csv("kaikai/iron_daily/data/train.csv", parse_dates=["timestamp"])
time_col = "timestamp"
target_col = "value"
feature_cols = [c for c in df.columns if c not in {time_col, target_col}]
```

All `feature_cols` already represent information from `t-1`, so they can be aligned directly with `value` at `t`. Further feature lags are optional but **not required**.


## 4. **Validation**
   - Use a time-based holdout or rolling-origin CV. Recommended: last ~200 trading days as validation.  
   - Report RMSE per horizon and averaged RMSE. Log the exact validation window (start/end).

## 5. Forecast Generation (No test.csv)

1. **Determine forecast timestamps**
   - Let `t_last = df['timestamp'].max()` (currently 2025-08-29).  
   - Generate the next 12 trading days or a custom trading-calendar util.
2. **Prepare input features**
   - Use the **latest available row** (or multiple recent rows if the model expects sequences).  
3. **Predict horizons**
   - Model outputs a length-12 vector.  
   - Combine with generated timestamps to build a dataframe:
     ```
     forecast = pd.DataFrame({
         "timestamp": future_dates,
         "value": predictions
     })
     ```
## Task Overview

We aim to build a multi-step forecasting pipeline for daily iron ore futures (contract code `FU00002776`). The goal is to predict the next 12 trading-day prices based on historical market features while respecting realistic data availability (no same-day leakage).

- **Input root**: `kaikai/iron_daily`
- **Training set**: `data/train_data` (tabular time series)
- **Test set**: `data/test_data` (contains future 12 trading dates; column `value` must be filled with forecasts)
- **Column dictionary**: `data/index` (describes feature names)
- **Target column**: `FU00002776`

## Data & Feature Handling

1. All non-target columns are candidate predictors. When forecasting day *t*, only use features up to *t-1* (strict one-day lag on every feature) to avoid leakage.
2. Training data contains missing values and possible anomalies; design a preprocessing pipeline that cleans/filters before modeling.
3. After preprocessing, save the cleaned dataset (feature matrix + target) to `processed_data/` for reproducibility.

## Modeling Requirements

- Construct a supervised learning setup where the model ingests **N historical days of lagged features** (you decide the optimal window length) and outputs the target for the next **12 consecutive trading days**.
- Split the historical data into training/validation by sampling one or more representative “forecast segments” (rolling or hold-out windows). Report **RMSE** on the chosen validation segment(s) to track generalization.
- Provide clear documentation of how the validation window is selected (e.g., last K trading days, rolling origin, etc.).

## Expected Deliverables

1. Preprocessing script/notebook that:
   - Loads raw data, aligns timestamps, applies the one-day lag, handles missing values/outliers.
   - Exports the cleaned dataset (and any feature engineering artifacts) to `processed_data/`.
2. Modeling script/notebook that:
   - Trains the multi-step forecaster.
   - Evaluates on the validation slice(s) and prints RMSE.
   - Generates `results.csv` matching `data/test_data` (same schema, `value` column filled with predictions for the 12 future trading days).
3. Brief summary of:
   - Preprocessing steps.
   - Model architecture/algorithm and chosen history window *N*.
   - Validation strategy and RMSE.

Keep all outputs under `runs/iron_daily/...` so the experiment trail is easy to inspect. Feel free to add extra diagnostics (feature importance, residual plots) if helpful.

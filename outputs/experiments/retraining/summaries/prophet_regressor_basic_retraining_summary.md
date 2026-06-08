# Revisi 1.6 - Retraining Prophet-Regressor-Basic

Run UTC: 2026-05-25T10:56:54.719256+00:00

## Scope

Model Prophet-Regressor-Basic dilatih ulang memakai seluruh `train_val.csv` dengan best params dari tuning revisi. Final test tidak dibaca atau digunakan pada tahap ini.

## Training Data

- Source train_val rows: 10198
- Training rows after history drop: 10030
- History rows dropped: 168
- Training period UTC: 2025-01-08T05:00:00+00:00 to 2026-03-02T03:00:00+00:00
- Regressors: lag_1, lag_24, lag_168, hour, day_of_week

## Best Configuration

- Parameter set: prophet_regressor_basic_003
- Params: {"changepoint_prior_scale": 0.01, "seasonality_mode": "additive", "seasonality_prior_scale": 5.0}

## Time Cost Computing

- Training time seconds: 3.790971
- Prediction time seconds: 0.000000 karena tahap ini hanya training.
- Total runtime seconds: 3.931210

## Output Files

- Model artifact: `F:\Research\Research BDTS\outputs\experiments\retraining\models\prophet_regressor_basic_retrained.json`
- Metadata: `F:\Research\Research BDTS\outputs\experiments\retraining\metadata\prophet_regressor_basic_metadata.json`
- Training summary: `F:\Research\Research BDTS\outputs\experiments\retraining\prophet_regressor_basic_training_summary.csv`
- Runtime summary: `F:\Research\Research BDTS\outputs\experiments\retraining\prophet_regressor_basic_runtime_summary.csv`
- Best params used: `F:\Research\Research BDTS\outputs\experiments\retraining\prophet_regressor_basic_best_params_used.json`

## Leakage Note

Lag regressor training dibuat dari actual train_val masa lalu saja. Final testing nantinya tetap wajib memakai recursive forecasting untuk membangun lag regressor masa depan.

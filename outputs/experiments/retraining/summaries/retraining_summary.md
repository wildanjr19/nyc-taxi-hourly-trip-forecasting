# Retraining Best Configurations

Run UTC: 2026-05-17T23:47:03.195756+00:00

## Scope

Model dilatih ulang memakai seluruh `train_val.csv` dan best params hasil tuning. Script ini tidak membaca `final_test.csv`, tidak melakukan tuning tambahan, dan tidak memilih ulang parameter.

## Training Summary

| Model | Feature set | Parameter set | Train rows | Train time seconds | Model path |
|---|---|---:|---:|---:|---|
| Prophet | prophet_internal | prophet_005 | 10198 | 3.525288 | `F:\Research\Research BDTS\outputs\experiments\retraining\models\prophet_retrained.json` |
| XGBoost-Basic | xgb_basic | xgb_basic_292 | 10030 | 12.990324 | `F:\Research\Research BDTS\outputs\experiments\retraining\models\xgb_basic_retrained.json` |
| XGBoost-Advanced | xgb_advanced | xgb_advanced_210 | 10030 | 1.151408 | `F:\Research\Research BDTS\outputs\experiments\retraining\models\xgb_advanced_retrained.json` |

## Time Cost Computing

- Total retraining runtime seconds: 18.033218
- Prediction time seconds: 0.000000 karena tahap ini hanya training.

## Output Files

- Training summary: `F:\Research\Research BDTS\outputs\experiments\retraining\training_summary.csv`
- Runtime summary: `F:\Research\Research BDTS\outputs\experiments\retraining\runtime_summary.csv`
- Best params used: `F:\Research\Research BDTS\outputs\experiments\retraining\best_params_used.json`
- Model registry: `F:\Research\Research BDTS\outputs\experiments\retraining\model_registry.json`
- Experiment metadata: `F:\Research\Research BDTS\outputs\experiments\retraining\experiment_metadata.json`

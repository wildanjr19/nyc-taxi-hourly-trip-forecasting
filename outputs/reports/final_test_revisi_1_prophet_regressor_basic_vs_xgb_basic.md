# Revisi 1.7 - Final Testing

Run UTC: 2026-05-25T11:13:54.727741+00:00

## Scope

Tahap ini mengevaluasi Prophet-Regressor-Basic pada final_test dengan recursive forecasting. XGBoost-Basic diambil dari artefak final test lama tanpa prediksi ulang karena konfigurasi dan output final test-nya tetap sama.

## Leakage Guardrail

- Tidak ada tuning ulang setelah final test.
- Final test tidak diubah.
- Lag regressor Prophet-Regressor-Basic dibangun dari history masa lalu dan prediksi recursive di dalam blok 24 jam.
- Actual final_test hanya dipakai sebagai label evaluasi dan masuk ke history setelah blok selesai.
- Semua prediction rows memiliki `used_actual_future_for_features=False`.

## Final Metric Ranking Revisi

| rank_by_primary_metric | model_label | parameter_set_id | mae | rmse | mape | smape | prediction_time_seconds | retraining_train_time_seconds | source_artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | XGBoost-Basic | xgb_basic_292 | 519.595949 | 720.224252 | 12.069298 | 12.169547 | 2.686983 | 12.990324 | F:\Research\Research BDTS\outputs\final_test\metrics\final_metrics.csv |
| 2 | Prophet-Regressor-Basic | prophet_regressor_basic_003 | 837.648179 | 1123.936750 | 32.502400 | 31.312555 | 17.941776 | 3.790971 | new_recursive_forecast_revisi_1_7 |

## Improvement Summary

| comparison | metric | prophet_regressor_basic | xgb_basic | absolute_difference_xgb_minus_prophet | xgb_improvement_percent_vs_prophet | winner |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost-Basic vs Prophet-Regressor-Basic | mae | 837.648179 | 519.595949 | -318.052230 | 37.969668 | XGBoost-Basic |
| XGBoost-Basic vs Prophet-Regressor-Basic | rmse | 1123.936750 | 720.224252 | -403.712498 | 35.919503 | XGBoost-Basic |
| XGBoost-Basic vs Prophet-Regressor-Basic | mape | 32.502400 | 12.069298 | -20.433102 | 62.866441 | XGBoost-Basic |
| XGBoost-Basic vs Prophet-Regressor-Basic | smape | 31.312555 | 12.169547 | -19.143008 | 61.135247 | XGBoost-Basic |

## Residual Summary

| model_label | n_predictions | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | residual_std | mean_absolute_error | rmse | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet-Regressor-Basic | 720 | 112.570706 | 11.505491 | 1119.062537 | 837.648179 | 1123.936750 | 0.493056 | 0.506944 | 5299.931706 |
| XGBoost-Basic | 720 | -8.104758 | -69.922791 | 720.679294 | 519.595949 | 720.224252 | 0.550000 | 0.450000 | 4017.453613 |

## Horizon Behavior

Best horizon step by MAE:

| model_label | horizon_step | n_predictions | mae | rmse | mean_error_actual_minus_predicted |
| --- | --- | --- | --- | --- | --- |
| Prophet-Regressor-Basic | 1 | 30 | 564.119107 | 699.151115 | -191.019191 |
| XGBoost-Basic | 5 | 30 | 166.291864 | 248.859075 | 54.795351 |

Worst horizon step by MAE:

| model_label | horizon_step | n_predictions | mae | rmse | mean_error_actual_minus_predicted |
| --- | --- | --- | --- | --- | --- |
| Prophet-Regressor-Basic | 24 | 30 | 1345.882766 | 1558.335736 | -342.382598 |
| XGBoost-Basic | 9 | 30 | 913.969377 | 1132.857587 | 136.851489 |

## Time Cost Computing

| experiment_name | model_key | model_label | model_name | feature_set | parameter_set_id | n_train_rows | n_prediction_rows | train_time_seconds | prediction_time_seconds | model_predict_time_seconds | total_runtime_seconds | retraining_train_time_seconds | source_artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_test_revisi_1 | prophet_regressor_basic | Prophet-Regressor-Basic | prophet_regressor_basic | prophet_basic_regressors | prophet_regressor_basic_003 | 10030 | 720 | 0.000000 | 17.941776 | 15.906961 | 17.941776 | 3.790971 | new_recursive_forecast_revisi_1_7 |
| final_test_revisi_1 | xgb_basic | XGBoost-Basic | xgboost | xgb_basic | xgb_basic_292 | 10198 | 720 | 0.000000 | 2.686983 | 0.774728 | 20.616370 | 12.990324 | F:\Research\Research BDTS\outputs\final_test\metrics\final_runtime.csv |

## Interpretation

Berdasarkan final test revisi, model terbaik pada metric utama mae adalah XGBoost-Basic. Hasil ini menjadi input untuk comparative evaluation revisi berikutnya.

## Output Files

- Prophet-Regressor-Basic predictions: `F:\Research\Research BDTS\outputs\final_test\revisi_1\predictions\prophet_regressor_basic_final_predictions.csv`
- Prophet-Regressor-Basic metrics: `F:\Research\Research BDTS\outputs\final_test\revisi_1\metrics\prophet_regressor_basic_final_metrics.csv`
- Prophet-Regressor-Basic runtime: `F:\Research\Research BDTS\outputs\final_test\revisi_1\metrics\prophet_regressor_basic_final_runtime.csv`
- Comparison predictions: `F:\Research\Research BDTS\outputs\final_test\revisi_1\predictions\final_predictions_revisi_1.csv`
- Comparison metrics: `F:\Research\Research BDTS\outputs\final_test\revisi_1\metrics\final_metrics_revisi_1.csv`
- Comparison runtime: `F:\Research\Research BDTS\outputs\final_test\revisi_1\metrics\final_runtime_revisi_1.csv`
- Improvement summary: `F:\Research\Research BDTS\outputs\final_test\revisi_1\metrics\improvement_summary_revisi_1.csv`
- Experiment metadata: `F:\Research\Research BDTS\outputs\final_test\revisi_1\experiment_metadata.json`
- Report mirror: `F:\Research\Research BDTS\outputs\reports\final_test_revisi_1_prophet_regressor_basic_vs_xgb_basic.md`
- Actual vs predicted plot: `F:\Research\Research BDTS\outputs\final_test\revisi_1\figures\actual_vs_predicted_revisi_1.png`
- Residuals plot: `F:\Research\Research BDTS\outputs\final_test\revisi_1\figures\residuals_revisi_1.png`

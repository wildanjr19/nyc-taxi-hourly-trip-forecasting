# Final Testing

Run UTC: 2026-05-18T04:24:39.244822+00:00

## Scope

Tahap ini mengevaluasi model hasil retraining pada final_test yang dipisahkan sejak awal. Tidak ada tuning tambahan dan tidak ada pemilihan ulang parameter setelah melihat test metric.

## Methodology

Forecast horizon utama adalah 24 jam. XGBoost diprediksi dengan recursive forecasting dalam blok 24 jam; actual final_test hanya masuk ke history setelah blok selesai dievaluasi. Prophet dievaluasi pada blok timestamp yang sama tanpa refit.

## Final Metric Ranking

| rank_by_primary_metric | model_label | parameter_set_id | mae | rmse | mape | smape | prediction_time_seconds | retraining_train_time_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | XGBoost-Basic | xgb_basic_292 | 519.595949 | 720.224252 | 12.069298 | 12.169547 | 2.686983 | 12.990324 |
| 2 | XGBoost-Advanced | xgb_advanced_210 | 577.166385 | 942.704153 | 12.358912 | 13.238262 | 4.067967 | 1.151408 |
| 3 | Prophet | prophet_005 | 1265.318768 | 1644.596324 | 40.685479 | 41.841144 | 2.644693 | 3.525288 |

## Residual Summary

| model_label | n_predictions | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | residual_std | mean_absolute_error | rmse | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | 720 | 739.155880 | 620.866652 | 1470.152212 | 1265.318768 | 1644.596324 | 0.309722 | 0.690278 | 5103.161471 |
| XGBoost-Basic | 720 | -8.104758 | -69.922791 | 720.679294 | 519.595949 | 720.224252 | 0.550000 | 0.450000 | 4017.453613 |
| XGBoost-Advanced | 720 | 61.359600 | -27.764435 | 941.359067 | 577.166385 | 942.704153 | 0.536111 | 0.463889 | 6248.800293 |

## Horizon Behavior

Best horizon step by MAE:

| model_label | horizon_step | n_predictions | mae | rmse | mean_error_actual_minus_predicted |
| --- | --- | --- | --- | --- | --- |
| Prophet | 17 | 30 | 536.441573 | 820.635151 | 142.605985 |
| XGBoost-Advanced | 5 | 30 | 163.001272 | 249.593599 | 95.521312 |
| XGBoost-Basic | 5 | 30 | 166.291864 | 248.859075 | 54.795351 |

Worst horizon step by MAE:

| model_label | horizon_step | n_predictions | mae | rmse | mean_error_actual_minus_predicted |
| --- | --- | --- | --- | --- | --- |
| Prophet | 9 | 30 | 2580.110112 | 2858.764329 | 1798.256766 |
| XGBoost-Advanced | 18 | 30 | 1007.918815 | 1421.098097 | 163.563509 |
| XGBoost-Basic | 9 | 30 | 913.969377 | 1132.857587 | 136.851489 |

## Time Cost Computing

| experiment_name | model_key | model_label | model_name | feature_set | parameter_set_id | train_start | train_end | validation_start | validation_end | n_train_rows | n_prediction_rows | model_load_time_seconds | train_time_seconds | prediction_time_seconds | model_predict_time_seconds | total_runtime_seconds | retraining_train_time_seconds | status | error_message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_test | prophet | Prophet | prophet | prophet_internal | prophet_005 | 2025-01-01T05:00:00+00:00 | 2026-03-02T03:00:00+00:00 | 2026-03-02T04:00:00+00:00 | 2026-04-01T03:00:00+00:00 | 10198 | 720 | 3.559717 | 0.000000 | 2.644693 | 2.429126 | 6.204434 | 3.525288 | success |  |
| final_test | xgb_basic | XGBoost-Basic | xgboost | xgb_basic | xgb_basic_292 | 2025-01-01T05:00:00+00:00 | 2026-03-02T03:00:00+00:00 | 2026-03-02T04:00:00+00:00 | 2026-04-01T03:00:00+00:00 | 10198 | 720 | 17.929373 | 0.000000 | 2.686983 | 0.774728 | 20.616370 | 12.990324 | success |  |
| final_test | xgb_advanced | XGBoost-Advanced | xgboost | xgb_advanced | xgb_advanced_210 | 2025-01-01T05:00:00+00:00 | 2026-03-02T03:00:00+00:00 | 2026-03-02T04:00:00+00:00 | 2026-04-01T03:00:00+00:00 | 10198 | 720 | 0.181118 | 0.000000 | 4.067967 | 1.270581 | 4.249097 | 1.151408 | success |  |

## Interpretation

Berdasarkan final test, model terbaik pada metric utama mae adalah XGBoost-Basic. Hasil ini menjadi dasar tahap berikutnya untuk comparative evaluation dan error pattern analysis.

## Output Files

- Final metrics: `F:\Research\Research BDTS\outputs\final_test\metrics\final_metrics.csv`
- Final runtime: `F:\Research\Research BDTS\outputs\final_test\metrics\final_runtime.csv`
- Final predictions: `F:\Research\Research BDTS\outputs\final_test\predictions\final_predictions.csv`
- Residual summary: `F:\Research\Research BDTS\outputs\final_test\metrics\residual_summary.csv`
- Horizon error summary: `F:\Research\Research BDTS\outputs\final_test\metrics\horizon_error_summary.csv`
- Experiment metadata: `F:\Research\Research BDTS\outputs\final_test\experiment_metadata.json`
- Report mirror: `F:\Research\Research BDTS\outputs\reports\final_test_report.md`
- Actual vs predicted plot: `F:\Research\Research BDTS\outputs\final_test\figures\actual_vs_predicted.png`
- Residuals plot: `F:\Research\Research BDTS\outputs\final_test\figures\residuals.png`

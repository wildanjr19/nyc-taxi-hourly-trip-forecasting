# Experiment A: Prophet vs XGBoost-Basic

Run UTC: 2026-05-17T11:48:08.719674+00:00

## Scope

Experiment A memakai artefak tuning/CV terbaik dari tahap 10. Final test tidak dibaca dan tidak digunakan pada tahap ini, sehingga benchmark test tetap tersimpan untuk tahap final testing setelah retraining.

## Best Configurations

| Model | Parameter set | Params |
|---|---:|---|
| Prophet | prophet_005 | `{"changepoint_prior_scale": 0.01, "seasonality_mode": "additive", "seasonality_prior_scale": 10.0}` |
| XGBoost-Basic | xgb_basic_292 | `{"colsample_bytree": 1.0, "learning_rate": 0.1, "max_depth": 3, "min_child_weight": 1, "n_estimators": 1000, "subsample": 0.8}` |

## CV Metric Comparison

| rank_by_primary_metric | model_label | parameter_set_id | mae_mean | mae_std | rmse_mean | rmse_std | mape_mean | smape_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | XGBoost-Basic | xgb_basic_292 | 638.659 | 148.293 | 961.206 | 274.498 | 27.083 | 17.451 |
| 2 | Prophet | prophet_005 | 1172.115 | 151.247 | 1582.783 | 204.997 | 45.823 | 35.046 |

## Improvement vs Prophet

| metric | baseline_model | challenger_model | baseline_mean | challenger_mean | absolute_reduction | percent_reduction_vs_baseline | winner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mae | Prophet | XGBoost-Basic | 1172.115 | 638.659 | 533.455 | 45.512 | XGBoost-Basic |
| rmse | Prophet | XGBoost-Basic | 1582.783 | 961.206 | 621.577 | 39.271 | XGBoost-Basic |
| mape | Prophet | XGBoost-Basic | 45.823 | 27.083 | 18.740 | 40.897 | XGBoost-Basic |
| smape | Prophet | XGBoost-Basic | 35.046 | 17.451 | 17.594 | 50.204 | XGBoost-Basic |

## Stability Across Folds

Winner by fold based on MAE: {"XGBoost-Basic": 5}.

| fold | mae_Prophet | mae_XGBoost-Basic | mae_absolute_reduction | mae_percent_reduction_vs_Prophet | mae_winner |
| --- | --- | --- | --- | --- | --- |
| 1 | 1189.504 | 750.670 | 438.834 | 36.892 | XGBoost-Basic |
| 2 | 1049.755 | 475.774 | 573.981 | 54.678 | XGBoost-Basic |
| 3 | 1072.619 | 507.691 | 564.929 | 52.668 | XGBoost-Basic |
| 4 | 1123.640 | 642.406 | 481.234 | 42.828 | XGBoost-Basic |
| 5 | 1425.055 | 816.756 | 608.300 | 42.686 | XGBoost-Basic |

## Time Cost Computing

| model_label | parameter_set_id | n_folds | best_train_time_seconds_sum | best_train_time_seconds_mean | best_prediction_time_seconds_sum | best_prediction_time_seconds_mean | best_total_runtime_seconds_sum | best_total_runtime_seconds_mean | full_tuning_fold_runtime_seconds_sum | full_tuning_train_time_seconds_sum | full_tuning_prediction_time_seconds_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | prophet_005 | 5 | 4.192262 | 0.838452 | 0.811391 | 0.162278 | 5.003716 | 1.000743 | 250.689377 | 229.350280 | 21.337121 |
| XGBoost-Basic | xgb_basic_292 | 5 | 1.679109 | 0.335822 | 3.616443 | 0.723289 | 5.307297 | 1.061459 | 1663.681676 | 555.956835 | 1104.093295 |

## Residual Behavior

| model_label | n_predictions | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | residual_std | mean_absolute_error | rmse | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | 840 | 120.342 | -87.997 | 1589.763 | 1172.115 | 1593.368 | 0.533 | 0.467 | 5289.655 |
| XGBoost-Basic | 840 | -29.745 | -3.994 | 992.211 | 638.659 | 992.067 | 0.504 | 0.496 | 4472.905 |

Best horizon step by MAE:

| model_label | horizon_step | n_predictions | mae | rmse | mean_error_actual_minus_predicted |
| --- | --- | --- | --- | --- | --- |
| Prophet | 7 | 35 | 469.416 | 588.534 | -280.043 |
| XGBoost-Basic | 5 | 35 | 205.153 | 295.260 | 70.441 |

Worst horizon step by MAE:

| model_label | horizon_step | n_predictions | mae | rmse | mean_error_actual_minus_predicted |
| --- | --- | --- | --- | --- | --- |
| Prophet | 1 | 35 | 1873.356 | 2341.504 | 714.929 |
| XGBoost-Basic | 20 | 35 | 1232.909 | 1572.711 | 622.423 |

## Interpretation

Berdasarkan CV best configuration, XGBoost-Basic menjadi model terbaik untuk Experiment A pada metric utama mae. XGBoost-Basic menurunkan MAE sebesar 45.51% dibanding Prophet, RMSE sebesar 39.27%, dan sMAPE sebesar 50.20%.

Hasil ini menunjukkan bahwa fitur lag sederhana dan calendar minimal sudah menangkap struktur temporal jangka pendek dengan lebih kuat daripada trend/seasonality internal Prophet pada window CV ini. Kesimpulan final tetap menunggu retraining dan final testing yang dijalankan pada tahap berikutnya.

## Output Files

- Metric comparison: `F:\Research\Research BDTS\outputs\experiments\experiment_a\metrics\metric_comparison.csv`
- Fold comparison: `F:\Research\Research BDTS\outputs\experiments\experiment_a\metrics\fold_comparison.csv`
- Runtime comparison: `F:\Research\Research BDTS\outputs\experiments\experiment_a\metrics\runtime_comparison.csv`
- Best CV predictions: `F:\Research\Research BDTS\outputs\experiments\experiment_a\predictions\best_cv_predictions.csv`
- Residual summary: `F:\Research\Research BDTS\outputs\experiments\experiment_a\metrics\residual_summary.csv`
- Report mirror: `F:\Research\Research BDTS\outputs\reports\experiment_a_prophet_vs_xgb_basic.md`
- Actual vs predicted plot: `F:\Research\Research BDTS\outputs\experiments\experiment_a\figures\actual_vs_predicted_best_cv.png`
- Residual distribution plot: `F:\Research\Research BDTS\outputs\experiments\experiment_a\figures\residual_distribution_best_cv.png`
- Horizon error plot: `F:\Research\Research BDTS\outputs\experiments\experiment_a\figures\absolute_error_by_horizon.png`

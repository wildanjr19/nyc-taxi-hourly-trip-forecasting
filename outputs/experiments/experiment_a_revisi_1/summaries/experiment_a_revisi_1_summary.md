# Experiment A Revisi 1: Prophet-Regressor-Basic vs XGBoost-Basic

Run UTC: 2026-06-07T05:01:35.133856+00:00

## Scope

Experiment A revisi memakai artefak tuning/CV terbaik untuk Prophet-Regressor-Basic dan XGBoost-Basic. Final test tidak dibaca dan tidak digunakan pada tahap ini, sehingga benchmark final tetap tersimpan untuk tahap final testing revisi.

Perbandingan ini menggantikan Experiment A lama sebagai pembanding utama revisi karena Prophet kini menerima regressor yang sebanding dengan feature set XGBoost-Basic: lag_1, lag_24, lag_168, hour, dan day_of_week. Report lama tetap dipertahankan sebagai baseline historis timestamp-only Prophet.

## Best Configurations

| Model | Parameter set | Params |
|---|---:|---|
| Prophet-Regressor-Basic | prophet_regressor_basic_003 | `{"changepoint_prior_scale": 0.01, "seasonality_mode": "additive", "seasonality_prior_scale": 5.0}` |
| XGBoost-Basic | xgb_basic_292 | `{"colsample_bytree": 1.0, "learning_rate": 0.1, "max_depth": 3, "min_child_weight": 1, "n_estimators": 1000, "subsample": 0.8}` |

## CV Metric Comparison

| rank_by_primary_metric | model_label | parameter_set_id | mae_mean | mae_std | rmse_mean | rmse_std | mape_mean | smape_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | XGBoost-Basic | xgb_basic_292 | 638.659 | 148.293 | 961.206 | 274.498 | 27.083 | 17.451 |
| 2 | Prophet-Regressor-Basic | prophet_regressor_basic_003 | 852.044 | 150.242 | 1155.411 | 221.665 | 41.879 | 27.892 |

## Improvement vs Prophet-Regressor-Basic

| metric | baseline_model | challenger_model | baseline_mean | challenger_mean | absolute_reduction | percent_reduction_vs_baseline | winner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mae | Prophet-Regressor-Basic | XGBoost-Basic | 852.044 | 638.659 | 213.384 | 25.044 | XGBoost-Basic |
| rmse | Prophet-Regressor-Basic | XGBoost-Basic | 1155.411 | 961.206 | 194.205 | 16.808 | XGBoost-Basic |
| mape | Prophet-Regressor-Basic | XGBoost-Basic | 41.879 | 27.083 | 14.797 | 35.332 | XGBoost-Basic |
| smape | Prophet-Regressor-Basic | XGBoost-Basic | 27.892 | 17.451 | 10.441 | 37.433 | XGBoost-Basic |

## Stability Across Folds

Winner by fold based on MAE: {"XGBoost-Basic": 5}.

| fold | mae_Prophet-Regressor-Basic | mae_XGBoost-Basic | mae_absolute_reduction | mae_percent_reduction_vs_Prophet-Regressor-Basic | mae_winner |
| --- | --- | --- | --- | --- | --- |
| 1 | 937.320 | 750.670 | 186.650 | 19.913 | XGBoost-Basic |
| 2 | 789.308 | 475.774 | 313.534 | 39.723 | XGBoost-Basic |
| 3 | 719.937 | 507.691 | 212.246 | 29.481 | XGBoost-Basic |
| 4 | 740.051 | 642.406 | 97.645 | 13.194 | XGBoost-Basic |
| 5 | 1073.602 | 816.756 | 256.847 | 23.924 | XGBoost-Basic |

## Time Cost Computing

| model_label | parameter_set_id | n_folds | best_train_time_seconds_sum | best_train_time_seconds_mean | best_prediction_time_seconds_sum | best_prediction_time_seconds_mean | best_total_runtime_seconds_sum | best_total_runtime_seconds_mean | full_tuning_fold_runtime_seconds_sum | full_tuning_train_time_seconds_sum | full_tuning_prediction_time_seconds_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet-Regressor-Basic | prophet_regressor_basic_003 | 5 | 4.672313 | 0.934463 | 23.460732 | 4.692146 | 28.133143 | 5.626629 | 835.459919 | 247.747393 | 587.710155 |
| XGBoost-Basic | xgb_basic_292 | 5 | 1.679109 | 0.335822 | 3.616443 | 0.723289 | 5.307297 | 1.061459 | 1663.681676 | 555.956835 | 1104.093295 |

## Residual Behavior

| model_label | n_predictions | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | residual_std | mean_absolute_error | rmse | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet-Regressor-Basic | 840 | 36.163 | -35.020 | 1172.438 | 852.044 | 1172.298 | 0.511 | 0.489 | 4568.538 |
| XGBoost-Basic | 840 | -29.745 | -3.994 | 992.211 | 638.659 | 992.067 | 0.504 | 0.496 | 4472.905 |

Best horizon step by MAE:

| model_label | horizon_step | n_predictions | mae | rmse | mean_error_actual_minus_predicted |
| --- | --- | --- | --- | --- | --- |
| Prophet-Regressor-Basic | 6 | 35 | 414.975 | 583.517 | -234.796 |
| XGBoost-Basic | 5 | 35 | 205.153 | 295.260 | 70.441 |

Worst horizon step by MAE:

| model_label | horizon_step | n_predictions | mae | rmse | mean_error_actual_minus_predicted |
| --- | --- | --- | --- | --- | --- |
| Prophet-Regressor-Basic | 20 | 35 | 1527.843 | 1821.587 | 1018.108 |
| XGBoost-Basic | 20 | 35 | 1232.909 | 1572.711 | 622.423 |

## Relation to Old Experiment A Report

Report lama `F:\Research\Research BDTS\outputs\reports\experiment_a_prophet_vs_xgb_basic.md` dan report revisi ini tidak sama. Perbedaannya metodologis: report lama membandingkan Prophet timestamp-only vs XGBoost-Basic, sedangkan report revisi membandingkan Prophet-Regressor-Basic vs XGBoost-Basic.

## Interpretation

Berdasarkan CV best configuration, XGBoost-Basic menjadi model terbaik untuk Experiment A revisi pada metric utama mae. XGBoost-Basic menurunkan MAE sebesar 25.04% dibanding Prophet-Regressor-Basic, RMSE sebesar 16.81%, dan sMAPE sebesar 37.43%.

Penambahan regressor basic memperbaiki fairness perbandingan CV karena Prophet dan XGBoost-Basic memakai informasi input temporal yang sebanding. Pada window CV ini, XGBoost-Basic tetap unggul, tetapi gap terhadap Prophet lebih kecil dibanding report lama timestamp-only Prophet.

## Output Files

- Metric comparison: `F:\Research\Research BDTS\outputs\experiments\experiment_a_revisi_1\metrics\metric_comparison.csv`
- Improvement summary: `F:\Research\Research BDTS\outputs\experiments\experiment_a_revisi_1\metrics\improvement_summary.csv`
- Fold comparison: `F:\Research\Research BDTS\outputs\experiments\experiment_a_revisi_1\metrics\fold_comparison.csv`
- Runtime comparison: `F:\Research\Research BDTS\outputs\experiments\experiment_a_revisi_1\metrics\runtime_comparison.csv`
- Best CV predictions: `F:\Research\Research BDTS\outputs\experiments\experiment_a_revisi_1\predictions\best_cv_predictions.csv`
- Residual summary: `F:\Research\Research BDTS\outputs\experiments\experiment_a_revisi_1\metrics\residual_summary.csv`
- Baseline report comparison: `F:\Research\Research BDTS\outputs\experiments\experiment_a_revisi_1\summaries\baseline_report_comparison.md`
- Report mirror: `F:\Research\Research BDTS\outputs\reports\experiment_a_revisi_1_prophet_regressor_basic_vs_xgb_basic.md`
- Actual vs predicted plot: `F:\Research\Research BDTS\outputs\experiments\experiment_a_revisi_1\figures\actual_vs_predicted_best_cv.png`
- Residual distribution plot: `F:\Research\Research BDTS\outputs\experiments\experiment_a_revisi_1\figures\residual_distribution_best_cv.png`
- Horizon error plot: `F:\Research\Research BDTS\outputs\experiments\experiment_a_revisi_1\figures\absolute_error_by_horizon.png`
- Fold error plot: `F:\Research\Research BDTS\outputs\experiments\experiment_a_revisi_1\figures\prophet_regressor_basic_vs_xgb_basic_error_by_fold.png`

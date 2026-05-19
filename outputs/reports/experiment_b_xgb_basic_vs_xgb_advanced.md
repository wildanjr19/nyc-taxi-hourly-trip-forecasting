# Experiment B: XGBoost-Basic vs XGBoost-Advanced

Run UTC: 2026-05-17T11:58:23.570314+00:00

## Scope

Experiment B memakai artefak tuning/CV terbaik dari tahap 10. Final test tidak dibaca dan tidak digunakan pada tahap ini. Fokusnya adalah mengukur dampak advanced feature engineering pada performa XGBoost dengan recursive forecasting CV.

## Feature Set Comparison

| feature_set | n_features | feature_columns | n_lag_features | n_rolling_features | n_calendar_features |
| --- | --- | --- | --- | --- | --- |
| xgb_basic | 5 | lag_1, lag_24, lag_168, hour, day_of_week | 3 | 0 | 2 |
| xgb_advanced | 17 | lag_1, lag_2, lag_3, lag_6, lag_12, lag_24, lag_48, lag_72, lag_168, rolling_mean_3, rolling_mean_24, rolling_mean_168, rolling_std_24, hour, day_of_week, is_weekend, month | 9 | 4 | 4 |

## Best Configurations

| Model | Parameter set | Params |
|---|---:|---|
| XGBoost-Basic | xgb_basic_292 | `{"colsample_bytree": 1.0, "learning_rate": 0.1, "max_depth": 3, "min_child_weight": 1, "n_estimators": 1000, "subsample": 0.8}` |
| XGBoost-Advanced | xgb_advanced_210 | `{"colsample_bytree": 1.0, "learning_rate": 0.1, "max_depth": 7, "min_child_weight": 10, "n_estimators": 600, "subsample": 0.8}` |

## CV Metric Comparison

| rank_by_primary_metric | model_label | feature_set | parameter_set_id | mae_mean | mae_std | rmse_mean | rmse_std | mape_mean | smape_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | XGBoost-Basic | xgb_basic | xgb_basic_292 | 638.659 | 148.293 | 961.206 | 274.498 | 27.083 | 17.451 |
| 2 | XGBoost-Advanced | xgb_advanced | xgb_advanced_210 | 644.175 | 162.871 | 1007.703 | 279.682 | 27.911 | 17.699 |

## Advanced vs Basic Improvement

| metric | baseline_model | challenger_model | baseline_mean | challenger_mean | absolute_reduction | percent_reduction_vs_baseline | winner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mae | XGBoost-Basic | XGBoost-Advanced | 638.659 | 644.175 | -5.516 | -0.864 | XGBoost-Basic |
| rmse | XGBoost-Basic | XGBoost-Advanced | 961.206 | 1007.703 | -46.498 | -4.837 | XGBoost-Basic |
| mape | XGBoost-Basic | XGBoost-Advanced | 27.083 | 27.911 | -0.828 | -3.058 | XGBoost-Basic |
| smape | XGBoost-Basic | XGBoost-Advanced | 17.451 | 17.699 | -0.247 | -1.418 | XGBoost-Basic |

## Stability Across Folds

Winner by fold based on MAE: {"XGBoost-Advanced": 3, "XGBoost-Basic": 2}.

| fold | mae_XGBoost-Basic | mae_XGBoost-Advanced | mae_absolute_reduction | mae_percent_reduction_vs_XGBoost-Basic | mae_winner |
| --- | --- | --- | --- | --- | --- |
| 1 | 750.670 | 838.212 | -87.542 | -11.662 | XGBoost-Basic |
| 2 | 475.774 | 503.431 | -27.657 | -5.813 | XGBoost-Basic |
| 3 | 507.691 | 473.495 | 34.196 | 6.736 | XGBoost-Advanced |
| 4 | 642.406 | 623.413 | 18.993 | 2.957 | XGBoost-Advanced |
| 5 | 816.756 | 782.325 | 34.431 | 4.216 | XGBoost-Advanced |

## Behavioral Error

| segment | XGBoost-Advanced | XGBoost-Basic | mae_absolute_reduction_advanced_vs_basic | mae_percent_reduction_advanced_vs_basic |
| --- | --- | --- | --- | --- |
| all | 644.175 | 638.659 | -5.516 | -0.864 |
| high_demand_spike_p90 | 772.982 | 874.292 | 101.310 | 11.588 |
| night_local_00_05 | 271.149 | 255.140 | -16.009 | -6.274 |
| rush_hour_local_07_09_16_19 | 873.772 | 886.243 | 12.471 | 1.407 |
| weekday_local | 581.368 | 584.634 | 3.265 | 0.559 |
| weekend_local | 801.193 | 773.723 | -27.470 | -3.550 |

## Time Cost Computing

| model_label | parameter_set_id | n_folds | best_train_time_seconds_sum | best_train_time_seconds_mean | best_prediction_time_seconds_sum | best_prediction_time_seconds_mean | best_total_runtime_seconds_sum | best_total_runtime_seconds_mean | full_tuning_fold_runtime_seconds_sum | full_tuning_train_time_seconds_sum | full_tuning_prediction_time_seconds_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost-Basic | xgb_basic_292 | 5 | 1.679109 | 0.335822 | 3.616443 | 0.723289 | 5.307297 | 1.061459 | 1663.681676 | 555.956835 | 1104.093295 |
| XGBoost-Advanced | xgb_advanced_210 | 5 | 4.574346 | 0.914869 | 5.259900 | 1.051980 | 9.847993 | 1.969599 | 2653.842677 | 1011.092424 | 1638.275169 |

## Residual Behavior

| model_label | n_predictions | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | residual_std | mean_absolute_error | rmse | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost-Advanced | 840 | -115.396 | -67.185 | 1032.471 | 644.175 | 1038.289 | 0.563 | 0.437 | 4421.833 |
| XGBoost-Basic | 840 | -29.745 | -3.994 | 992.211 | 638.659 | 992.067 | 0.504 | 0.496 | 4472.905 |

Best horizon step by MAE:

| model_label | horizon_step | n_predictions | mae | rmse | mean_error_actual_minus_predicted |
| --- | --- | --- | --- | --- | --- |
| XGBoost-Advanced | 6 | 35 | 178.101 | 254.164 | 54.967 |
| XGBoost-Basic | 5 | 35 | 205.153 | 295.260 | 70.441 |

Worst horizon step by MAE:

| model_label | horizon_step | n_predictions | mae | rmse | mean_error_actual_minus_predicted |
| --- | --- | --- | --- | --- | --- |
| XGBoost-Advanced | 20 | 35 | 1206.683 | 1680.071 | 162.424 |
| XGBoost-Basic | 20 | 35 | 1232.909 | 1572.711 | 622.423 |

## Interpretation

Berdasarkan CV best configuration, XGBoost-Basic menjadi model terbaik untuk Experiment B pada metric utama mae. Advanced feature engineering belum memberi peningkatan pada metric utama. Perubahan MAE advanced terhadap basic adalah -0.86%, RMSE -4.84%, dan sMAPE -1.42%.

Pada window CV ini, tambahan lag, rolling statistics, dan calendar features belum otomatis memperbaiki generalisasi. Ini bisa terjadi karena feature set yang lebih kaya meningkatkan kompleksitas model dan sensitivitas terhadap pola fold tertentu. Kesimpulan ini tetap perlu dibawa ke tahap retraining dan final testing tanpa tuning ulang.

## Output Files

- Feature set comparison: `F:\Research\Research BDTS\outputs\experiments\experiment_b\metrics\feature_set_comparison.csv`
- Metric comparison: `F:\Research\Research BDTS\outputs\experiments\experiment_b\metrics\metric_comparison.csv`
- Improvement summary: `F:\Research\Research BDTS\outputs\experiments\experiment_b\metrics\advanced_vs_basic_improvement.csv`
- Fold comparison: `F:\Research\Research BDTS\outputs\experiments\experiment_b\metrics\fold_comparison.csv`
- Behavioral error summary: `F:\Research\Research BDTS\outputs\experiments\experiment_b\metrics\behavioral_error_summary.csv`
- Runtime comparison: `F:\Research\Research BDTS\outputs\experiments\experiment_b\metrics\runtime_comparison.csv`
- Best CV predictions: `F:\Research\Research BDTS\outputs\experiments\experiment_b\predictions\best_cv_predictions.csv`
- Report mirror: `F:\Research\Research BDTS\outputs\reports\experiment_b_xgb_basic_vs_xgb_advanced.md`
- Actual vs predicted plot: `F:\Research\Research BDTS\outputs\experiments\experiment_b\figures\actual_vs_predicted_best_cv.png`
- Residual distribution plot: `F:\Research\Research BDTS\outputs\experiments\experiment_b\figures\residual_distribution_best_cv.png`
- Horizon error plot: `F:\Research\Research BDTS\outputs\experiments\experiment_b\figures\absolute_error_by_horizon.png`
- Behavioral error plot: `F:\Research\Research BDTS\outputs\experiments\experiment_b\figures\mae_by_behavior_segment.png`

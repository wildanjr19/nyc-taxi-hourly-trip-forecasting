# Comparative Evaluation

Run UTC: 2026-05-18T10:13:09.456058+00:00

## Scope

Tahap ini membandingkan Prophet, XGBoost-Basic, dan XGBoost-Advanced memakai artefak yang sudah selesai dibuat. Tidak ada tuning, retraining, atau prediksi final test baru pada tahap ini.

## Leakage Guardrail

Final test tetap dipakai hanya sebagai label evaluasi dari output tahap 14. Semua prediksi final memiliki `used_actual_future_for_features=False`, dan metric final memiliki `final_test_used_for_tuning=False`.

## Final Benchmark Ranking

| rank_by_mae | model_label | parameter_set_id | mae | rmse | mape | smape | prediction_time_seconds | retraining_train_time_seconds | negative_prediction_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | XGBoost-Basic | xgb_basic_292 | 519.595949 | 720.224252 | 12.069298 | 12.169547 | 2.686983 | 12.990324 | 0 |
| 2 | XGBoost-Advanced | xgb_advanced_210 | 577.166385 | 942.704153 | 12.358912 | 13.238262 | 4.067967 | 1.151408 | 0 |
| 3 | Prophet | prophet_005 | 1265.318768 | 1644.596324 | 40.685479 | 41.841144 | 2.644693 | 3.525288 | 34 |

Ranking tiap metric:

| rank | mae | mape | rmse | smape |
| --- | --- | --- | --- | --- |
| 1 | XGBoost-Basic | XGBoost-Basic | XGBoost-Basic | XGBoost-Basic |
| 2 | XGBoost-Advanced | XGBoost-Advanced | XGBoost-Advanced | XGBoost-Advanced |
| 3 | Prophet | Prophet | Prophet | Prophet |

## Research Question A

Apakah machine learning sederhana mampu mengungguli forecasting klasik? Pada final test, XGBoost-Basic mengungguli Prophet. MAE turun 58.94%, RMSE turun 56.21%, dan sMAPE turun 70.91%.

## Research Question B

Apakah advanced feature engineering meningkatkan performa XGBoost? Pada final test ini, XGBoost-Advanced belum mengungguli XGBoost-Basic pada metric utama. Perubahan MAE advanced terhadap basic adalah -11.08%, RMSE -30.89%, dan sMAPE -8.78%.

## Improvement Summary

| comparison | metric | baseline_model | challenger_model | baseline_value | challenger_value | absolute_reduction | percent_reduction_vs_baseline | winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Experiment A final: XGBoost-Basic vs Prophet | mae | Prophet | XGBoost-Basic | 1265.318768 | 519.595949 | 745.722819 | 58.935569 | XGBoost-Basic |
| Experiment A final: XGBoost-Basic vs Prophet | rmse | Prophet | XGBoost-Basic | 1644.596324 | 720.224252 | 924.372073 | 56.206624 | XGBoost-Basic |
| Experiment A final: XGBoost-Basic vs Prophet | mape | Prophet | XGBoost-Basic | 40.685479 | 12.069298 | 28.616182 | 70.335122 | XGBoost-Basic |
| Experiment A final: XGBoost-Basic vs Prophet | smape | Prophet | XGBoost-Basic | 41.841144 | 12.169547 | 29.671597 | 70.914879 | XGBoost-Basic |
| Experiment B final: XGBoost-Advanced vs XGBoost-Basic | mae | XGBoost-Basic | XGBoost-Advanced | 519.595949 | 577.166385 | -57.570436 | -11.079847 | XGBoost-Basic |
| Experiment B final: XGBoost-Advanced vs XGBoost-Basic | rmse | XGBoost-Basic | XGBoost-Advanced | 720.224252 | 942.704153 | -222.479902 | -30.890365 | XGBoost-Basic |
| Experiment B final: XGBoost-Advanced vs XGBoost-Basic | mape | XGBoost-Basic | XGBoost-Advanced | 12.069298 | 12.358912 | -0.289615 | -2.399597 | XGBoost-Basic |
| Experiment B final: XGBoost-Advanced vs XGBoost-Basic | smape | XGBoost-Basic | XGBoost-Advanced | 12.169547 | 13.238262 | -1.068714 | -8.781874 | XGBoost-Basic |
| Reference final: XGBoost-Advanced vs Prophet | mae | Prophet | XGBoost-Advanced | 1265.318768 | 577.166385 | 688.152384 | 54.385693 | XGBoost-Advanced |
| Reference final: XGBoost-Advanced vs Prophet | rmse | Prophet | XGBoost-Advanced | 1644.596324 | 942.704153 | 701.892171 | 42.678690 | XGBoost-Advanced |
| Reference final: XGBoost-Advanced vs Prophet | mape | Prophet | XGBoost-Advanced | 40.685479 | 12.358912 | 28.326567 | 69.623285 | XGBoost-Advanced |
| Reference final: XGBoost-Advanced vs Prophet | smape | Prophet | XGBoost-Advanced | 41.841144 | 13.238262 | 28.602882 | 68.360661 | XGBoost-Advanced |

## Stability Across CV Folds

Winner by fold berdasarkan MAE: {"XGBoost-Advanced": 3, "XGBoost-Basic": 2}.

| model_label | parameter_set_id | mae_mean | mae_std | mae_coefficient_of_variation | rmse_mean | rmse_std | smape_mean | smape_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost-Basic | xgb_basic_292 | 638.659351 | 148.292658 | 0.232194 | 961.205915 | 274.497856 | 17.451291 | 7.402655 |
| XGBoost-Advanced | xgb_advanced_210 | 644.175365 | 162.870517 | 0.252836 | 1007.703421 | 279.681791 | 17.698783 | 7.783693 |
| Prophet | prophet_005 | 1172.114830 | 151.246823 | 0.129038 | 1582.782696 | 204.997334 | 35.045682 | 5.479097 |

| fold | mae_Prophet | mae_XGBoost-Basic | mae_XGBoost-Advanced | mae_winner |
| --- | --- | --- | --- | --- |
| 1 | 1189.504098 | 750.669697 | 838.212125 | XGBoost-Basic |
| 2 | 1049.755113 | 475.774481 | 503.431284 | XGBoost-Basic |
| 3 | 1072.619296 | 507.690768 | 473.495164 | XGBoost-Advanced |
| 4 | 1123.640177 | 642.406298 | 623.413433 | XGBoost-Advanced |
| 5 | 1425.055466 | 816.755509 | 782.324818 | XGBoost-Advanced |

## Behavioral Comparison

| segment | Prophet | XGBoost-Advanced | XGBoost-Basic | mae_percent_reduction_xgb_basic_vs_prophet | mae_percent_reduction_xgb_advanced_vs_basic |
| --- | --- | --- | --- | --- | --- |
| all | 1265.318768 | 577.166385 | 519.595949 | 58.935569 | -11.079847 |
| weekday_local | 1252.277032 | 616.101937 | 523.850133 | 58.168191 | -17.610343 |
| weekend_local | 1301.183543 | 470.093614 | 507.896943 | 60.966541 | 7.443110 |
| rush_hour_local_07_09_16_19 | 1564.880112 | 784.360564 | 758.554404 | 51.526357 | -3.402018 |
| night_local_00_05 | 1288.223466 | 241.456941 | 227.477429 | 82.341773 | -6.145450 |
| high_demand_spike_p90 | 2431.964206 | 1001.142354 | 986.759542 | 59.425409 | -1.457580 |

## Residual And Horizon Behavior

| model_label | n_predictions | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | residual_std | mean_absolute_error | rmse | overprediction_rate | underprediction_rate | max_absolute_error | negative_prediction_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | 720 | 739.155880 | 620.866652 | 1470.152212 | 1265.318768 | 1644.596324 | 0.309722 | 0.690278 | 5103.161471 | 34 |
| XGBoost-Advanced | 720 | 61.359600 | -27.764435 | 941.359067 | 577.166385 | 942.704153 | 0.536111 | 0.463889 | 6248.800293 | 0 |
| XGBoost-Basic | 720 | -8.104758 | -69.922791 | 720.679294 | 519.595949 | 720.224252 | 0.550000 | 0.450000 | 4017.453613 | 0 |

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

| model_label | cv_best_total_runtime_seconds_sum | full_tuning_runtime_seconds_sum | retraining_train_time_seconds | final_model_load_time_seconds | final_prediction_time_seconds | known_end_to_end_runtime_seconds | rank_by_final_mae | rank_by_final_prediction_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost-Basic | 5.307297 | 1663.681676 | 12.990324 | 17.929373 | 2.686983 | 1697.288356 | 1 | 2 |
| XGBoost-Advanced | 9.847993 | 2653.842677 | 1.151408 | 0.181118 | 4.067967 | 2659.243170 | 2 | 3 |
| Prophet | 5.003716 | 250.689377 | 3.525288 | 3.559717 | 2.644693 | 260.419075 | 3 | 1 |

## Interpretation

Model terbaik secara final test berdasarkan mae adalah XGBoost-Basic. XGBoost-Basic juga menjadi pilihan paling seimbang pada benchmark ini: akurasinya terbaik, bias rata-ratanya kecil, dan waktu prediksi finalnya hampir sama dengan Prophet meskipun biaya tuning dan retrainingnya lebih besar.

Advanced feature set tidak otomatis meningkatkan generalisasi. Tambahan lag, rolling statistics, dan calendar feature memberi kapasitas model yang lebih besar, tetapi pada final test performanya lebih lemah daripada XGBoost-Basic, terutama terlihat dari RMSE dan max error yang lebih besar.

Prophet memiliki biaya prediksi final yang kompetitif, tetapi error final test jauh lebih tinggi dan menghasilkan sejumlah prediksi negatif. Temuan ini perlu dibahas lebih rinci pada tahap error pattern analysis.

## Output Files

- Final metric ranking: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation\metrics\final_metric_ranking.csv`
- Improvement summary: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation\metrics\final_improvement_summary.csv`
- CV stability summary: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation\metrics\cv_stability_summary.csv`
- Fold stability comparison: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation\metrics\fold_stability_comparison.csv`
- Time cost computing: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation\metrics\time_cost_computing.csv`
- Behavioral error summary: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation\metrics\behavioral_error_summary.csv`
- Report mirror: `F:\Research\Research BDTS\outputs\reports\comparative_evaluation.md`
- Actual vs predicted plot: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation\figures\actual_vs_predicted_final.png`
- Residual time series plot: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation\figures\residual_time_series_final.png`
- Residual distribution plot: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation\figures\residual_distribution_final.png`
- Scatter plot: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation\figures\actual_vs_predicted_scatter.png`
- Accuracy/runtime plot: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation\figures\accuracy_vs_runtime.png`
- Behavioral error plot: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation\figures\mae_by_behavior_segment.png`
- Horizon error plot: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation\figures\mae_by_horizon_step.png`
- CV fold MAE plot: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation\figures\cv_fold_mae.png`

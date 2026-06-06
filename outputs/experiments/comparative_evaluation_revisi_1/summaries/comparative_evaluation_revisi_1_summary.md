# Comparative Evaluation Revisi 1

Run UTC: 2026-05-25T11:38:05.593138+00:00

## Scope

Tahap Revisi 1.8 membandingkan empat model final: Prophet timestamp-only, Prophet-Regressor-Basic, XGBoost-Basic, dan XGBoost-Advanced. Prophet timestamp-only dipertahankan sebagai baseline tambahan / ablation, sedangkan Experiment A revisi memakai Prophet-Regressor-Basic vs XGBoost-Basic.

## Leakage Guardrail

Tahap ini hanya membaca artefak yang sudah dibuat. Tidak ada tuning, retraining, atau prediksi ulang final test. Prophet-Regressor-Basic memakai prediksi recursive dari Revisi 1.7; semua final predictions memiliki `used_actual_future_for_features=False`, dan metric final memiliki `final_test_used_for_tuning=False`.

## Final Benchmark Ranking

| rank_by_mae | model_label | parameter_set_id | mae | rmse | mape | smape | prediction_time_seconds | retraining_train_time_seconds | negative_prediction_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | XGBoost-Basic | xgb_basic_292 | 519.595949 | 720.224252 | 12.069298 | 12.169547 | 2.686983 | 12.990324 | 0 |
| 2 | XGBoost-Advanced | xgb_advanced_210 | 577.166385 | 942.704153 | 12.358912 | 13.238262 | 4.067967 | 1.151408 | 0 |
| 3 | Prophet-Regressor-Basic | prophet_regressor_basic_003 | 837.648179 | 1123.936750 | 32.502400 | 31.312555 | 17.941776 | 3.790971 | 33 |
| 4 | Prophet | prophet_005 | 1265.318768 | 1644.596324 | 40.685479 | 41.841144 | 2.644693 | 3.525288 | 34 |

Ranking tiap metric:

| rank | mae | mape | rmse | smape |
| --- | --- | --- | --- | --- |
| 1 | XGBoost-Basic | XGBoost-Basic | XGBoost-Basic | XGBoost-Basic |
| 2 | XGBoost-Advanced | XGBoost-Advanced | XGBoost-Advanced | XGBoost-Advanced |
| 3 | Prophet-Regressor-Basic | Prophet-Regressor-Basic | Prophet-Regressor-Basic | Prophet-Regressor-Basic |
| 4 | Prophet | Prophet | Prophet | Prophet |

## Prophet Ablation

Penambahan regressor basic memperbaiki Prophet timestamp-only pada final test. Dibanding Prophet timestamp-only, Prophet-Regressor-Basic menurunkan MAE 33.80%, RMSE 31.66%, dan sMAPE 25.16%.

## Research Question A Revisi

Apakah machine learning sederhana mampu mengungguli forecasting klasik ketika informasi input dibuat lebih sebanding? Pada final test, XGBoost-Basic tetap mengungguli Prophet-Regressor-Basic. MAE turun 37.97%, RMSE turun 35.92%, dan sMAPE turun 61.14%.

## Research Question B

Apakah advanced feature engineering meningkatkan performa XGBoost? Pada final test ini, XGBoost-Advanced belum mengungguli XGBoost-Basic pada metric utama. Perubahan MAE advanced terhadap basic adalah -11.08%, RMSE -30.89%, dan sMAPE -8.78%.

## Improvement Summary

| comparison | metric | baseline_model | challenger_model | baseline_value | challenger_value | absolute_reduction | percent_reduction_vs_baseline | winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ablation final: Prophet-Regressor-Basic vs Prophet timestamp-only | mae | Prophet | Prophet-Regressor-Basic | 1265.318768 | 837.648179 | 427.670589 | 33.799435 | Prophet-Regressor-Basic |
| Ablation final: Prophet-Regressor-Basic vs Prophet timestamp-only | rmse | Prophet | Prophet-Regressor-Basic | 1644.596324 | 1123.936750 | 520.659574 | 31.658807 | Prophet-Regressor-Basic |
| Ablation final: Prophet-Regressor-Basic vs Prophet timestamp-only | mape | Prophet | Prophet-Regressor-Basic | 40.685479 | 32.502400 | 8.183080 | 20.113023 | Prophet-Regressor-Basic |
| Ablation final: Prophet-Regressor-Basic vs Prophet timestamp-only | smape | Prophet | Prophet-Regressor-Basic | 41.841144 | 31.312555 | 10.528589 | 25.163243 | Prophet-Regressor-Basic |
| Experiment A revisi final: XGBoost-Basic vs Prophet-Regressor-Basic | mae | Prophet-Regressor-Basic | XGBoost-Basic | 837.648179 | 519.595949 | 318.052230 | 37.969668 | XGBoost-Basic |
| Experiment A revisi final: XGBoost-Basic vs Prophet-Regressor-Basic | rmse | Prophet-Regressor-Basic | XGBoost-Basic | 1123.936750 | 720.224252 | 403.712498 | 35.919503 | XGBoost-Basic |
| Experiment A revisi final: XGBoost-Basic vs Prophet-Regressor-Basic | mape | Prophet-Regressor-Basic | XGBoost-Basic | 32.502400 | 12.069298 | 20.433102 | 62.866441 | XGBoost-Basic |
| Experiment A revisi final: XGBoost-Basic vs Prophet-Regressor-Basic | smape | Prophet-Regressor-Basic | XGBoost-Basic | 31.312555 | 12.169547 | 19.143008 | 61.135247 | XGBoost-Basic |
| Experiment A historical final: XGBoost-Basic vs Prophet timestamp-only | mae | Prophet | XGBoost-Basic | 1265.318768 | 519.595949 | 745.722819 | 58.935569 | XGBoost-Basic |
| Experiment A historical final: XGBoost-Basic vs Prophet timestamp-only | rmse | Prophet | XGBoost-Basic | 1644.596324 | 720.224252 | 924.372073 | 56.206624 | XGBoost-Basic |
| Experiment A historical final: XGBoost-Basic vs Prophet timestamp-only | mape | Prophet | XGBoost-Basic | 40.685479 | 12.069298 | 28.616182 | 70.335122 | XGBoost-Basic |
| Experiment A historical final: XGBoost-Basic vs Prophet timestamp-only | smape | Prophet | XGBoost-Basic | 41.841144 | 12.169547 | 29.671597 | 70.914879 | XGBoost-Basic |
| Experiment B final: XGBoost-Advanced vs XGBoost-Basic | mae | XGBoost-Basic | XGBoost-Advanced | 519.595949 | 577.166385 | -57.570436 | -11.079847 | XGBoost-Basic |
| Experiment B final: XGBoost-Advanced vs XGBoost-Basic | rmse | XGBoost-Basic | XGBoost-Advanced | 720.224252 | 942.704153 | -222.479902 | -30.890365 | XGBoost-Basic |
| Experiment B final: XGBoost-Advanced vs XGBoost-Basic | mape | XGBoost-Basic | XGBoost-Advanced | 12.069298 | 12.358912 | -0.289615 | -2.399597 | XGBoost-Basic |
| Experiment B final: XGBoost-Advanced vs XGBoost-Basic | smape | XGBoost-Basic | XGBoost-Advanced | 12.169547 | 13.238262 | -1.068714 | -8.781874 | XGBoost-Basic |
| Reference final: XGBoost-Advanced vs Prophet-Regressor-Basic | mae | Prophet-Regressor-Basic | XGBoost-Advanced | 837.648179 | 577.166385 | 260.481794 | 31.096802 | XGBoost-Advanced |
| Reference final: XGBoost-Advanced vs Prophet-Regressor-Basic | rmse | Prophet-Regressor-Basic | XGBoost-Advanced | 1123.936750 | 942.704153 | 181.232596 | 16.124804 | XGBoost-Advanced |
| Reference final: XGBoost-Advanced vs Prophet-Regressor-Basic | mape | Prophet-Regressor-Basic | XGBoost-Advanced | 32.502400 | 12.358912 | 20.143487 | 61.975385 | XGBoost-Advanced |
| Reference final: XGBoost-Advanced vs Prophet-Regressor-Basic | smape | Prophet-Regressor-Basic | XGBoost-Advanced | 31.312555 | 13.238262 | 18.074294 | 57.722194 | XGBoost-Advanced |

## Stability Across CV Folds

Winner by fold berdasarkan MAE: {"XGBoost-Advanced": 3, "XGBoost-Basic": 2}.

| model_label | parameter_set_id | mae_mean | mae_std | mae_coefficient_of_variation | rmse_mean | rmse_std | smape_mean | smape_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost-Basic | xgb_basic_292 | 638.659351 | 148.292658 | 0.232194 | 961.205915 | 274.497856 | 17.451291 | 7.402655 |
| XGBoost-Advanced | xgb_advanced_210 | 644.175365 | 162.870517 | 0.252836 | 1007.703421 | 279.681791 | 17.698783 | 7.783693 |
| Prophet-Regressor-Basic | prophet_regressor_basic_003 | 852.043712 | 150.241885 | 0.176331 | 1155.411238 | 221.664560 | 27.892088 | 9.831884 |
| Prophet | prophet_005 | 1172.114830 | 151.246823 | 0.129038 | 1582.782696 | 204.997334 | 35.045682 | 5.479097 |

| fold | mae_Prophet | mae_Prophet-Regressor-Basic | mae_XGBoost-Basic | mae_XGBoost-Advanced | mae_winner |
| --- | --- | --- | --- | --- | --- |
| 1 | 1189.504098 | 937.319773 | 750.669697 | 838.212125 | XGBoost-Basic |
| 2 | 1049.755113 | 789.308121 | 475.774481 | 503.431284 | XGBoost-Basic |
| 3 | 1072.619296 | 719.937183 | 507.690768 | 473.495164 | XGBoost-Advanced |
| 4 | 1123.640177 | 740.051084 | 642.406298 | 623.413433 | XGBoost-Advanced |
| 5 | 1425.055466 | 1073.602398 | 816.755509 | 782.324818 | XGBoost-Advanced |

## Behavioral Comparison

| segment | Prophet | Prophet-Regressor-Basic | XGBoost-Advanced | XGBoost-Basic | mae_percent_reduction_prophet_regressor_vs_prophet | mae_percent_reduction_xgb_basic_vs_prophet_regressor | mae_percent_reduction_xgb_advanced_vs_basic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 1265.318768 | 837.648179 | 577.166385 | 519.595949 | 33.799435 | 37.969668 | -11.079847 |
| weekday_local | 1252.277032 | 861.764508 | 616.101937 | 523.850133 | 31.184196 | 39.211916 | -17.610343 |
| weekend_local | 1301.183543 | 771.328273 | 470.093614 | 507.896943 | 40.721025 | 34.152946 | 7.443110 |
| rush_hour_local_07_09_16_19 | 1564.880112 | 1015.715149 | 784.360564 | 758.554404 | 35.093101 | 25.318195 | -3.402018 |
| night_local_00_05 | 1288.223466 | 634.456642 | 241.456941 | 227.477429 | 50.749489 | 64.146103 | -6.145450 |
| high_demand_spike_p90 | 2431.964206 | 1191.884010 | 1001.142354 | 986.759542 | 50.990890 | 17.210103 | -1.457580 |

## Residual And Horizon Behavior

| model_label | n_predictions | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | residual_std | mean_absolute_error | rmse | overprediction_rate | underprediction_rate | max_absolute_error | negative_prediction_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | 720 | 739.155880 | 620.866652 | 1470.152212 | 1265.318768 | 1644.596324 | 0.309722 | 0.690278 | 5103.161471 | 34 |
| Prophet-Regressor-Basic | 720 | 112.570706 | 11.505491 | 1119.062537 | 837.648179 | 1123.936750 | 0.493056 | 0.506944 | 5299.931706 | 33 |
| XGBoost-Advanced | 720 | 61.359600 | -27.764435 | 941.359067 | 577.166385 | 942.704153 | 0.536111 | 0.463889 | 6248.800293 | 0 |
| XGBoost-Basic | 720 | -8.104758 | -69.922791 | 720.679294 | 519.595949 | 720.224252 | 0.550000 | 0.450000 | 4017.453613 | 0 |

Best horizon step by MAE:

| model_label | horizon_step | n_predictions | mae | rmse | mean_error_actual_minus_predicted |
| --- | --- | --- | --- | --- | --- |
| Prophet | 17 | 30 | 536.441573 | 820.635151 | 142.605985 |
| Prophet-Regressor-Basic | 1 | 30 | 564.119107 | 699.151115 | -191.019191 |
| XGBoost-Advanced | 5 | 30 | 163.001272 | 249.593599 | 95.521312 |
| XGBoost-Basic | 5 | 30 | 166.291864 | 248.859075 | 54.795351 |

Worst horizon step by MAE:

| model_label | horizon_step | n_predictions | mae | rmse | mean_error_actual_minus_predicted |
| --- | --- | --- | --- | --- | --- |
| Prophet | 9 | 30 | 2580.110112 | 2858.764329 | 1798.256766 |
| Prophet-Regressor-Basic | 24 | 30 | 1345.882766 | 1558.335736 | -342.382598 |
| XGBoost-Advanced | 18 | 30 | 1007.918815 | 1421.098097 | 163.563509 |
| XGBoost-Basic | 9 | 30 | 913.969377 | 1132.857587 | 136.851489 |

## Accuracy vs Runtime

| model_label | cv_best_total_runtime_seconds_sum | full_tuning_runtime_seconds_sum | retraining_train_time_seconds | final_model_load_time_seconds | final_prediction_time_seconds | known_end_to_end_runtime_seconds | rank_by_final_mae | rank_by_final_prediction_time | rank_by_known_end_to_end_runtime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost-Basic | 5.307297 | 1663.681676 | 12.990324 | 17.929373 | 2.686983 | 1697.288356 | 1 | 2 | 3 |
| XGBoost-Advanced | 9.847993 | 2653.842677 | 1.151408 | 0.181118 | 4.067967 | 2659.243170 | 2 | 3 | 4 |
| Prophet-Regressor-Basic | 28.133143 | 835.459919 | 3.790971 | 0.000000 | 17.941776 | 857.192666 | 3 | 4 | 2 |
| Prophet | 5.003716 | 250.689377 | 3.525288 | 3.559717 | 2.644693 | 260.419075 | 4 | 1 | 1 |

XGBoost-Basic menjadi titik terbaik pada trade-off final test: MAE paling rendah dan waktu prediksi final tetap kecil. Prophet-Regressor-Basic lebih akurat daripada Prophet timestamp-only, tetapi waktu prediksi finalnya naik karena setiap step harus membangun lag regressor secara recursive dan memanggil Prophet. Prophet timestamp-only tetap cepat, tetapi error dan prediksi negatifnya paling besar.

## Interpretation

Model terbaik secara final test berdasarkan mae adalah XGBoost-Basic. Hasil revisi memperjelas dua hal: penambahan regressor basic memang memperbaiki Prophet, tetapi XGBoost-Basic masih lebih kuat pada benchmark final test yang tidak tersentuh sebelumnya.

Advanced feature set tidak otomatis meningkatkan generalisasi. Tambahan lag, rolling statistics, dan calendar feature memberi kapasitas model yang lebih besar, tetapi pada final test performanya masih lebih lemah daripada XGBoost-Basic.

## Output Files

- Final metric ranking: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation_revisi_1\metrics\final_metric_ranking_revisi_1.csv`
- Improvement summary: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation_revisi_1\metrics\final_improvement_summary_revisi_1.csv`
- CV stability summary: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation_revisi_1\metrics\cv_stability_summary_revisi_1.csv`
- Fold stability comparison: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation_revisi_1\metrics\fold_stability_comparison_revisi_1.csv`
- Time cost computing: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation_revisi_1\metrics\time_cost_computing_revisi_1.csv`
- Behavioral error summary: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation_revisi_1\metrics\behavioral_error_summary_revisi_1.csv`
- Report mirror: `F:\Research\Research BDTS\outputs\reports\comparative_evaluation_revisi_1.md`
- Actual vs predicted plot: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation_revisi_1\figures\actual_vs_predicted_final_revisi_1.png`
- Residual time series plot: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation_revisi_1\figures\residual_time_series_final_revisi_1.png`
- Residual distribution plot: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation_revisi_1\figures\residual_distribution_final_revisi_1.png`
- Scatter plot: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation_revisi_1\figures\actual_vs_predicted_scatter_revisi_1.png`
- Accuracy/runtime plot: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation_revisi_1\figures\accuracy_vs_runtime_revisi_1.png`
- Behavioral error plot: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation_revisi_1\figures\mae_by_behavior_segment_revisi_1.png`
- Horizon error plot: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation_revisi_1\figures\mae_by_horizon_step_revisi_1.png`
- CV fold MAE plot: `F:\Research\Research BDTS\outputs\experiments\comparative_evaluation_revisi_1\figures\cv_fold_mae_revisi_1.png`

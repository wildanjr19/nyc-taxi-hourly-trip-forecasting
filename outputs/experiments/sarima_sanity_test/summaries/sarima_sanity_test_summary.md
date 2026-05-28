# SARIMA Sanity Test

Run UTC: 2026-05-25T04:21:29.519793+00:00

## Scope

Eksperimen ini adalah post-hoc sanity test atas permintaan dosen. SARIMA tidak dimasukkan ke flow penelitian utama, tidak dipakai untuk mengubah ranking comparative evaluation, dan outputnya disimpan terpisah.

## Methodology

Model auto-SARIMA dilatih pada train_val lalu dievaluasi pada final_test dengan rolling-origin block 24 jam. Actual final_test hanya digunakan sebagai label evaluasi dan baru dimasukkan ke state model setelah block 24 jam selesai diprediksi.
Konfigurasi evaluasi SARIMA: train window policy `full_train_val`, `update_maxiter=0`, dan final_test tidak digunakan untuk tuning.

## SARIMA Metrics

| model_label | parameter_set_id | order | seasonal_order | mae | rmse | mape | smape | negative_prediction_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SARIMA-Auto | auto_order_(1,0,1)_seasonal_(1,0,1,24) | (1, 0, 1) | (1, 0, 1, 24) | 1193.792793 | 1552.074343 | 55.733057 | 33.651731 | 1 |

## Reference Comparison

Tabel ini hanya referensi post-hoc terhadap hasil final_test yang sudah ada. Ini bukan ranking resmi baru.

| rank_by_mae_reference_only | model_label | source | official_workflow_member | mae | rmse | mape | smape | prediction_time_seconds | train_time_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | XGBoost-Basic | official_final_test_reference | True | 519.595949 | 720.224252 | 12.069298 | 12.169547 | 2.686983 | 12.990324 |
| 2 | XGBoost-Advanced | official_final_test_reference | True | 577.166385 | 942.704153 | 12.358912 | 13.238262 | 4.067967 | 1.151408 |
| 3 | SARIMA-Auto | standalone_sarima_sanity_test | False | 1193.792793 | 1552.074343 | 55.733057 | 33.651731 | 233.504210 | 645.353683 |
| 4 | Prophet | official_final_test_reference | True | 1265.318768 | 1644.596324 | 40.685479 | 41.841144 | 2.644693 | 3.525288 |

## Residual Summary

| model_label | n_predictions | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | residual_std | mean_absolute_error | rmse | overprediction_rate | underprediction_rate | negative_prediction_count | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SARIMA-Auto | 720 | -51.268275 | -29.482416 | 1552.305727 | 1193.792793 | 1552.074343 | 0.513889 | 0.486111 | 1 | 5992.075591 |

## Residual Diagnostics

Diagnostik ini dipakai untuk menilai apakah residual SARIMA sudah menyerupai white noise: berpusat di sekitar nol, tidak memiliki autokorelasi tersisa, dan mendekati distribusi normal.

| model_label | residual_mean | residual_std | mean_zero_p_value | sign_balance_p_value | runs_test_p_value | acf_lag_1 | acf_lag_24 | acf_lag_168 | significant_acf_lags_count_1_to_max_lag | significant_pacf_lags_count_1_to_max_lag | ljung_box_all_checked_lags_pass_white_noise | normality_all_tests_pass | residual_randomness_conclusion | sarima_suitability_conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SARIMA-Auto | -51.268275 | 1552.305727 | 0.375800 | 0.478916 | 0.000000 | 0.828027 | 0.273035 | 0.646508 | 104 | 35 | False | False | residuals_centered_but_autocorrelated | residual_autocorrelation_indicates_sarima_underfits_temporal_structure |

Ljung-Box test:

| lag | lb_stat | lb_pvalue | reject_white_noise_null | interpretation |
| --- | --- | --- | --- | --- |
| 24 | 1252.942734 | 0.000000 | True | residual_autocorrelation_detected |
| 48 | 1580.407621 | 0.000000 | True | residual_autocorrelation_detected |
| 72 | 1789.016559 | 0.000000 | True | residual_autocorrelation_detected |
| 168 | 3642.149720 | 0.000000 | True | residual_autocorrelation_detected |

Normality tests:

| test_name | statistic | p_value | reject_normality_null | interpretation |
| --- | --- | --- | --- | --- |
| Jarque-Bera | 20.302673 | 0.000039 | True | normality_rejected |
| D'Agostino K^2 | 11.930092 | 0.002567 | True | normality_rejected |
| Shapiro-Wilk | 0.992234 | 0.000813 | True | normality_rejected |

## Horizon Behavior

Best horizon step by MAE:

| model_label | horizon_step | n_predictions | mae | rmse | mean_error_actual_minus_predicted |
| --- | --- | --- | --- | --- | --- |
| SARIMA-Auto | 13 | 30 | 414.311765 | 511.380706 | 145.332583 |

Worst horizon step by MAE:

| model_label | horizon_step | n_predictions | mae | rmse | mean_error_actual_minus_predicted |
| --- | --- | --- | --- | --- | --- |
| SARIMA-Auto | 24 | 30 | 2033.206286 | 2629.083205 | -40.118112 |

## Time Cost Computing

| experiment_name | model_label | model_name | feature_set | parameter_set_id | n_train_rows | n_prediction_rows | train_time_seconds | prediction_time_seconds | forecast_time_seconds | update_time_seconds | total_runtime_seconds | status | error_message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sarima_sanity_test | SARIMA-Auto | sarima | univariate_auto_sarima_m24 | auto_order_(1,0,1)_seasonal_(1,0,1,24) | 10198 | 720 | 645.353683 | 233.504210 | 0.157404 | 233.207835 | 878.857893 | success |  |

Residual diagnostics runtime seconds: `2.151381`

## Interpretation

Secara teknis model konvensional SARIMA bisa digunakan pada data hourly ini. Namun pada sanity test ini SARIMA berada di rank 3 dari 4 berdasarkan MAE referensi, dengan MAE 1193.793. Dibanding model resmi terbaik (XGBoost-Basic, MAE 519.596), gap MAE SARIMA sekitar 129.75%. Jadi kesimpulan yang lebih tepat bukan 'SARIMA tidak bisa dipakai', melainkan SARIMA univariate harian ini kurang kompetitif untuk pola demand NYC Taxi dibanding model ML yang memakai lag/calendar features.

Diagnostik residual menunjukkan: Residual tidak menunjukkan bias mean yang kuat (t-test p=0.3758) dan urutan tanda residual belum terlihat acak (runs test p=1.507e-77), tetapi Ljung-Box menolak white-noise pada 4 dari 4 lag yang dicek. ACF residual masih penting untuk dibaca, terutama ACF lag-24=0.2730 dan lag-168=0.6465. Selain itu, normalitas residual ditolak oleh 3 dari 3 uji. Jadi untuk pertanyaan kecocokan model konvensional, auto-SARIMA univariate ini dapat dipakai sebagai baseline, tetapi residualnya belum ideal sebagai white noise; masih ada struktur temporal/distribusional yang belum ditangkap model.

## Output Files

- Params: `F:\Research\Research BDTS\outputs\experiments\sarima_sanity_test\params.json`
- Metrics: `F:\Research\Research BDTS\outputs\experiments\sarima_sanity_test\metrics\final_metrics.csv`
- Runtime summary: `F:\Research\Research BDTS\outputs\experiments\sarima_sanity_test\metrics\runtime_summary.csv`
- Predictions: `F:\Research\Research BDTS\outputs\experiments\sarima_sanity_test\predictions\final_predictions.csv`
- Residual summary: `F:\Research\Research BDTS\outputs\experiments\sarima_sanity_test\metrics\residual_summary.csv`
- Residual diagnostic summary: `F:\Research\Research BDTS\outputs\experiments\sarima_sanity_test\metrics\residual_diagnostic_summary.csv`
- Residual ACF/PACF values: `F:\Research\Research BDTS\outputs\experiments\sarima_sanity_test\metrics\residual_acf_pacf.csv`
- Ljung-Box test: `F:\Research\Research BDTS\outputs\experiments\sarima_sanity_test\metrics\residual_ljung_box.csv`
- Normality tests: `F:\Research\Research BDTS\outputs\experiments\sarima_sanity_test\metrics\residual_normality_tests.csv`
- Horizon error summary: `F:\Research\Research BDTS\outputs\experiments\sarima_sanity_test\metrics\horizon_error_summary.csv`
- Reference comparison: `F:\Research\Research BDTS\outputs\experiments\sarima_sanity_test\metrics\reference_metric_comparison.csv`
- Model summary: `F:\Research\Research BDTS\outputs\experiments\sarima_sanity_test\summaries\model_summary.txt`
- Metadata: `F:\Research\Research BDTS\outputs\experiments\sarima_sanity_test\experiment_metadata.json`
- Actual vs predicted plot: `F:\Research\Research BDTS\outputs\experiments\sarima_sanity_test\figures\actual_vs_predicted.png`
- Residuals plot: `F:\Research\Research BDTS\outputs\experiments\sarima_sanity_test\figures\residuals.png`
- Residual ACF/PACF plot: `F:\Research\Research BDTS\outputs\experiments\sarima_sanity_test\figures\residual_acf_pacf.png`
- Residual normality/QQ plot: `F:\Research\Research BDTS\outputs\experiments\sarima_sanity_test\figures\residual_normality_qq.png`

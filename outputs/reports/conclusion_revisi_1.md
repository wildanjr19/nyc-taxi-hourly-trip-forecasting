# Conclusion Revisi 1

Run UTC: 2026-05-25T12:25:38.679485+00:00

## Scope

Tahap Revisi 1.11 menyintesis hasil revisi pipeline penelitian NYC Taxi hourly `trip_count`. Conclusion ini memakai artefak Revisi 1.7 sampai Revisi 1.10: final testing, comparative evaluation, error analysis, dan interpretability.

Tidak ada tuning ulang, retraining ulang, atau prediksi final test baru pada tahap ini. Final test tetap diperlakukan sebagai unseen hold-out; semua kesimpulan final diambil dari artefak yang sudah dibuat sebelumnya.

## Metodologi Revisi

Revisi 1 memperbaiki fairness Research Question A. Prophet timestamp-only tidak lagi menjadi pembanding utama, melainkan baseline awal / ablation. Pembanding utama revisi adalah `Prophet-Regressor-Basic` vs `XGBoost-Basic`, karena keduanya memakai informasi lag dan calendar basic yang sebanding.

Prophet-Regressor-Basic memakai regressor `lag_1`, `lag_24`, `lag_168`, `hour`, dan `day_of_week`. Pada validation dan final test, lag regressor tidak diisi dari actual masa depan, tetapi dibangun secara recursive dari history masa lalu dan prediksi sebelumnya. Calendar regressor adalah known-future, sedangkan lag regressor bukan known-future.

## Final Test Ranking

| rank_by_mae | model_label             | parameter_set_id            | mae         | rmse        | mape      | smape     | prediction_time_seconds | negative_prediction_count |
| ----------- | ----------------------- | --------------------------- | ----------- | ----------- | --------- | --------- | ----------------------- | ------------------------- |
| 1           | XGBoost-Basic           | xgb_basic_292               | 519.595949  | 720.224252  | 12.069298 | 12.169547 | 2.686983                | 0                         |
| 2           | XGBoost-Advanced        | xgb_advanced_210            | 577.166385  | 942.704153  | 12.358912 | 13.238262 | 4.067967                | 0                         |
| 3           | Prophet-Regressor-Basic | prophet_regressor_basic_003 | 837.648179  | 1123.936750 | 32.502400 | 31.312555 | 17.941776               | 33                        |
| 4           | Prophet                 | prophet_005                 | 1265.318768 | 1644.596324 | 40.685479 | 41.841144 | 2.644693                | 34                        |

Model terbaik final test tetap **XGBoost-Basic** dengan MAE 519.595949, RMSE 720.224252, MAPE 12.069298, dan sMAPE 12.169547. Prophet-Regressor-Basic berada di rank 3 dengan MAE 837.648179, lebih baik dari Prophet timestamp-only MAE 1265.318768, tetapi belum mengungguli XGBoost-Basic.

## Jawaban Research Question

### Research Question A Revisi

Pertanyaan revisi: apakah machine learning forecasting sederhana masih mengungguli Prophet ketika Prophet diberi regressor basic yang sebanding?

Jawabannya: **ya**. Pada final test, XGBoost-Basic menurunkan MAE sebesar 37.969668%, RMSE sebesar 35.919503%, dan sMAPE sebesar 61.135247% dibanding Prophet-Regressor-Basic.

Artinya, keunggulan XGBoost-Basic pada benchmark ini bukan sekadar karena Prophet lama tidak memakai lag features. Setelah Prophet diberi regressor lag dan calendar basic secara leakage-safe, XGBoost-Basic tetap lebih adaptif terhadap dinamika hourly lokal.

### Posisi Prophet Timestamp-Only

Prophet timestamp-only dipertahankan sebagai baseline awal / ablation, bukan pembanding utama revisi. Baseline ini berguna untuk mengukur dampak penambahan regressor basic pada Prophet.

Dibanding Prophet timestamp-only, Prophet-Regressor-Basic menurunkan MAE 33.799435%, RMSE 31.658807%, dan sMAPE 25.163243%. Jadi penambahan regressor basic memang memperbaiki Prophet secara substansial.

### Research Question B

Research Question B tetap tidak berubah: apakah advanced feature engineering meningkatkan performa XGBoost?

Pada final test, jawabannya **belum**. XGBoost-Advanced memiliki MAE 577.166385, lebih buruk daripada XGBoost-Basic MAE 519.595949. Perubahan MAE advanced terhadap basic adalah -11.079847%, sehingga advanced feature set tidak memberi peningkatan overall final test meskipun membantu pada beberapa segmen.

## Stabilitas CV

| model_label             | parameter_set_id            | mae_mean    | mae_std    | rmse_mean   | smape_mean |
| ----------------------- | --------------------------- | ----------- | ---------- | ----------- | ---------- |
| XGBoost-Basic           | xgb_basic_292               | 638.659351  | 148.292658 | 961.205915  | 17.451291  |
| XGBoost-Advanced        | xgb_advanced_210            | 644.175365  | 162.870517 | 1007.703421 | 17.698783  |
| Prophet-Regressor-Basic | prophet_regressor_basic_003 | 852.043712  | 150.241885 | 1155.411238 | 27.892088  |
| Prophet                 | prophet_005                 | 1172.114830 | 151.246823 | 1582.782696 | 35.045682  |

Pada CV, XGBoost-Basic dan XGBoost-Advanced relatif dekat. XGBoost-Advanced menang di sebagian fold, tetapi rata-rata MAE CV dan final test tetap mendukung XGBoost-Basic sebagai pilihan utama yang lebih seimbang.

## Error Pattern dan Robustness

| segment                     | Prophet     | Prophet-Regressor-Basic | XGBoost-Basic | XGBoost-Advanced |
| --------------------------- | ----------- | ----------------------- | ------------- | ---------------- |
| all                         | 1265.318768 | 837.648179              | 519.595949    | 577.166385       |
| high_demand_spike_p90       | 2431.964206 | 1191.884010             | 986.759542    | 1001.142354      |
| night_local_00_05           | 1288.223466 | 634.456642              | 227.477429    | 241.456941       |
| rush_hour_local_07_09_16_19 | 1564.880112 | 1015.715149             | 758.554404    | 784.360564       |

Regressor basic memperbaiki kelemahan utama Prophet, terutama pada rush hour, high-demand spike P90, underprediction, dan residual autocorrelation lag harian. Namun XGBoost-Basic masih lebih rendah error-nya pada semua segment utama yang dianalisis.

| weakness                                  | prophet_timestamp_only | prophet_regressor_basic | xgb_basic  | percent_reduction_prophet_regressor_vs_prophet | percent_reduction_xgb_basic_vs_prophet_regressor |
| ----------------------------------------- | ---------------------- | ----------------------- | ---------- | ---------------------------------------------- | ------------------------------------------------ |
| overall_mae                               | 1265.318768            | 837.648179              | 519.595949 | 33.799435                                      | 37.969668                                        |
| overall_mean_error_actual_minus_predicted | 739.155880             | 112.570706              | 8.104758   | 84.770370                                      | 92.800295                                        |
| overall_underprediction_rate              | 0.690278               | 0.506944                | 0.450000   | 26.559356                                      | 11.232877                                        |
| negative_prediction_count                 | 34.000000              | 33.000000               | 0.000000   | 2.941176                                       | 100.000000                                       |
| rush_hour_local_07_09_16_19_mae           | 1564.880112            | 1015.715149             | 758.554404 | 35.093101                                      | 25.318195                                        |
| high_demand_spike_p90_mae                 | 2431.964206            | 1191.884010             | 986.759542 | 50.990890                                      | 17.210103                                        |
| residual_acf_lag_24                       | 0.467375               | 0.292124                | 0.155764   | 37.496827                                      | 46.678944                                        |

Residual summary:

| model_label             | mean_error_actual_minus_predicted | residual_std | mean_absolute_error | negative_prediction_count |
| ----------------------- | --------------------------------- | ------------ | ------------------- | ------------------------- |
| Prophet                 | 739.155880                        | 1470.152212  | 1265.318768         | 34                        |
| Prophet-Regressor-Basic | 112.570706                        | 1119.062537  | 837.648179          | 33                        |
| XGBoost-Advanced        | 61.359600                         | 941.359067   | 577.166385          | 0                         |
| XGBoost-Basic           | -8.104758                         | 720.679294   | 519.595949          | 0                         |

Residual autocorrelation terpilih:

| model_label             | lag | residual_autocorrelation |
| ----------------------- | --- | ------------------------ |
| Prophet                 | 1   | 0.838328                 |
| Prophet                 | 24  | 0.467375                 |
| Prophet                 | 48  | -0.069379                |
| Prophet-Regressor-Basic | 1   | 0.791725                 |
| Prophet-Regressor-Basic | 24  | 0.292124                 |
| Prophet-Regressor-Basic | 48  | 0.027649                 |
| XGBoost-Basic           | 1   | 0.768105                 |
| XGBoost-Basic           | 24  | 0.155764                 |
| XGBoost-Basic           | 48  | 0.164323                 |
| XGBoost-Advanced        | 1   | 0.862656                 |
| XGBoost-Advanced        | 24  | 0.185163                 |
| XGBoost-Advanced        | 48  | 0.055780                 |

Kelemahan Prophet-Regressor-Basic yang paling jelas adalah prediksi negatif masih muncul 33 kali pada final test, hampir sama dengan Prophet timestamp-only yang memiliki 34 prediksi negatif. XGBoost-Basic dan XGBoost-Advanced tidak menghasilkan prediksi negatif pada final test.

## Interpretability

| regressor   | rank_prophet_effect | prophet_share_of_total_mean_abs_regressor_effect | rank_xgb_basic_shap | xgb_basic_share_of_total_mean_abs_shap |
| ----------- | ------------------- | ------------------------------------------------ | ------------------- | -------------------------------------- |
| lag_1       | 1                   | 0.572330                                         | 1                   | 0.576958                               |
| lag_168     | 2                   | 0.219513                                         | 3                   | 0.145479                               |
| lag_24      | 3                   | 0.158801                                         | 4                   | 0.089782                               |
| day_of_week | 4                   | 0.035155                                         | 5                   | 0.035896                               |
| hour        | 5                   | 0.014200                                         | 2                   | 0.151885                               |

Interpretability Revisi 1 menunjukkan bahwa Prophet-Regressor-Basic dan XGBoost-Basic sama-sama menempatkan `lag_1` sebagai fitur paling dominan. Pada Prophet-Regressor-Basic, share mean absolute regressor effect `lag_1` adalah sekitar 57.23%; pada SHAP XGBoost-Basic, share mean absolute SHAP `lag_1` sekitar 57.70%.

Perbandingan ini bersifat directional. Komponen Prophet adalah dekomposisi model aditif dengan scaling internal regressor, sedangkan SHAP XGBoost adalah atribusi marginal TreeExplainer. Keduanya mendukung insight yang sama, tetapi tidak boleh disetarakan sebagai ukuran kontribusi yang identik.

## Trade-Off Runtime

| model_label             | full_tuning_runtime_seconds_sum | retraining_train_time_seconds | final_prediction_time_seconds | known_end_to_end_runtime_seconds | rank_by_final_mae | rank_by_final_prediction_time |
| ----------------------- | ------------------------------- | ----------------------------- | ----------------------------- | -------------------------------- | ----------------- | ----------------------------- |
| XGBoost-Basic           | 1663.681676                     | 12.990324                     | 2.686983                      | 1697.288356                      | 1                 | 2                             |
| XGBoost-Advanced        | 2653.842677                     | 1.151408                      | 4.067967                      | 2659.243170                      | 2                 | 3                             |
| Prophet-Regressor-Basic | 835.459919                      | 3.790971                      | 17.941776                     | 857.192666                       | 3                 | 4                             |
| Prophet                 | 250.689377                      | 3.525288                      | 2.644693                      | 260.419075                       | 4                 | 1                             |

Prophet timestamp-only tetap paling murah secara komputasi dengan known end-to-end runtime 260.419075 detik pada artefak comparative evaluation, tetapi akurasinya paling lemah. Prophet-Regressor-Basic lebih akurat daripada Prophet timestamp-only, namun final prediction time naik menjadi 17.941776 detik karena lag regressor harus dibangun recursive.

XGBoost-Basic membutuhkan full tuning runtime lebih besar (1663.681676 detik), tetapi final prediction time hanya 2.686983 detik dan memberi MAE final terbaik. Jadi trade-off terbaik pada benchmark ini adalah XGBoost-Basic: tuning mahal, inference final cepat, dan akurasi paling kuat.

## Keterbatasan Revisi

1. Prophet dengan lag regressors membutuhkan future regressor yang dibangun recursive karena nilai lag masa depan tidak diketahui saat forecasting.
2. Recursive lag regressors dapat mengakumulasi error di dalam horizon 24 jam, terutama ketika prediksi awal meleset pada perubahan level demand yang tajam.
3. Calendar regressors seperti `hour` dan `day_of_week` bersifat known-future, tetapi lag regressors seperti `lag_1`, `lag_24`, dan `lag_168` tidak known-future.
4. Prophet-Regressor-Basic masih menghasilkan prediksi negatif; perlu eksperimen post-processing non-negativity terpisah jika ingin dipakai dalam konteks operasional.
5. Final test hanya mencakup 30 hari terakhir, sehingga robustness lintas musim atau event kota yang lebih panjang belum diuji.
6. Interpretability Prophet component dan SHAP XGBoost tidak dapat dibandingkan secara satu-ke-satu karena metodologi atribusinya berbeda.

## Time Cost Computing Tahap Conclusion

Revisi 1.11 adalah post-hoc conclusion. Training time dan prediction time pada tahap ini adalah 0 detik karena tidak ada model yang dilatih atau diprediksi. Runtime tahap ini hanya mencakup pembacaan artefak, penyusunan report, metadata, dan logging.

Runtime rujukan post-hoc sebelumnya: error analysis 4.365908 detik dan interpretability 23.707272 detik.

## Kesimpulan Akhir Revisi

Kesimpulan akhir revisi adalah bahwa penambahan regressor basic membuat evaluasi Prophet jauh lebih fair dan secara nyata memperbaiki Prophet timestamp-only. Namun, setelah fairness ditingkatkan, XGBoost-Basic tetap menjadi model terbaik untuk forecasting NYC Taxi hourly `trip_count` horizon 24 jam.

Jawaban akhir revisi:

- **Experiment A Revisi:** XGBoost-Basic mengungguli Prophet-Regressor-Basic pada CV dan final test.
- **Prophet Ablation:** Prophet-Regressor-Basic memperbaiki Prophet timestamp-only, tetapi belum menutup gap terhadap XGBoost-Basic.
- **Experiment B:** XGBoost-Advanced tidak mengungguli XGBoost-Basic secara overall final test.

Dengan demikian, konfigurasi terbaik untuk benchmark revisi ini tetap **XGBoost-Basic dengan recursive forecasting horizon 24 jam**, menggunakan fitur lag utama dan calendar minimal yang dibangun tanpa data leakage.

## Artefak Rujukan

- Comparative evaluation revisi: `outputs/reports/comparative_evaluation_revisi_1.md`
- Final testing revisi: `outputs/reports/final_test_revisi_1_prophet_regressor_basic_vs_xgb_basic.md`
- Error analysis revisi: `outputs/reports/error_analysis_revisi_1.md`
- Interpretability revisi: `outputs/reports/model_interpretability_revisi_1.md`
- Final metric ranking: `outputs/experiments/comparative_evaluation_revisi_1/metrics/final_metric_ranking_revisi_1.csv`
- Time cost computing: `outputs/experiments/comparative_evaluation_revisi_1/metrics/time_cost_computing_revisi_1.csv`
- Runtime conclusion: `outputs/experiments/conclusion_revisi_1/metrics/runtime_summary_revisi_1.csv`

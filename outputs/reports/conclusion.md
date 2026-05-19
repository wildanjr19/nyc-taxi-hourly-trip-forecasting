# Conclusion dan Research Summary

## Scope

Tahap conclusion ini menyintesis seluruh hasil pipeline penelitian NYC Taxi hourly `trip_count`, mulai dari EDA, preprocessing time series, hold-out split, feature engineering, time series CV, tuning, retraining, final testing, comparative evaluation, interpretability, dan error pattern analysis.

Kesimpulan diambil dari artefak yang sudah selesai dibuat. Tidak ada tuning tambahan, retraining ulang, atau evaluasi final test baru pada tahap ini. Final test tetap diperlakukan sebagai unseen hold-out yang hanya dipakai untuk evaluasi akhir.

## Ringkasan Metodologi

- Dataset: NYC Taxi hourly, target `trip_count`.
- Horizon utama: 24 hourly steps.
- Final test: 30 hari terakhir, UTC `2026-03-02T04:00:00+00:00` sampai `2026-04-01T03:00:00+00:00`, 720 observasi.
- Model yang dibandingkan: Prophet, XGBoost-Basic, XGBoost-Advanced.
- Tuning: time series cross validation dengan expanding window.
- Evaluasi XGBoost: recursive forecasting, bukan prediksi langsung dari feature matrix validation/test.
- Guardrail leakage: actual validation/test hanya dipakai sebagai label evaluasi dan baru masuk ke history setelah blok horizon 24 jam selesai.

## Jawaban Utama Penelitian

### Model terbaik final test

Model terbaik pada final test adalah **XGBoost-Basic**. Model ini menempati rank 1 untuk seluruh metric utama: MAE, RMSE, MAPE, dan sMAPE.

| rank | model | parameter_set_id | MAE | RMSE | MAPE | sMAPE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | XGBoost-Basic | xgb_basic_292 | 519.595949 | 720.224252 | 12.069298 | 12.169547 |
| 2 | XGBoost-Advanced | xgb_advanced_210 | 577.166385 | 942.704153 | 12.358912 | 13.238262 |
| 3 | Prophet | prophet_005 | 1265.318768 | 1644.596324 | 40.685479 | 41.841144 |

XGBoost-Basic juga memiliki residual final test paling stabil: bias rata-rata `actual - predicted` sebesar -8.104758, lebih dekat ke nol dibanding XGBoost-Advanced 61.359600 dan Prophet 739.155880.

### Experiment A: Apakah XGBoost-Basic mengungguli Prophet?

Ya. XGBoost-Basic mengungguli Prophet secara konsisten pada CV dan final test.

Pada final test, XGBoost-Basic menurunkan:

- MAE sebesar 58.94% dibanding Prophet.
- RMSE sebesar 56.21% dibanding Prophet.
- MAPE sebesar 70.34% dibanding Prophet.
- sMAPE sebesar 70.91% dibanding Prophet.

Secara interpretatif, fitur sederhana seperti `lag_1`, `lag_24`, `lag_168`, `hour`, dan `day_of_week` sudah cukup kuat untuk menangkap dependensi jangka pendek, pola harian, dan pola mingguan pada data hourly NYC Taxi. Hasil SHAP juga mendukung hal ini: pada XGBoost-Basic, grup lag menyumbang 81.22% dari total mean absolute SHAP, dengan `lag_1` sebagai fitur paling dominan.

### Experiment B: Apakah XGBoost-Advanced mengungguli XGBoost-Basic?

Tidak untuk performa overall final test. XGBoost-Advanced tidak mengungguli XGBoost-Basic pada metric utama.

Pada final test, dibanding XGBoost-Basic, XGBoost-Advanced memiliki:

- MAE lebih buruk 11.08%.
- RMSE lebih buruk 30.89%.
- MAPE lebih buruk 2.40%.
- sMAPE lebih buruk 8.78%.

Advanced feature engineering tetap memberi informasi tambahan, tetapi manfaatnya tidak cukup kuat untuk mengalahkan model basic secara keseluruhan. Pada SHAP XGBoost-Advanced, grup lag masih mendominasi 87.29% kontribusi, sedangkan rolling statistics hanya 3.66%. Ini menunjukkan bahwa tambahan rolling features dipakai oleh model, tetapi kontribusinya relatif kecil dan dapat menambah kompleksitas tanpa memperbaiki generalisasi final test.

## Dampak Feature Engineering

Hasil penelitian menunjukkan bahwa "lebih banyak fitur" tidak otomatis berarti "forecast lebih baik". XGBoost-Advanced memakai 17 fitur, termasuk lag tambahan, rolling mean, rolling standard deviation, weekend flag, dan month. Namun, XGBoost-Basic dengan 5 fitur justru lebih robust pada final test.

Nuansanya penting:

- XGBoost-Advanced sempat menang pada 3 dari 5 fold CV berdasarkan MAE, tetapi rata-rata CV dan final test tetap lebih baik untuk XGBoost-Basic.
- XGBoost-Advanced lebih baik pada segmen weekend final test, dengan MAE 470.093614 dibanding XGBoost-Basic 507.896943.
- XGBoost-Advanced juga lebih baik pada low-demand P10, dengan MAE 98.833694 dibanding XGBoost-Basic 115.109321.
- Namun, XGBoost-Advanced memiliki tail error yang lebih besar, termasuk largest single absolute error 6248.800293 pada final test.

Kesimpulannya, advanced features membantu pada beberapa segmen temporal, tetapi belum memberikan peningkatan yang stabil secara keseluruhan. Untuk dataset dan horizon ini, feature set basic menjadi pilihan paling seimbang.

## Insight Temporal dan Error Pattern

Error analysis menunjukkan bahwa forecasting hourly NYC Taxi paling sulit pada kondisi demand tinggi dan jam sibuk.

Temuan utama:

- High-demand spike P90 adalah segmen error paling sulit untuk semua model.
- XGBoost-Basic adalah model terbaik pada high-demand P90 dengan MAE 986.759542, sedikit lebih baik dari XGBoost-Advanced 1001.142354 dan jauh lebih baik dari Prophet 2431.964206.
- Rush hour lokal juga sulit: XGBoost-Basic tetap terbaik dengan MAE 758.554404, XGBoost-Advanced 784.360564, dan Prophet 1564.880112.
- Night period lebih mudah untuk XGBoost karena level demand lebih rendah dan pola temporal lebih stabil.
- Prophet cenderung underpredict pada demand tinggi dan menghasilkan 34 prediksi negatif pada final test.
- Residual autocorrelation masih terlihat, terutama pada lag pendek dan sebagian lag harian, sehingga ada pola temporal yang belum sepenuhnya ditangkap.

Secara praktis, error recursive dapat terpropagasi di dalam horizon 24 jam. Ini terlihat dari variasi error berdasarkan `horizon_step`; XGBoost-Basic memiliki worst horizon step MAE 913.969377 pada step 9, sedangkan XGBoost-Advanced mencapai 1007.918815 pada step 18.

## Time Cost Computing

Perbandingan akurasi perlu dibaca bersama biaya komputasi. Prophet paling efisien dari sisi known end-to-end runtime, tetapi akurasinya paling lemah. XGBoost-Basic membutuhkan biaya tuning lebih besar, tetapi memberi hasil final test terbaik.

| model | full tuning runtime (s) | retraining train time (s) | final prediction time (s) | known end-to-end runtime (s) | final MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prophet | 250.689377 | 3.525288 | 2.644693 | 260.419075 | 1265.318768 |
| XGBoost-Basic | 1663.681676 | 12.990324 | 2.686983 | 1697.288356 | 519.595949 |
| XGBoost-Advanced | 2653.842677 | 1.151408 | 4.067967 | 2659.243170 | 577.166385 |

Trade-off utama:

- Prophet paling murah secara komputasi, tetapi error terlalu besar untuk menjadi model terbaik pada benchmark ini.
- XGBoost-Basic menawarkan akurasi terbaik dengan final prediction time yang hampir sama dengan Prophet.
- XGBoost-Advanced memiliki biaya tuning dan prediction lebih tinggi, tetapi tidak memberi peningkatan final test overall.

## Keterbatasan Penelitian

1. Final test hanya mencakup 30 hari terakhir, sehingga generalisasi lintas musim atau event khusus yang lebih panjang belum diuji.
2. Horizon utama adalah 24 jam dengan recursive forecasting; strategi direct multi-horizon atau hybrid belum dibandingkan.
3. Prophet digunakan tanpa regressor eksternal seperti cuaca, event kota, holiday khusus, atau indikator operasional taxi.
4. Feature engineering XGBoost masih berbasis lag, rolling, dan calendar features; belum mencakup event-aware features.
5. SHAP final-test recursive belum dihitung karena feature vector valid untuk final test harus berasal dari history masa lalu dan prediksi recursive, bukan precomputed actual final test.
6. XGBoost-Advanced menunjukkan manfaat pada beberapa segmen, tetapi belum ada mekanisme segment-specific modeling untuk memanfaatkan keunggulan tersebut.
7. Evaluasi dilakukan pada satu dataset agregasi hourly; hasil belum tentu langsung berlaku untuk zona pickup, borough, atau resolusi temporal lain.

## Rekomendasi Future Work

1. Tambahkan fitur eksternal yang relevan, seperti cuaca, holiday/event NYC, airport activity, atau public transit disruption.
2. Bandingkan recursive forecasting dengan direct multi-output forecasting untuk mengurangi propagasi error di horizon 24 jam.
3. Uji model segment-aware, misalnya model khusus weekend, rush hour, atau high-demand regimes.
4. Simpan feature vectors yang benar-benar dipakai saat recursive final test agar SHAP final-test dapat dihitung secara leakage-safe.
5. Lakukan backtesting pada beberapa periode hold-out agar robustness tidak hanya bergantung pada satu bulan final test.
6. Uji post-processing non-negativity, terutama untuk Prophet, lalu evaluasi sebagai eksperimen terpisah tanpa mengubah benchmark utama.
7. Pertimbangkan model tambahan seperti LightGBM, CatBoost, SARIMAX, N-BEATS, Temporal Fusion Transformer, atau ensemble XGBoost-Basic dan XGBoost-Advanced.

## Kesimpulan Akhir

Penelitian ini menunjukkan bahwa untuk forecasting hourly NYC Taxi 24 jam ke depan, pendekatan machine learning berbasis lag sederhana lebih unggul daripada Prophet dan lebih robust daripada XGBoost dengan feature set yang lebih kompleks. XGBoost-Basic menjadi model terbaik karena berhasil menyeimbangkan akurasi, stabilitas residual, dan biaya prediksi final.

Jawaban akhir atas dua research question adalah:

- **Experiment A:** XGBoost-Basic mengungguli Prophet secara jelas pada CV dan final test.
- **Experiment B:** XGBoost-Advanced tidak mengungguli XGBoost-Basic secara overall, meskipun memberi manfaat pada segmen tertentu seperti weekend dan low-demand.

Dengan demikian, konfigurasi terbaik untuk benchmark penelitian ini adalah **XGBoost-Basic dengan recursive forecasting horizon 24 jam**, menggunakan fitur lag utama dan calendar minimal yang dibuat tanpa leakage.

## Artefak Rujukan

- Final test report: `outputs/reports/final_test_report.md`
- Comparative evaluation: `outputs/reports/comparative_evaluation.md`
- Error pattern analysis: `outputs/reports/error_analysis.md`
- Model interpretability: `outputs/reports/model_interpretability.md`
- Final metrics: `outputs/final_test/metrics/final_metrics.csv`
- Final predictions: `outputs/final_test/predictions/final_predictions.csv`
- Time cost computing: `outputs/experiments/comparative_evaluation/metrics/time_cost_computing.csv`

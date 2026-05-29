# ALUR PENELITIAN
> ini adalah rincian dari alur penelitian  
> mungkin ini akan sedikit berbeda dengan @PLAN.md karena di dokumen ini adalah gambaran umum atau utama 

## 1. Data Collection
Data sudah dikumpulkan dan siap untuk digunakan. Lokasi data di folder [data](data\yellow_taxi_hourly_2025_2026_clean.csv).

## 2. Exploratory Data Analysis
### Tujuan 
Untuk memahami : 
- trend,
- seasonality,
- pola temporal,
- anomali,
- dan karakteristik statistik data.

### Analisis Yang Wajib Dilakukan
#### Analisi Struktur Data
- shape dataset
- missing values
- duplicate timestamp
- missing hours
#### Analisis Distribusi
- distribusi target
- skewness
- outlier
#### Analisis Temporal
- overall trend
- seasonality harian
- seasonality mingguan
- perilaku weekday vs weekend
#### Analisis Statistik
- rolling mean
- rolling standard deviation
- autocorrelation
- partial autocorrelation

### Output yang Diharapkan
- visualisasi EDA
- insight temporal
- observasi seasonality
- justifikasi feature engineering

Apakah tahapan EDA lebih cocok dilakukan di Jupyter Notebook? Jika iya mungkin tahapan ini dilakukan terpisah di Notebook dan di folder tersendiri.

## 3. Time Basaed Preprocessing
### Tujuan
Membersihkan data time series tanpa menyebabkan data leakage.
### Proses yang wajib dilakukan
#### Datetime Handling
- konversi ke datetime
- sorting berdasarkan waktu
#### Missing Value Handling
- cek dulu apakah ada missing
- kalau tidak, skip bagian ini
### Aturan Penting
- JANGAN menggunakan future valeus saat preprocessing
- selalu pertahankan chronological order
### Output yang diharapkan
- cleaned time series
- datetime index yang lengkap dan urut
> Karena NYC menggunakan Daylight Saving Time, jika timestamp menggunakan timezone lokal America/New_York, maka dapat terjadi missing hour saat spring forward dan duplicated/ambiguous hour saat fall back. Oleh karena itu, timestamp perlu dicek terlebih dahulu apakah menggunakan UTC atau local timezone.

## 4. Hold-Out Time-based Split
### Tujuan
Membuat final test set yang benar-benar unseen sebelum eksperimen dilakukan.
### Jenis Split
- Train + Validation
- Final Test
### Split Policy
- Final test: 1 Bulan atau 30 hari terakhir.
- Train + validation: seluruh data sebelum final test.
- Test set dibuat sekali di awal dan disimpan.
### Aturan Penting
- final test TIDAK BOLEH digunakan saat tuning
- random split dilarang
### Output yang diharapkan
- train dataset
- validation-ready dataset
- untouched final test dataset

## 5. Feature Engineering
### Tujuan
Pembuatan fitur untuk modelling
lengkapnya ada di [FEATURE_SET.md](notes/FEATURE_SET.md)

## 6. Time Series Cross Validation
### Tujuan
Melakukan validasi model secara realistis untuk time series
### CV Configuration
- Strategy: expanding window
- Number of folds: 5
- Validation horizon per fold: 168 jam (7 hari)
- Gap: optional, default 0
### Workflow Validasi
Pada setiap fold:
1. Train model
2. Predict validation horizon
3. Hitung metrics
### Aturan Penting
- Fold tidak boleh dirandom
- Validation harus tetap mempertahankan urutan waktu
### Output yang diharapkan
- Metrics tiap fold
- Rata-rata validation metrics
- Analisis stabilitas model

## 7. Hyperparameter Tuning
### Tujuan
Mencari konfigurasi model terbaik.
### Prophet Parameters
Contoh parameters:
- changepoint_prior_scale
- seasonality_prior_scale
- seasonality_mode
### Prophet Regressors
Pada alur utama, Prophet menggunakan regressor basic agar Experiment A fair
terhadap XGBoost-Basic:
- lag_1
- lag_24
- lag_168
- hour
- day_of_week

Regressor lag untuk validation dan final test dibangun secara recursive dari
history masa lalu dan prediksi sebelumnya. Actual validation/test hanya boleh
menjadi label evaluasi dan baru masuk history setelah blok horizon 24 jam
selesai dievaluasi.
### XGBoost Parameters
Contoh parameters:
- n_estimators
- learning_rate
- max_depth
- subsample
- colsample_bytree
- min_child_weight
### Aturan Tuning
- Tuning HARUS menggunakan time series cross validation
- Final test tidak boleh digunakan saat tuning
### Output yang diharapkan 
- best parameter configuration
- validation metric comparison

## 8. Experiment A
Prophet-Regressor-Basic vs XGBoost-basic
### Tujuan
Membandingkan:
- forecasting klasik
vs
- machine learning forecasting sederhana
### Karakteristik Prophet-Regressor-Basic
- automatic trend modeling
- automatic seasonality modeling
- external regressors basic yang sebanding dengan XGBoost-Basic
### Karakteristik XGBoost-basic
Menggunakan:
- simple lag features
- basic calendar features
- Output yang Diharapkan
- perbandingan metrics
- visualisasi prediction
- baseline benchmarking

## 9. Experiment B
XGBoost-basic vs XGBoost-advanced
### Tujuan
Menganalisis pengaruh advanced feature engineering.
### Karakteristik XGBoost-advanced
Tambahan fitur:
- lebih banyak lag
- rolling statistics
- richer calendar features
### Fokus Analisis
Mengukur apakah advanced temporal features meningkatkan akurasi forecasting.
### Output yang Diharapkan
- performance improvement analysis
- diskusi dampak feature engineering

## 10. Retraining Best Configurations
### Tujuan
Melatih ulang model terbaik menggunakan:
- seluruh training data
- seluruh validation data
sebelum final testing dilakukan.
### Aturan Penting
- Gunakan best parameter hasil tuning.
- Tidak boleh melakukan tuning tambahan.
### Output yang Diharapkan
- final trained models

## 11. Final Testing
### Tujuan
Mengukur kemampuan generalisasi model pada unseen data.
### Metrics yang Wajib Digunakan
- MAE
- RMSE
- MAPE
- sMAPE
### Aturan Penting
- Evaluasi test set hanya dilakukan SATU KALI.
- Tidak boleh ada penyesuaian model setelah testing.
### Output yang Diharapkan
- final benchmark results
- final prediction plots

## 12. Comparative Evaluation
### Tujuan
Membandingkan seluruh model secara menyeluruh.
### Perbandingan yang Wajib Dilakukan
#### Quantitative Comparison
- perbandingan metrics
- ranking model
#### Visual Comparison
- actual vs predicted
- residual plots
#### Behavioral Comparison
- stabilitas antar fold
- konsistensi temporal
#### Output yang Diharapkan
- tabel perbandingan
- diskusi performa model

## 13. Error Pattern Analysis
### Tujuan
Menganalisis kapan dan mengapa model gagal.
### Analisis yang Wajib Dilakukan
#### Temporal Error Analysis
- error saat rush hour
- error saat weekend
- error malam hari
#### Residual Analysis
- distribusi residual
- residual autocorrelation
#### Extreme Event Analysis
- kegagalan prediksi spike
- underprediction
- overprediction
### Output yang Diharapkan
- temporal error insights
- diskusi robustness model
- analisis limitasi model

## 14. Conclusion
### Wajib Membahas
- model terbaik
- pengaruh feature engineering
- insight forecasting
- keterbatasan penelitian
- rekomendasi future work

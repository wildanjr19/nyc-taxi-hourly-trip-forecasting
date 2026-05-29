# List Feature Untuk Feature Engineering
Markdown ini berisi list atau daftar feature-feature yang akan digenerate pada tahap Feature Engineering

# A. Lag Features
## XGBoost-Basic
Lag:
- lag_1
- lag_24
- lag_168
Calendar
- hour
- day_of_week

## XGBoost-Advanced
Lag
- lag_1
- lag_2
- lag_3
- lag_6
- lag_12
- lag_24
- lag_48
- lag_72
- lag_168

# B. Rolling Features
## Hanya Untuk XGBoost-advanced
- rolling_mean_3
- rolling_mean_24
- rolling_mean_168
- rolling_std_24
### Aturan Rolling Features
- HARUS shift terlebih dahulu sebelum rolling
- TIDAK BOLEH terjadi leakage
konsep yang benar:
shift -> rolling
BUKAN:
rolling langsung pada target saat ini

# C. Calendar Features
## XGBoost-basic
- hour
- day_of_week
## XGBoost-advanced
- hour
- day_of_week
- is_weekend
- month

# Catatan Prophet
Prophet pada alur utama menggunakan:
- internal trend modelling
- internal seasonality modelling
- regressor basic yang sebanding dengan XGBoost-Basic:
  - lag_1
  - lag_24
  - lag_168
  - hour
  - day_of_week

Lag regressor Prophet untuk validation/final test tetap dibangun secara
recursive dari history masa lalu dan prediksi sebelumnya, bukan dari actual
masa depan.

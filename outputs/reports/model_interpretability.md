# Model Interpretability

Run UTC: 2026-05-19T13:11:22.774273+00:00

## Scope

Tahap 15b ini bersifat post-hoc interpretability. Script hanya membaca model retrained tahap 13 dan feature matrix train_val. Tidak ada tuning ulang, retraining ulang, perubahan final predictions, perubahan final metrics, atau perubahan ranking model.

## Leakage Guardrail

SHAP untuk XGBoost dihitung pada feature matrix train_val, bukan pada precomputed feature matrix final test. Final-test SHAP recursive belum dihitung pada tahap ini. Prophet final-test component view, jika ada, hanya memakai timestamp sebagai post-hoc inspection dan tidak memakai label aktual.

## Prophet Components

Komponen Prophet yang valid untuk konfigurasi saat ini adalah trend, daily seasonality, dan weekly seasonality. Yearly seasonality tidak diaktifkan (`yearly_seasonality=False`), dan monthly seasonality tidak ditambahkan sebagai custom seasonality sehingga tidak ada klaim learned monthly component.

Ringkasan Prophet:

| component_rows_train_val | component_rows_final_test | components_present | components_absent |
| --- | --- | --- | --- |
| 10198 | 720 | trend, daily, weekly | yearly, monthly |

## XGBoost-Basic SHAP

SHAP XGBoost-Basic memakai 2000 sample dari 10030 feature rows train_val dengan strategi chronological_even_spacing. Kontribusi grup lag mencakup 81.22% dari total mean absolute SHAP.

| rank | feature | feature_group | mean_abs_shap | share_of_total_mean_abs_shap |
| --- | --- | --- | --- | --- |
| 1 | lag_1 | lag | 1837.259135 | 0.576958 |
| 2 | hour | calendar | 483.659477 | 0.151885 |
| 3 | lag_168 | lag | 463.261971 | 0.145479 |
| 4 | lag_24 | lag | 285.901542 | 0.089782 |
| 5 | day_of_week | calendar | 114.306360 | 0.035896 |

## XGBoost-Advanced SHAP

SHAP XGBoost-Advanced memakai 2000 sample dari 10030 feature rows train_val dengan strategi chronological_even_spacing. Kontribusi grup lag adalah 87.29%, sedangkan rolling statistics adalah 3.66% dari total mean absolute SHAP.

| rank | feature | feature_group | mean_abs_shap | share_of_total_mean_abs_shap |
| --- | --- | --- | --- | --- |
| 1 | lag_1 | lag | 1941.719682 | 0.543090 |
| 2 | lag_168 | lag | 534.319531 | 0.149447 |
| 3 | hour | calendar | 259.488890 | 0.072578 |
| 4 | lag_24 | lag | 256.658597 | 0.071786 |
| 5 | lag_2 | lag | 123.197899 | 0.034458 |
| 6 | lag_3 | lag | 84.415418 | 0.023611 |
| 7 | lag_6 | lag | 63.357618 | 0.017721 |
| 8 | lag_12 | lag | 57.874905 | 0.016187 |
| 9 | day_of_week | calendar | 42.730311 | 0.011951 |
| 10 | rolling_mean_168 | rolling_mean | 41.591859 | 0.011633 |

## Link to Comparative Evaluation

Pada comparative evaluation, model terbaik final test berdasarkan MAE adalah XGBoost-Basic (MAE 519.595949). Hasil SHAP membantu menjelaskan mengapa XGBoost-Basic dapat tetap robust walaupun fiturnya lebih sederhana: sinyal dominan berada pada lag utama, terutama lag jangka pendek dan pola harian/mingguan. Pada advanced model, tambahan rolling statistics memang dipakai, tetapi kontribusinya perlu dibaca sebagai kompleksitas tambahan yang tidak otomatis meningkatkan generalisasi final test.

## Time Cost Computing

| model_key | model_label | model_name | feature_set | model_load_time_seconds | component_compute_train_val_seconds | component_compute_final_test_seconds | shap_compute_time_seconds | plot_time_seconds | total_runtime_seconds | sample_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prophet | Prophet | prophet | prophet_internal | 1.069873 | 1.437510 | 0.150747 | 0.000000 | 0.897369 | 3.555499 |  |
| xgb_basic | XGBoost-Basic | xgboost | xgb_basic | 1.612460 | 0.000000 | 0.000000 | 1.486407 | 0.617838 | 3.716705 | 2000 |
| xgb_advanced | XGBoost-Advanced | xgboost | xgb_advanced | 0.192768 | 0.000000 | 0.000000 | 2.301004 | 1.914067 | 4.407839 | 2000 |
| total | Total | all_models | interpretability | 2.875101 | 1.437510 | 0.150747 | 3.787411 | 3.429274 | 11.680043 | 4000 |

## Output Files

- Prophet components train_val: `F:\Research\Research BDTS\outputs\experiments\prophet\components\prophet_components_train_val.csv`
- Prophet components final-test view: `F:\Research\Research BDTS\outputs\experiments\prophet\components\prophet_components_final_test.csv`
- XGBoost-Basic SHAP importance: `F:\Research\Research BDTS\outputs\experiments\xgb_basic\shap\shap_importance.csv`
- XGBoost-Advanced SHAP importance: `F:\Research\Research BDTS\outputs\experiments\xgb_advanced\shap\shap_importance.csv`
- Runtime summary: `F:\Research\Research BDTS\outputs\experiments\model_interpretability\metrics\interpretability_runtime_summary.csv`
- Report mirror: `F:\Research\Research BDTS\outputs\reports\model_interpretability.md`

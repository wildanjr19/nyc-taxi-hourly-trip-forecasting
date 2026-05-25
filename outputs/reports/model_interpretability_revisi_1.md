# Model Interpretability Revisi 1

Run UTC: 2026-05-25T12:17:40.456578+00:00

## Scope

Tahap Revisi 1.10 melakukan interpretability post-hoc untuk Prophet-Regressor-Basic dan membandingkannya secara hati-hati dengan SHAP XGBoost-Basic. Tahap ini hanya membaca artefak model, train_val, final_test, final predictions revisi, dan SHAP XGBoost yang sudah ada.

## Leakage Guardrail

- Tidak ada tuning ulang, retraining ulang, atau prediksi final baru untuk ranking.
- Komponen final Prophet-Regressor-Basic dihitung dengan lag regressor recursive.
- Actual final_test hanya dipakai sebagai label evaluasi dan validasi alignment.
- Actual final_test baru masuk ke history setelah blok horizon 24 jam selesai.
- SHAP XGBoost-Basic yang dirujuk berasal dari feature matrix train_val, bukan final test precomputed.

## Prophet-Regressor-Basic Components

Kolom komponen yang tersedia dari Prophet-Regressor-Basic meliputi `trend`, `daily`, `weekly`, `extra_regressors_additive`, dan efek masing-masing regressor basic. Karena konfigurasi memakai mode additive, efek regressor masuk ke `extra_regressors_additive`.

| component | mean | median | mean_abs | std | min | max | share_of_component_mean_abs_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trend | 5504.783585 | 5504.783585 | 5504.783585 | 0.079710 | 5504.645811 | 5504.921359 | 0.339781 |
| yhat | 5224.127906 | 5863.396734 | 5262.544145 | 2917.396969 | -908.500214 | 11209.766632 | 0.324829 |
| extra_regressors_additive | -275.991196 | 327.164866 | 2492.526506 | 2991.445166 | -6101.270774 | 5729.537506 | 0.153850 |
| additive_terms | -280.655679 | 358.506992 | 2436.773686 | 2917.392111 | -6413.285140 | 5704.957562 | 0.150409 |
| daily | -0.000000 | 74.151666 | 414.160531 | 498.476711 | -780.550125 | 900.676480 | 0.025564 |
| weekly | -4.664483 | 28.975804 | 90.187767 | 118.564794 | -275.922760 | 157.598242 | 0.005567 |
| multiplicative_terms | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Prophet Regressor Effects

Regressor dengan mean absolute component effect terbesar pada final test adalah `lag_1` (1562.328875).

| regressor | mean_effect | median_effect | mean_abs_effect | std_effect | min_effect | max_effect | feature_mean | feature_min | feature_max | share_of_total_mean_abs_effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lag_1 | -194.828884 | 201.016019 | 1562.328875 | 1864.857596 | -4075.861246 | 3611.490413 | 5209.521686 | -908.500214 | 11209.766632 | 0.572330 |
| lag_168 | -59.739711 | 57.236086 | 599.219893 | 724.211135 | -1278.922068 | 1629.493932 | 5250.026389 | 112.000000 | 12369.000000 | 0.219513 |
| lag_24 | -31.270542 | 65.095277 | 433.490365 | 531.013149 | -966.804065 | 1251.044862 | 5343.537500 | 218.000000 | 12369.000000 | 0.158801 |
| day_of_week | 9.658310 | 0.616603 | 95.965415 | 110.444824 | -162.134123 | 163.367329 | 2.833333 | 0.000000 | 6.000000 | 0.035155 |
| hour | 0.189630 | 3.230018 | 38.762710 | 44.757650 | -74.259520 | 74.262095 | 11.529167 | 0.000000 | 23.000000 | 0.014200 |

## Comparison With XGBoost-Basic SHAP

Tabel berikut bersifat directional, bukan perbandingan satu-ke-satu. Komponen Prophet adalah kontribusi model aditif berdasarkan struktur Prophet dan scaling internal regressors, sedangkan SHAP XGBoost adalah atribusi marginal berbasis TreeExplainer pada sample train_val. Karena metodologinya berbeda, ranking dan share hanya digunakan untuk membaca pola dominasi fitur, bukan menyatakan efek yang identik.

Pada Prophet-Regressor-Basic, fitur teratas adalah `lag_1`. Pada SHAP XGBoost-Basic, fitur teratas adalah `lag_1`.

| regressor | rank_prophet_effect | prophet_mean_abs_regressor_effect | prophet_share_of_total_mean_abs_regressor_effect | rank_xgb_basic_shap | xgb_basic_mean_abs_shap | xgb_basic_share_of_total_mean_abs_shap |
| --- | --- | --- | --- | --- | --- | --- |
| lag_1 | 1 | 1562.328875 | 0.572330 | 1 | 1837.259135 | 0.576958 |
| lag_168 | 2 | 599.219893 | 0.219513 | 3 | 463.261971 | 0.145479 |
| lag_24 | 3 | 433.490365 | 0.158801 | 4 | 285.901542 | 0.089782 |
| day_of_week | 4 | 95.965415 | 0.035155 | 5 | 114.306360 | 0.035896 |
| hour | 5 | 38.762710 | 0.014200 | 2 | 483.659477 | 0.151885 |

SHAP group reference XGBoost-Basic:

| rank | feature_group | mean_abs_shap | share_of_total_mean_abs_shap |
| --- | --- | --- | --- |
| 1 | lag | 2586.422648 | 0.812220 |
| 2 | calendar | 597.965838 | 0.187780 |

## Temporal Component View

Rata-rata komponen berdasarkan local hour:

| local_hour | trend | daily | weekly | extra_regressors_additive | yhat | actual | absolute_error | n_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | 5504.779267 | -771.172244 | -9.903874 | -814.552396 | 3909.150754 | 3766.533333 | 550.182705 | 30.000000 |
| 1.000000 | 5504.779650 | -705.479160 | -9.570384 | -2272.242536 | 2517.487571 | 2443.733333 | 610.451731 | 30.000000 |
| 2.000000 | 5504.782719 | -535.592290 | -11.670489 | -3615.223199 | 1342.296742 | 1535.241379 | 657.442926 | 29.000000 |
| 3.000000 | 5504.780404 | -251.235110 | -8.280906 | -4340.187211 | 905.077177 | 1244.233333 | 681.595372 | 30.000000 |
| 4.000000 | 5504.780787 | 116.759127 | -7.828628 | -4790.757545 | 822.953742 | 1105.166667 | 606.597270 | 30.000000 |
| 5.000000 | 5504.781171 | 493.453269 | -7.356352 | -4818.669604 | 1172.208483 | 1193.766667 | 701.236060 | 30.000000 |
| 6.000000 | 5504.781554 | 779.261874 | -6.867584 | -4211.891714 | 2065.284130 | 2261.733333 | 852.914836 | 30.000000 |
| 7.000000 | 5504.781937 | 889.608807 | -6.365860 | -2929.692394 | 3458.332491 | 4113.600000 | 1420.012339 | 30.000000 |
| 8.000000 | 5504.782320 | 802.020606 | -5.854718 | -1478.703851 | 4822.244356 | 5460.200000 | 1341.923588 | 30.000000 |
| 9.000000 | 5504.782704 | 575.562595 | -5.337667 | -512.516312 | 5562.491321 | 5503.800000 | 803.600650 | 30.000000 |
| 10.000000 | 5504.783087 | 323.199629 | -4.818149 | -68.259337 | 5754.905230 | 5410.133333 | 860.744413 | 30.000000 |
| 11.000000 | 5504.783470 | 150.610545 | -4.299520 | 169.000030 | 5820.094526 | 5721.033333 | 777.673568 | 30.000000 |
| 12.000000 | 5504.783853 | 100.955577 | -3.785010 | 389.486081 | 5991.440501 | 6169.700000 | 665.757105 | 30.000000 |
| 13.000000 | 5504.784237 | 141.633176 | -3.277706 | 586.960492 | 6230.100199 | 6334.800000 | 698.785170 | 30.000000 |
| 14.000000 | 5504.784620 | 197.976062 | -2.780520 | 971.112533 | 6671.092696 | 6890.766667 | 647.905369 | 30.000000 |
| 15.000000 | 5504.785003 | 205.393384 | -2.296172 | 1405.184108 | 7113.066323 | 7206.466667 | 708.643188 | 30.000000 |
| 16.000000 | 5504.785386 | 142.363969 | -1.827169 | 1822.946438 | 7468.268624 | 7444.500000 | 742.581533 | 30.000000 |
| 17.000000 | 5504.785769 | 26.977651 | -1.375785 | 2477.664992 | 8008.052627 | 8542.800000 | 886.480050 | 30.000000 |
| 18.000000 | 5504.786153 | -110.261713 | -0.944051 | 3096.919004 | 8490.499393 | 9083.900000 | 1000.225331 | 30.000000 |
| 19.000000 | 5504.786536 | -250.361823 | -0.533741 | 3219.319909 | 8473.210882 | 8336.333333 | 915.182551 | 30.000000 |
| 20.000000 | 5504.786919 | -389.587337 | -0.146363 | 2999.941916 | 8114.995136 | 7868.700000 | 958.191082 | 30.000000 |
| 21.000000 | 5504.787302 | -527.164667 | 0.216844 | 2762.995118 | 7740.834597 | 8045.066667 | 966.987838 | 30.000000 |
| 22.000000 | 5504.787686 | -652.740862 | 0.554911 | 2167.463131 | 7020.064866 | 7064.100000 | 809.626355 | 30.000000 |
| 23.000000 | 5504.783480 | -745.194332 | -7.730057 | 1005.876967 | 5757.736059 | 5211.903226 | 1220.061315 | 31.000000 |

Rata-rata komponen berdasarkan local day_of_week:

| local_day_of_week | trend | daily | weekly | extra_regressors_additive | yhat | actual | absolute_error | n_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | 5504.779063 | 0.000000 | -197.706163 | -1280.059162 | 4027.013738 | 4534.475000 | 1124.541837 | 120.000000 |
| 1.000000 | 5504.788261 | 0.000000 | 60.493375 | -540.040615 | 5025.241021 | 5214.341667 | 788.056872 | 120.000000 |
| 2.000000 | 5504.765285 | -0.000000 | 20.244624 | -90.397861 | 5434.612048 | 5459.770833 | 713.794149 | 96.000000 |
| 3.000000 | 5504.774483 | -0.000000 | 50.160106 | 295.937113 | 5850.871702 | 6161.572917 | 965.183822 | 96.000000 |
| 4.000000 | 5504.783681 | 0.000000 | 27.000489 | 342.610708 | 5874.394877 | 5723.416667 | 669.978438 | 96.000000 |
| 5.000000 | 5504.792879 | 0.000000 | 128.892376 | 175.372643 | 5809.057897 | 5679.229167 | 761.953461 | 96.000000 |
| 6.000000 | 5504.801406 | -0.000000 | -89.765236 | -518.331851 | 4896.704319 | 4815.229167 | 780.703085 | 96.000000 |

## Interpretation

Hasil revisi menunjukkan bahwa Prophet-Regressor-Basic memang memanfaatkan regressor lag dan calendar secara eksplisit melalui komponen aditif. Namun interpretasi ini tetap berbeda dari SHAP XGBoost-Basic: Prophet memecah prediksi menurut komponen model aditif, sedangkan SHAP menjelaskan kontribusi fitur terhadap output model tree relatif terhadap expected value. Keduanya konsisten dalam menunjukkan pentingnya informasi lag, terutama `lag_1`, tetapi tidak boleh disetarakan sebagai ukuran kontribusi yang sama.

## Time Cost Computing

| model_key | model_label | model_name | feature_set | model_load_time_seconds | component_compute_train_val_seconds | component_compute_final_test_seconds | shap_compute_time_seconds | plot_time_seconds | total_runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prophet_regressor_basic | Prophet-Regressor-Basic | prophet_regressor_basic | prophet_basic_regressors | 0.598960 | 0.964384 | 21.691090 | 0.000000 | 0.452838 | 23.707272 |

## Output Files

- Prophet-Regressor train_val components: `F:\Research\Research BDTS\outputs\experiments\model_interpretability_revisi_1\components\prophet_regressor_basic_components_train_val.csv`
- Prophet-Regressor final component prediction frame: `F:\Research\Research BDTS\outputs\experiments\model_interpretability_revisi_1\components\prophet_regressor_basic_final_component_predictions.csv`
- Component summary: `F:\Research\Research BDTS\outputs\experiments\model_interpretability_revisi_1\metrics\prophet_regressor_basic_component_summary_revisi_1.csv`
- Regressor effect summary: `F:\Research\Research BDTS\outputs\experiments\model_interpretability_revisi_1\metrics\prophet_regressor_basic_regressor_effect_summary_revisi_1.csv`
- Prophet vs SHAP comparison: `F:\Research\Research BDTS\outputs\experiments\model_interpretability_revisi_1\metrics\prophet_regressor_effect_vs_xgb_basic_shap_revisi_1.csv`
- Runtime summary: `F:\Research\Research BDTS\outputs\experiments\model_interpretability_revisi_1\metrics\interpretability_runtime_summary_revisi_1.csv`
- Report mirror: `F:\Research\Research BDTS\outputs\reports\model_interpretability_revisi_1.md`

# Error Pattern Analysis

Run UTC: 2026-05-19T13:33:49.801649+00:00

## Scope

Tahap ini melakukan analisis post-hoc atas prediksi final test dari tahap 14. Tidak ada tuning, retraining, ataupun prediksi baru; final test dipakai hanya sebagai label evaluasi dan konteks pola error.

## Leakage Guardrail

Semua baris final prediction yang dianalisis memiliki `used_actual_future_for_features=False`. Analisis ini membaca artefak final test yang sudah dibekukan, sehingga tidak mengubah ranking model atau konfigurasi hasil benchmark.

## Overall Error Summary

| rank_by_mae | model_label | parameter_set_id | mae | rmse | mape | smape | mean_error_actual_minus_predicted | overprediction_rate | underprediction_rate | negative_prediction_count | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | XGBoost-Basic | xgb_basic_292 | 519.595949 | 720.224252 | 12.069298 | 12.169547 | -8.104758 | 0.550000 | 0.450000 | 0 | 4017.453613 |
| 2 | XGBoost-Advanced | xgb_advanced_210 | 577.166385 | 942.704153 | 12.358912 | 13.238262 | 61.359600 | 0.536111 | 0.463889 | 0 | 6248.800293 |
| 3 | Prophet | prophet_005 | 1265.318768 | 1644.596324 | 40.685479 | 41.841144 | 739.155880 | 0.309722 | 0.690278 | 34 | 5103.161471 |

Interpretasi ringkas:

Model dengan MAE final test terbaik tetap XGBoost-Basic. XGBoost-Basic memiliki bias rata-rata paling kecil, sedangkan Prophet menunjukkan underprediction kuat dan prediksi negatif. XGBoost-Advanced lebih dekat dari Prophet, tetapi tail error dan RMSE-nya lebih besar daripada XGBoost-Basic.

## Temporal Error Analysis

Worst segment per model:

| model_label | segment | n_predictions | spike_threshold_p90_actual | low_threshold_p10_actual | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | high_demand_spike_p90 | 72 | 9080.300000 |  | 2431.964206 | 2609.987414 | 23.835664 | 27.574690 | 2431.964206 | 2238.102989 | 0.000000 | 1.000000 | 5103.161471 |
| XGBoost-Advanced | high_demand_spike_p90 | 72 | 9080.300000 |  | 1001.142354 | 1367.483804 | 9.866048 | 10.776727 | 556.728373 | 605.458008 | 0.319444 | 0.680556 | 6248.800293 |
| XGBoost-Basic | high_demand_spike_p90 | 72 | 9080.300000 |  | 986.759542 | 1247.081220 | 9.582497 | 10.296909 | 787.251621 | 696.878906 | 0.236111 | 0.763889 | 4017.453613 |

Best hour per model by MAE:

| model_label | local_hour | n_predictions | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | 16 | 30 | 526.795362 | 807.693877 | 6.605766 | 6.784817 | 115.367004 | 6.486379 | 0.500000 | 0.500000 | 3077.975518 |
| XGBoost-Advanced | 4 | 30 | 174.721908 | 255.704047 | 17.007863 | 18.667567 | 101.654899 | 83.018570 | 0.266667 | 0.733333 | 889.616211 |
| XGBoost-Basic | 4 | 30 | 170.856813 | 251.493764 | 16.173501 | 17.336046 | 62.365016 | 57.394592 | 0.366667 | 0.633333 | 842.867188 |

Worst hour per model by MAE:

| model_label | local_hour | n_predictions | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | 8 | 30 | 2708.378009 | 2959.992572 | 50.923134 | 59.371984 | 1935.485482 | 2882.106923 | 0.266667 | 0.733333 | 4658.005753 |
| XGBoost-Advanced | 17 | 30 | 1047.705754 | 1493.797600 | 11.958238 | 13.033864 | 184.583358 | -281.020508 | 0.633333 | 0.366667 | 5100.479736 |
| XGBoost-Basic | 8 | 30 | 968.650293 | 1192.567549 | 21.011889 | 20.205793 | 146.535856 | -365.819580 | 0.533333 | 0.466667 | 2793.296387 |

Best local day per model by MAE:

| model_label | local_day_of_week | n_predictions | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error | local_day_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | 4 | 96 | 871.733871 | 1200.217962 | 19.069360 | 21.384592 | 665.008217 | 424.014263 | 0.250000 | 0.750000 | 3256.629153 | Friday |
| XGBoost-Advanced | 1 | 120 | 389.767622 | 538.076683 | 9.633564 | 10.026904 | 63.372194 | 15.702988 | 0.433333 | 0.566667 | 1552.010254 | Tuesday |
| XGBoost-Basic | 1 | 120 | 395.642443 | 587.650631 | 9.349301 | 9.818060 | 133.331724 | 27.817764 | 0.458333 | 0.541667 | 1728.820801 | Tuesday |

Worst local day per model by MAE:

| model_label | local_day_of_week | n_predictions | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error | local_day_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | 3 | 96 | 1489.638565 | 1892.710545 | 38.696095 | 34.737853 | 1054.551324 | 884.491685 | 0.250000 | 0.750000 | 4418.568307 | Thursday |
| XGBoost-Advanced | 0 | 120 | 972.111544 | 1755.027989 | 21.502854 | 26.519602 | 676.417563 | 96.300385 | 0.416667 | 0.583333 | 6248.800293 | Monday |
| XGBoost-Basic | 3 | 96 | 736.302184 | 1013.357465 | 12.895752 | 13.338400 | 298.340520 | 49.818970 | 0.427083 | 0.572917 | 2940.094727 | Thursday |

Weekday vs weekend:

| model_label | weekday_weekend | n_predictions | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | weekday | 528 | 1252.277032 | 1612.296139 | 44.081749 | 43.886415 | 847.154931 | 715.214562 | 0.242424 | 0.757576 | 5103.161471 |
| Prophet | weekend | 192 | 1301.183543 | 1730.315637 | 31.345738 | 36.216648 | 442.158487 | 19.873581 | 0.494792 | 0.505208 | 4939.681048 |
| XGBoost-Basic | weekday | 528 | 523.850133 | 754.204441 | 11.800534 | 12.346207 | 95.029627 | 18.479797 | 0.481061 | 0.518939 | 4017.453613 |
| XGBoost-Basic | weekend | 192 | 507.896943 | 617.206786 | 12.808398 | 11.683732 | -291.724318 | -384.686523 | 0.739583 | 0.260417 | 1390.103760 |
| XGBoost-Advanced | weekday | 528 | 616.101937 | 1037.316012 | 13.175236 | 14.508114 | 149.510515 | 4.123230 | 0.494318 | 0.505682 | 6248.800293 |
| XGBoost-Advanced | weekend | 192 | 470.093614 | 611.166349 | 10.114022 | 9.746167 | -181.055415 | -196.829102 | 0.651042 | 0.348958 | 1728.214844 |

Rush hour dan high-demand spike menjadi zona error yang paling kritis. Untuk XGBoost, night period relatif lebih mudah karena level demand rendah dan pola jamnya lebih stabil. Weekend memberi sinyal menarik: advanced feature set lebih kompetitif di weekend, tetapi belum cukup untuk mengalahkan basic secara keseluruhan.

## Residual Analysis

| model_label | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | residual_std | residual_skew | residual_p05 | residual_p95 | absolute_error_p90 | absolute_error_p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | 739.155880 | 620.866652 | 1470.152212 | 0.432425 | -1508.868242 | 3433.344598 | 2811.002828 | 4437.352663 |
| XGBoost-Basic | -8.104758 | -69.922791 | 720.679294 | 1.166737 | -974.406592 | 1405.128223 | 1193.896313 | 2212.646965 |
| XGBoost-Advanced | 61.359600 | -27.764435 | 941.359067 | 2.503225 | -1066.324023 | 1421.661987 | 1171.452393 | 4457.805676 |

Selected residual autocorrelation:

| model_label | lag | residual_autocorrelation | is_hourly_lag_24 | is_two_day_lag_48 |
| --- | --- | --- | --- | --- |
| Prophet | 1 | 0.838328 | 0 | 0 |
| Prophet | 24 | 0.467375 | 1 | 0 |
| Prophet | 48 | -0.069379 | 0 | 1 |
| XGBoost-Basic | 1 | 0.768105 | 0 | 0 |
| XGBoost-Basic | 24 | 0.155764 | 1 | 0 |
| XGBoost-Basic | 48 | 0.164323 | 0 | 1 |
| XGBoost-Advanced | 1 | 0.862656 | 0 | 0 |
| XGBoost-Advanced | 24 | 0.185163 | 1 | 0 |
| XGBoost-Advanced | 48 | 0.055780 | 0 | 1 |

Worst recursive horizon step per model:

| model_label | horizon_step | n_predictions | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | 9 | 30 | 2580.110112 | 2858.764329 | 52.989929 | 59.667169 | 1798.256766 | 2587.962351 | 0.266667 | 0.733333 | 4658.005753 |
| XGBoost-Advanced | 18 | 30 | 1007.918815 | 1421.098097 | 11.949606 | 13.049613 | 163.563509 | -281.020508 | 0.633333 | 0.366667 | 4463.627441 |
| XGBoost-Basic | 9 | 30 | 913.969377 | 1132.857587 | 21.573095 | 20.975756 | 136.851489 | -274.081543 | 0.533333 | 0.466667 | 2793.296387 |

Worst 24-hour origin block per model:

| model_label | origin_block | block_start | block_end | n_predictions | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | 11 | 2026-03-12T04:00:00+00:00 | 2026-03-13T03:00:00+00:00 | 24 | 1976.370940 | 2263.132742 | 43.123689 | 40.893341 | 1568.784927 | 1917.608779 | 0.166667 | 0.833333 | 4371.808044 |
| XGBoost-Advanced | 1 | 2026-03-02T04:00:00+00:00 | 2026-03-03T03:00:00+00:00 | 24 | 3066.778400 | 3674.238674 | 53.784095 | 79.451087 | 3066.778400 | 4049.895142 | 0.000000 | 1.000000 | 6248.800293 |
| XGBoost-Basic | 11 | 2026-03-12T04:00:00+00:00 | 2026-03-13T03:00:00+00:00 | 24 | 1065.740318 | 1361.580853 | 17.207610 | 19.123427 | 857.081718 | 681.510010 | 0.250000 | 0.750000 | 2940.094727 |

Residual autocorrelation yang masih tampak, terutama pada lag harian, menunjukkan bahwa sebagian pola temporal belum sepenuhnya ditangkap oleh model. Ini penting untuk dibahas sebagai limitasi forecasting recursive: error pada awal horizon dapat terbawa ke step berikutnya.

## Extreme Event Analysis

| model_label | event | threshold_operator | actual_threshold | n_predictions | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | low_demand_p10 | <= | 948.900000 | 72 | 674.640886 | 851.269640 | 133.187003 | 106.749717 | 89.407445 | 89.082991 | 0.416667 | 0.583333 | 1998.065549 |
| Prophet | high_demand_p90 | >= | 9080.300000 | 72 | 2431.964206 | 2609.987414 | 23.835664 | 27.574690 | 2431.964206 | 2238.102989 | 0.000000 | 1.000000 | 5103.161471 |
| Prophet | high_demand_p95 | >= | 9955.500000 | 36 | 2995.347807 | 3123.579448 | 27.984858 | 33.027465 | 2995.347807 | 2979.305877 | 0.000000 | 1.000000 | 5103.161471 |
| XGBoost-Basic | low_demand_p10 | <= | 948.900000 | 72 | 115.109321 | 151.217418 | 21.195757 | 22.488579 | 21.977759 | 30.557678 | 0.375000 | 0.625000 | 478.494751 |
| XGBoost-Basic | high_demand_p90 | >= | 9080.300000 | 72 | 986.759542 | 1247.081220 | 9.582497 | 10.296909 | 787.251621 | 696.878906 | 0.236111 | 0.763889 | 4017.453613 |
| XGBoost-Basic | high_demand_p95 | >= | 9955.500000 | 36 | 1195.923516 | 1481.259691 | 10.988093 | 11.923057 | 1008.104831 | 968.783203 | 0.194444 | 0.805556 | 4017.453613 |
| XGBoost-Advanced | low_demand_p10 | <= | 948.900000 | 72 | 98.833694 | 135.044105 | 17.409184 | 18.021250 | 17.745648 | 17.057999 | 0.402778 | 0.597222 | 498.118408 |
| XGBoost-Advanced | high_demand_p90 | >= | 9080.300000 | 72 | 1001.142354 | 1367.483804 | 9.866048 | 10.776727 | 556.728373 | 605.458008 | 0.319444 | 0.680556 | 6248.800293 |
| XGBoost-Advanced | high_demand_p95 | >= | 9955.500000 | 36 | 1059.797282 | 1341.207961 | 9.768592 | 10.355044 | 639.922607 | 732.099121 | 0.277778 | 0.722222 | 3959.882324 |

High-demand P90 summary:

| model_label | demand_tail | actual_threshold | n_predictions | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | high_demand_p90 | 9080.300000 | 72 | 2431.964206 | 2609.987414 | 23.835664 | 27.574690 | 2431.964206 | 2238.102989 | 0.000000 | 1.000000 | 5103.161471 |
| XGBoost-Basic | high_demand_p90 | 9080.300000 | 72 | 986.759542 | 1247.081220 | 9.582497 | 10.296909 | 787.251621 | 696.878906 | 0.236111 | 0.763889 | 4017.453613 |
| XGBoost-Advanced | high_demand_p90 | 9080.300000 | 72 | 1001.142354 | 1367.483804 | 9.866048 | 10.776727 | 556.728373 | 605.458008 | 0.319444 | 0.680556 | 6248.800293 |

Low-demand P10 summary:

| model_label | demand_tail | actual_threshold | n_predictions | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | low_demand_p10 | 948.900000 | 72 | 674.640886 | 851.269640 | 133.187003 | 106.749717 | 89.407445 | 89.082991 | 0.416667 | 0.583333 | 1998.065549 |
| XGBoost-Basic | low_demand_p10 | 948.900000 | 72 | 115.109321 | 151.217418 | 21.195757 | 22.488579 | 21.977759 | 30.557678 | 0.375000 | 0.625000 | 478.494751 |
| XGBoost-Advanced | low_demand_p10 | 948.900000 | 72 | 98.833694 | 135.044105 | 17.409184 | 18.021250 | 17.745648 | 17.057999 | 0.402778 | 0.597222 | 498.118408 |

Top absolute errors per model:

| rank_within_model | timestamp | model_label | actual | predicted | residual | absolute_error | error_direction | horizon_step | origin_block | local_hour | local_day_name | period_type | actual_demand_band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-03-16T21:00:00+00:00 | Prophet | 11789.000000 | 6685.838529 | 5103.161471 | 5103.161471 | underprediction | 18 | 15 | 17 | Monday | weekday | high_demand_p90 |
| 2 | 2026-03-08T06:00:00+00:00 | Prophet | 6819.000000 | 1879.318952 | 4939.681048 | 4939.681048 | underprediction | 3 | 7 | 1 | Sunday | weekend | middle_demand |
| 3 | 2026-03-08T07:00:00+00:00 | Prophet | 5781.000000 | 855.625093 | 4925.374907 | 4925.374907 | underprediction | 4 | 7 | 3 | Sunday | weekend | middle_demand |
| 4 | 2026-03-08T05:00:00+00:00 | Prophet | 8014.000000 | 3238.183943 | 4775.816057 | 4775.816057 | underprediction | 2 | 7 | 0 | Sunday | weekend | middle_demand |
| 5 | 2026-03-16T12:00:00+00:00 | Prophet | 6803.000000 | 2144.994247 | 4658.005753 | 4658.005753 | underprediction | 9 | 15 | 8 | Monday | weekday | middle_demand |
| 1 | 2026-03-02T23:00:00+00:00 | XGBoost-Advanced | 9083.000000 | 2834.199707 | 6248.800293 | 6248.800293 | underprediction | 20 | 1 | 18 | Monday | weekday | high_demand_p90 |
| 2 | 2026-03-02T22:00:00+00:00 | XGBoost-Advanced | 8159.000000 | 3058.520264 | 5100.479736 | 5100.479736 | underprediction | 19 | 1 | 17 | Monday | weekday | middle_demand |
| 3 | 2026-03-03T00:00:00+00:00 | XGBoost-Advanced | 7809.000000 | 2777.533203 | 5031.466797 | 5031.466797 | underprediction | 21 | 1 | 19 | Monday | weekday | middle_demand |
| 4 | 2026-03-02T13:00:00+00:00 | XGBoost-Advanced | 6572.000000 | 1748.881226 | 4823.118774 | 4823.118774 | underprediction | 10 | 1 | 8 | Monday | weekday | middle_demand |
| 5 | 2026-03-03T01:00:00+00:00 | XGBoost-Advanced | 7339.000000 | 2570.887939 | 4768.112061 | 4768.112061 | underprediction | 22 | 1 | 20 | Monday | weekday | middle_demand |
| 1 | 2026-03-16T21:00:00+00:00 | XGBoost-Basic | 11789.000000 | 7771.546387 | 4017.453613 | 4017.453613 | underprediction | 18 | 15 | 17 | Monday | weekday | high_demand_p90 |
| 2 | 2026-03-12T20:00:00+00:00 | XGBoost-Basic | 10916.000000 | 7975.905273 | 2940.094727 | 2940.094727 | underprediction | 17 | 11 | 16 | Thursday | weekday | high_demand_p90 |
| 3 | 2026-03-06T00:00:00+00:00 | XGBoost-Basic | 11834.000000 | 9016.886719 | 2817.113281 | 2817.113281 | underprediction | 21 | 4 | 19 | Thursday | weekday | high_demand_p90 |
| 4 | 2026-03-12T12:00:00+00:00 | XGBoost-Basic | 8268.000000 | 5474.703613 | 2793.296387 | 2793.296387 | underprediction | 9 | 11 | 8 | Thursday | weekday | middle_demand |
| 5 | 2026-03-02T13:00:00+00:00 | XGBoost-Basic | 6572.000000 | 3789.413330 | 2782.586670 | 2782.586670 | underprediction | 10 | 1 | 8 | Monday | weekday | middle_demand |

Largest signed errors:

| signed_error_type | rank_within_model_and_type | timestamp | model_label | actual | predicted | residual | absolute_error | horizon_step | origin_block | local_hour | local_day_name | period_type | actual_demand_band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| largest_overprediction | 1 | 2026-03-17T03:00:00+00:00 | Prophet | 1782.000000 | 4662.576131 | -2880.576131 | 2880.576131 | 24 | 15 | 23 | Monday | weekday | middle_demand |
| largest_overprediction | 2 | 2026-03-07T12:00:00+00:00 | Prophet | 1547.000000 | 4217.085077 | -2670.085077 | 2670.085077 | 9 | 6 | 7 | Saturday | weekend | middle_demand |
| largest_overprediction | 3 | 2026-03-07T13:00:00+00:00 | Prophet | 2621.000000 | 5156.672782 | -2535.672782 | 2535.672782 | 10 | 6 | 8 | Saturday | weekend | middle_demand |
| largest_underprediction | 1 | 2026-03-16T21:00:00+00:00 | Prophet | 11789.000000 | 6685.838529 | 5103.161471 | 5103.161471 | 18 | 15 | 17 | Monday | weekday | high_demand_p90 |
| largest_underprediction | 2 | 2026-03-08T06:00:00+00:00 | Prophet | 6819.000000 | 1879.318952 | 4939.681048 | 4939.681048 | 3 | 7 | 1 | Sunday | weekend | middle_demand |
| largest_underprediction | 3 | 2026-03-08T07:00:00+00:00 | Prophet | 5781.000000 | 855.625093 | 4925.374907 | 4925.374907 | 4 | 7 | 3 | Sunday | weekend | middle_demand |
| largest_overprediction | 1 | 2026-03-19T22:00:00+00:00 | XGBoost-Advanced | 10680.000000 | 12856.866211 | -2176.866211 | 2176.866211 | 19 | 18 | 18 | Thursday | weekday | high_demand_p90 |
| largest_overprediction | 2 | 2026-03-07T03:00:00+00:00 | XGBoost-Advanced | 7746.000000 | 9699.543945 | -1953.543945 | 1953.543945 | 24 | 5 | 22 | Friday | weekday | middle_demand |
| largest_overprediction | 3 | 2026-03-19T21:00:00+00:00 | XGBoost-Advanced | 9905.000000 | 11850.211914 | -1945.211914 | 1945.211914 | 18 | 18 | 17 | Thursday | weekday | high_demand_p90 |
| largest_underprediction | 1 | 2026-03-02T23:00:00+00:00 | XGBoost-Advanced | 9083.000000 | 2834.199707 | 6248.800293 | 6248.800293 | 20 | 1 | 18 | Monday | weekday | high_demand_p90 |
| largest_underprediction | 2 | 2026-03-02T22:00:00+00:00 | XGBoost-Advanced | 8159.000000 | 3058.520264 | 5100.479736 | 5100.479736 | 19 | 1 | 17 | Monday | weekday | middle_demand |
| largest_underprediction | 3 | 2026-03-03T00:00:00+00:00 | XGBoost-Advanced | 7809.000000 | 2777.533203 | 5031.466797 | 5031.466797 | 21 | 1 | 19 | Monday | weekday | middle_demand |
| largest_overprediction | 1 | 2026-03-07T03:00:00+00:00 | XGBoost-Basic | 7746.000000 | 9619.408203 | -1873.408203 | 1873.408203 | 24 | 5 | 22 | Friday | weekday | middle_demand |
| largest_overprediction | 2 | 2026-03-05T03:00:00+00:00 | XGBoost-Basic | 6740.000000 | 8482.576172 | -1742.576172 | 1742.576172 | 24 | 3 | 22 | Wednesday | weekday | middle_demand |
| largest_overprediction | 3 | 2026-03-07T02:00:00+00:00 | XGBoost-Basic | 7238.000000 | 8873.736328 | -1635.736328 | 1635.736328 | 23 | 5 | 21 | Friday | weekday | middle_demand |
| largest_underprediction | 1 | 2026-03-16T21:00:00+00:00 | XGBoost-Basic | 11789.000000 | 7771.546387 | 4017.453613 | 4017.453613 | 18 | 15 | 17 | Monday | weekday | high_demand_p90 |
| largest_underprediction | 2 | 2026-03-12T20:00:00+00:00 | XGBoost-Basic | 10916.000000 | 7975.905273 | 2940.094727 | 2940.094727 | 17 | 11 | 16 | Thursday | weekday | high_demand_p90 |
| largest_underprediction | 3 | 2026-03-06T00:00:00+00:00 | XGBoost-Basic | 11834.000000 | 9016.886719 | 2817.113281 | 2817.113281 | 21 | 4 | 19 | Thursday | weekday | high_demand_p90 |

Timestamps yang sulit untuk semua model:

| timestamp | actual | local_hour | local_day_name | period_type | mean_absolute_error_across_models | max_absolute_error_across_models | n_models | absolute_error_Prophet | residual_Prophet | absolute_error_XGBoost_Basic | residual_XGBoost_Basic | absolute_error_XGBoost_Advanced | residual_XGBoost_Advanced |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-16T21:00:00+00:00 | 11789.000000 | 17 | Monday | weekday | 4360.165803 | 5103.161471 | 3 | 5103.161471 | 5103.161471 | 4017.453613 | 4017.453613 | 3959.882324 | 3959.882324 |
| 2026-03-02T13:00:00+00:00 | 6572.000000 | 8 | Monday | weekday | 3651.496012 | 4823.118774 | 3 | 3348.782593 | 3348.782593 | 2782.586670 | 2782.586670 | 4823.118774 | 4823.118774 |
| 2026-03-02T23:00:00+00:00 | 9083.000000 | 18 | Monday | weekday | 3416.485876 | 6248.800293 | 3 | 2000.928818 | 2000.928818 | 1999.728516 | 1999.728516 | 6248.800293 | 6248.800293 |
| 2026-03-12T12:00:00+00:00 | 8268.000000 | 8 | Thursday | weekday | 3246.220194 | 4371.808044 | 3 | 4371.808044 | 4371.808044 | 2793.296387 | 2793.296387 | 2573.556152 | 2573.556152 |
| 2026-03-12T20:00:00+00:00 | 10916.000000 | 16 | Thursday | weekday | 3040.962380 | 3104.816895 | 3 | 3077.975518 | 3077.975518 | 2940.094727 | 2940.094727 | 3104.816895 | 3104.816895 |
| 2026-03-02T12:00:00+00:00 | 5078.000000 | 7 | Monday | weekday | 3014.303086 | 3612.269653 | 3 | 2789.257768 | 2789.257768 | 2641.381836 | 2641.381836 | 3612.269653 | 3612.269653 |
| 2026-03-06T00:00:00+00:00 | 11834.000000 | 19 | Thursday | weekday | 2885.489058 | 3633.331432 | 3 | 3633.331432 | 3633.331432 | 2817.113281 | 2817.113281 | 2206.022461 | 2206.022461 |
| 2026-03-03T02:00:00+00:00 | 7023.000000 | 21 | Monday | weekday | 2572.114263 | 4633.344238 | 3 | 1244.405776 | 1244.405776 | 1838.592773 | 1838.592773 | 4633.344238 | 4633.344238 |
| 2026-03-03T00:00:00+00:00 | 7809.000000 | 19 | Monday | weekday | 2538.866084 | 5031.466797 | 3 | 920.474718 | 920.474718 | 1664.656738 | 1664.656738 | 5031.466797 | 5031.466797 |
| 2026-03-05T23:00:00+00:00 | 12181.000000 | 18 | Thursday | weekday | 2534.256576 | 3731.107620 | 3 | 3731.107620 | 3731.107620 | 2390.006836 | 2390.006836 | 1481.655273 | 1481.655273 |

Negative predictions:

| model_label | negative_prediction_count |
| --- | --- |
| Prophet | 34 |
| XGBoost-Basic | 0 |
| XGBoost-Advanced | 0 |

Spike demand cenderung menghasilkan underprediction, terutama pada Prophet. XGBoost-Basic tetap menjadi model paling kuat pada high-demand P90 secara rata-rata, tetapi masih memiliki error besar pada timestamp tertentu. XGBoost-Advanced menunjukkan beberapa kegagalan tail yang lebih ekstrem, mengindikasikan tambahan fitur belum otomatis membuat model lebih robust pada perubahan demand yang tajam.

## Time Cost Computing

Tahap 16 adalah analisis post-hoc, sehingga `train_time_seconds=0` dan `prediction_time_seconds=0`. Runtime yang dicatat adalah waktu analisis dan penulisan output: 3.008733 detik.

## Research Interpretation

Pola error mendukung hasil tahap 15: XGBoost-Basic bukan hanya unggul secara rata-rata, tetapi juga lebih stabil secara residual. Advanced features memberi sinyal tambahan pada beberapa segmen seperti weekend, namun manfaatnya kalah oleh error tail yang lebih besar. Prophet menangkap pola musiman umum, tetapi kurang adaptif terhadap level demand lokal pada horizon final test dan bahkan menghasilkan prediksi negatif.

Untuk tahap conclusion, poin utama yang perlu dibawa adalah bahwa forecasting hourly NYC Taxi sangat sensitif pada jam sibuk, spike demand, dan propagasi error recursive. Model terbaik pada penelitian ini adalah model yang paling konsisten menghadapi segmen tersebut, bukan hanya yang memiliki kapasitas fitur paling besar.

## Output Files

- Model summary: `F:\Research\Research BDTS\outputs\experiments\error_analysis\metrics\model_error_summary.csv`
- Temporal hour error: `F:\Research\Research BDTS\outputs\experiments\error_analysis\metrics\temporal_error_by_hour.csv`
- Temporal day error: `F:\Research\Research BDTS\outputs\experiments\error_analysis\metrics\temporal_error_by_day_of_week.csv`
- Behavioral error summary: `F:\Research\Research BDTS\outputs\experiments\error_analysis\metrics\behavioral_error_summary.csv`
- Residual distribution summary: `F:\Research\Research BDTS\outputs\experiments\error_analysis\metrics\residual_distribution_summary.csv`
- Residual autocorrelation: `F:\Research\Research BDTS\outputs\experiments\error_analysis\metrics\residual_autocorrelation.csv`
- Extreme event summary: `F:\Research\Research BDTS\outputs\experiments\error_analysis\metrics\extreme_event_summary.csv`
- Top absolute errors: `F:\Research\Research BDTS\outputs\experiments\error_analysis\metrics\top_absolute_errors.csv`
- Runtime summary: `F:\Research\Research BDTS\outputs\experiments\error_analysis\metrics\runtime_summary.csv`
- Report mirror: `F:\Research\Research BDTS\outputs\reports\error_analysis.md`
- Hourly error plot: `F:\Research\Research BDTS\outputs\experiments\error_analysis\figures\mae_by_hour.png`
- Day error plot: `F:\Research\Research BDTS\outputs\experiments\error_analysis\figures\mae_by_day_of_week.png`
- Behavioral error plot: `F:\Research\Research BDTS\outputs\experiments\error_analysis\figures\mae_by_behavior_segment.png`
- Residual distribution plot: `F:\Research\Research BDTS\outputs\experiments\error_analysis\figures\residual_distribution.png`
- Residual ACF plot: `F:\Research\Research BDTS\outputs\experiments\error_analysis\figures\residual_autocorrelation.png`
- High demand prediction plot: `F:\Research\Research BDTS\outputs\experiments\error_analysis\figures\high_demand_actual_vs_predicted.png`
- Top error plot: `F:\Research\Research BDTS\outputs\experiments\error_analysis\figures\top_error_timestamps.png`
- Origin block error plot: `F:\Research\Research BDTS\outputs\experiments\error_analysis\figures\mae_by_origin_block.png`

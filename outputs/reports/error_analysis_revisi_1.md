# Error Analysis Revisi 1

Run UTC: 2026-05-25T12:06:44.254755+00:00

## Scope

Tahap Revisi 1.9 melakukan error pattern analysis atas empat model final: Prophet timestamp-only, Prophet-Regressor-Basic, XGBoost-Basic, dan XGBoost-Advanced. Fokus revisi adalah melihat apakah regressor basic memperbaiki kelemahan Prophet lama, lalu membandingkan pola temporal Prophet-Regressor-Basic dengan XGBoost-Basic.

## Leakage Guardrail

Tahap ini hanya membaca artefak final test yang sudah dibuat. Tidak ada tuning, retraining, atau prediksi ulang final test. Prophet-Regressor-Basic memakai prediksi recursive dari Revisi 1.7; semua final predictions memiliki `used_actual_future_for_features=False`.

## Overall Error Summary

| rank_by_mae | model_label | parameter_set_id | mae | rmse | mape | smape | mean_error_actual_minus_predicted | overprediction_rate | underprediction_rate | negative_prediction_count | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | XGBoost-Basic | xgb_basic_292 | 519.595949 | 720.224252 | 12.069298 | 12.169547 | -8.104758 | 0.550000 | 0.450000 | 0 | 4017.453613 |
| 2 | XGBoost-Advanced | xgb_advanced_210 | 577.166385 | 942.704153 | 12.358912 | 13.238262 | 61.359600 | 0.536111 | 0.463889 | 0 | 6248.800293 |
| 3 | Prophet-Regressor-Basic | prophet_regressor_basic_003 | 837.648179 | 1123.936750 | 32.502400 | 31.312555 | 112.570706 | 0.493056 | 0.506944 | 33 | 5299.931706 |
| 4 | Prophet | prophet_005 | 1265.318768 | 1644.596324 | 40.685479 | 41.841144 | 739.155880 | 0.309722 | 0.690278 | 34 | 5103.161471 |

Interpretasi ringkas:

Model dengan MAE final test terbaik tetap XGBoost-Basic. Prophet-Regressor-Basic memperbaiki Prophet timestamp-only secara jelas, tetapi belum menutup gap terhadap XGBoost-Basic. Secara bias, Prophet-Regressor-Basic jauh lebih seimbang daripada Prophet lama, namun masih memiliki prediksi negatif dan tail error besar.

## Prophet Weakness Revisited

| weakness | prophet_timestamp_only | prophet_regressor_basic | xgb_basic | percent_reduction_prophet_regressor_vs_prophet | percent_reduction_xgb_basic_vs_prophet_regressor | prophet_regressor_improved_vs_prophet | xgb_basic_better_than_prophet_regressor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall_mae | 1265.318768 | 837.648179 | 519.595949 | 33.799435 | 37.969668 | True | True |
| overall_mean_error_actual_minus_predicted | 739.155880 | 112.570706 | 8.104758 | 84.770370 | 92.800295 | True | True |
| overall_underprediction_rate | 0.690278 | 0.506944 | 0.450000 | 26.559356 | 11.232877 | True | True |
| negative_prediction_count | 34.000000 | 33.000000 | 0.000000 | 2.941176 | 100.000000 | True | True |
| rush_hour_local_07_09_16_19_mae | 1564.880112 | 1015.715149 | 758.554404 | 35.093101 | 25.318195 | True | True |
| rush_hour_local_07_09_16_19_underprediction_rate | 0.685714 | 0.542857 | 0.428571 | 20.833333 | 21.052632 | True | True |
| high_demand_spike_p90_mae | 2431.964206 | 1191.884010 | 986.759542 | 50.990890 | 17.210103 | True | True |
| high_demand_spike_p90_underprediction_rate | 1.000000 | 0.847222 | 0.763889 | 15.277778 | 9.836066 | True | True |
| residual_acf_lag_1 | 0.838328 | 0.791725 | 0.768105 | 5.558947 | 2.983380 | True | True |
| residual_acf_lag_24 | 0.467375 | 0.292124 | 0.155764 | 37.496827 | 46.678944 | True | True |
| residual_acf_lag_48 | 0.069379 | 0.027649 | 0.164323 | 60.148050 | -494.315943 | True | False |

Regressor basic memperbaiki kelemahan utama Prophet lama: MAE rush hour turun 35.09%, MAE high-demand P90 turun 50.99%, dan residual ACF lag 24 turun 37.50% jika dilihat dari nilai absolut. Namun prediksi negatif belum benar-benar hilang karena Prophet-Regressor-Basic masih menghasilkan 33 nilai negatif pada final test.

## Temporal Error: Prophet-Regressor-Basic vs XGBoost-Basic

Behavior segment comparison:

| segment | mae_prophet_regressor_basic | mae_xgb_basic | mae_gap_prophet_regressor_minus_xgb_basic | percent_reduction_xgb_basic_vs_prophet_regressor | winner_by_mae |
| --- | --- | --- | --- | --- | --- |
| all | 837.648179 | 519.595949 | 318.052230 | 37.969668 | XGBoost-Basic |
| high_demand_spike_p90 | 1191.884010 | 986.759542 | 205.124468 | 17.210103 | XGBoost-Basic |
| low_demand_p10 | 683.054900 | 115.109321 | 567.945579 | 83.147867 | XGBoost-Basic |
| night_local_00_05 | 634.456642 | 227.477429 | 406.979214 | 64.146103 | XGBoost-Basic |
| rush_hour_local_07_09_16_19 | 1015.715149 | 758.554404 | 257.160745 | 25.318195 | XGBoost-Basic |
| weekday_local | 861.764508 | 523.850133 | 337.914375 | 39.211916 | XGBoost-Basic |
| weekend_local | 771.328273 | 507.896943 | 263.431330 | 34.152946 | XGBoost-Basic |

Largest local-hour gaps where XGBoost-Basic is better:

| local_hour | mae_prophet_regressor_basic | mae_xgb_basic | mae_gap_prophet_regressor_minus_xgb_basic | percent_reduction_xgb_basic_vs_prophet_regressor | winner_by_mae |
| --- | --- | --- | --- | --- | --- |
| 23 | 1220.061315 | 590.400178 | 629.661137 | 51.608975 | XGBoost-Basic |
| 7 | 1420.012339 | 827.766105 | 592.246234 | 41.707119 | XGBoost-Basic |
| 6 | 852.914836 | 330.895223 | 522.019613 | 61.204190 | XGBoost-Basic |
| 5 | 701.236060 | 219.827047 | 481.409013 | 68.651491 | XGBoost-Basic |
| 2 | 657.442926 | 178.696984 | 478.745943 | 72.819392 | XGBoost-Basic |
| 3 | 681.595372 | 204.917169 | 476.678203 | 69.935657 | XGBoost-Basic |
| 4 | 606.597270 | 170.856813 | 435.740457 | 71.833567 | XGBoost-Basic |
| 10 | 860.744413 | 450.726253 | 410.018160 | 47.635297 | XGBoost-Basic |

Largest horizon-step gaps where XGBoost-Basic is better:

| horizon_step | mae_prophet_regressor_basic | mae_xgb_basic | mae_gap_prophet_regressor_minus_xgb_basic | percent_reduction_xgb_basic_vs_prophet_regressor | winner_by_mae |
| --- | --- | --- | --- | --- | --- |
| 24 | 1345.882766 | 753.149438 | 592.733327 | 44.040487 | XGBoost-Basic |
| 8 | 1265.382705 | 710.639099 | 554.743606 | 43.839986 | XGBoost-Basic |
| 7 | 786.688784 | 274.196954 | 512.491830 | 65.145435 | XGBoost-Basic |
| 4 | 705.201500 | 204.011025 | 501.190475 | 71.070534 | XGBoost-Basic |
| 6 | 690.813031 | 193.835442 | 496.977589 | 71.940969 | XGBoost-Basic |
| 5 | 638.996168 | 166.291864 | 472.704304 | 73.976078 | XGBoost-Basic |
| 3 | 686.998026 | 215.926766 | 471.071260 | 68.569522 | XGBoost-Basic |
| 12 | 808.741901 | 395.324056 | 413.417845 | 51.118638 | XGBoost-Basic |

XGBoost-Basic memiliki MAE lebih rendah pada 24 dari 24 local hour. Gap terbesar tetap muncul pada jam-jam dengan demand tinggi atau perubahan level tajam, sehingga hasil ini mendukung interpretasi bahwa XGBoost-Basic lebih adaptif terhadap dinamika lokal hourly daripada Prophet-Regressor-Basic.

## Temporal Error Analysis

Worst segment per model:

| model_label | segment | n_predictions | spike_threshold_p90_actual | low_threshold_p10_actual | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | high_demand_spike_p90 | 72 | 9080.300000 |  | 2431.964206 | 2609.987414 | 23.835664 | 27.574690 | 2431.964206 | 2238.102989 | 0.000000 | 1.000000 | 5103.161471 |
| Prophet-Regressor-Basic | high_demand_spike_p90 | 72 | 9080.300000 |  | 1191.884010 | 1603.268211 | 11.559241 | 12.916655 | 1051.146318 | 680.736870 | 0.152778 | 0.847222 | 5299.931706 |
| XGBoost-Advanced | high_demand_spike_p90 | 72 | 9080.300000 |  | 1001.142354 | 1367.483804 | 9.866048 | 10.776727 | 556.728373 | 605.458008 | 0.319444 | 0.680556 | 6248.800293 |
| XGBoost-Basic | high_demand_spike_p90 | 72 | 9080.300000 |  | 986.759542 | 1247.081220 | 9.582497 | 10.296909 | 787.251621 | 696.878906 | 0.236111 | 0.763889 | 4017.453613 |

Best hour per model by MAE:

| model_label | local_hour | n_predictions | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | 16 | 30 | 526.795362 | 807.693877 | 6.605766 | 6.784817 | 115.367004 | 6.486379 | 0.500000 | 0.500000 | 3077.975518 |
| Prophet-Regressor-Basic | 0 | 30 | 550.182705 | 673.674805 | 23.123031 | 20.470358 | -142.617420 | -0.261564 | 0.500000 | 0.500000 | 1150.385948 |
| XGBoost-Advanced | 4 | 30 | 174.721908 | 255.704047 | 17.007863 | 18.667567 | 101.654899 | 83.018570 | 0.266667 | 0.733333 | 889.616211 |
| XGBoost-Basic | 4 | 30 | 170.856813 | 251.493764 | 16.173501 | 17.336046 | 62.365016 | 57.394592 | 0.366667 | 0.633333 | 842.867188 |

Worst hour per model by MAE:

| model_label | local_hour | n_predictions | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | 8 | 30 | 2708.378009 | 2959.992572 | 50.923134 | 59.371984 | 1935.485482 | 2882.106923 | 0.266667 | 0.733333 | 4658.005753 |
| Prophet-Regressor-Basic | 7 | 30 | 1420.012339 | 1608.916086 | 43.466191 | 40.874762 | 655.267509 | 795.329563 | 0.300000 | 0.700000 | 3329.732340 |
| XGBoost-Advanced | 17 | 30 | 1047.705754 | 1493.797600 | 11.958238 | 13.033864 | 184.583358 | -281.020508 | 0.633333 | 0.366667 | 5100.479736 |
| XGBoost-Basic | 8 | 30 | 968.650293 | 1192.567549 | 21.011889 | 20.205793 | 146.535856 | -365.819580 | 0.533333 | 0.466667 | 2793.296387 |

Best local day per model by MAE:

| model_label | local_day_of_week | n_predictions | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error | local_day_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | 4 | 96 | 871.733871 | 1200.217962 | 19.069360 | 21.384592 | 665.008217 | 424.014263 | 0.250000 | 0.750000 | 3256.629153 | Friday |
| Prophet-Regressor-Basic | 4 | 96 | 669.978438 | 835.020175 | 17.853213 | 20.974861 | -150.978211 | -143.052960 | 0.572917 | 0.427083 | 2342.577394 | Friday |
| XGBoost-Advanced | 1 | 120 | 389.767622 | 538.076683 | 9.633564 | 10.026904 | 63.372194 | 15.702988 | 0.433333 | 0.566667 | 1552.010254 | Tuesday |
| XGBoost-Basic | 1 | 120 | 395.642443 | 587.650631 | 9.349301 | 9.818060 | 133.331724 | 27.817764 | 0.458333 | 0.541667 | 1728.820801 | Tuesday |

Worst local day per model by MAE:

| model_label | local_day_of_week | n_predictions | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error | local_day_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | 3 | 96 | 1489.638565 | 1892.710545 | 38.696095 | 34.737853 | 1054.551324 | 884.491685 | 0.250000 | 0.750000 | 4418.568307 | Thursday |
| Prophet-Regressor-Basic | 0 | 120 | 1124.541837 | 1697.562627 | 42.094350 | 35.709814 | 507.461262 | -22.182377 | 0.508333 | 0.491667 | 5299.931706 | Monday |
| XGBoost-Advanced | 0 | 120 | 972.111544 | 1755.027989 | 21.502854 | 26.519602 | 676.417563 | 96.300385 | 0.416667 | 0.583333 | 6248.800293 | Monday |
| XGBoost-Basic | 3 | 96 | 736.302184 | 1013.357465 | 12.895752 | 13.338400 | 298.340520 | 49.818970 | 0.427083 | 0.572917 | 2940.094727 | Thursday |

Weekday vs weekend:

| model_label | weekday_weekend | n_predictions | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | weekday | 528 | 1252.277032 | 1612.296139 | 44.081749 | 43.886415 | 847.154931 | 715.214562 | 0.242424 | 0.757576 | 5103.161471 |
| Prophet | weekend | 192 | 1301.183543 | 1730.315637 | 31.345738 | 36.216648 | 442.158487 | 19.873581 | 0.494792 | 0.505208 | 4939.681048 |
| Prophet-Regressor-Basic | weekday | 528 | 861.764508 | 1184.270393 | 35.417877 | 35.402554 | 191.924395 | 92.289394 | 0.465909 | 0.534091 | 5299.931706 |
| Prophet-Regressor-Basic | weekend | 192 | 771.328273 | 938.222689 | 24.484838 | 20.065059 | -105.651941 | -198.695719 | 0.567708 | 0.432292 | 2984.829132 |
| XGBoost-Basic | weekday | 528 | 523.850133 | 754.204441 | 11.800534 | 12.346207 | 95.029627 | 18.479797 | 0.481061 | 0.518939 | 4017.453613 |
| XGBoost-Basic | weekend | 192 | 507.896943 | 617.206786 | 12.808398 | 11.683732 | -291.724318 | -384.686523 | 0.739583 | 0.260417 | 1390.103760 |
| XGBoost-Advanced | weekday | 528 | 616.101937 | 1037.316012 | 13.175236 | 14.508114 | 149.510515 | 4.123230 | 0.494318 | 0.505682 | 6248.800293 |
| XGBoost-Advanced | weekend | 192 | 470.093614 | 611.166349 | 10.114022 | 9.746167 | -181.055415 | -196.829102 | 0.651042 | 0.348958 | 1728.214844 |

## Residual Analysis

| model_label | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | residual_std | residual_skew | residual_p05 | residual_p95 | absolute_error_p90 | absolute_error_p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | 739.155880 | 620.866652 | 1470.152212 | 0.432425 | -1508.868242 | 3433.344598 | 2811.002828 | 4437.352663 |
| Prophet-Regressor-Basic | 112.570706 | 11.505491 | 1119.062537 | 1.051782 | -1437.976685 | 2235.110875 | 1644.159013 | 3946.805479 |
| XGBoost-Basic | -8.104758 | -69.922791 | 720.679294 | 1.166737 | -974.406592 | 1405.128223 | 1193.896313 | 2212.646965 |
| XGBoost-Advanced | 61.359600 | -27.764435 | 941.359067 | 2.503225 | -1066.324023 | 1421.661987 | 1171.452393 | 4457.805676 |

Selected residual autocorrelation:

| model_label | lag | residual_autocorrelation | is_hourly_lag_24 | is_two_day_lag_48 |
| --- | --- | --- | --- | --- |
| Prophet | 1 | 0.838328 | 0 | 0 |
| Prophet | 24 | 0.467375 | 1 | 0 |
| Prophet | 48 | -0.069379 | 0 | 1 |
| Prophet-Regressor-Basic | 1 | 0.791725 | 0 | 0 |
| Prophet-Regressor-Basic | 24 | 0.292124 | 1 | 0 |
| Prophet-Regressor-Basic | 48 | 0.027649 | 0 | 1 |
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
| Prophet-Regressor-Basic | 24 | 30 | 1345.882766 | 1558.335736 | 30.071796 | 26.717063 | -342.382598 | -714.885112 | 0.666667 | 0.333333 | 2984.829132 |
| XGBoost-Advanced | 18 | 30 | 1007.918815 | 1421.098097 | 11.949606 | 13.049613 | 163.563509 | -281.020508 | 0.633333 | 0.366667 | 4463.627441 |
| XGBoost-Basic | 9 | 30 | 913.969377 | 1132.857587 | 21.573095 | 20.975756 | 136.851489 | -274.081543 | 0.533333 | 0.466667 | 2793.296387 |

Worst 24-hour origin block per model:

| model_label | origin_block | block_start | block_end | n_predictions | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | 11 | 2026-03-12T04:00:00+00:00 | 2026-03-13T03:00:00+00:00 | 24 | 1976.370940 | 2263.132742 | 43.123689 | 40.893341 | 1568.784927 | 1917.608779 | 0.166667 | 0.833333 | 4371.808044 |
| Prophet-Regressor-Basic | 1 | 2026-03-02T04:00:00+00:00 | 2026-03-03T03:00:00+00:00 | 24 | 2763.399605 | 3191.266656 | 73.869467 | 76.122235 | 2421.476030 | 3326.312883 | 0.291667 | 0.708333 | 5299.931706 |
| XGBoost-Advanced | 1 | 2026-03-02T04:00:00+00:00 | 2026-03-03T03:00:00+00:00 | 24 | 3066.778400 | 3674.238674 | 53.784095 | 79.451087 | 3066.778400 | 4049.895142 | 0.000000 | 1.000000 | 6248.800293 |
| XGBoost-Basic | 11 | 2026-03-12T04:00:00+00:00 | 2026-03-13T03:00:00+00:00 | 24 | 1065.740318 | 1361.580853 | 17.207610 | 19.123427 | 857.081718 | 681.510010 | 0.250000 | 0.750000 | 2940.094727 |

Residual autocorrelation masih tampak pada semua model. Prophet-Regressor-Basic menurunkan pola lag harian dibanding Prophet timestamp-only, tetapi residualnya masih lebih besar dan lebih berstruktur daripada XGBoost-Basic pada beberapa segmen.

## Extreme Event Analysis

| model_label | event | threshold_operator | actual_threshold | n_predictions | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | low_demand_p10 | <= | 948.900000 | 72 | 674.640886 | 851.269640 | 133.187003 | 106.749717 | 89.407445 | 89.082991 | 0.416667 | 0.583333 | 1998.065549 |
| Prophet | high_demand_p90 | >= | 9080.300000 | 72 | 2431.964206 | 2609.987414 | 23.835664 | 27.574690 | 2431.964206 | 2238.102989 | 0.000000 | 1.000000 | 5103.161471 |
| Prophet | high_demand_p95 | >= | 9955.500000 | 36 | 2995.347807 | 3123.579448 | 27.984858 | 33.027465 | 2995.347807 | 2979.305877 | 0.000000 | 1.000000 | 5103.161471 |
| Prophet-Regressor-Basic | low_demand_p10 | <= | 948.900000 | 72 | 683.054900 | 776.346138 | 145.883757 | 130.790636 | 282.252649 | 425.650022 | 0.291667 | 0.708333 | 1482.646312 |
| Prophet-Regressor-Basic | high_demand_p90 | >= | 9080.300000 | 72 | 1191.884010 | 1603.268211 | 11.559241 | 12.916655 | 1051.146318 | 680.736870 | 0.152778 | 0.847222 | 5299.931706 |
| Prophet-Regressor-Basic | high_demand_p95 | >= | 9955.500000 | 36 | 1561.814030 | 1905.927002 | 14.404625 | 16.082553 | 1444.654106 | 1153.267871 | 0.138889 | 0.861111 | 4533.303839 |
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
| Prophet-Regressor-Basic | high_demand_p90 | 9080.300000 | 72 | 1191.884010 | 1603.268211 | 11.559241 | 12.916655 | 1051.146318 | 680.736870 | 0.152778 | 0.847222 | 5299.931706 |
| XGBoost-Basic | high_demand_p90 | 9080.300000 | 72 | 986.759542 | 1247.081220 | 9.582497 | 10.296909 | 787.251621 | 696.878906 | 0.236111 | 0.763889 | 4017.453613 |
| XGBoost-Advanced | high_demand_p90 | 9080.300000 | 72 | 1001.142354 | 1367.483804 | 9.866048 | 10.776727 | 556.728373 | 605.458008 | 0.319444 | 0.680556 | 6248.800293 |

Low-demand P10 summary:

| model_label | demand_tail | actual_threshold | n_predictions | mae | rmse | mape | smape | mean_error_actual_minus_predicted | median_error_actual_minus_predicted | overprediction_rate | underprediction_rate | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prophet | low_demand_p10 | 948.900000 | 72 | 674.640886 | 851.269640 | 133.187003 | 106.749717 | 89.407445 | 89.082991 | 0.416667 | 0.583333 | 1998.065549 |
| Prophet-Regressor-Basic | low_demand_p10 | 948.900000 | 72 | 683.054900 | 776.346138 | 145.883757 | 130.790636 | 282.252649 | 425.650022 | 0.291667 | 0.708333 | 1482.646312 |
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
| 1 | 2026-03-02T23:00:00+00:00 | Prophet-Regressor-Basic | 9083.000000 | 3783.068294 | 5299.931706 | 5299.931706 | underprediction | 20 | 1 | 18 | Monday | weekday | high_demand_p90 |
| 2 | 2026-03-02T13:00:00+00:00 | Prophet-Regressor-Basic | 6572.000000 | 1853.322765 | 4718.677235 | 4718.677235 | underprediction | 10 | 1 | 8 | Monday | weekday | middle_demand |
| 3 | 2026-03-16T21:00:00+00:00 | Prophet-Regressor-Basic | 11789.000000 | 7255.696161 | 4533.303839 | 4533.303839 | underprediction | 18 | 15 | 17 | Monday | weekday | high_demand_p90 |
| 4 | 2026-03-02T22:00:00+00:00 | Prophet-Regressor-Basic | 8159.000000 | 3695.191197 | 4463.808803 | 4463.808803 | underprediction | 19 | 1 | 17 | Monday | weekday | middle_demand |
| 5 | 2026-03-03T02:00:00+00:00 | Prophet-Regressor-Basic | 7023.000000 | 2641.810263 | 4381.189737 | 4381.189737 | underprediction | 23 | 1 | 21 | Monday | weekday | middle_demand |
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
| largest_overprediction | 1 | 2026-03-13T03:00:00+00:00 | Prophet-Regressor-Basic | 6042.000000 | 8503.437210 | -2461.437210 | 2461.437210 | 24 | 11 | 23 | Thursday | weekday | middle_demand |
| largest_overprediction | 2 | 2026-03-07T01:00:00+00:00 | Prophet-Regressor-Basic | 7622.000000 | 9964.577394 | -2342.577394 | 2342.577394 | 22 | 5 | 20 | Friday | weekday | middle_demand |
| largest_overprediction | 3 | 2026-03-12T03:00:00+00:00 | Prophet-Regressor-Basic | 4406.000000 | 6697.150531 | -2291.150531 | 2291.150531 | 24 | 10 | 23 | Wednesday | weekday | middle_demand |
| largest_underprediction | 1 | 2026-03-02T23:00:00+00:00 | Prophet-Regressor-Basic | 9083.000000 | 3783.068294 | 5299.931706 | 5299.931706 | 20 | 1 | 18 | Monday | weekday | high_demand_p90 |
| largest_underprediction | 2 | 2026-03-02T13:00:00+00:00 | Prophet-Regressor-Basic | 6572.000000 | 1853.322765 | 4718.677235 | 4718.677235 | 10 | 1 | 8 | Monday | weekday | middle_demand |
| largest_underprediction | 3 | 2026-03-16T21:00:00+00:00 | Prophet-Regressor-Basic | 11789.000000 | 7255.696161 | 4533.303839 | 4533.303839 | 18 | 15 | 17 | Monday | weekday | high_demand_p90 |
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

| timestamp | actual | local_hour | local_day_name | period_type | mean_absolute_error_across_models | max_absolute_error_across_models | n_models | absolute_error_Prophet | residual_Prophet | absolute_error_Prophet_Regressor_Basic | residual_Prophet_Regressor_Basic | absolute_error_XGBoost_Advanced | residual_XGBoost_Advanced | absolute_error_XGBoost_Basic | residual_XGBoost_Basic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-16T21:00:00+00:00 | 11789.000000 | 17 | Monday | weekday | 4403.450312 | 5103.161471 | 4 | 5103.161471 | 5103.161471 | 4533.303839 | 4533.303839 | 3959.882324 | 3959.882324 | 4017.453613 | 4017.453613 |
| 2026-03-02T13:00:00+00:00 | 6572.000000 | 8 | Monday | weekday | 3918.291318 | 4823.118774 | 4 | 3348.782593 | 3348.782593 | 4718.677235 | 4718.677235 | 4823.118774 | 4823.118774 | 2782.586670 | 2782.586670 |
| 2026-03-02T23:00:00+00:00 | 9083.000000 | 18 | Monday | weekday | 3887.347333 | 6248.800293 | 4 | 2000.928818 | 2000.928818 | 5299.931706 | 5299.931706 | 6248.800293 | 6248.800293 | 1999.728516 | 1999.728516 |
| 2026-03-12T12:00:00+00:00 | 8268.000000 | 8 | Thursday | weekday | 3201.877414 | 4371.808044 | 4 | 4371.808044 | 4371.808044 | 3068.849074 | 3068.849074 | 2573.556152 | 2573.556152 | 2793.296387 | 2793.296387 |
| 2026-03-02T12:00:00+00:00 | 5078.000000 | 7 | Monday | weekday | 3093.160399 | 3612.269653 | 4 | 2789.257768 | 2789.257768 | 3329.732340 | 3329.732340 | 3612.269653 | 3612.269653 | 2641.381836 | 2641.381836 |
| 2026-03-03T02:00:00+00:00 | 7023.000000 | 21 | Monday | weekday | 3024.383131 | 4633.344238 | 4 | 1244.405776 | 1244.405776 | 4381.189737 | 4381.189737 | 4633.344238 | 4633.344238 | 1838.592773 | 1838.592773 |
| 2026-03-12T20:00:00+00:00 | 10916.000000 | 16 | Thursday | weekday | 2998.707483 | 3104.816895 | 4 | 3077.975518 | 3077.975518 | 2871.942793 | 2871.942793 | 3104.816895 | 3104.816895 | 2940.094727 | 2940.094727 |
| 2026-03-02T22:00:00+00:00 | 8159.000000 | 17 | Monday | weekday | 2987.399648 | 5100.479736 | 4 | 1096.011225 | 1096.011225 | 4463.808803 | 4463.808803 | 5100.479736 | 5100.479736 | 1289.298828 | 1289.298828 |
| 2026-03-03T00:00:00+00:00 | 7809.000000 | 19 | Monday | weekday | 2970.982567 | 5031.466797 | 4 | 920.474718 | 920.474718 | 4267.332014 | 4267.332014 | 5031.466797 | 5031.466797 | 1664.656738 | 1664.656738 |
| 2026-03-03T01:00:00+00:00 | 7339.000000 | 20 | Monday | weekday | 2915.453622 | 4768.112061 | 4 | 873.267874 | 873.267874 | 4197.463362 | 4197.463362 | 4768.112061 | 4768.112061 | 1822.971191 | 1822.971191 |

Negative predictions:

| model_label | negative_prediction_count |
| --- | --- |
| Prophet | 34 |
| Prophet-Regressor-Basic | 33 |
| XGBoost-Basic | 0 |
| XGBoost-Advanced | 0 |

## Time Cost Computing

Revisi 1.9 adalah analisis post-hoc, sehingga `train_time_seconds=0` dan `prediction_time_seconds=0`. Runtime yang dicatat adalah waktu analisis dan penulisan output: 4.365908 detik.

## Research Interpretation

Hasil error analysis revisi memperjelas bahwa penambahan regressor basic membuat Prophet lebih fair dan lebih kuat daripada baseline timestamp-only. Perbaikannya terutama terlihat pada high-demand P90, rush hour, bias underprediction, dan residual autocorrelation lag harian.

Meski begitu, XGBoost-Basic tetap lebih robust pada final test. Keunggulannya bukan hanya rata-rata MAE, tetapi juga konsistensi pada local hour, horizon recursive, dan segmen high-demand. Ini mendukung kesimpulan Revisi 1 bahwa machine learning forecasting sederhana masih mengungguli Prophet-Regressor-Basic pada benchmark NYC Taxi hourly ini.

## Output Files

- Model summary: `F:\Research\Research BDTS\outputs\experiments\error_analysis_revisi_1\metrics\model_error_summary_revisi_1.csv`
- Prophet weakness comparison: `F:\Research\Research BDTS\outputs\experiments\error_analysis_revisi_1\metrics\prophet_weakness_comparison_revisi_1.csv`
- PR vs XGB hour comparison: `F:\Research\Research BDTS\outputs\experiments\error_analysis_revisi_1\metrics\prophet_regressor_vs_xgb_basic_by_hour_revisi_1.csv`
- PR vs XGB segment comparison: `F:\Research\Research BDTS\outputs\experiments\error_analysis_revisi_1\metrics\prophet_regressor_vs_xgb_basic_by_segment_revisi_1.csv`
- Residual autocorrelation: `F:\Research\Research BDTS\outputs\experiments\error_analysis_revisi_1\metrics\residual_autocorrelation_revisi_1.csv`
- Extreme event summary: `F:\Research\Research BDTS\outputs\experiments\error_analysis_revisi_1\metrics\extreme_event_summary_revisi_1.csv`
- Top absolute errors: `F:\Research\Research BDTS\outputs\experiments\error_analysis_revisi_1\metrics\top_absolute_errors_revisi_1.csv`
- Runtime summary: `F:\Research\Research BDTS\outputs\experiments\error_analysis_revisi_1\metrics\runtime_summary_revisi_1.csv`
- Report mirror: `F:\Research\Research BDTS\outputs\reports\error_analysis_revisi_1.md`
- Hourly error plot: `F:\Research\Research BDTS\outputs\experiments\error_analysis_revisi_1\figures\mae_by_hour_revisi_1.png`
- Day error plot: `F:\Research\Research BDTS\outputs\experiments\error_analysis_revisi_1\figures\mae_by_day_of_week_revisi_1.png`
- Behavioral error plot: `F:\Research\Research BDTS\outputs\experiments\error_analysis_revisi_1\figures\mae_by_behavior_segment_revisi_1.png`
- Residual distribution plot: `F:\Research\Research BDTS\outputs\experiments\error_analysis_revisi_1\figures\residual_distribution_revisi_1.png`
- Residual ACF plot: `F:\Research\Research BDTS\outputs\experiments\error_analysis_revisi_1\figures\residual_autocorrelation_revisi_1.png`
- High demand prediction plot: `F:\Research\Research BDTS\outputs\experiments\error_analysis_revisi_1\figures\high_demand_actual_vs_predicted_revisi_1.png`
- PR vs XGB hour plot: `F:\Research\Research BDTS\outputs\experiments\error_analysis_revisi_1\figures\prophet_regressor_vs_xgb_basic_mae_by_hour_revisi_1.png`

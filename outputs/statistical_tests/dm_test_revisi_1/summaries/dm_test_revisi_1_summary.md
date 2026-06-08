# Diebold-Mariano Test Revisi 1

Run UTC: 2026-06-08T03:51:26.428400+00:00

## Scope

Tahap ini menambahkan uji signifikansi statistik pada hasil final test. Script hanya membaca artefak prediksi final test yang sudah ada; tidak ada tuning, retraining, atau prediksi ulang.

## Methodology

- Primary analysis: DM test pada rata-rata squared error per origin_block 24 jam.
- Robustness check: DM test pada rata-rata absolute error per origin_block 24 jam.
- Secondary analysis: DM test per horizon step dengan koreksi Holm dan Benjamini-Hochberg.
- Sensitivity analysis: timestamp-level DM test dengan HAC lag 23.
- Loss differential d_t = loss_baseline - loss_challenger; nilai positif berarti challenger lebih baik.

## Primary DM Test

| comparison | baseline_model | challenger_model | winner_by_mean_loss | dm_statistic | p_value | significant_alpha_0_05 | dm_test_conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Experiment A revisi: XGBoost-Basic vs Prophet-Regressor-Basic | Prophet-Regressor-Basic | XGBoost-Basic | XGBoost-Basic | 2.673709 | 0.012191 | True | Signifikan pada alpha 0.05; XGBoost-Basic memiliki rata-rata loss lebih rendah pada level blok 24 jam. |
| Experiment B: XGBoost-Advanced vs XGBoost-Basic | XGBoost-Basic | XGBoost-Advanced | XGBoost-Basic | -0.915965 | 0.367240 | False | Tidak signifikan pada alpha 0.05; belum cukup bukti statistik bahwa loss kedua model berbeda pada level blok 24 jam. |

## Block-Level Robustness

Squared error:

| comparison | loss_type | baseline_model | challenger_model | mean_loss_difference_baseline_minus_challenger | percent_loss_reduction_challenger_vs_baseline | winner_by_mean_loss | dm_statistic | p_value | significant_alpha_0_05 | n_obs | hac_lags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Experiment A revisi: XGBoost-Basic vs Prophet-Regressor-Basic | squared_error | Prophet-Regressor-Basic | XGBoost-Basic | 744510.844994 | 58.936899 | XGBoost-Basic | 2.673709 | 0.012191 | True | 30 | 0 |
| Experiment B: XGBoost-Advanced vs XGBoost-Basic | squared_error | XGBoost-Basic | XGBoost-Advanced | -369968.148358 | -71.322877 | XGBoost-Basic | -0.915965 | 0.367240 | False | 30 | 0 |

Absolute error:

| comparison | loss_type | baseline_model | challenger_model | mean_loss_difference_baseline_minus_challenger | percent_loss_reduction_challenger_vs_baseline | winner_by_mean_loss | dm_statistic | p_value | significant_alpha_0_05 | n_obs | hac_lags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Experiment A revisi: XGBoost-Basic vs Prophet-Regressor-Basic | absolute_error | Prophet-Regressor-Basic | XGBoost-Basic | 318.052230 | 37.969668 | XGBoost-Basic | 5.381688 | 0.000009 | True | 30 | 0 |
| Experiment B: XGBoost-Advanced vs XGBoost-Basic | absolute_error | XGBoost-Basic | XGBoost-Advanced | -57.570436 | -11.079847 | XGBoost-Basic | -0.763669 | 0.451233 | False | 30 | 0 |

## Horizon-Level Summary

| comparison | n_horizons | n_significant_holm | n_significant_bh | mean_effect |
| --- | --- | --- | --- | --- |
| Experiment A revisi: XGBoost-Basic vs Prophet-Regressor-Basic | 24 | 9 | 10 | 744510.844994 |
| Experiment B: XGBoost-Advanced vs XGBoost-Basic | 24 | 0 | 0 | -369968.148358 |

## Timestamp-Level Sensitivity

| comparison | loss_type | baseline_model | challenger_model | mean_loss_difference_baseline_minus_challenger | percent_loss_reduction_challenger_vs_baseline | winner_by_mean_loss | dm_statistic | p_value | significant_alpha_0_05 | n_obs | hac_lags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Experiment A revisi: XGBoost-Basic vs Prophet-Regressor-Basic | squared_error | Prophet-Regressor-Basic | XGBoost-Basic | 744510.844994 | 58.936899 | XGBoost-Basic | 2.957230 | 0.003206 | True | 720 | 23 |
| Experiment B: XGBoost-Advanced vs XGBoost-Basic | squared_error | XGBoost-Basic | XGBoost-Advanced | -369968.148358 | -71.322877 | XGBoost-Basic | -1.004811 | 0.315326 | False | 720 | 23 |

## Time Cost Computing

| experiment_name | stage | statistical_test | n_prediction_rows | n_models | n_comparisons | n_loss_types | n_block_level_tests | n_by_horizon_tests | n_timestamp_level_tests | prediction_time_seconds | train_time_seconds | total_runtime_seconds | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dm_test_revisi_1 | statistical_significance_testing_final_test | Diebold-Mariano | 2880 | 4 | 4 | 2 | 8 | 192 | 8 | 0.000000 | 0.000000 | 0.909599 | success |

## Interpretation Notes

Hasil utama yang sebaiknya dibahas di laporan adalah block-level squared error karena unit evaluasinya konsisten dengan task recursive 24-hour forecasting. Hasil per horizon dipakai untuk menganalisis di horizon mana perbedaan model paling kuat.

## Output Files

- DM summary: `F:\Research\Research BDTS\outputs\statistical_tests\dm_test_revisi_1\metrics\dm_test_summary_revisi_1.csv`
- Block-level DM: `F:\Research\Research BDTS\outputs\statistical_tests\dm_test_revisi_1\metrics\dm_test_block_level_revisi_1.csv`
- Horizon-level DM: `F:\Research\Research BDTS\outputs\statistical_tests\dm_test_revisi_1\metrics\dm_test_by_horizon_revisi_1.csv`
- Timestamp-level sensitivity: `F:\Research\Research BDTS\outputs\statistical_tests\dm_test_revisi_1\metrics\dm_test_timestamp_level_revisi_1.csv`
- Time cost computing: `F:\Research\Research BDTS\outputs\statistical_tests\dm_test_revisi_1\metrics\time_cost_computing_revisi_1.csv`
- Metadata: `F:\Research\Research BDTS\outputs\statistical_tests\dm_test_revisi_1\experiment_metadata.json`
- Report mirror: `F:\Research\Research BDTS\outputs\reports\dm_test_revisi_1.md`
- Horizon p-value plot: `F:\Research\Research BDTS\outputs\statistical_tests\dm_test_revisi_1\figures\dm_pvalues_by_horizon_revisi_1.png`
- Horizon effect plot: `F:\Research\Research BDTS\outputs\statistical_tests\dm_test_revisi_1\figures\dm_effect_size_by_horizon_revisi_1.png`
- Block effect plot: `F:\Research\Research BDTS\outputs\statistical_tests\dm_test_revisi_1\figures\dm_block_level_effect_revisi_1.png`

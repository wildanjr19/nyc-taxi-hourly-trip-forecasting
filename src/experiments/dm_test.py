"""
Script: Diebold-Mariano Test Revisi 1.

Scope:
- Membaca artefak prediksi final test yang sudah dibuat.
- Tidak melakukan tuning, retraining, atau prediksi ulang.
- Menjalankan Diebold-Mariano test pada level blok 24 jam, per horizon,
  dan timestamp-level sensitivity analysis.

Contoh:
    python -m src.experiments.dm_test
    python -m src.experiments.dm_test --skip-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import (
    FINAL_TEST_DIR,
    FORECAST_HORIZON,
    OUTPUT_DIR as PROJECT_OUTPUT_DIR,
    REPORTS_DIR,
    TIMESTAMP_COL,
    ensure_dirs,
)
from src.experiments.compare_models import (
    coerce_bool_series,
    dataframe_to_markdown,
    load_final_predictions,
    stringify_paths,
    write_text,
)
from src.statistical_tests import (
    adjust_pvalues,
    diebold_mariano_test,
    loss_from_predictions,
)
from src.tracking import (
    append_experiment_run,
    elapsed_seconds,
    log_runtime,
    make_runtime_record,
    save_experiment_metadata,
    start_timer,
    utc_now_iso,
)


EXPERIMENT_NAME = "dm_test_revisi_1"
OUTPUT_DIR = PROJECT_OUTPUT_DIR / "statistical_tests" / EXPERIMENT_NAME
REPORT_PATH = REPORTS_DIR / "dm_test_revisi_1.md"

OLD_FINAL_PREDICTIONS_PATH = FINAL_TEST_DIR / "predictions" / "final_predictions.csv"
REVISI_1_PROPHET_REGRESSOR_PREDICTIONS_PATH = (
    FINAL_TEST_DIR
    / "revisi_1"
    / "predictions"
    / "prophet_regressor_basic_final_predictions.csv"
)

PROPHET = "Prophet"
PROPHET_REGRESSOR = "Prophet-Regressor-Basic"
XGB_BASIC = "XGBoost-Basic"
XGB_ADVANCED = "XGBoost-Advanced"

EXPECTED_MODELS = [PROPHET, PROPHET_REGRESSOR, XGB_BASIC, XGB_ADVANCED]
LOSS_TYPES = ["squared_error", "absolute_error"]
ALPHA = 0.05

COMPARISONS = [
    {
        "comparison_id": "experiment_a_revisi",
        "comparison": "Experiment A revisi: XGBoost-Basic vs Prophet-Regressor-Basic",
        "baseline_model": PROPHET_REGRESSOR,
        "challenger_model": XGB_BASIC,
        "research_role": "primary",
    },
    {
        "comparison_id": "experiment_b",
        "comparison": "Experiment B: XGBoost-Advanced vs XGBoost-Basic",
        "baseline_model": XGB_BASIC,
        "challenger_model": XGB_ADVANCED,
        "research_role": "primary",
    },
    {
        "comparison_id": "experiment_a_historical",
        "comparison": "Experiment A historical: XGBoost-Basic vs Prophet",
        "baseline_model": PROPHET,
        "challenger_model": XGB_BASIC,
        "research_role": "reference",
    },
    {
        "comparison_id": "reference_advanced_vs_prophet_regressor",
        "comparison": "Reference: XGBoost-Advanced vs Prophet-Regressor-Basic",
        "baseline_model": PROPHET_REGRESSOR,
        "challenger_model": XGB_ADVANCED,
        "research_role": "reference",
    },
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Diebold-Mariano tests on final test predictions."
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip matplotlib figure generation.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    metadata = run_dm_test_revisi_1(skip_plots=args.skip_plots)
    print("Diebold-Mariano test revisi 1 selesai.")
    print(f"Output dir: {metadata['outputs']['base']}")
    print(f"Report: {metadata['outputs']['report']}")


def run_dm_test_revisi_1(*, skip_plots: bool = False) -> dict[str, Any]:
    timer = start_timer()
    started_at = utc_now_iso()
    outputs = dm_output_paths()
    status = "success"
    error_message = ""

    try:
        ensure_dirs()
        ensure_dm_dirs(outputs)

        predictions = load_revisi_1_prediction_artifacts()
        validate_prediction_artifacts(predictions)
        loss_wide_by_type = {
            loss_type: build_loss_wide(predictions, loss_type=loss_type)
            for loss_type in LOSS_TYPES
        }

        block_level = build_block_level_dm(loss_wide_by_type)
        by_horizon = build_by_horizon_dm(loss_wide_by_type)
        timestamp_level = build_timestamp_level_dm(loss_wide_by_type)
        summary = build_dm_summary(block_level)
        runtime_summary = build_runtime_summary(
            timer=timer,
            predictions=predictions,
            block_level=block_level,
            by_horizon=by_horizon,
            timestamp_level=timestamp_level,
        )

        block_level.to_csv(outputs["block_level"], index=False)
        by_horizon.to_csv(outputs["by_horizon"], index=False)
        timestamp_level.to_csv(outputs["timestamp_level"], index=False)
        summary.to_csv(outputs["summary_table"], index=False)
        runtime_summary.to_csv(outputs["time_cost_computing"], index=False)

        if not skip_plots:
            save_horizon_pvalue_plot(by_horizon, outputs["horizon_pvalue_plot"])
            save_horizon_effect_plot(by_horizon, outputs["horizon_effect_plot"])
            save_block_effect_plot(block_level, outputs["block_effect_plot"])

        total_runtime_seconds = elapsed_seconds(timer)
        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "stage": "statistical_significance_testing_final_test",
            "status": status,
            "error_message": error_message,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "total_runtime_seconds": total_runtime_seconds,
            "alpha": ALPHA,
            "forecast_horizon_hours": int(FORECAST_HORIZON),
            "models_compared": EXPECTED_MODELS,
            "loss_types": LOSS_TYPES,
            "comparisons": COMPARISONS,
            "input_artifacts": {
                "old_final_predictions": str(OLD_FINAL_PREDICTIONS_PATH),
                "revisi_1_prophet_regressor_predictions": str(
                    REVISI_1_PROPHET_REGRESSOR_PREDICTIONS_PATH
                ),
            },
            "methodology": [
                "Primary DM test memakai rata-rata loss tiap origin_block 24 jam.",
                "Secondary DM test dilakukan per horizon step 1 sampai 24.",
                "Timestamp-level DM memakai HAC lag 23 sebagai sensitivity analysis.",
                "Loss differential d_t = loss_baseline - loss_challenger.",
                "Mean loss differential positif berarti challenger lebih baik.",
                "Semua input berasal dari final test recursive forecasting artifacts.",
            ],
            "leakage_guardrail": [
                "Tidak ada tuning, retraining, atau prediksi baru.",
                "Actual final test hanya dipakai sebagai label evaluasi.",
                "Final predictions wajib memiliki used_actual_future_for_features=False.",
            ],
            "outputs": stringify_paths(outputs),
        }

        report_text = render_report(
            metadata=metadata,
            summary=summary,
            block_level=block_level,
            by_horizon=by_horizon,
            timestamp_level=timestamp_level,
            runtime_summary=runtime_summary,
            skip_plots=skip_plots,
            outputs=outputs,
        )
        write_text(outputs["summary_report"], report_text)
        write_text(outputs["report"], report_text)
        save_experiment_metadata(metadata, outputs["metadata"])
        log_runtime(
            make_runtime_record(
                experiment_name=EXPERIMENT_NAME,
                model_name="diebold_mariano",
                feature_set="final_test_revisi_1",
                n_prediction_rows=int(predictions.shape[0]),
                prediction_time_seconds=0.0,
                total_runtime_seconds=total_runtime_seconds,
                status=status,
            )
        )
        append_experiment_run(metadata)
        return metadata
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        total_runtime_seconds = elapsed_seconds(timer)
        failure_metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "stage": "statistical_significance_testing_final_test",
            "status": status,
            "error_message": error_message,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "total_runtime_seconds": total_runtime_seconds,
            "outputs": stringify_paths(outputs),
        }
        save_experiment_metadata(failure_metadata, outputs["metadata"])
        log_runtime(
            make_runtime_record(
                experiment_name=EXPERIMENT_NAME,
                model_name="diebold_mariano",
                feature_set="final_test_revisi_1",
                total_runtime_seconds=total_runtime_seconds,
                status=status,
                error_message=error_message,
            )
        )
        append_experiment_run(failure_metadata)
        raise


def dm_output_paths() -> dict[str, Path]:
    metrics_dir = OUTPUT_DIR / "metrics"
    figures_dir = OUTPUT_DIR / "figures"
    summaries_dir = OUTPUT_DIR / "summaries"
    return {
        "base": OUTPUT_DIR,
        "metrics": metrics_dir,
        "figures": figures_dir,
        "summaries": summaries_dir,
        "block_level": metrics_dir / "dm_test_block_level_revisi_1.csv",
        "by_horizon": metrics_dir / "dm_test_by_horizon_revisi_1.csv",
        "timestamp_level": metrics_dir / "dm_test_timestamp_level_revisi_1.csv",
        "summary_table": metrics_dir / "dm_test_summary_revisi_1.csv",
        "time_cost_computing": metrics_dir / "time_cost_computing_revisi_1.csv",
        "horizon_pvalue_plot": figures_dir / "dm_pvalues_by_horizon_revisi_1.png",
        "horizon_effect_plot": figures_dir / "dm_effect_size_by_horizon_revisi_1.png",
        "block_effect_plot": figures_dir / "dm_block_level_effect_revisi_1.png",
        "metadata": OUTPUT_DIR / "experiment_metadata.json",
        "summary_report": summaries_dir / "dm_test_revisi_1_summary.md",
        "report": REPORT_PATH,
    }


def ensure_dm_dirs(paths: Mapping[str, Path]) -> None:
    for key in ["base", "metrics", "figures", "summaries"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_revisi_1_prediction_artifacts() -> pd.DataFrame:
    old_predictions = load_final_predictions(OLD_FINAL_PREDICTIONS_PATH)
    prophet_regressor = load_final_predictions(
        REVISI_1_PROPHET_REGRESSOR_PREDICTIONS_PATH
    )
    combined = pd.concat(
        [old_predictions, prophet_regressor],
        ignore_index=True,
        sort=False,
    )
    return combined.sort_values(
        ["model_label", TIMESTAMP_COL],
        kind="mergesort",
    ).reset_index(drop=True)


def validate_prediction_artifacts(predictions: pd.DataFrame) -> None:
    labels = set(predictions["model_label"].astype(str).unique().tolist())
    expected = set(EXPECTED_MODELS)
    if labels != expected:
        raise ValueError(
            "Label model prediksi final tidak sesuai. "
            f"Expected={sorted(expected)}, found={sorted(labels)}"
        )
    predictions["used_actual_future_for_features"] = coerce_bool_series(
        predictions["used_actual_future_for_features"],
        column_name="used_actual_future_for_features",
    )
    if predictions["used_actual_future_for_features"].any():
        raise ValueError("Final predictions menandai leakage flag aktif.")

    reference: Optional[pd.DataFrame] = None
    expected_rows: Optional[int] = None
    for model_label, group in predictions.groupby("model_label", sort=False):
        group = group.sort_values(TIMESTAMP_COL, kind="mergesort")
        if group[TIMESTAMP_COL].duplicated().any():
            raise ValueError(f"Duplicate timestamp pada model {model_label}.")
        if group["horizon_step"].min() != 1:
            raise ValueError(f"horizon_step minimum {model_label} bukan 1.")
        if group["horizon_step"].max() != FORECAST_HORIZON:
            raise ValueError(
                f"horizon_step maksimum {model_label} bukan {FORECAST_HORIZON}."
            )
        if expected_rows is None:
            expected_rows = int(group.shape[0])
        elif group.shape[0] != expected_rows:
            raise ValueError("Jumlah prediksi tidak sejajar antar model.")
        candidate = group[[TIMESTAMP_COL, "actual", "origin_block", "horizon_step"]]
        candidate = candidate.reset_index(drop=True)
        if reference is None:
            reference = candidate
        elif not candidate.equals(reference):
            raise ValueError(
                f"Timestamp, actual, origin_block, atau horizon {model_label} "
                "tidak sejajar dengan model referensi."
            )
    if expected_rows != 720:
        raise ValueError(f"Jumlah prediksi per model bukan 720 rows: {expected_rows}")


def build_loss_wide(predictions: pd.DataFrame, *, loss_type: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model_label, group in predictions.groupby("model_label", sort=False):
        frame = group[
            [TIMESTAMP_COL, "actual", "origin_block", "horizon_step"]
        ].copy()
        frame["model_label"] = model_label
        frame["loss"] = loss_from_predictions(group, loss_type=loss_type)
        rows.append(frame)
    long = pd.concat(rows, ignore_index=True)
    wide = long.pivot_table(
        index=[TIMESTAMP_COL, "actual", "origin_block", "horizon_step"],
        columns="model_label",
        values="loss",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    missing_models = [model for model in EXPECTED_MODELS if model not in wide.columns]
    if missing_models:
        raise ValueError(f"Kolom loss model hilang: {missing_models}")
    if wide[EXPECTED_MODELS].isna().any().any():
        raise ValueError("Loss wide mengandung NaN.")
    return wide.sort_values(
        ["origin_block", "horizon_step"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_block_level_dm(loss_wide_by_type: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for loss_type, wide in loss_wide_by_type.items():
        block_losses = (
            wide.groupby("origin_block", sort=True)[EXPECTED_MODELS]
            .mean()
            .reset_index()
        )
        for comparison in COMPARISONS:
            rows.append(
                run_dm_row(
                    frame=block_losses,
                    comparison=comparison,
                    loss_type=loss_type,
                    analysis_level="block_24h_mean",
                    forecast_horizon_for_dm=1,
                    hac_lags=0,
                )
            )
    return pd.DataFrame(rows)


def build_by_horizon_dm(loss_wide_by_type: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for loss_type, wide in loss_wide_by_type.items():
        for horizon_step, horizon_df in wide.groupby("horizon_step", sort=True):
            horizon_df = horizon_df.sort_values("origin_block", kind="mergesort")
            for comparison in COMPARISONS:
                row = run_dm_row(
                    frame=horizon_df,
                    comparison=comparison,
                    loss_type=loss_type,
                    analysis_level="by_horizon",
                    forecast_horizon_for_dm=1,
                    hac_lags=0,
                )
                row["horizon_step"] = int(horizon_step)
                rows.append(row)
    result = pd.DataFrame(rows)
    result["p_value_holm_by_comparison_loss"] = np.nan
    result["p_value_bh_by_comparison_loss"] = np.nan
    for _, index in result.groupby(["comparison_id", "loss_type"]).groups.items():
        idx = list(index)
        p_values = result.loc[idx, "p_value"]
        result.loc[idx, "p_value_holm_by_comparison_loss"] = adjust_pvalues(
            p_values,
            method="holm",
        )
        result.loc[idx, "p_value_bh_by_comparison_loss"] = adjust_pvalues(
            p_values,
            method="benjamini_hochberg",
        )
    result["significant_holm_alpha_0_05"] = (
        result["p_value_holm_by_comparison_loss"] < ALPHA
    )
    result["significant_bh_alpha_0_05"] = (
        result["p_value_bh_by_comparison_loss"] < ALPHA
    )
    return result


def build_timestamp_level_dm(
    loss_wide_by_type: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for loss_type, wide in loss_wide_by_type.items():
        timestamp_losses = wide.sort_values(TIMESTAMP_COL, kind="mergesort")
        for comparison in COMPARISONS:
            rows.append(
                run_dm_row(
                    frame=timestamp_losses,
                    comparison=comparison,
                    loss_type=loss_type,
                    analysis_level="timestamp_level_hac23_sensitivity",
                    forecast_horizon_for_dm=FORECAST_HORIZON,
                    hac_lags=FORECAST_HORIZON - 1,
                )
            )
    return pd.DataFrame(rows)


def run_dm_row(
    *,
    frame: pd.DataFrame,
    comparison: Mapping[str, Any],
    loss_type: str,
    analysis_level: str,
    forecast_horizon_for_dm: int,
    hac_lags: int,
) -> dict[str, Any]:
    baseline = str(comparison["baseline_model"])
    challenger = str(comparison["challenger_model"])
    result = diebold_mariano_test(
        frame[baseline],
        frame[challenger],
        forecast_horizon=forecast_horizon_for_dm,
        hac_lags=hac_lags,
        alternative="two_sided",
        variance_estimator="newey_west",
        apply_hln_correction=True,
    )
    mean_baseline = float(pd.to_numeric(frame[baseline], errors="raise").mean())
    mean_challenger = float(pd.to_numeric(frame[challenger], errors="raise").mean())
    percent_reduction = (
        (mean_baseline - mean_challenger) / mean_baseline * 100.0
        if mean_baseline != 0
        else np.nan
    )
    winner = challenger if mean_challenger < mean_baseline else baseline
    direction = (
        "challenger_lower_loss"
        if result.mean_loss_difference > 0
        else "baseline_lower_loss"
        if result.mean_loss_difference < 0
        else "equal_mean_loss"
    )
    return {
        "comparison_id": comparison["comparison_id"],
        "comparison": comparison["comparison"],
        "research_role": comparison["research_role"],
        "analysis_level": analysis_level,
        "loss_type": loss_type,
        "baseline_model": baseline,
        "challenger_model": challenger,
        "mean_loss_baseline": mean_baseline,
        "mean_loss_challenger": mean_challenger,
        "mean_loss_difference_baseline_minus_challenger": (
            result.mean_loss_difference
        ),
        "percent_loss_reduction_challenger_vs_baseline": percent_reduction,
        "winner_by_mean_loss": winner,
        "direction": direction,
        "dm_statistic": result.statistic,
        "p_value": result.p_value,
        "significant_alpha_0_05": result.p_value < ALPHA,
        "n_obs": result.n_obs,
        "forecast_horizon_for_dm": result.forecast_horizon,
        "hac_lags": result.hac_lags,
        "variance_estimator": result.variance_estimator,
        "pvalue_distribution": result.pvalue_distribution,
        "hln_correction_applied": result.hln_correction_applied,
        "alternative": result.alternative,
        "long_run_variance": result.long_run_variance,
        "variance_loss_difference": result.variance_loss_difference,
        "alpha": ALPHA,
    }


def build_dm_summary(block_level: pd.DataFrame) -> pd.DataFrame:
    summary = block_level[
        (block_level["research_role"] == "primary")
        & (block_level["loss_type"] == "squared_error")
    ].copy()
    summary = summary.sort_values("comparison_id", kind="mergesort").reset_index(
        drop=True
    )
    summary["dm_test_conclusion"] = summary.apply(format_dm_conclusion, axis=1)
    return summary


def format_dm_conclusion(row: pd.Series) -> str:
    if not bool(row["significant_alpha_0_05"]):
        return (
            "Tidak signifikan pada alpha 0.05; belum cukup bukti statistik "
            "bahwa loss kedua model berbeda pada level blok 24 jam."
        )
    winner = row["winner_by_mean_loss"]
    return (
        f"Signifikan pada alpha 0.05; {winner} memiliki rata-rata loss lebih rendah "
        "pada level blok 24 jam."
    )


def build_runtime_summary(
    *,
    timer: float,
    predictions: pd.DataFrame,
    block_level: pd.DataFrame,
    by_horizon: pd.DataFrame,
    timestamp_level: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_name": EXPERIMENT_NAME,
                "stage": "statistical_significance_testing_final_test",
                "statistical_test": "Diebold-Mariano",
                "n_prediction_rows": int(predictions.shape[0]),
                "n_models": int(predictions["model_label"].nunique()),
                "n_comparisons": len(COMPARISONS),
                "n_loss_types": len(LOSS_TYPES),
                "n_block_level_tests": int(block_level.shape[0]),
                "n_by_horizon_tests": int(by_horizon.shape[0]),
                "n_timestamp_level_tests": int(timestamp_level.shape[0]),
                "prediction_time_seconds": 0.0,
                "train_time_seconds": 0.0,
                "total_runtime_seconds": elapsed_seconds(timer),
                "status": "success",
            }
        ]
    )


def save_horizon_pvalue_plot(by_horizon: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_df = by_horizon[by_horizon["loss_type"] == "squared_error"].copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    for comparison, group in plot_df.groupby("comparison", sort=False):
        group = group.sort_values("horizon_step", kind="mergesort")
        ax.plot(
            group["horizon_step"],
            group["p_value_holm_by_comparison_loss"],
            marker="o",
            linewidth=1.3,
            label=comparison,
        )
    ax.axhline(ALPHA, color="black", linewidth=1.0, linestyle="--")
    ax.set_title("DM Test Holm-Adjusted P-Values by Horizon")
    ax.set_xlabel("Horizon step")
    ax.set_ylabel("Holm-adjusted p-value")
    ax.set_xticks(list(range(1, FORECAST_HORIZON + 1)))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_horizon_effect_plot(by_horizon: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_df = by_horizon[by_horizon["loss_type"] == "squared_error"].copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    for comparison, group in plot_df.groupby("comparison", sort=False):
        group = group.sort_values("horizon_step", kind="mergesort")
        ax.plot(
            group["horizon_step"],
            group["mean_loss_difference_baseline_minus_challenger"],
            marker="o",
            linewidth=1.3,
            label=comparison,
        )
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_title("DM Test Mean Loss Differential by Horizon")
    ax.set_xlabel("Horizon step")
    ax.set_ylabel("Baseline loss minus challenger loss")
    ax.set_xticks(list(range(1, FORECAST_HORIZON + 1)))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_block_effect_plot(block_level: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_df = block_level[block_level["loss_type"] == "squared_error"].copy()
    labels = plot_df["comparison_id"].astype(str)
    values = plot_df["mean_loss_difference_baseline_minus_challenger"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_title("DM Test Block-Level Mean Loss Differential")
    ax.set_xlabel("Comparison")
    ax.set_ylabel("Baseline loss minus challenger loss")
    ax.grid(True, axis="y", alpha=0.25)
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_report(
    *,
    metadata: Mapping[str, Any],
    summary: pd.DataFrame,
    block_level: pd.DataFrame,
    by_horizon: pd.DataFrame,
    timestamp_level: pd.DataFrame,
    runtime_summary: pd.DataFrame,
    skip_plots: bool,
    outputs: Mapping[str, Path],
) -> str:
    primary_abs = block_level[
        (block_level["research_role"] == "primary")
        & (block_level["loss_type"] == "absolute_error")
    ]
    timestamp_primary = timestamp_level[
        (timestamp_level["research_role"] == "primary")
        & (timestamp_level["loss_type"] == "squared_error")
    ]
    horizon_primary = by_horizon[
        (by_horizon["research_role"] == "primary")
        & (by_horizon["loss_type"] == "squared_error")
    ].copy()
    horizon_significance = (
        horizon_primary.groupby("comparison", sort=False)
        .agg(
            n_horizons=("horizon_step", "count"),
            n_significant_holm=("significant_holm_alpha_0_05", "sum"),
            n_significant_bh=("significant_bh_alpha_0_05", "sum"),
            mean_effect=("mean_loss_difference_baseline_minus_challenger", "mean"),
        )
        .reset_index()
    )

    block_columns = [
        "comparison",
        "loss_type",
        "baseline_model",
        "challenger_model",
        "mean_loss_difference_baseline_minus_challenger",
        "percent_loss_reduction_challenger_vs_baseline",
        "winner_by_mean_loss",
        "dm_statistic",
        "p_value",
        "significant_alpha_0_05",
        "n_obs",
        "hac_lags",
    ]
    summary_columns = [
        "comparison",
        "baseline_model",
        "challenger_model",
        "winner_by_mean_loss",
        "dm_statistic",
        "p_value",
        "significant_alpha_0_05",
        "dm_test_conclusion",
    ]

    lines = [
        "# Diebold-Mariano Test Revisi 1",
        "",
        f"Run UTC: {metadata['started_at_utc']}",
        "",
        "## Scope",
        "",
        (
            "Tahap ini menambahkan uji signifikansi statistik pada hasil final "
            "test. Script hanya membaca artefak prediksi final test yang sudah "
            "ada; tidak ada tuning, retraining, atau prediksi ulang."
        ),
        "",
        "## Methodology",
        "",
        "- Primary analysis: DM test pada rata-rata squared error per origin_block 24 jam.",
        "- Robustness check: DM test pada rata-rata absolute error per origin_block 24 jam.",
        "- Secondary analysis: DM test per horizon step dengan koreksi Holm dan Benjamini-Hochberg.",
        "- Sensitivity analysis: timestamp-level DM test dengan HAC lag 23.",
        "- Loss differential d_t = loss_baseline - loss_challenger; nilai positif berarti challenger lebih baik.",
        "",
        "## Primary DM Test",
        "",
        dataframe_to_markdown(summary[summary_columns], float_digits=6),
        "",
        "## Block-Level Robustness",
        "",
        "Squared error:",
        "",
        dataframe_to_markdown(
            block_level[
                (block_level["research_role"] == "primary")
                & (block_level["loss_type"] == "squared_error")
            ][block_columns],
            float_digits=6,
        ),
        "",
        "Absolute error:",
        "",
        dataframe_to_markdown(primary_abs[block_columns], float_digits=6),
        "",
        "## Horizon-Level Summary",
        "",
        dataframe_to_markdown(horizon_significance, float_digits=6),
        "",
        "## Timestamp-Level Sensitivity",
        "",
        dataframe_to_markdown(timestamp_primary[block_columns], float_digits=6),
        "",
        "## Time Cost Computing",
        "",
        dataframe_to_markdown(runtime_summary, float_digits=6),
        "",
        "## Interpretation Notes",
        "",
        (
            "Hasil utama yang sebaiknya dibahas di laporan adalah block-level "
            "squared error karena unit evaluasinya konsisten dengan task "
            "recursive 24-hour forecasting. Hasil per horizon dipakai untuk "
            "menganalisis di horizon mana perbedaan model paling kuat."
        ),
        "",
        "## Output Files",
        "",
        f"- DM summary: `{outputs['summary_table']}`",
        f"- Block-level DM: `{outputs['block_level']}`",
        f"- Horizon-level DM: `{outputs['by_horizon']}`",
        f"- Timestamp-level sensitivity: `{outputs['timestamp_level']}`",
        f"- Time cost computing: `{outputs['time_cost_computing']}`",
        f"- Metadata: `{outputs['metadata']}`",
        f"- Report mirror: `{outputs['report']}`",
    ]
    if not skip_plots:
        lines.extend(
            [
                f"- Horizon p-value plot: `{outputs['horizon_pvalue_plot']}`",
                f"- Horizon effect plot: `{outputs['horizon_effect_plot']}`",
                f"- Block effect plot: `{outputs['block_effect_plot']}`",
            ]
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

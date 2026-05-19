"""
Script tahap 12: Experiment B - XGBoost-Basic vs XGBoost-Advanced.

Experiment B menjawab pertanyaan:
Apakah advanced feature engineering meningkatkan performa XGBoost?

Scope tahap ini:
- Membandingkan hasil CV dari best configuration hasil tuning.
- Tidak membaca final_test dan tidak melakukan pemilihan parameter baru.
- Kedua model diaudit dari output recursive forecasting CV.

Contoh:
    python -m src.experiments.experiment_b
    python -m src.experiments.experiment_b --skip-plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import (
    EXPERIMENTS_DIR,
    FEATURE_COLUMNS_PATH,
    FORECAST_HORIZON,
    PRIMARY_METRIC,
    REPORTS_DIR,
    TARGET_COL,
    TIMESTAMP_COL,
    XGB_ADVANCED_OUTPUT_DIR,
    XGB_BASIC_OUTPUT_DIR,
    ensure_dirs,
)
from src.experiments.experiment_a import (
    build_combined_best_predictions,
    build_fold_comparison,
    build_horizon_error_summary,
    build_improvement_summary,
    build_local_hour_error_summary,
    build_metric_comparison,
    build_residual_summary,
    build_runtime_comparison,
    dataframe_to_markdown,
    load_best_tuning_artifacts,
    save_actual_vs_predicted_plot,
    save_horizon_error_plot,
    save_residual_distribution_plot,
    select_winner,
    stringify_paths,
    validate_best_artifacts,
    write_text,
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


EXPERIMENT_NAME = "experiment_b_xgb_basic_vs_xgb_advanced"
OUTPUT_DIR = EXPERIMENTS_DIR / "experiment_b"
REPORT_PATH = REPORTS_DIR / "experiment_b_xgb_basic_vs_xgb_advanced.md"
BASELINE_LABEL = "XGBoost-Basic"
CHALLENGER_LABEL = "XGBoost-Advanced"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment B comparison: XGBoost-Basic vs XGBoost-Advanced."
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip matplotlib figure generation.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    metadata = run_experiment_b(skip_plots=args.skip_plots)
    print("Experiment B selesai.")
    print(f"Winner by {PRIMARY_METRIC}: {metadata['winner_by_primary_metric']}")
    print(f"Advanced improves primary metric: {metadata['advanced_improves_primary_metric']}")
    print(f"Output dir: {metadata['outputs']['base']}")
    print(f"Report: {metadata['outputs']['report']}")


def run_experiment_b(*, skip_plots: bool = False) -> dict[str, Any]:
    timer = start_timer()
    started_at = utc_now_iso()
    status = "success"
    error_message = ""
    outputs = experiment_b_output_paths()

    try:
        ensure_dirs()
        ensure_experiment_b_dirs(outputs)

        xgb_basic = load_best_tuning_artifacts(
            model_label=BASELINE_LABEL,
            output_dir=XGB_BASIC_OUTPUT_DIR,
        )
        xgb_advanced = load_best_tuning_artifacts(
            model_label=CHALLENGER_LABEL,
            output_dir=XGB_ADVANCED_OUTPUT_DIR,
        )
        validate_best_artifacts([xgb_basic, xgb_advanced])

        feature_comparison = build_feature_set_comparison()
        comparison = build_metric_comparison([xgb_basic, xgb_advanced])
        improvements = build_improvement_summary(
            comparison,
            baseline_label=BASELINE_LABEL,
            challenger_label=CHALLENGER_LABEL,
        )
        fold_comparison = build_fold_comparison(
            xgb_basic["best_metrics"],
            xgb_advanced["best_metrics"],
            baseline_label=BASELINE_LABEL,
            challenger_label=CHALLENGER_LABEL,
        )
        runtime_comparison = build_runtime_comparison([xgb_basic, xgb_advanced])
        predictions = build_combined_best_predictions([xgb_basic, xgb_advanced])
        residual_summary = build_residual_summary(predictions)
        horizon_error = build_horizon_error_summary(predictions)
        local_hour_error = build_local_hour_error_summary(predictions)
        behavioral_error = build_behavioral_error_summary(predictions)

        feature_comparison.to_csv(outputs["feature_set_comparison"], index=False)
        comparison.to_csv(outputs["metric_comparison"], index=False)
        improvements.to_csv(outputs["improvement_summary"], index=False)
        fold_comparison.to_csv(outputs["fold_comparison"], index=False)
        runtime_comparison.to_csv(outputs["runtime_comparison"], index=False)
        predictions.to_csv(outputs["best_predictions"], index=False)
        residual_summary.to_csv(outputs["residual_summary"], index=False)
        horizon_error.to_csv(outputs["horizon_error_summary"], index=False)
        local_hour_error.to_csv(outputs["local_hour_error_summary"], index=False)
        behavioral_error.to_csv(outputs["behavioral_error_summary"], index=False)

        if not skip_plots:
            save_actual_vs_predicted_plot(
                predictions,
                outputs["actual_vs_predicted_plot"],
                title="Experiment B - Best CV Predictions",
            )
            save_residual_distribution_plot(
                predictions,
                outputs["residual_distribution_plot"],
                title="Experiment B - Residual Distribution",
            )
            save_horizon_error_plot(
                horizon_error,
                outputs["horizon_error_plot"],
                title="Experiment B - MAE by Recursive Horizon Step",
            )
            save_behavioral_error_plot(
                behavioral_error,
                outputs["behavioral_error_plot"],
            )

        advanced_improves = advanced_improves_primary_metric(improvements)
        report_text = render_experiment_b_report(
            feature_comparison=feature_comparison,
            comparison=comparison,
            improvements=improvements,
            fold_comparison=fold_comparison,
            runtime_comparison=runtime_comparison,
            residual_summary=residual_summary,
            horizon_error=horizon_error,
            behavioral_error=behavioral_error,
            artifacts=[xgb_basic, xgb_advanced],
            started_at=started_at,
            skip_plots=skip_plots,
            outputs=outputs,
        )
        write_text(outputs["summary"], report_text)
        write_text(outputs["report"], report_text)

        total_runtime_seconds = elapsed_seconds(timer)
        winner = select_winner(comparison, metric=PRIMARY_METRIC)
        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "status": status,
            "error_message": error_message,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "total_runtime_seconds": total_runtime_seconds,
            "scope": "best tuned CV artifacts only; final_test is not used",
            "primary_metric": PRIMARY_METRIC,
            "winner_by_primary_metric": winner,
            "advanced_improves_primary_metric": advanced_improves,
            "models_compared": [BASELINE_LABEL, CHALLENGER_LABEL],
            "best_parameter_sets": {
                xgb_basic["model_label"]: xgb_basic["parameter_set_id"],
                xgb_advanced["model_label"]: xgb_advanced["parameter_set_id"],
            },
            "outputs": stringify_paths(outputs),
        }
        save_experiment_metadata(metadata, outputs["metadata"])
        log_runtime(
            make_runtime_record(
                experiment_name=EXPERIMENT_NAME,
                model_name="comparison",
                feature_set="xgb_basic_vs_xgb_advanced",
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
                model_name="comparison",
                feature_set="xgb_basic_vs_xgb_advanced",
                total_runtime_seconds=total_runtime_seconds,
                status=status,
                error_message=error_message,
            )
        )
        append_experiment_run(failure_metadata)
        raise


def experiment_b_output_paths() -> dict[str, Path]:
    base = OUTPUT_DIR
    metrics_dir = base / "metrics"
    predictions_dir = base / "predictions"
    figures_dir = base / "figures"
    summaries_dir = base / "summaries"
    return {
        "base": base,
        "metrics": metrics_dir,
        "predictions": predictions_dir,
        "figures": figures_dir,
        "summaries": summaries_dir,
        "feature_set_comparison": metrics_dir / "feature_set_comparison.csv",
        "metric_comparison": metrics_dir / "metric_comparison.csv",
        "improvement_summary": metrics_dir / "advanced_vs_basic_improvement.csv",
        "fold_comparison": metrics_dir / "fold_comparison.csv",
        "runtime_comparison": metrics_dir / "runtime_comparison.csv",
        "residual_summary": metrics_dir / "residual_summary.csv",
        "horizon_error_summary": metrics_dir / "horizon_error_summary.csv",
        "local_hour_error_summary": metrics_dir / "local_hour_error_summary.csv",
        "behavioral_error_summary": metrics_dir / "behavioral_error_summary.csv",
        "best_predictions": predictions_dir / "best_cv_predictions.csv",
        "actual_vs_predicted_plot": figures_dir / "actual_vs_predicted_best_cv.png",
        "residual_distribution_plot": figures_dir / "residual_distribution_best_cv.png",
        "horizon_error_plot": figures_dir / "absolute_error_by_horizon.png",
        "behavioral_error_plot": figures_dir / "mae_by_behavior_segment.png",
        "summary": summaries_dir / "experiment_b_summary.md",
        "metadata": base / "experiment_metadata.json",
        "report": REPORT_PATH,
    }


def ensure_experiment_b_dirs(paths: Mapping[str, Path]) -> None:
    for key in ["base", "metrics", "predictions", "figures", "summaries"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def build_feature_set_comparison() -> pd.DataFrame:
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Daftar feature columns tidak ditemukan: {FEATURE_COLUMNS_PATH}"
        )
    with FEATURE_COLUMNS_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows: list[dict[str, Any]] = []
    for feature_set in ["xgb_basic", "xgb_advanced"]:
        columns = list(payload.get(feature_set, []))
        if not columns:
            raise ValueError(f"Feature columns kosong untuk {feature_set}.")
        rows.append(
            {
                "feature_set": feature_set,
                "n_features": len(columns),
                "feature_columns": ", ".join(columns),
                "n_lag_features": sum(column.startswith("lag_") for column in columns),
                "n_rolling_features": sum(
                    column.startswith("rolling_") for column in columns
                ),
                "n_calendar_features": sum(
                    not column.startswith("lag_") and not column.startswith("rolling_")
                    for column in columns
                ),
            }
        )
    return pd.DataFrame(rows)


def build_behavioral_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    actual_reference = predictions.drop_duplicates(subset=["fold", TIMESTAMP_COL])
    spike_threshold = float(actual_reference["actual"].quantile(0.90))

    segments = {
        "all": pd.Series(True, index=predictions.index),
        "rush_hour_local_07_09_16_19": predictions["local_hour"].isin(
            [7, 8, 9, 16, 17, 18, 19]
        ),
        "night_local_00_05": predictions["local_hour"].between(0, 5),
        "weekend_local": predictions["local_is_weekend"].astype(bool),
        "weekday_local": ~predictions["local_is_weekend"].astype(bool),
        "high_demand_spike_p90": predictions["actual"] >= spike_threshold,
    }

    rows: list[dict[str, Any]] = []
    for model_label, model_df in predictions.groupby("model_label", sort=False):
        for segment_name, mask in segments.items():
            group = model_df[mask.loc[model_df.index]]
            if group.empty:
                continue
            residual = pd.to_numeric(group["residual"], errors="raise")
            predicted = pd.to_numeric(group["predicted"], errors="raise")
            actual = pd.to_numeric(group["actual"], errors="raise")
            rows.append(
                {
                    "model_label": model_label,
                    "segment": segment_name,
                    "n_predictions": int(group.shape[0]),
                    "spike_threshold_p90_actual": spike_threshold
                    if segment_name == "high_demand_spike_p90"
                    else np.nan,
                    "mae": float(np.abs(residual).mean()),
                    "rmse": float(np.sqrt(np.square(residual).mean())),
                    "mean_error_actual_minus_predicted": float(residual.mean()),
                    "overprediction_rate": float((predicted > actual).mean()),
                    "underprediction_rate": float((predicted < actual).mean()),
                }
            )
    return pd.DataFrame(rows)


def save_behavioral_error_plot(
    behavioral_error: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    segments = [
        "rush_hour_local_07_09_16_19",
        "night_local_00_05",
        "weekend_local",
        "high_demand_spike_p90",
    ]
    plot_df = behavioral_error[behavioral_error["segment"].isin(segments)].copy()
    pivot = plot_df.pivot(index="segment", columns="model_label", values="mae")
    pivot = pivot.reindex(segments)

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Experiment B - MAE by Behavior Segment")
    ax.set_xlabel("Behavior segment")
    ax.set_ylabel("MAE")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def advanced_improves_primary_metric(improvements: pd.DataFrame) -> bool:
    primary = improvements[improvements["metric"] == PRIMARY_METRIC]
    if primary.empty:
        raise ValueError(f"Metric utama tidak ditemukan: {PRIMARY_METRIC}")
    return bool(float(primary.iloc[0]["absolute_reduction"]) > 0)


def render_experiment_b_report(
    *,
    feature_comparison: pd.DataFrame,
    comparison: pd.DataFrame,
    improvements: pd.DataFrame,
    fold_comparison: pd.DataFrame,
    runtime_comparison: pd.DataFrame,
    residual_summary: pd.DataFrame,
    horizon_error: pd.DataFrame,
    behavioral_error: pd.DataFrame,
    artifacts: Sequence[Mapping[str, Any]],
    started_at: str,
    skip_plots: bool,
    outputs: Mapping[str, Path],
) -> str:
    winner = select_winner(comparison, metric=PRIMARY_METRIC)
    primary_improvement = improvements[
        improvements["metric"] == PRIMARY_METRIC
    ].iloc[0]
    rmse_improvement = improvements[improvements["metric"] == "rmse"].iloc[0]
    smape_improvement = improvements[improvements["metric"] == "smape"].iloc[0]
    fold_mae_wins = fold_comparison["mae_winner"].value_counts().to_dict()
    advanced_improves = advanced_improves_primary_metric(improvements)
    interpretation_sentence = (
        "Advanced feature engineering memberi peningkatan pada metric utama."
        if advanced_improves
        else "Advanced feature engineering belum memberi peningkatan pada metric utama."
    )

    best_horizon = (
        horizon_error.sort_values(["model_label", "mae"], kind="mergesort")
        .groupby("model_label", sort=False)
        .head(1)
    )
    worst_horizon = (
        horizon_error.sort_values(["model_label", "mae"], ascending=[True, False])
        .groupby("model_label", sort=False)
        .head(1)
    )

    behavior_pivot = behavioral_error.pivot(
        index="segment",
        columns="model_label",
        values="mae",
    ).reset_index()
    if BASELINE_LABEL in behavior_pivot.columns and CHALLENGER_LABEL in behavior_pivot.columns:
        behavior_pivot["mae_absolute_reduction_advanced_vs_basic"] = (
            behavior_pivot[BASELINE_LABEL] - behavior_pivot[CHALLENGER_LABEL]
        )
        behavior_pivot["mae_percent_reduction_advanced_vs_basic"] = (
            behavior_pivot["mae_absolute_reduction_advanced_vs_basic"]
            / behavior_pivot[BASELINE_LABEL]
            * 100.0
        )

    lines = [
        "# Experiment B: XGBoost-Basic vs XGBoost-Advanced",
        "",
        f"Run UTC: {started_at}",
        "",
        "## Scope",
        "",
        (
            "Experiment B memakai artefak tuning/CV terbaik dari tahap 10. "
            "Final test tidak dibaca dan tidak digunakan pada tahap ini. "
            "Fokusnya adalah mengukur dampak advanced feature engineering "
            "pada performa XGBoost dengan recursive forecasting CV."
        ),
        "",
        "## Feature Set Comparison",
        "",
        dataframe_to_markdown(feature_comparison, float_digits=3),
        "",
        "## Best Configurations",
        "",
        "| Model | Parameter set | Params |",
        "|---|---:|---|",
    ]
    for artifact in artifacts:
        lines.append(
            "| {model} | {param_id} | `{params}` |".format(
                model=artifact["model_label"],
                param_id=artifact["parameter_set_id"],
                params=json.dumps(artifact["params"], sort_keys=True),
            )
        )

    lines.extend(
        [
            "",
            "## CV Metric Comparison",
            "",
            dataframe_to_markdown(
                comparison[
                    [
                        "rank_by_primary_metric",
                        "model_label",
                        "feature_set",
                        "parameter_set_id",
                        "mae_mean",
                        "mae_std",
                        "rmse_mean",
                        "rmse_std",
                        "mape_mean",
                        "smape_mean",
                    ]
                ],
                float_digits=3,
            ),
            "",
            "## Advanced vs Basic Improvement",
            "",
            dataframe_to_markdown(improvements, float_digits=3),
            "",
            "## Stability Across Folds",
            "",
            (
                "Winner by fold based on MAE: "
                f"{json.dumps(fold_mae_wins, sort_keys=True)}."
            ),
            "",
            dataframe_to_markdown(
                fold_comparison[
                    [
                        "fold",
                        f"mae_{BASELINE_LABEL}",
                        f"mae_{CHALLENGER_LABEL}",
                        "mae_absolute_reduction",
                        f"mae_percent_reduction_vs_{BASELINE_LABEL}",
                        "mae_winner",
                    ]
                ],
                float_digits=3,
            ),
            "",
            "## Behavioral Error",
            "",
            dataframe_to_markdown(behavior_pivot, float_digits=3),
            "",
            "## Time Cost Computing",
            "",
            dataframe_to_markdown(runtime_comparison, float_digits=6),
            "",
            "## Residual Behavior",
            "",
            dataframe_to_markdown(residual_summary, float_digits=3),
            "",
            "Best horizon step by MAE:",
            "",
            dataframe_to_markdown(best_horizon, float_digits=3),
            "",
            "Worst horizon step by MAE:",
            "",
            dataframe_to_markdown(worst_horizon, float_digits=3),
            "",
            "## Interpretation",
            "",
            (
                f"Berdasarkan CV best configuration, {winner} menjadi model terbaik "
                f"untuk Experiment B pada metric utama {PRIMARY_METRIC}. "
                f"{interpretation_sentence} "
                f"Perubahan MAE advanced terhadap basic adalah "
                f"{primary_improvement['percent_reduction_vs_baseline']:.2f}%, "
                f"RMSE {rmse_improvement['percent_reduction_vs_baseline']:.2f}%, "
                f"dan sMAPE {smape_improvement['percent_reduction_vs_baseline']:.2f}%."
            ),
            "",
            (
                "Pada window CV ini, tambahan lag, rolling statistics, dan calendar "
                "features belum otomatis memperbaiki generalisasi. Ini bisa terjadi "
                "karena feature set yang lebih kaya meningkatkan kompleksitas model "
                "dan sensitivitas terhadap pola fold tertentu. Kesimpulan ini tetap "
                "perlu dibawa ke tahap retraining dan final testing tanpa tuning ulang."
            ),
            "",
            "## Output Files",
            "",
            f"- Feature set comparison: `{outputs['feature_set_comparison']}`",
            f"- Metric comparison: `{outputs['metric_comparison']}`",
            f"- Improvement summary: `{outputs['improvement_summary']}`",
            f"- Fold comparison: `{outputs['fold_comparison']}`",
            f"- Behavioral error summary: `{outputs['behavioral_error_summary']}`",
            f"- Runtime comparison: `{outputs['runtime_comparison']}`",
            f"- Best CV predictions: `{outputs['best_predictions']}`",
            f"- Report mirror: `{outputs['report']}`",
        ]
    )

    if not skip_plots:
        lines.extend(
            [
                f"- Actual vs predicted plot: `{outputs['actual_vs_predicted_plot']}`",
                f"- Residual distribution plot: `{outputs['residual_distribution_plot']}`",
                f"- Horizon error plot: `{outputs['horizon_error_plot']}`",
                f"- Behavioral error plot: `{outputs['behavioral_error_plot']}`",
            ]
        )

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()


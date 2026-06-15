"""
Script tahap 11: Experiment A - Prophet-Regressor-Basic vs XGBoost-Basic.

Experiment A menjawab pertanyaan:
Apakah machine learning sederhana mampu mengungguli forecasting klasik?

Scope tahap ini:
- Membandingkan hasil CV dari best configuration hasil tuning.
- Model Prophet pada alur utama memakai regressor basic yang sebanding dengan
  XGBoost-Basic.
- Tidak membaca final_test dan tidak melakukan pemilihan parameter baru.
- XGBoost-Basic tetap diaudit dari output recursive forecasting CV.

Contoh:
    python -m src.experiments.experiment_a
    python -m src.experiments.experiment_a --skip-plots
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
    FORECAST_HORIZON,
    METRICS,
    PRIMARY_METRIC,
    PROPHET_REGRESSOR_BASIC_OUTPUT_DIR,
    REPORTS_DIR,
    TARGET_COL,
    TIMESTAMP_COL,
    XGB_BASIC_OUTPUT_DIR,
    ensure_dirs,
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


EXPERIMENT_NAME = "experiment_a_prophet_regressor_basic_vs_xgb_basic"
OUTPUT_DIR = EXPERIMENTS_DIR / "experiment_a"
REPORT_PATH = REPORTS_DIR / "experiment_a_prophet_regressor_basic_vs_xgb_basic.md"
BASELINE_LABEL = "Prophet-Regressor-Basic"
CHALLENGER_LABEL = "XGBoost-Basic"

REQUIRED_PREDICTION_COLUMNS = {
    TIMESTAMP_COL,
    "actual",
    "predicted",
    "model_name",
    "feature_set",
    "fold",
    "parameter_set_id",
    "horizon_step",
    "prediction_time_seconds",
    "used_actual_future_for_features",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Experiment A comparison: "
            "Prophet-Regressor-Basic vs XGBoost-Basic."
        )
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip matplotlib figure generation.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    metadata = run_experiment_a(skip_plots=args.skip_plots)
    print("Experiment A selesai.")
    print(f"Winner by {PRIMARY_METRIC}: {metadata['winner_by_primary_metric']}")
    print(f"Output dir: {metadata['outputs']['base']}")
    print(f"Report: {metadata['outputs']['report']}")


def run_experiment_a(*, skip_plots: bool = False) -> dict[str, Any]:
    timer = start_timer()
    started_at = utc_now_iso()
    status = "success"
    error_message = ""
    outputs = experiment_a_output_paths()

    try:
        ensure_dirs()
        ensure_experiment_a_dirs(outputs)

        prophet = load_best_tuning_artifacts(
            model_label=BASELINE_LABEL,
            output_dir=PROPHET_REGRESSOR_BASIC_OUTPUT_DIR,
        )
        xgb_basic = load_best_tuning_artifacts(
            model_label=CHALLENGER_LABEL,
            output_dir=XGB_BASIC_OUTPUT_DIR,
        )

        validate_best_artifacts([prophet, xgb_basic])

        comparison = build_metric_comparison([prophet, xgb_basic])
        improvements = build_improvement_summary(
            comparison,
            baseline_label=BASELINE_LABEL,
            challenger_label=CHALLENGER_LABEL,
        )
        fold_comparison = build_fold_comparison(
            prophet["best_metrics"],
            xgb_basic["best_metrics"],
            baseline_label=BASELINE_LABEL,
            challenger_label=CHALLENGER_LABEL,
        )
        runtime_comparison = build_runtime_comparison([prophet, xgb_basic])
        predictions = build_combined_best_predictions([prophet, xgb_basic])
        residual_summary = build_residual_summary(predictions)
        horizon_error = build_horizon_error_summary(predictions)
        local_hour_error = build_local_hour_error_summary(predictions)

        comparison.to_csv(outputs["metric_comparison"], index=False)
        improvements.to_csv(outputs["improvement_summary"], index=False)
        fold_comparison.to_csv(outputs["fold_comparison"], index=False)
        runtime_comparison.to_csv(outputs["runtime_comparison"], index=False)
        predictions.to_csv(outputs["best_predictions"], index=False)
        residual_summary.to_csv(outputs["residual_summary"], index=False)
        horizon_error.to_csv(outputs["horizon_error_summary"], index=False)
        local_hour_error.to_csv(outputs["local_hour_error_summary"], index=False)

        if not skip_plots:
            save_actual_vs_predicted_plot(
                predictions,
                outputs["actual_vs_predicted_plot"],
            )
            save_residual_distribution_plot(
                predictions,
                outputs["residual_distribution_plot"],
            )
            save_horizon_error_plot(
                horizon_error,
                outputs["horizon_error_plot"],
            )
            save_fold_error_plot(
                fold_comparison,
                outputs["fold_error_plot"],
            )

        report_text = render_experiment_a_report(
            comparison=comparison,
            improvements=improvements,
            fold_comparison=fold_comparison,
            runtime_comparison=runtime_comparison,
            residual_summary=residual_summary,
            horizon_error=horizon_error,
            artifacts=[prophet, xgb_basic],
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
            "models_compared": [artifact["model_label"] for artifact in [prophet, xgb_basic]],
            "best_parameter_sets": {
                artifact["model_label"]: artifact["parameter_set_id"]
                for artifact in [prophet, xgb_basic]
            },
            "outputs": stringify_paths(outputs),
        }
        save_experiment_metadata(metadata, outputs["metadata"])
        log_runtime(
            make_runtime_record(
                experiment_name=EXPERIMENT_NAME,
                model_name="comparison",
                feature_set="prophet_basic_regressors_vs_xgb_basic",
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
                feature_set="prophet_basic_regressors_vs_xgb_basic",
                total_runtime_seconds=total_runtime_seconds,
                status=status,
                error_message=error_message,
            )
        )
        append_experiment_run(failure_metadata)
        raise


def experiment_a_output_paths() -> dict[str, Path]:
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
        "metric_comparison": metrics_dir / "metric_comparison.csv",
        "improvement_summary": metrics_dir / "improvement_summary.csv",
        "fold_comparison": metrics_dir / "fold_comparison.csv",
        "runtime_comparison": metrics_dir / "runtime_comparison.csv",
        "residual_summary": metrics_dir / "residual_summary.csv",
        "horizon_error_summary": metrics_dir / "horizon_error_summary.csv",
        "local_hour_error_summary": metrics_dir / "local_hour_error_summary.csv",
        "best_predictions": predictions_dir / "best_cv_predictions.csv",
        "actual_vs_predicted_plot": figures_dir / "actual_vs_predicted_best_cv.png",
        "residual_distribution_plot": figures_dir / "residual_distribution_best_cv.png",
        "horizon_error_plot": figures_dir / "absolute_error_by_horizon.png",
        "fold_error_plot": figures_dir
        / "prophet_regressor_basic_vs_xgb_basic_error_by_fold.png",
        "summary": summaries_dir / "experiment_a_summary.md",
        "metadata": base / "experiment_metadata.json",
        "report": REPORT_PATH,
    }


def ensure_experiment_a_dirs(paths: Mapping[str, Path]) -> None:
    for key in ["base", "metrics", "predictions", "figures", "summaries"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_best_tuning_artifacts(
    *,
    model_label: str,
    output_dir: Path,
) -> dict[str, Any]:
    best_params_path = output_dir / "best_params.json"
    metrics_path = output_dir / "cv_metrics.csv"
    metrics_summary_path = output_dir / "cv_metrics_summary.csv"
    predictions_path = output_dir / "cv_predictions.csv"
    runtime_summary_path = output_dir / "runtime_summary.csv"
    metadata_path = output_dir / "experiment_metadata.json"

    for path in [
        best_params_path,
        metrics_path,
        metrics_summary_path,
        predictions_path,
        runtime_summary_path,
        metadata_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Artefak tuning untuk {model_label} tidak ditemukan: {path}"
            )

    best_payload = load_json(best_params_path)
    tuning_metadata = load_json(metadata_path)
    parameter_set_id = str(best_payload["parameter_set_id"])

    metrics_summary = pd.read_csv(metrics_summary_path)
    best_summary = select_parameter_rows(
        metrics_summary,
        parameter_set_id=parameter_set_id,
        source=metrics_summary_path,
    )

    metrics = pd.read_csv(metrics_path)
    best_metrics = select_parameter_rows(
        metrics,
        parameter_set_id=parameter_set_id,
        source=metrics_path,
    )

    runtime_summary = pd.read_csv(runtime_summary_path)
    best_runtime = select_parameter_rows(
        runtime_summary,
        parameter_set_id=parameter_set_id,
        source=runtime_summary_path,
    )

    predictions = pd.read_csv(predictions_path)
    validate_prediction_columns(predictions, source=predictions_path)
    best_predictions = select_parameter_rows(
        predictions,
        parameter_set_id=parameter_set_id,
        source=predictions_path,
    )
    best_predictions = prepare_prediction_frame(best_predictions, model_label=model_label)

    return {
        "model_label": model_label,
        "output_dir": output_dir,
        "parameter_set_id": parameter_set_id,
        "params": best_payload.get("params", {}),
        "selection_rule": best_payload.get("selection_rule", ""),
        "best_payload": best_payload,
        "tuning_metadata": tuning_metadata,
        "best_summary": best_summary,
        "best_metrics": best_metrics,
        "best_runtime": best_runtime,
        "best_predictions": best_predictions,
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON harus berupa object: {path}")
    return payload


def select_parameter_rows(
    frame: pd.DataFrame,
    *,
    parameter_set_id: str,
    source: Path,
) -> pd.DataFrame:
    if "parameter_set_id" not in frame.columns:
        raise ValueError(f"Kolom parameter_set_id tidak ditemukan: {source}")
    selected = frame[frame["parameter_set_id"].astype(str) == str(parameter_set_id)].copy()
    if selected.empty:
        raise ValueError(
            f"Parameter set {parameter_set_id} tidak ditemukan dalam {source}"
        )
    return selected


def validate_prediction_columns(predictions: pd.DataFrame, *, source: Path) -> None:
    missing = sorted(REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns))
    if missing:
        raise ValueError(f"Kolom prediksi tidak lengkap pada {source}: {missing}")


def prepare_prediction_frame(predictions: pd.DataFrame, *, model_label: str) -> pd.DataFrame:
    prepared = predictions.copy()
    prepared[TIMESTAMP_COL] = pd.to_datetime(
        prepared[TIMESTAMP_COL],
        utc=True,
        errors="raise",
    )
    for column in ["actual", "predicted", "prediction_time_seconds"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="raise")
    prepared["fold"] = pd.to_numeric(prepared["fold"], errors="raise").astype(int)
    prepared["horizon_step"] = pd.to_numeric(
        prepared["horizon_step"],
        errors="raise",
    ).astype(int)
    prepared["used_actual_future_for_features"] = coerce_bool_series(
        prepared["used_actual_future_for_features"],
        column_name="used_actual_future_for_features",
    )
    prepared["model_label"] = model_label
    prepared["residual"] = prepared["actual"] - prepared["predicted"]
    prepared["absolute_error"] = prepared["residual"].abs()
    prepared["squared_error"] = np.square(prepared["residual"])
    local_timestamp = prepared[TIMESTAMP_COL].dt.tz_convert("America/New_York")
    prepared["local_hour"] = local_timestamp.dt.hour
    prepared["local_day_of_week"] = local_timestamp.dt.dayofweek
    prepared["local_is_weekend"] = prepared["local_day_of_week"].isin([5, 6]).astype(int)
    return prepared.sort_values(["fold", TIMESTAMP_COL], kind="mergesort").reset_index(
        drop=True
    )


def validate_best_artifacts(artifacts: Sequence[Mapping[str, Any]]) -> None:
    if len(artifacts) != 2:
        raise ValueError("Perbandingan eksperimen membutuhkan tepat dua model.")

    for artifact in artifacts:
        predictions = artifact["best_predictions"]
        metrics = artifact["best_metrics"]
        summary = artifact["best_summary"]
        runtime = artifact["best_runtime"]

        if predictions.empty or metrics.empty or summary.empty or runtime.empty:
            raise ValueError(f"Artefak best config kosong: {artifact['model_label']}")
        leakage_flags = coerce_bool_series(
            predictions["used_actual_future_for_features"],
            column_name="used_actual_future_for_features",
        )
        if leakage_flags.any():
            raise ValueError(
                f"Leakage flag aktif pada prediksi {artifact['model_label']}."
            )
        if predictions[TIMESTAMP_COL].duplicated().any():
            raise ValueError(
                f"Prediksi best config memiliki duplicate timestamp: "
                f"{artifact['model_label']}"
            )
        if predictions["horizon_step"].min() < 1:
            raise ValueError(f"horizon_step tidak valid: {artifact['model_label']}")
        if predictions["horizon_step"].max() > FORECAST_HORIZON:
            raise ValueError(
                f"horizon_step melebihi FORECAST_HORIZON pada {artifact['model_label']}"
            )
        if metrics["fold"].nunique() != int(summary.iloc[0]["n_folds"]):
            raise ValueError(
                f"Jumlah fold metrics tidak sesuai summary: {artifact['model_label']}"
            )

    left = artifacts[0]["best_predictions"]
    right = artifacts[1]["best_predictions"]
    left_keys = left[["fold", TIMESTAMP_COL]].sort_values(["fold", TIMESTAMP_COL])
    right_keys = right[["fold", TIMESTAMP_COL]].sort_values(["fold", TIMESTAMP_COL])
    if not left_keys.reset_index(drop=True).equals(right_keys.reset_index(drop=True)):
        labels = " dan ".join(str(artifact["model_label"]) for artifact in artifacts)
        raise ValueError(f"Timestamp/fold prediksi {labels} tidak sejajar.")

    merged_actuals = left.merge(
        right,
        on=["fold", TIMESTAMP_COL],
        suffixes=("_left", "_right"),
    )
    if not np.allclose(merged_actuals["actual_left"], merged_actuals["actual_right"]):
        labels = " dan ".join(str(artifact["model_label"]) for artifact in artifacts)
        raise ValueError(f"Nilai actual {labels} tidak identik.")


def build_metric_comparison(artifacts: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        summary_row = artifact["best_summary"].iloc[0].to_dict()
        row = {
            "model_label": artifact["model_label"],
            "model_name": summary_row.get("model_name", ""),
            "feature_set": summary_row.get("feature_set", ""),
            "parameter_set_id": artifact["parameter_set_id"],
            "selection_rule": artifact["selection_rule"],
            "params_json": json.dumps(artifact["params"], sort_keys=True),
            "n_rows": int(summary_row.get("n_rows", 0)),
            "n_folds": int(summary_row.get("n_folds", 0)),
        }
        for metric in METRICS:
            for suffix in ["mean", "std", "min", "max"]:
                column = f"{metric}_{suffix}"
                row[column] = float(summary_row[column])
        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(
        f"{PRIMARY_METRIC}_mean",
        kind="mergesort",
    ).reset_index(drop=True)
    comparison.insert(0, "rank_by_primary_metric", np.arange(1, len(comparison) + 1))
    return comparison


def build_improvement_summary(
    comparison: pd.DataFrame,
    *,
    baseline_label: str,
    challenger_label: str,
) -> pd.DataFrame:
    baseline = comparison[comparison["model_label"] == baseline_label]
    challenger = comparison[comparison["model_label"] == challenger_label]
    if baseline.empty or challenger.empty:
        raise ValueError("Baseline atau challenger tidak ditemukan pada metric comparison.")

    baseline_row = baseline.iloc[0]
    challenger_row = challenger.iloc[0]
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        baseline_value = float(baseline_row[f"{metric}_mean"])
        challenger_value = float(challenger_row[f"{metric}_mean"])
        absolute_reduction = baseline_value - challenger_value
        percent_reduction = (
            absolute_reduction / baseline_value * 100.0
            if baseline_value != 0
            else np.nan
        )
        rows.append(
            {
                "metric": metric,
                "baseline_model": baseline_label,
                "challenger_model": challenger_label,
                "baseline_mean": baseline_value,
                "challenger_mean": challenger_value,
                "absolute_reduction": absolute_reduction,
                "percent_reduction_vs_baseline": percent_reduction,
                "winner": challenger_label
                if challenger_value < baseline_value
                else baseline_label,
            }
        )
    return pd.DataFrame(rows)


def build_fold_comparison(
    baseline_metrics: pd.DataFrame,
    challenger_metrics: pd.DataFrame,
    *,
    baseline_label: str,
    challenger_label: str,
) -> pd.DataFrame:
    baseline = baseline_metrics.copy()
    challenger = challenger_metrics.copy()
    baseline["fold"] = pd.to_numeric(baseline["fold"], errors="raise").astype(int)
    challenger["fold"] = pd.to_numeric(challenger["fold"], errors="raise").astype(int)

    rows: list[dict[str, Any]] = []
    for fold in sorted(set(baseline["fold"]).intersection(challenger["fold"])):
        base_row = baseline[baseline["fold"] == fold].iloc[0]
        chal_row = challenger[challenger["fold"] == fold].iloc[0]
        row: dict[str, Any] = {"fold": int(fold)}
        for metric in METRICS:
            baseline_value = float(base_row[metric])
            challenger_value = float(chal_row[metric])
            reduction = baseline_value - challenger_value
            row[f"{metric}_{baseline_label}"] = baseline_value
            row[f"{metric}_{challenger_label}"] = challenger_value
            row[f"{metric}_absolute_reduction"] = reduction
            row[f"{metric}_percent_reduction_vs_{baseline_label}"] = (
                reduction / baseline_value * 100.0
                if baseline_value != 0
                else np.nan
            )
            row[f"{metric}_winner"] = (
                challenger_label if challenger_value < baseline_value else baseline_label
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_runtime_comparison(artifacts: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        runtime = artifact["best_runtime"].iloc[0].to_dict()
        tuning_time = artifact["tuning_metadata"].get("time_cost_computing", {})
        row = {
            "model_label": artifact["model_label"],
            "parameter_set_id": artifact["parameter_set_id"],
            "n_folds": int(runtime.get("n_folds", 0)),
            "best_train_time_seconds_sum": float(runtime["train_time_seconds_sum"]),
            "best_train_time_seconds_mean": float(runtime["train_time_seconds_mean"]),
            "best_prediction_time_seconds_sum": float(
                runtime["prediction_time_seconds_sum"]
            ),
            "best_prediction_time_seconds_mean": float(
                runtime["prediction_time_seconds_mean"]
            ),
            "best_total_runtime_seconds_sum": float(runtime["total_runtime_seconds_sum"]),
            "best_total_runtime_seconds_mean": float(runtime["total_runtime_seconds_mean"]),
            "full_tuning_fold_runtime_seconds_sum": float(
                tuning_time.get("sum_fold_runtime_seconds", np.nan)
            ),
            "full_tuning_train_time_seconds_sum": float(
                tuning_time.get("sum_train_time_seconds", np.nan)
            ),
            "full_tuning_prediction_time_seconds_sum": float(
                tuning_time.get("sum_prediction_time_seconds", np.nan)
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_combined_best_predictions(
    artifacts: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    columns = [
        "model_label",
        "model_name",
        "feature_set",
        "parameter_set_id",
        "fold",
        TIMESTAMP_COL,
        "actual",
        "predicted",
        "residual",
        "absolute_error",
        "squared_error",
        "horizon_step",
        "forecast_origin",
        "origin_block",
        "validation_start",
        "validation_end",
        "prediction_time_seconds",
        "used_actual_future_for_features",
        "local_hour",
        "local_day_of_week",
        "local_is_weekend",
    ]
    frames = []
    for artifact in artifacts:
        frame = artifact["best_predictions"].copy()
        available_columns = [column for column in columns if column in frame.columns]
        frames.append(frame[available_columns])
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(
        ["model_label", "fold", TIMESTAMP_COL],
        kind="mergesort",
    ).reset_index(drop=True)


def build_residual_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_label, group in predictions.groupby("model_label", sort=False):
        residual = pd.to_numeric(group["residual"], errors="raise")
        absolute_error = pd.to_numeric(group["absolute_error"], errors="raise")
        predicted = pd.to_numeric(group["predicted"], errors="raise")
        actual = pd.to_numeric(group["actual"], errors="raise")
        rows.append(
            {
                "model_label": model_label,
                "n_predictions": int(group.shape[0]),
                "mean_error_actual_minus_predicted": float(residual.mean()),
                "median_error_actual_minus_predicted": float(residual.median()),
                "residual_std": float(residual.std(ddof=1)),
                "mean_absolute_error": float(absolute_error.mean()),
                "rmse": float(np.sqrt(np.square(residual).mean())),
                "overprediction_rate": float((predicted > actual).mean()),
                "underprediction_rate": float((predicted < actual).mean()),
                "max_absolute_error": float(absolute_error.max()),
            }
        )
    return pd.DataFrame(rows)


def build_horizon_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_label, horizon_step), group in predictions.groupby(
        ["model_label", "horizon_step"],
        sort=True,
    ):
        residual = pd.to_numeric(group["residual"], errors="raise")
        rows.append(
            {
                "model_label": model_label,
                "horizon_step": int(horizon_step),
                "n_predictions": int(group.shape[0]),
                "mae": float(np.abs(residual).mean()),
                "rmse": float(np.sqrt(np.square(residual).mean())),
                "mean_error_actual_minus_predicted": float(residual.mean()),
            }
        )
    return pd.DataFrame(rows)


def build_local_hour_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_label, local_hour), group in predictions.groupby(
        ["model_label", "local_hour"],
        sort=True,
    ):
        residual = pd.to_numeric(group["residual"], errors="raise")
        rows.append(
            {
                "model_label": model_label,
                "local_hour": int(local_hour),
                "n_predictions": int(group.shape[0]),
                "mae": float(np.abs(residual).mean()),
                "rmse": float(np.sqrt(np.square(residual).mean())),
                "mean_error_actual_minus_predicted": float(residual.mean()),
            }
        )
    return pd.DataFrame(rows)


def save_actual_vs_predicted_plot(
    predictions: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Experiment A - Best CV Predictions",
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_df = predictions.sort_values(TIMESTAMP_COL, kind="mergesort")
    actual = (
        plot_df[[TIMESTAMP_COL, "actual"]]
        .drop_duplicates(subset=[TIMESTAMP_COL])
        .sort_values(TIMESTAMP_COL, kind="mergesort")
    )

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(actual[TIMESTAMP_COL], actual["actual"], label="Actual", linewidth=1.4)
    for model_label, group in plot_df.groupby("model_label", sort=False):
        ax.plot(
            group[TIMESTAMP_COL],
            group["predicted"],
            label=model_label,
            linewidth=1.1,
            alpha=0.85,
        )
    ax.set_title(title)
    ax.set_xlabel("UTC timestamp")
    ax.set_ylabel(TARGET_COL)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_residual_distribution_plot(
    predictions: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Experiment A - Residual Distribution",
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for model_label, group in predictions.groupby("model_label", sort=False):
        ax.hist(
            group["residual"],
            bins=40,
            alpha=0.55,
            label=model_label,
        )
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_horizon_error_plot(
    horizon_error: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Experiment A - MAE by Recursive Horizon Step",
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for model_label, group in horizon_error.groupby("model_label", sort=False):
        ax.plot(
            group["horizon_step"],
            group["mae"],
            marker="o",
            linewidth=1.4,
            label=model_label,
        )
    ax.set_title(title)
    ax.set_xlabel("Horizon step")
    ax.set_ylabel("MAE")
    ax.set_xticks(list(range(1, FORECAST_HORIZON + 1)))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_fold_error_plot(
    fold_comparison: pd.DataFrame,
    output_path: Path,
    *,
    title: str = (
        "Perbandingan Error Prophet-Regressor-Basic dan XGBoost-Basic pada Setiap Fold"
    ),
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_df = fold_comparison.sort_values("fold", kind="mergesort").copy()
    x = np.arange(plot_df.shape[0])
    width = 0.36

    prophet_col = f"mae_{BASELINE_LABEL}"
    xgb_col = f"mae_{CHALLENGER_LABEL}"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        plot_df[prophet_col],
        width,
        label=BASELINE_LABEL,
        color="#2f6fbb",
    )
    ax.bar(
        x + width / 2,
        plot_df[xgb_col],
        width,
        label=CHALLENGER_LABEL,
        color="#d77a27",
    )

    ax.set_title(title)
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["fold"].astype(str))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_experiment_a_report(
    *,
    comparison: pd.DataFrame,
    improvements: pd.DataFrame,
    fold_comparison: pd.DataFrame,
    runtime_comparison: pd.DataFrame,
    residual_summary: pd.DataFrame,
    horizon_error: pd.DataFrame,
    artifacts: Sequence[Mapping[str, Any]],
    started_at: str,
    skip_plots: bool,
    outputs: Mapping[str, Path],
) -> str:
    winner = select_winner(comparison, metric=PRIMARY_METRIC)
    mae_improvement = improvements[improvements["metric"] == "mae"].iloc[0]
    rmse_improvement = improvements[improvements["metric"] == "rmse"].iloc[0]
    smape_improvement = improvements[improvements["metric"] == "smape"].iloc[0]
    fold_mae_wins = fold_comparison["mae_winner"].value_counts().to_dict()
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

    lines = [
        "# Experiment A: Prophet-Regressor-Basic vs XGBoost-Basic",
        "",
        f"Run UTC: {started_at}",
        "",
        "## Scope",
        "",
        (
            "Experiment A memakai artefak tuning/CV terbaik dari tahap 10. "
            "Final test tidak dibaca dan tidak digunakan pada tahap ini, sehingga "
            "benchmark test tetap tersimpan untuk tahap final testing setelah retraining."
        ),
        "",
        (
            "Prophet pada alur utama memakai regressor basic (`lag_1`, `lag_24`, "
            "`lag_168`, `hour`, `day_of_week`) agar perbandingan dengan "
            "XGBoost-Basic lebih fair dan menjadi satu metodologi canonical."
        ),
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
            "## Improvement vs Prophet-Regressor-Basic",
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
                f"untuk Experiment A pada metric utama {PRIMARY_METRIC}. "
                f"{CHALLENGER_LABEL} menurunkan MAE sebesar "
                f"{mae_improvement['percent_reduction_vs_baseline']:.2f}% "
                f"dibanding {BASELINE_LABEL}, RMSE sebesar "
                f"{rmse_improvement['percent_reduction_vs_baseline']:.2f}%, "
                f"dan sMAPE sebesar "
                f"{smape_improvement['percent_reduction_vs_baseline']:.2f}%."
            ),
            "",
            (
                "Hasil ini menunjukkan apakah model machine learning sederhana "
                "masih unggul setelah Prophet diberi informasi lag dan calendar "
                "basic yang sebanding. "
                "Kesimpulan final tetap menunggu retraining dan final testing yang "
                "dijalankan pada tahap berikutnya."
            ),
            "",
            "## Output Files",
            "",
            f"- Metric comparison: `{outputs['metric_comparison']}`",
            f"- Fold comparison: `{outputs['fold_comparison']}`",
            f"- Runtime comparison: `{outputs['runtime_comparison']}`",
            f"- Best CV predictions: `{outputs['best_predictions']}`",
            f"- Residual summary: `{outputs['residual_summary']}`",
            f"- Report mirror: `{outputs['report']}`",
        ]
    )

    if not skip_plots:
        lines.extend(
            [
                f"- Actual vs predicted plot: `{outputs['actual_vs_predicted_plot']}`",
                f"- Residual distribution plot: `{outputs['residual_distribution_plot']}`",
                f"- Horizon error plot: `{outputs['horizon_error_plot']}`",
                f"- Fold error plot: `{outputs['fold_error_plot']}`",
            ]
        )

    return "\n".join(lines) + "\n"


def dataframe_to_markdown(df: pd.DataFrame, *, float_digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"

    def format_cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}f}"
        text = str(value).replace("\n", " ")
        return text.replace("|", "\\|")

    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(format_cell(row[column]) for column in df.columns) + " |")
    return "\n".join(lines)


def coerce_bool_series(series: pd.Series, *, column_name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
    }
    parsed = normalized.map(mapping)
    if parsed.isna().any():
        invalid = sorted(normalized[parsed.isna()].unique().tolist())
        raise ValueError(f"Kolom boolean {column_name} memiliki nilai tidak valid: {invalid}")
    return parsed.astype(bool)


def select_winner(comparison: pd.DataFrame, *, metric: str) -> str:
    if comparison.empty:
        raise ValueError("Metric comparison kosong.")
    return str(
        comparison.sort_values(f"{metric}_mean", kind="mergesort").iloc[0][
            "model_label"
        ]
    )


def stringify_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    return {key: str(value) for key, value in paths.items()}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

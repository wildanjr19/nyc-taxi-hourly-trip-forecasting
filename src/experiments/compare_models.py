"""
Script tahap 15: Comparative Evaluation.

Scope:
- Membaca artefak CV best configuration, retraining runtime, dan final test.
- Tidak melakukan tuning, retraining, atau evaluasi ulang final test.
- Menyimpan ranking, improvement, behavioral summary, visual comparison,
  report Markdown, metadata, dan time cost computing tahap komparatif.

Contoh:
    python -m src.experiments.compare_models
    python -m src.experiments.compare_models --skip-plots
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
    FINAL_TEST_DIR,
    FORECAST_HORIZON,
    LOCAL_TZ,
    METRICS,
    PRIMARY_METRIC,
    PROPHET_OUTPUT_DIR,
    REPORTS_DIR,
    TARGET_COL,
    TIMESTAMP_COL,
    XGB_ADVANCED_OUTPUT_DIR,
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


EXPERIMENT_NAME = "comparative_evaluation"
OUTPUT_DIR = EXPERIMENTS_DIR / "comparative_evaluation"
REPORT_PATH = REPORTS_DIR / "comparative_evaluation.md"

MODEL_SOURCES = [
    ("Prophet", PROPHET_OUTPUT_DIR),
    ("XGBoost-Basic", XGB_BASIC_OUTPUT_DIR),
    ("XGBoost-Advanced", XGB_ADVANCED_OUTPUT_DIR),
]

FINAL_METRICS_PATH = FINAL_TEST_DIR / "metrics" / "final_metrics.csv"
FINAL_RUNTIME_PATH = FINAL_TEST_DIR / "metrics" / "final_runtime.csv"
FINAL_PREDICTIONS_PATH = FINAL_TEST_DIR / "predictions" / "final_predictions.csv"

EXPERIMENT_A_BASELINE = "Prophet"
EXPERIMENT_A_CHALLENGER = "XGBoost-Basic"
EXPERIMENT_B_BASELINE = "XGBoost-Basic"
EXPERIMENT_B_CHALLENGER = "XGBoost-Advanced"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run comparative evaluation across final tuned models."
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip matplotlib figure generation.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    metadata = run_comparative_evaluation(skip_plots=args.skip_plots)
    print("Comparative evaluation selesai.")
    print(f"Winner by {PRIMARY_METRIC}: {metadata['winner_by_primary_metric']}")
    print(f"Output dir: {metadata['outputs']['base']}")
    print(f"Report: {metadata['outputs']['report']}")


def run_comparative_evaluation(*, skip_plots: bool = False) -> dict[str, Any]:
    timer = start_timer()
    started_at = utc_now_iso()
    status = "success"
    error_message = ""
    outputs = comparative_output_paths()

    try:
        ensure_dirs()
        ensure_comparative_dirs(outputs)

        final_metrics = load_final_metrics(FINAL_METRICS_PATH)
        final_runtime = load_final_runtime(FINAL_RUNTIME_PATH)
        final_predictions = load_final_predictions(FINAL_PREDICTIONS_PATH)
        validate_final_artifacts(final_metrics, final_runtime, final_predictions)

        tuning_artifacts = [
            load_tuning_artifact(model_label=model_label, output_dir=output_dir)
            for model_label, output_dir in MODEL_SOURCES
        ]
        validate_tuning_artifacts(tuning_artifacts)

        final_ranking = build_final_metric_ranking(final_metrics)
        metric_rankings_long = build_metric_rankings_long(final_metrics)
        improvement_summary = build_final_improvement_summary(final_metrics)
        cv_stability = build_cv_stability_summary(tuning_artifacts)
        fold_stability = build_fold_stability_comparison(tuning_artifacts)
        time_cost = build_time_cost_computing(
            tuning_artifacts,
            final_metrics=final_metrics,
            final_runtime=final_runtime,
        )
        residual_summary = build_residual_summary(final_predictions)
        horizon_error = build_horizon_error_summary(final_predictions)
        local_hour_error = build_local_hour_error_summary(final_predictions)
        behavioral_error = build_behavioral_error_summary(final_predictions)
        behavioral_pivot = build_behavioral_error_pivot(behavioral_error)

        final_ranking.to_csv(outputs["final_metric_ranking"], index=False)
        metric_rankings_long.to_csv(outputs["metric_rankings_long"], index=False)
        improvement_summary.to_csv(outputs["final_improvement_summary"], index=False)
        cv_stability.to_csv(outputs["cv_stability_summary"], index=False)
        fold_stability.to_csv(outputs["fold_stability_comparison"], index=False)
        time_cost.to_csv(outputs["time_cost_computing"], index=False)
        residual_summary.to_csv(outputs["residual_summary"], index=False)
        horizon_error.to_csv(outputs["horizon_error_comparison"], index=False)
        local_hour_error.to_csv(outputs["local_hour_error_summary"], index=False)
        behavioral_error.to_csv(outputs["behavioral_error_summary"], index=False)
        behavioral_pivot.to_csv(outputs["behavioral_error_pivot"], index=False)

        if not skip_plots:
            save_actual_vs_predicted_plot(
                final_predictions,
                outputs["actual_vs_predicted_plot"],
            )
            save_residual_time_series_plot(
                final_predictions,
                outputs["residual_time_series_plot"],
            )
            save_residual_distribution_plot(
                final_predictions,
                outputs["residual_distribution_plot"],
            )
            save_actual_predicted_scatter_plot(
                final_predictions,
                outputs["actual_predicted_scatter_plot"],
            )
            save_accuracy_runtime_plot(
                final_ranking,
                time_cost,
                outputs["accuracy_runtime_plot"],
            )
            save_behavioral_error_plot(
                behavioral_error,
                outputs["behavioral_error_plot"],
            )
            save_horizon_error_plot(
                horizon_error,
                outputs["horizon_error_plot"],
            )
            save_cv_fold_mae_plot(
                tuning_artifacts,
                outputs["cv_fold_mae_plot"],
            )

        total_runtime_seconds = elapsed_seconds(timer)
        winner = select_final_winner(final_metrics, metric=PRIMARY_METRIC)
        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "stage": "comparative_evaluation",
            "status": status,
            "error_message": error_message,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "total_runtime_seconds": total_runtime_seconds,
            "primary_metric": PRIMARY_METRIC,
            "winner_by_primary_metric": winner,
            "models_compared": [model_label for model_label, _ in MODEL_SOURCES],
            "input_artifacts": {
                "final_metrics": str(FINAL_METRICS_PATH),
                "final_runtime": str(FINAL_RUNTIME_PATH),
                "final_predictions": str(FINAL_PREDICTIONS_PATH),
                "tuning_output_dirs": {
                    model_label: str(output_dir)
                    for model_label, output_dir in MODEL_SOURCES
                },
            },
            "leakage_guardrail": [
                "Tahap ini hanya membaca artefak final test yang sudah dibuat.",
                "Tidak ada tuning, retraining, atau prediksi baru.",
                "Final predictions wajib memiliki used_actual_future_for_features=False.",
                "Kolom final_test_used_for_tuning pada metric final wajib False.",
            ],
            "outputs": stringify_paths(outputs),
        }

        report_text = render_comparative_report(
            metadata=metadata,
            final_ranking=final_ranking,
            metric_rankings_long=metric_rankings_long,
            improvement_summary=improvement_summary,
            cv_stability=cv_stability,
            fold_stability=fold_stability,
            time_cost=time_cost,
            residual_summary=residual_summary,
            horizon_error=horizon_error,
            behavioral_pivot=behavioral_pivot,
            skip_plots=skip_plots,
            outputs=outputs,
        )
        write_text(outputs["summary"], report_text)
        write_text(outputs["report"], report_text)
        save_experiment_metadata(metadata, outputs["metadata"])
        log_runtime(
            make_runtime_record(
                experiment_name=EXPERIMENT_NAME,
                model_name="comparison",
                feature_set="all_models",
                n_prediction_rows=int(final_predictions.shape[0]),
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
            "stage": "comparative_evaluation",
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
                feature_set="all_models",
                total_runtime_seconds=total_runtime_seconds,
                status=status,
                error_message=error_message,
            )
        )
        append_experiment_run(failure_metadata)
        raise


def comparative_output_paths() -> dict[str, Path]:
    metrics_dir = OUTPUT_DIR / "metrics"
    figures_dir = OUTPUT_DIR / "figures"
    summaries_dir = OUTPUT_DIR / "summaries"
    return {
        "base": OUTPUT_DIR,
        "metrics": metrics_dir,
        "figures": figures_dir,
        "summaries": summaries_dir,
        "final_metric_ranking": metrics_dir / "final_metric_ranking.csv",
        "metric_rankings_long": metrics_dir / "metric_rankings_long.csv",
        "final_improvement_summary": metrics_dir / "final_improvement_summary.csv",
        "cv_stability_summary": metrics_dir / "cv_stability_summary.csv",
        "fold_stability_comparison": metrics_dir / "fold_stability_comparison.csv",
        "time_cost_computing": metrics_dir / "time_cost_computing.csv",
        "residual_summary": metrics_dir / "residual_summary.csv",
        "horizon_error_comparison": metrics_dir / "horizon_error_comparison.csv",
        "local_hour_error_summary": metrics_dir / "local_hour_error_summary.csv",
        "behavioral_error_summary": metrics_dir / "behavioral_error_summary.csv",
        "behavioral_error_pivot": metrics_dir / "behavioral_error_pivot.csv",
        "actual_vs_predicted_plot": figures_dir / "actual_vs_predicted_final.png",
        "residual_time_series_plot": figures_dir / "residual_time_series_final.png",
        "residual_distribution_plot": figures_dir / "residual_distribution_final.png",
        "actual_predicted_scatter_plot": figures_dir
        / "actual_vs_predicted_scatter.png",
        "accuracy_runtime_plot": figures_dir / "accuracy_vs_runtime.png",
        "behavioral_error_plot": figures_dir / "mae_by_behavior_segment.png",
        "horizon_error_plot": figures_dir / "mae_by_horizon_step.png",
        "cv_fold_mae_plot": figures_dir / "cv_fold_mae.png",
        "metadata": OUTPUT_DIR / "experiment_metadata.json",
        "summary": summaries_dir / "comparative_evaluation_summary.md",
        "report": REPORT_PATH,
    }


def ensure_comparative_dirs(paths: Mapping[str, Path]) -> None:
    for key in ["base", "metrics", "figures", "summaries"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_final_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Final metrics tidak ditemukan: {path}")
    frame = pd.read_csv(path)
    required = {
        "model_label",
        "model_key",
        "feature_set",
        "parameter_set_id",
        "prediction_time_seconds",
        "retraining_train_time_seconds",
        "used_actual_future_for_features",
        "final_test_used_for_tuning",
        *METRICS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Kolom final metrics tidak lengkap: {missing}")
    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")
    for column in [
        "prediction_time_seconds",
        "retraining_train_time_seconds",
        "model_load_time_seconds",
        "total_runtime_seconds",
        "negative_prediction_count",
        "mean_error_actual_minus_predicted",
        "max_absolute_error",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["used_actual_future_for_features"] = coerce_bool_series(
        frame["used_actual_future_for_features"],
        column_name="used_actual_future_for_features",
    )
    frame["final_test_used_for_tuning"] = coerce_bool_series(
        frame["final_test_used_for_tuning"],
        column_name="final_test_used_for_tuning",
    )
    return frame


def load_final_runtime(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Final runtime tidak ditemukan: {path}")
    frame = pd.read_csv(path)
    required = {
        "model_label",
        "model_key",
        "parameter_set_id",
        "model_load_time_seconds",
        "prediction_time_seconds",
        "total_runtime_seconds",
        "retraining_train_time_seconds",
        "status",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Kolom final runtime tidak lengkap: {missing}")
    for column in [
        "model_load_time_seconds",
        "prediction_time_seconds",
        "model_predict_time_seconds",
        "total_runtime_seconds",
        "retraining_train_time_seconds",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def load_final_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Final predictions tidak ditemukan: {path}")
    frame = pd.read_csv(path)
    required = {
        TIMESTAMP_COL,
        "actual",
        "predicted",
        "model_label",
        "model_key",
        "feature_set",
        "parameter_set_id",
        "horizon_step",
        "origin_block",
        "prediction_time_seconds",
        "used_actual_future_for_features",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Kolom final predictions tidak lengkap: {missing}")
    frame[TIMESTAMP_COL] = pd.to_datetime(frame[TIMESTAMP_COL], utc=True, errors="raise")
    for column in ["actual", "predicted", "prediction_time_seconds"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["horizon_step"] = pd.to_numeric(
        frame["horizon_step"],
        errors="raise",
    ).astype(int)
    frame["origin_block"] = pd.to_numeric(
        frame["origin_block"],
        errors="raise",
    ).astype(int)
    frame["used_actual_future_for_features"] = coerce_bool_series(
        frame["used_actual_future_for_features"],
        column_name="used_actual_future_for_features",
    )
    if "residual" not in frame.columns:
        frame["residual"] = frame["actual"] - frame["predicted"]
    else:
        frame["residual"] = pd.to_numeric(frame["residual"], errors="raise")
    if "absolute_error" not in frame.columns:
        frame["absolute_error"] = frame["residual"].abs()
    else:
        frame["absolute_error"] = pd.to_numeric(frame["absolute_error"], errors="raise")
    if "squared_error" not in frame.columns:
        frame["squared_error"] = np.square(frame["residual"])
    else:
        frame["squared_error"] = pd.to_numeric(frame["squared_error"], errors="raise")
    if "local_hour" not in frame.columns or "local_day_of_week" not in frame.columns:
        local_timestamp = frame[TIMESTAMP_COL].dt.tz_convert(LOCAL_TZ)
        frame["local_hour"] = local_timestamp.dt.hour
        frame["local_day_of_week"] = local_timestamp.dt.dayofweek
    else:
        frame["local_hour"] = pd.to_numeric(frame["local_hour"], errors="raise").astype(
            int
        )
        frame["local_day_of_week"] = pd.to_numeric(
            frame["local_day_of_week"],
            errors="raise",
        ).astype(int)
    if "local_is_weekend" not in frame.columns:
        frame["local_is_weekend"] = frame["local_day_of_week"].isin([5, 6]).astype(int)
    else:
        frame["local_is_weekend"] = pd.to_numeric(
            frame["local_is_weekend"],
            errors="raise",
        ).astype(int)
    return frame.sort_values(["model_label", TIMESTAMP_COL], kind="mergesort").reset_index(
        drop=True
    )


def validate_final_artifacts(
    final_metrics: pd.DataFrame,
    final_runtime: pd.DataFrame,
    final_predictions: pd.DataFrame,
) -> None:
    expected_labels = {model_label for model_label, _ in MODEL_SOURCES}
    metric_labels = set(final_metrics["model_label"].astype(str))
    runtime_labels = set(final_runtime["model_label"].astype(str))
    prediction_labels = set(final_predictions["model_label"].astype(str))
    for source_name, labels in [
        ("final_metrics", metric_labels),
        ("final_runtime", runtime_labels),
        ("final_predictions", prediction_labels),
    ]:
        if labels != expected_labels:
            raise ValueError(
                f"Label model pada {source_name} tidak sesuai. "
                f"Expected={sorted(expected_labels)}, found={sorted(labels)}"
            )
    if final_metrics["used_actual_future_for_features"].any():
        raise ValueError("Final metrics menandai used_actual_future_for_features=True.")
    if final_metrics["final_test_used_for_tuning"].any():
        raise ValueError("Final metrics menandai final_test_used_for_tuning=True.")
    if final_predictions["used_actual_future_for_features"].any():
        raise ValueError("Final predictions menandai leakage flag aktif.")
    if final_runtime["status"].astype(str).str.lower().ne("success").any():
        raise ValueError("Final runtime mengandung status bukan success.")

    reference: Optional[pd.DataFrame] = None
    for model_label, group in final_predictions.groupby("model_label", sort=False):
        group = group.sort_values(TIMESTAMP_COL, kind="mergesort")
        if group[TIMESTAMP_COL].duplicated().any():
            raise ValueError(f"Duplicate timestamp pada prediksi {model_label}.")
        if group["horizon_step"].min() < 1:
            raise ValueError(f"horizon_step tidak valid pada {model_label}.")
        if group["horizon_step"].max() > FORECAST_HORIZON:
            raise ValueError(f"horizon_step melebihi horizon pada {model_label}.")
        if reference is None:
            reference = group[[TIMESTAMP_COL, "actual"]].reset_index(drop=True)
        else:
            candidate = group[[TIMESTAMP_COL, "actual"]].reset_index(drop=True)
            if not candidate.equals(reference):
                raise ValueError(
                    f"Timestamp/actual {model_label} tidak sejajar dengan model lain."
                )


def load_tuning_artifact(*, model_label: str, output_dir: Path) -> dict[str, Any]:
    best_params_path = output_dir / "best_params.json"
    metrics_path = output_dir / "cv_metrics.csv"
    metrics_summary_path = output_dir / "cv_metrics_summary.csv"
    runtime_summary_path = output_dir / "runtime_summary.csv"
    metadata_path = output_dir / "experiment_metadata.json"
    for path in [
        best_params_path,
        metrics_path,
        metrics_summary_path,
        runtime_summary_path,
        metadata_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Artefak tuning tidak ditemukan: {path}")

    best_payload = load_json(best_params_path)
    parameter_set_id = str(best_payload["parameter_set_id"])
    metrics = select_parameter_rows(
        pd.read_csv(metrics_path),
        parameter_set_id=parameter_set_id,
        source=metrics_path,
    )
    summary = select_parameter_rows(
        pd.read_csv(metrics_summary_path),
        parameter_set_id=parameter_set_id,
        source=metrics_summary_path,
    )
    runtime = select_parameter_rows(
        pd.read_csv(runtime_summary_path),
        parameter_set_id=parameter_set_id,
        source=runtime_summary_path,
    )
    for metric in METRICS:
        metrics[metric] = pd.to_numeric(metrics[metric], errors="raise")
    metrics["fold"] = pd.to_numeric(metrics["fold"], errors="raise").astype(int)
    return {
        "model_label": model_label,
        "output_dir": output_dir,
        "parameter_set_id": parameter_set_id,
        "params": dict(best_payload.get("params", {})),
        "best_payload": best_payload,
        "best_metrics": metrics,
        "best_summary": summary,
        "best_runtime": runtime,
        "tuning_metadata": load_json(metadata_path),
    }


def validate_tuning_artifacts(artifacts: Sequence[Mapping[str, Any]]) -> None:
    if len(artifacts) != len(MODEL_SOURCES):
        raise ValueError("Jumlah tuning artifacts tidak sesuai jumlah model.")
    expected_folds: Optional[set[int]] = None
    for artifact in artifacts:
        metrics = artifact["best_metrics"]
        summary = artifact["best_summary"]
        runtime = artifact["best_runtime"]
        if metrics.empty or summary.empty or runtime.empty:
            raise ValueError(f"Artefak tuning kosong: {artifact['model_label']}")
        if str(artifact["tuning_metadata"].get("status", "")).lower() != "success":
            raise ValueError(f"Tuning metadata bukan success: {artifact['model_label']}")
        folds = set(metrics["fold"].astype(int))
        if expected_folds is None:
            expected_folds = folds
        elif folds != expected_folds:
            raise ValueError("Fold best CV tidak sejajar antar model.")


def build_final_metric_ranking(final_metrics: pd.DataFrame) -> pd.DataFrame:
    ranking = final_metrics.copy()
    for metric in METRICS:
        ranking[f"rank_by_{metric}"] = (
            ranking[metric].rank(method="min", ascending=True).astype(int)
        )
    ordered_columns = [
        f"rank_by_{PRIMARY_METRIC}",
        "model_label",
        "model_key",
        "feature_set",
        "parameter_set_id",
        *[f"rank_by_{metric}" for metric in METRICS if metric != PRIMARY_METRIC],
        *METRICS,
        "prediction_time_seconds",
        "retraining_train_time_seconds",
        "model_load_time_seconds",
        "total_runtime_seconds",
        "negative_prediction_count",
        "mean_error_actual_minus_predicted",
        "max_absolute_error",
    ]
    available = [column for column in ordered_columns if column in ranking.columns]
    return ranking[available].sort_values(
        f"rank_by_{PRIMARY_METRIC}",
        kind="mergesort",
    ).reset_index(drop=True)


def build_metric_rankings_long(final_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        ordered = final_metrics.sort_values(metric, kind="mergesort").reset_index(
            drop=True
        )
        for rank, row in enumerate(ordered.to_dict("records"), start=1):
            rows.append(
                {
                    "metric": metric,
                    "rank": rank,
                    "model_label": row["model_label"],
                    "parameter_set_id": row["parameter_set_id"],
                    "value": float(row[metric]),
                }
            )
    return pd.DataFrame(rows)


def build_final_improvement_summary(final_metrics: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        (
            "Experiment A final: XGBoost-Basic vs Prophet",
            EXPERIMENT_A_BASELINE,
            EXPERIMENT_A_CHALLENGER,
        ),
        (
            "Experiment B final: XGBoost-Advanced vs XGBoost-Basic",
            EXPERIMENT_B_BASELINE,
            EXPERIMENT_B_CHALLENGER,
        ),
        (
            "Reference final: XGBoost-Advanced vs Prophet",
            "Prophet",
            "XGBoost-Advanced",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for comparison_name, baseline_label, challenger_label in pairs:
        baseline = select_model_row(final_metrics, baseline_label)
        challenger = select_model_row(final_metrics, challenger_label)
        for metric in METRICS:
            baseline_value = float(baseline[metric])
            challenger_value = float(challenger[metric])
            absolute_reduction = baseline_value - challenger_value
            percent_reduction = (
                absolute_reduction / baseline_value * 100.0
                if baseline_value != 0
                else np.nan
            )
            rows.append(
                {
                    "comparison": comparison_name,
                    "metric": metric,
                    "baseline_model": baseline_label,
                    "challenger_model": challenger_label,
                    "baseline_value": baseline_value,
                    "challenger_value": challenger_value,
                    "absolute_reduction": absolute_reduction,
                    "percent_reduction_vs_baseline": percent_reduction,
                    "winner": challenger_label
                    if challenger_value < baseline_value
                    else baseline_label,
                }
            )
    return pd.DataFrame(rows)


def build_cv_stability_summary(
    artifacts: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        metrics = artifact["best_metrics"]
        row: dict[str, Any] = {
            "model_label": artifact["model_label"],
            "parameter_set_id": artifact["parameter_set_id"],
            "n_folds": int(metrics["fold"].nunique()),
        }
        for metric in METRICS:
            values = pd.to_numeric(metrics[metric], errors="raise")
            mean = float(values.mean())
            std = float(values.std(ddof=1))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
            row[f"{metric}_coefficient_of_variation"] = (
                std / mean if mean != 0 else np.nan
            )
        rows.append(row)
    stability = pd.DataFrame(rows)
    return stability.sort_values(
        f"{PRIMARY_METRIC}_mean",
        kind="mergesort",
    ).reset_index(drop=True)


def build_fold_stability_comparison(
    artifacts: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    combined = []
    for artifact in artifacts:
        frame = artifact["best_metrics"].copy()
        frame["model_label"] = artifact["model_label"]
        frame["parameter_set_id"] = artifact["parameter_set_id"]
        combined.append(frame)
    metrics = pd.concat(combined, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for fold, group in metrics.groupby("fold", sort=True):
        row: dict[str, Any] = {"fold": int(fold)}
        for metric in METRICS:
            winner = group.sort_values(metric, kind="mergesort").iloc[0]
            row[f"{metric}_winner"] = winner["model_label"]
            for _, item in group.sort_values("model_label", kind="mergesort").iterrows():
                column_name = f"{metric}_{item['model_label']}"
                row[column_name] = float(item[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def build_time_cost_computing(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    final_metrics: pd.DataFrame,
    final_runtime: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        model_label = str(artifact["model_label"])
        runtime = artifact["best_runtime"].iloc[0].to_dict()
        metadata = artifact["tuning_metadata"]
        time_cost = dict(metadata.get("time_cost_computing", {}))
        final_metric_row = select_model_row(final_metrics, model_label)
        final_runtime_row = select_model_row(final_runtime, model_label)
        full_tuning_runtime = float(
            time_cost.get("sum_fold_runtime_seconds", np.nan)
        )
        final_model_load = float(final_runtime_row["model_load_time_seconds"])
        final_prediction = float(final_runtime_row["prediction_time_seconds"])
        retraining_train = float(final_runtime_row["retraining_train_time_seconds"])
        known_end_to_end = (
            full_tuning_runtime + retraining_train + final_model_load + final_prediction
        )
        rows.append(
            {
                "model_label": model_label,
                "parameter_set_id": artifact["parameter_set_id"],
                "cv_best_train_time_seconds_sum": float(
                    runtime["train_time_seconds_sum"]
                ),
                "cv_best_train_time_seconds_mean": float(
                    runtime["train_time_seconds_mean"]
                ),
                "cv_best_prediction_time_seconds_sum": float(
                    runtime["prediction_time_seconds_sum"]
                ),
                "cv_best_prediction_time_seconds_mean": float(
                    runtime["prediction_time_seconds_mean"]
                ),
                "cv_best_total_runtime_seconds_sum": float(
                    runtime["total_runtime_seconds_sum"]
                ),
                "full_tuning_runtime_seconds_sum": full_tuning_runtime,
                "full_tuning_train_time_seconds_sum": float(
                    time_cost.get("sum_train_time_seconds", np.nan)
                ),
                "full_tuning_prediction_time_seconds_sum": float(
                    time_cost.get("sum_prediction_time_seconds", np.nan)
                ),
                "retraining_train_time_seconds": retraining_train,
                "final_model_load_time_seconds": final_model_load,
                "final_prediction_time_seconds": final_prediction,
                "final_total_runtime_seconds": float(
                    final_runtime_row["total_runtime_seconds"]
                ),
                "known_end_to_end_runtime_seconds": known_end_to_end,
                "final_mae": float(final_metric_row["mae"]),
                "final_rmse": float(final_metric_row["rmse"]),
                "final_smape": float(final_metric_row["smape"]),
            }
        )
    result = pd.DataFrame(rows)
    result["rank_by_final_mae"] = result["final_mae"].rank(
        method="min",
        ascending=True,
    ).astype(int)
    result["rank_by_final_prediction_time"] = result[
        "final_prediction_time_seconds"
    ].rank(method="min", ascending=True).astype(int)
    result["rank_by_known_end_to_end_runtime"] = result[
        "known_end_to_end_runtime_seconds"
    ].rank(method="min", ascending=True).astype(int)
    return result.sort_values("rank_by_final_mae", kind="mergesort").reset_index(
        drop=True
    )


def build_residual_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_label, group in predictions.groupby("model_label", sort=False):
        residual = pd.to_numeric(group["residual"], errors="raise")
        predicted = pd.to_numeric(group["predicted"], errors="raise")
        actual = pd.to_numeric(group["actual"], errors="raise")
        rows.append(
            {
                "model_label": model_label,
                "n_predictions": int(group.shape[0]),
                "mean_error_actual_minus_predicted": float(residual.mean()),
                "median_error_actual_minus_predicted": float(residual.median()),
                "residual_std": float(residual.std(ddof=1)),
                "mean_absolute_error": float(group["absolute_error"].mean()),
                "rmse": float(np.sqrt(np.square(residual).mean())),
                "overprediction_rate": float((predicted > actual).mean()),
                "underprediction_rate": float((predicted < actual).mean()),
                "max_absolute_error": float(group["absolute_error"].max()),
                "negative_prediction_count": int((predicted < 0).sum()),
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


def build_behavioral_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    actual_reference = predictions.drop_duplicates(subset=[TIMESTAMP_COL])
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


def build_behavioral_error_pivot(behavioral_error: pd.DataFrame) -> pd.DataFrame:
    pivot = behavioral_error.pivot(
        index="segment",
        columns="model_label",
        values="mae",
    ).reset_index()
    if EXPERIMENT_A_BASELINE in pivot.columns and EXPERIMENT_A_CHALLENGER in pivot.columns:
        pivot["mae_percent_reduction_xgb_basic_vs_prophet"] = (
            (pivot[EXPERIMENT_A_BASELINE] - pivot[EXPERIMENT_A_CHALLENGER])
            / pivot[EXPERIMENT_A_BASELINE]
            * 100.0
        )
    if EXPERIMENT_B_BASELINE in pivot.columns and EXPERIMENT_B_CHALLENGER in pivot.columns:
        pivot["mae_percent_reduction_xgb_advanced_vs_basic"] = (
            (pivot[EXPERIMENT_B_BASELINE] - pivot[EXPERIMENT_B_CHALLENGER])
            / pivot[EXPERIMENT_B_BASELINE]
            * 100.0
        )
    segment_order = [
        "all",
        "weekday_local",
        "weekend_local",
        "rush_hour_local_07_09_16_19",
        "night_local_00_05",
        "high_demand_spike_p90",
    ]
    pivot["segment"] = pd.Categorical(
        pivot["segment"],
        categories=segment_order,
        ordered=True,
    )
    return pivot.sort_values("segment", kind="mergesort").reset_index(drop=True)


def save_actual_vs_predicted_plot(predictions: pd.DataFrame, output_path: Path) -> None:
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
    ax.set_title("Comparative Evaluation - Final Test Actual vs Predicted")
    ax.set_xlabel("UTC timestamp")
    ax.set_ylabel(TARGET_COL)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_residual_time_series_plot(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5.5))
    for model_label, group in predictions.groupby("model_label", sort=False):
        group = group.sort_values(TIMESTAMP_COL, kind="mergesort")
        ax.plot(
            group[TIMESTAMP_COL],
            group["residual"],
            label=model_label,
            linewidth=1.0,
            alpha=0.85,
        )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_title("Comparative Evaluation - Final Test Residual Time Series")
    ax.set_xlabel("UTC timestamp")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_residual_distribution_plot(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for model_label, group in predictions.groupby("model_label", sort=False):
        ax.hist(group["residual"], bins=45, alpha=0.5, label=model_label)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_title("Comparative Evaluation - Final Test Residual Distribution")
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_actual_predicted_scatter_plot(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    actual_min = float(predictions["actual"].min())
    actual_max = float(predictions["actual"].max())
    pred_min = float(predictions["predicted"].min())
    pred_max = float(predictions["predicted"].max())
    low = min(actual_min, pred_min)
    high = max(actual_max, pred_max)

    fig, ax = plt.subplots(figsize=(7, 7))
    for model_label, group in predictions.groupby("model_label", sort=False):
        ax.scatter(
            group["actual"],
            group["predicted"],
            s=15,
            alpha=0.45,
            label=model_label,
        )
    ax.plot([low, high], [low, high], color="black", linewidth=1.0)
    ax.set_title("Comparative Evaluation - Actual vs Predicted Scatter")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_accuracy_runtime_plot(
    final_ranking: pd.DataFrame,
    time_cost: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_df = final_ranking[["model_label", "mae"]].merge(
        time_cost[["model_label", "final_prediction_time_seconds"]],
        on="model_label",
        how="inner",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        plot_df["final_prediction_time_seconds"],
        plot_df["mae"],
        s=90,
        alpha=0.85,
    )
    for _, row in plot_df.iterrows():
        ax.annotate(
            str(row["model_label"]),
            (row["final_prediction_time_seconds"], row["mae"]),
            xytext=(6, 4),
            textcoords="offset points",
        )
    ax.set_title("Accuracy vs Final Prediction Time")
    ax.set_xlabel("Final prediction time (seconds)")
    ax.set_ylabel("Final test MAE")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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
    ax.set_title("Comparative Evaluation - MAE by Behavior Segment")
    ax.set_xlabel("Behavior segment")
    ax.set_ylabel("MAE")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_horizon_error_plot(horizon_error: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    for model_label, group in horizon_error.groupby("model_label", sort=False):
        ax.plot(
            group["horizon_step"],
            group["mae"],
            marker="o",
            linewidth=1.4,
            label=model_label,
        )
    ax.set_title("Comparative Evaluation - MAE by Recursive Horizon Step")
    ax.set_xlabel("Horizon step")
    ax.set_ylabel("MAE")
    ax.set_xticks(list(range(1, FORECAST_HORIZON + 1)))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_cv_fold_mae_plot(
    artifacts: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for artifact in artifacts:
        metrics = artifact["best_metrics"].sort_values("fold", kind="mergesort")
        ax.plot(
            metrics["fold"],
            metrics["mae"],
            marker="o",
            linewidth=1.4,
            label=artifact["model_label"],
        )
    ax.set_title("Comparative Evaluation - CV Fold MAE")
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_comparative_report(
    *,
    metadata: Mapping[str, Any],
    final_ranking: pd.DataFrame,
    metric_rankings_long: pd.DataFrame,
    improvement_summary: pd.DataFrame,
    cv_stability: pd.DataFrame,
    fold_stability: pd.DataFrame,
    time_cost: pd.DataFrame,
    residual_summary: pd.DataFrame,
    horizon_error: pd.DataFrame,
    behavioral_pivot: pd.DataFrame,
    skip_plots: bool,
    outputs: Mapping[str, Path],
) -> str:
    winner = metadata["winner_by_primary_metric"]
    exp_a_mae = select_improvement_row(
        improvement_summary,
        comparison="Experiment A final: XGBoost-Basic vs Prophet",
        metric="mae",
    )
    exp_a_rmse = select_improvement_row(
        improvement_summary,
        comparison="Experiment A final: XGBoost-Basic vs Prophet",
        metric="rmse",
    )
    exp_a_smape = select_improvement_row(
        improvement_summary,
        comparison="Experiment A final: XGBoost-Basic vs Prophet",
        metric="smape",
    )
    exp_b_mae = select_improvement_row(
        improvement_summary,
        comparison="Experiment B final: XGBoost-Advanced vs XGBoost-Basic",
        metric="mae",
    )
    exp_b_rmse = select_improvement_row(
        improvement_summary,
        comparison="Experiment B final: XGBoost-Advanced vs XGBoost-Basic",
        metric="rmse",
    )
    exp_b_smape = select_improvement_row(
        improvement_summary,
        comparison="Experiment B final: XGBoost-Advanced vs XGBoost-Basic",
        metric="smape",
    )
    fold_mae_wins = fold_stability["mae_winner"].value_counts().to_dict()
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
    metric_rank_view = metric_rankings_long.pivot(
        index="rank",
        columns="metric",
        values="model_label",
    ).reset_index()

    final_view_columns = [
        f"rank_by_{PRIMARY_METRIC}",
        "model_label",
        "parameter_set_id",
        "mae",
        "rmse",
        "mape",
        "smape",
        "prediction_time_seconds",
        "retraining_train_time_seconds",
        "negative_prediction_count",
    ]
    time_cost_view_columns = [
        "model_label",
        "cv_best_total_runtime_seconds_sum",
        "full_tuning_runtime_seconds_sum",
        "retraining_train_time_seconds",
        "final_model_load_time_seconds",
        "final_prediction_time_seconds",
        "known_end_to_end_runtime_seconds",
        "rank_by_final_mae",
        "rank_by_final_prediction_time",
    ]
    cv_view_columns = [
        "model_label",
        "parameter_set_id",
        "mae_mean",
        "mae_std",
        "mae_coefficient_of_variation",
        "rmse_mean",
        "rmse_std",
        "smape_mean",
        "smape_std",
    ]

    lines = [
        "# Comparative Evaluation",
        "",
        f"Run UTC: {metadata['started_at_utc']}",
        "",
        "## Scope",
        "",
        (
            "Tahap ini membandingkan Prophet, XGBoost-Basic, dan "
            "XGBoost-Advanced memakai artefak yang sudah selesai dibuat. "
            "Tidak ada tuning, retraining, atau prediksi final test baru pada "
            "tahap ini."
        ),
        "",
        "## Leakage Guardrail",
        "",
        (
            "Final test tetap dipakai hanya sebagai label evaluasi dari output "
            "tahap 14. Semua prediksi final memiliki "
            "`used_actual_future_for_features=False`, dan metric final memiliki "
            "`final_test_used_for_tuning=False`."
        ),
        "",
        "## Final Benchmark Ranking",
        "",
        dataframe_to_markdown(
            final_ranking[[column for column in final_view_columns if column in final_ranking.columns]],
            float_digits=6,
        ),
        "",
        "Ranking tiap metric:",
        "",
        dataframe_to_markdown(metric_rank_view, float_digits=6),
        "",
        "## Research Question A",
        "",
        (
            "Apakah machine learning sederhana mampu mengungguli forecasting "
            "klasik? Pada final test, XGBoost-Basic mengungguli Prophet. "
            f"MAE turun {exp_a_mae['percent_reduction_vs_baseline']:.2f}%, "
            f"RMSE turun {exp_a_rmse['percent_reduction_vs_baseline']:.2f}%, "
            f"dan sMAPE turun {exp_a_smape['percent_reduction_vs_baseline']:.2f}%."
        ),
        "",
        "## Research Question B",
        "",
        (
            "Apakah advanced feature engineering meningkatkan performa XGBoost? "
            "Pada final test ini, XGBoost-Advanced belum mengungguli "
            "XGBoost-Basic pada metric utama. "
            f"Perubahan MAE advanced terhadap basic adalah "
            f"{exp_b_mae['percent_reduction_vs_baseline']:.2f}%, RMSE "
            f"{exp_b_rmse['percent_reduction_vs_baseline']:.2f}%, dan sMAPE "
            f"{exp_b_smape['percent_reduction_vs_baseline']:.2f}%."
        ),
        "",
        "## Improvement Summary",
        "",
        dataframe_to_markdown(improvement_summary, float_digits=6),
        "",
        "## Stability Across CV Folds",
        "",
        f"Winner by fold berdasarkan MAE: {json.dumps(fold_mae_wins, sort_keys=True)}.",
        "",
        dataframe_to_markdown(cv_stability[cv_view_columns], float_digits=6),
        "",
        dataframe_to_markdown(
            fold_stability[
                [
                    "fold",
                    "mae_Prophet",
                    "mae_XGBoost-Basic",
                    "mae_XGBoost-Advanced",
                    "mae_winner",
                ]
            ],
            float_digits=6,
        ),
        "",
        "## Behavioral Comparison",
        "",
        dataframe_to_markdown(behavioral_pivot, float_digits=6),
        "",
        "## Residual And Horizon Behavior",
        "",
        dataframe_to_markdown(residual_summary, float_digits=6),
        "",
        "Best horizon step by MAE:",
        "",
        dataframe_to_markdown(best_horizon, float_digits=6),
        "",
        "Worst horizon step by MAE:",
        "",
        dataframe_to_markdown(worst_horizon, float_digits=6),
        "",
        "## Time Cost Computing",
        "",
        dataframe_to_markdown(
            time_cost[
                [column for column in time_cost_view_columns if column in time_cost.columns]
            ],
            float_digits=6,
        ),
        "",
        "## Interpretation",
        "",
        (
            f"Model terbaik secara final test berdasarkan {PRIMARY_METRIC} adalah "
            f"{winner}. XGBoost-Basic juga menjadi pilihan paling seimbang pada "
            "benchmark ini: akurasinya terbaik, bias rata-ratanya kecil, dan "
            "waktu prediksi finalnya hampir sama dengan Prophet meskipun biaya "
            "tuning dan retrainingnya lebih besar."
        ),
        "",
        (
            "Advanced feature set tidak otomatis meningkatkan generalisasi. "
            "Tambahan lag, rolling statistics, dan calendar feature memberi "
            "kapasitas model yang lebih besar, tetapi pada final test performanya "
            "lebih lemah daripada XGBoost-Basic, terutama terlihat dari RMSE dan "
            "max error yang lebih besar."
        ),
        "",
        (
            "Prophet memiliki biaya prediksi final yang kompetitif, tetapi error "
            "final test jauh lebih tinggi dan menghasilkan sejumlah prediksi "
            "negatif. Temuan ini perlu dibahas lebih rinci pada tahap error "
            "pattern analysis."
        ),
        "",
        "## Output Files",
        "",
        f"- Final metric ranking: `{outputs['final_metric_ranking']}`",
        f"- Improvement summary: `{outputs['final_improvement_summary']}`",
        f"- CV stability summary: `{outputs['cv_stability_summary']}`",
        f"- Fold stability comparison: `{outputs['fold_stability_comparison']}`",
        f"- Time cost computing: `{outputs['time_cost_computing']}`",
        f"- Behavioral error summary: `{outputs['behavioral_error_summary']}`",
        f"- Report mirror: `{outputs['report']}`",
    ]
    if not skip_plots:
        lines.extend(
            [
                f"- Actual vs predicted plot: `{outputs['actual_vs_predicted_plot']}`",
                f"- Residual time series plot: `{outputs['residual_time_series_plot']}`",
                f"- Residual distribution plot: `{outputs['residual_distribution_plot']}`",
                f"- Scatter plot: `{outputs['actual_predicted_scatter_plot']}`",
                f"- Accuracy/runtime plot: `{outputs['accuracy_runtime_plot']}`",
                f"- Behavioral error plot: `{outputs['behavioral_error_plot']}`",
                f"- Horizon error plot: `{outputs['horizon_error_plot']}`",
                f"- CV fold MAE plot: `{outputs['cv_fold_mae_plot']}`",
            ]
        )
    return "\n".join(lines) + "\n"


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
    selected = frame[frame["parameter_set_id"].astype(str) == str(parameter_set_id)]
    if selected.empty:
        raise ValueError(
            f"Parameter set {parameter_set_id} tidak ditemukan dalam {source}"
        )
    return selected.copy()


def select_model_row(frame: pd.DataFrame, model_label: str) -> pd.Series:
    selected = frame[frame["model_label"].astype(str) == str(model_label)]
    if selected.empty:
        raise ValueError(f"Model tidak ditemukan: {model_label}")
    return selected.iloc[0]


def select_final_winner(final_metrics: pd.DataFrame, *, metric: str) -> str:
    if final_metrics.empty:
        raise ValueError("Final metrics kosong.")
    return str(final_metrics.sort_values(metric, kind="mergesort").iloc[0]["model_label"])


def select_improvement_row(
    improvement_summary: pd.DataFrame,
    *,
    comparison: str,
    metric: str,
) -> pd.Series:
    selected = improvement_summary[
        (improvement_summary["comparison"] == comparison)
        & (improvement_summary["metric"] == metric)
    ]
    if selected.empty:
        raise ValueError(f"Improvement row tidak ditemukan: {comparison} / {metric}")
    return selected.iloc[0]


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
        raise ValueError(f"Kolom boolean {column_name} tidak valid: {invalid}")
    return parsed.astype(bool)


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
        lines.append(
            "| "
            + " | ".join(format_cell(row[column]) for column in df.columns)
            + " |"
        )
    return "\n".join(lines)


def stringify_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    return {key: str(value) for key, value in paths.items()}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

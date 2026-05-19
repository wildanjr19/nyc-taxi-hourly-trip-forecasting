"""
Script tahap 16: Error Pattern Analysis.

Scope:
- Membaca artefak final test dari tahap 14.
- Tidak melakukan tuning, retraining, atau prediksi ulang final test.
- Menganalisis pola error temporal, residual, dan extreme events.
- Menyimpan tables, plots, report Markdown, metadata, dan time cost computing.

Contoh:
    python -m src.experiments.error_analysis
    python -m src.experiments.error_analysis --skip-plots
"""

from __future__ import annotations

import argparse
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
    REPORTS_DIR,
    TARGET_COL,
    TIMESTAMP_COL,
    ensure_dirs,
)
from src.metrics import compute_all_metrics
from src.tracking import (
    append_experiment_run,
    elapsed_seconds,
    log_runtime,
    make_runtime_record,
    save_experiment_metadata,
    start_timer,
    utc_now_iso,
)


EXPERIMENT_NAME = "error_pattern_analysis"
OUTPUT_DIR = EXPERIMENTS_DIR / "error_analysis"
REPORT_PATH = REPORTS_DIR / "error_analysis.md"

FINAL_METRICS_PATH = FINAL_TEST_DIR / "metrics" / "final_metrics.csv"
FINAL_RUNTIME_PATH = FINAL_TEST_DIR / "metrics" / "final_runtime.csv"
FINAL_PREDICTIONS_PATH = FINAL_TEST_DIR / "predictions" / "final_predictions.csv"

MODEL_ORDER = ["Prophet", "XGBoost-Basic", "XGBoost-Advanced"]
LOCAL_DAY_NAMES = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run error pattern analysis on final test predictions."
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip matplotlib figure generation.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Jumlah timestamp top error yang disimpan per model. Default: 20.",
    )
    parser.add_argument(
        "--acf-max-lag",
        type=int,
        default=48,
        help="Lag maksimum residual autocorrelation. Default: 48.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    metadata = run_error_analysis(
        skip_plots=args.skip_plots,
        top_n=args.top_n,
        acf_max_lag=args.acf_max_lag,
    )
    print("Error pattern analysis selesai.")
    print(f"Status: {metadata['status']}")
    print(f"Worst segment overall: {metadata['key_findings']['worst_segment_overall']}")
    print(f"Output dir: {metadata['outputs']['base']}")
    print(f"Report: {metadata['outputs']['report']}")


def run_error_analysis(
    *,
    skip_plots: bool = False,
    top_n: int = 20,
    acf_max_lag: int = 48,
) -> dict[str, Any]:
    if int(top_n) <= 0:
        raise ValueError("top_n harus > 0.")
    if int(acf_max_lag) <= 0:
        raise ValueError("acf_max_lag harus > 0.")

    timer = start_timer()
    started_at = utc_now_iso()
    status = "success"
    error_message = ""
    outputs = error_analysis_output_paths()

    try:
        ensure_dirs()
        ensure_error_analysis_dirs(outputs)

        final_metrics = load_final_metrics(FINAL_METRICS_PATH)
        final_runtime = load_final_runtime(FINAL_RUNTIME_PATH)
        predictions = load_final_predictions(FINAL_PREDICTIONS_PATH)
        validate_final_artifacts(final_metrics, final_runtime, predictions)

        enriched = enrich_predictions(predictions)
        model_summary = build_model_summary(enriched, final_metrics, final_runtime)
        hourly_error = build_temporal_error_summary(enriched, "local_hour")
        day_error = build_temporal_error_summary(enriched, "local_day_of_week")
        weekday_weekend = build_weekday_weekend_summary(enriched)
        behavioral_error = build_behavioral_error_summary(enriched)
        residual_summary = build_residual_distribution_summary(enriched)
        residual_acf = build_residual_autocorrelation(enriched, max_lag=acf_max_lag)
        horizon_error = build_horizon_error_summary(enriched)
        origin_block_error = build_origin_block_error_summary(enriched)
        extreme_summary = build_extreme_event_summary(enriched)
        high_demand_errors = build_demand_tail_error_table(enriched, tail="high")
        low_demand_errors = build_demand_tail_error_table(enriched, tail="low")
        top_errors = build_top_absolute_errors(enriched, top_n=top_n)
        signed_extremes = build_signed_extreme_errors(enriched, top_n=top_n)
        negative_predictions = build_negative_prediction_table(enriched)
        timestamp_consensus = build_timestamp_consensus_error(enriched, top_n=top_n)
        key_findings = build_key_findings(
            model_summary=model_summary,
            behavioral_error=behavioral_error,
            residual_summary=residual_summary,
            residual_acf=residual_acf,
            high_demand_errors=high_demand_errors,
            top_errors=top_errors,
            negative_predictions=negative_predictions,
        )

        model_summary.to_csv(outputs["model_summary"], index=False)
        hourly_error.to_csv(outputs["hourly_error"], index=False)
        day_error.to_csv(outputs["day_of_week_error"], index=False)
        weekday_weekend.to_csv(outputs["weekday_weekend_error"], index=False)
        behavioral_error.to_csv(outputs["behavioral_error"], index=False)
        residual_summary.to_csv(outputs["residual_distribution_summary"], index=False)
        residual_acf.to_csv(outputs["residual_autocorrelation"], index=False)
        horizon_error.to_csv(outputs["horizon_error"], index=False)
        origin_block_error.to_csv(outputs["origin_block_error"], index=False)
        extreme_summary.to_csv(outputs["extreme_event_summary"], index=False)
        high_demand_errors.to_csv(outputs["high_demand_errors"], index=False)
        low_demand_errors.to_csv(outputs["low_demand_errors"], index=False)
        top_errors.to_csv(outputs["top_absolute_errors"], index=False)
        signed_extremes.to_csv(outputs["signed_extreme_errors"], index=False)
        negative_predictions.to_csv(outputs["negative_predictions"], index=False)
        timestamp_consensus.to_csv(outputs["timestamp_consensus_errors"], index=False)

        if not skip_plots:
            save_hourly_error_plot(hourly_error, outputs["hourly_error_plot"])
            save_day_of_week_error_plot(day_error, outputs["day_error_plot"])
            save_behavioral_error_plot(
                behavioral_error,
                outputs["behavioral_error_plot"],
            )
            save_residual_distribution_plot(
                enriched,
                outputs["residual_distribution_plot"],
            )
            save_residual_acf_plot(residual_acf, outputs["residual_acf_plot"])
            save_high_demand_prediction_plot(
                high_demand_errors,
                outputs["high_demand_prediction_plot"],
            )
            save_top_error_plot(top_errors, outputs["top_error_plot"])
            save_origin_block_error_plot(
                origin_block_error,
                outputs["origin_block_error_plot"],
            )

        total_runtime_seconds = elapsed_seconds(timer)
        runtime_summary = pd.DataFrame(
            [
                {
                    "experiment_name": EXPERIMENT_NAME,
                    "status": status,
                    "analysis_runtime_seconds": total_runtime_seconds,
                    "n_prediction_rows": int(enriched.shape[0]),
                    "n_models": int(enriched["model_label"].nunique()),
                    "top_n": int(top_n),
                    "acf_max_lag": int(acf_max_lag),
                    "plots_generated": not skip_plots,
                }
            ]
        )
        runtime_summary.to_csv(outputs["runtime_summary"], index=False)

        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "stage": "error_pattern_analysis",
            "status": status,
            "error_message": error_message,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "total_runtime_seconds": total_runtime_seconds,
            "primary_metric": PRIMARY_METRIC,
            "forecast_horizon_hours": int(FORECAST_HORIZON),
            "local_timezone": LOCAL_TZ,
            "models_analyzed": sorted(enriched["model_label"].unique().tolist()),
            "n_prediction_rows": int(enriched.shape[0]),
            "top_n": int(top_n),
            "acf_max_lag": int(acf_max_lag),
            "input_artifacts": {
                "final_metrics": str(FINAL_METRICS_PATH),
                "final_runtime": str(FINAL_RUNTIME_PATH),
                "final_predictions": str(FINAL_PREDICTIONS_PATH),
            },
            "leakage_guardrail": [
                "Tahap ini hanya membaca prediksi final test yang sudah dibuat.",
                "Tidak ada tuning, retraining, atau prediksi final test baru.",
                "Actual final test dipakai hanya sebagai label evaluasi dan konteks error.",
                "Final predictions wajib memiliki used_actual_future_for_features=False.",
            ],
            "key_findings": key_findings,
            "time_cost_computing": {
                "analysis_runtime_seconds": total_runtime_seconds,
                "training_time_seconds": 0.0,
                "prediction_time_seconds": 0.0,
                "note": (
                    "Tahap 16 adalah post-hoc analysis; tidak ada training atau "
                    "prediction baru."
                ),
            },
            "outputs": stringify_paths(outputs),
        }

        report_text = render_error_analysis_report(
            metadata=metadata,
            model_summary=model_summary,
            hourly_error=hourly_error,
            day_error=day_error,
            weekday_weekend=weekday_weekend,
            behavioral_error=behavioral_error,
            residual_summary=residual_summary,
            residual_acf=residual_acf,
            horizon_error=horizon_error,
            origin_block_error=origin_block_error,
            extreme_summary=extreme_summary,
            high_demand_errors=high_demand_errors,
            low_demand_errors=low_demand_errors,
            top_errors=top_errors,
            signed_extremes=signed_extremes,
            negative_predictions=negative_predictions,
            timestamp_consensus=timestamp_consensus,
            skip_plots=skip_plots,
            outputs=outputs,
        )
        outputs["summary"].write_text(report_text, encoding="utf-8")
        outputs["report"].write_text(report_text, encoding="utf-8")
        save_experiment_metadata(metadata, outputs["metadata"])
        log_runtime(
            make_runtime_record(
                experiment_name=EXPERIMENT_NAME,
                model_name="all_final_models",
                feature_set="mixed",
                n_prediction_rows=int(enriched.shape[0]),
                train_time_seconds=0.0,
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
            "stage": "error_pattern_analysis",
            "status": status,
            "error_message": error_message,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "total_runtime_seconds": total_runtime_seconds,
            "outputs": stringify_paths(outputs),
        }
        log_runtime(
            make_runtime_record(
                experiment_name=EXPERIMENT_NAME,
                model_name="all_final_models",
                feature_set="mixed",
                train_time_seconds=0.0,
                prediction_time_seconds=0.0,
                total_runtime_seconds=total_runtime_seconds,
                status=status,
                error_message=error_message,
            )
        )
        append_experiment_run(failure_metadata)
        raise


def error_analysis_output_paths() -> dict[str, Path]:
    metrics_dir = OUTPUT_DIR / "metrics"
    figures_dir = OUTPUT_DIR / "figures"
    summaries_dir = OUTPUT_DIR / "summaries"
    return {
        "base": OUTPUT_DIR,
        "metrics": metrics_dir,
        "figures": figures_dir,
        "summaries": summaries_dir,
        "model_summary": metrics_dir / "model_error_summary.csv",
        "hourly_error": metrics_dir / "temporal_error_by_hour.csv",
        "day_of_week_error": metrics_dir / "temporal_error_by_day_of_week.csv",
        "weekday_weekend_error": metrics_dir / "weekday_weekend_error_summary.csv",
        "behavioral_error": metrics_dir / "behavioral_error_summary.csv",
        "residual_distribution_summary": metrics_dir
        / "residual_distribution_summary.csv",
        "residual_autocorrelation": metrics_dir / "residual_autocorrelation.csv",
        "horizon_error": metrics_dir / "horizon_error_summary.csv",
        "origin_block_error": metrics_dir / "origin_block_error_summary.csv",
        "extreme_event_summary": metrics_dir / "extreme_event_summary.csv",
        "high_demand_errors": metrics_dir / "high_demand_error_details.csv",
        "low_demand_errors": metrics_dir / "low_demand_error_details.csv",
        "top_absolute_errors": metrics_dir / "top_absolute_errors.csv",
        "signed_extreme_errors": metrics_dir / "signed_extreme_errors.csv",
        "negative_predictions": metrics_dir / "negative_predictions.csv",
        "timestamp_consensus_errors": metrics_dir / "timestamp_consensus_errors.csv",
        "runtime_summary": metrics_dir / "runtime_summary.csv",
        "hourly_error_plot": figures_dir / "mae_by_hour.png",
        "day_error_plot": figures_dir / "mae_by_day_of_week.png",
        "behavioral_error_plot": figures_dir / "mae_by_behavior_segment.png",
        "residual_distribution_plot": figures_dir / "residual_distribution.png",
        "residual_acf_plot": figures_dir / "residual_autocorrelation.png",
        "high_demand_prediction_plot": figures_dir / "high_demand_actual_vs_predicted.png",
        "top_error_plot": figures_dir / "top_error_timestamps.png",
        "origin_block_error_plot": figures_dir / "mae_by_origin_block.png",
        "metadata": OUTPUT_DIR / "experiment_metadata.json",
        "summary": summaries_dir / "error_analysis_summary.md",
        "report": REPORT_PATH,
    }


def ensure_error_analysis_dirs(paths: Mapping[str, Path]) -> None:
    for key in ["base", "metrics", "figures", "summaries"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_final_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Final metrics tidak ditemukan: {path}")
    frame = pd.read_csv(path)
    required = {
        "model_label",
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
    return order_by_model(frame)


def load_final_runtime(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Final runtime tidak ditemukan: {path}")
    frame = pd.read_csv(path)
    required = {
        "model_label",
        "parameter_set_id",
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
    return order_by_model(frame)


def load_final_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Final predictions tidak ditemukan: {path}")
    frame = pd.read_csv(path)
    required = {
        TIMESTAMP_COL,
        "actual",
        "predicted",
        "model_label",
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
    local_timestamp = frame[TIMESTAMP_COL].dt.tz_convert(LOCAL_TZ)
    if "local_hour" not in frame.columns:
        frame["local_hour"] = local_timestamp.dt.hour
    else:
        frame["local_hour"] = pd.to_numeric(frame["local_hour"], errors="raise").astype(
            int
        )
    if "local_day_of_week" not in frame.columns:
        frame["local_day_of_week"] = local_timestamp.dt.dayofweek
    else:
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
    if "local_month" not in frame.columns:
        frame["local_month"] = local_timestamp.dt.month
    else:
        frame["local_month"] = pd.to_numeric(
            frame["local_month"],
            errors="raise",
        ).astype(int)
    frame["local_day_name"] = frame["local_day_of_week"].map(LOCAL_DAY_NAMES)
    return order_by_model(frame).sort_values(
        ["model_order", TIMESTAMP_COL],
        kind="mergesort",
    ).drop(columns=["model_order"]).reset_index(drop=True)


def validate_final_artifacts(
    final_metrics: pd.DataFrame,
    final_runtime: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    labels = set(predictions["model_label"].astype(str).unique())
    metric_labels = set(final_metrics["model_label"].astype(str).unique())
    runtime_labels = set(final_runtime["model_label"].astype(str).unique())
    if labels != metric_labels:
        raise ValueError(
            "Model pada final_predictions dan final_metrics tidak sama. "
            f"predictions={sorted(labels)}, metrics={sorted(metric_labels)}"
        )
    if labels != runtime_labels:
        raise ValueError(
            "Model pada final_predictions dan final_runtime tidak sama. "
            f"predictions={sorted(labels)}, runtime={sorted(runtime_labels)}"
        )
    if predictions["used_actual_future_for_features"].any():
        raise ValueError("Final predictions menandai leakage flag aktif.")
    if final_metrics["used_actual_future_for_features"].any():
        raise ValueError("Final metrics menandai used_actual_future_for_features=True.")
    if final_metrics["final_test_used_for_tuning"].any():
        raise ValueError("Final metrics menandai final_test_used_for_tuning=True.")
    if final_runtime["status"].astype(str).str.lower().ne("success").any():
        raise ValueError("Final runtime mengandung status bukan success.")
    if predictions["horizon_step"].min() < 1:
        raise ValueError("horizon_step minimum harus >= 1.")
    if predictions["horizon_step"].max() > FORECAST_HORIZON:
        raise ValueError("horizon_step maksimum melebihi FORECAST_HORIZON.")

    reference: Optional[pd.DataFrame] = None
    for model_label, group in predictions.groupby("model_label", sort=False):
        group = group.sort_values(TIMESTAMP_COL, kind="mergesort")
        if group[TIMESTAMP_COL].duplicated().any():
            raise ValueError(f"Duplicate timestamp pada model {model_label}.")
        candidate = group[[TIMESTAMP_COL, "actual"]].reset_index(drop=True)
        if reference is None:
            reference = candidate
        elif not candidate.equals(reference):
            raise ValueError(
                f"Timestamp/actual {model_label} tidak sejajar dengan model lain."
            )


def enrich_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["absolute_percentage_error"] = np.where(
        frame["actual"] != 0,
        frame["absolute_error"] / frame["actual"].abs() * 100.0,
        np.nan,
    )
    denominator = frame["actual"].abs() + frame["predicted"].abs()
    frame["symmetric_absolute_percentage_error"] = np.where(
        denominator != 0,
        2.0 * frame["absolute_error"] / denominator * 100.0,
        0.0,
    )
    frame["error_direction"] = np.select(
        [frame["predicted"] > frame["actual"], frame["predicted"] < frame["actual"]],
        ["overprediction", "underprediction"],
        default="exact",
    )
    frame["period_type"] = np.where(
        frame["local_is_weekend"].astype(int).eq(1),
        "weekend",
        "weekday",
    )
    frame["time_of_day_segment"] = pd.cut(
        frame["local_hour"],
        bins=[-1, 5, 9, 15, 19, 23],
        labels=["night_00_05", "morning_06_09", "midday_10_15", "evening_16_19", "late_20_23"],
    ).astype(str)
    frame["is_rush_hour"] = frame["local_hour"].isin([7, 8, 9, 16, 17, 18, 19])
    frame["is_night"] = frame["local_hour"].between(0, 5)

    actual_reference = frame.drop_duplicates(subset=[TIMESTAMP_COL])
    high_threshold = float(actual_reference["actual"].quantile(0.90))
    low_threshold = float(actual_reference["actual"].quantile(0.10))
    frame["is_high_demand_p90"] = frame["actual"] >= high_threshold
    frame["is_low_demand_p10"] = frame["actual"] <= low_threshold
    frame["actual_demand_band"] = np.select(
        [frame["is_high_demand_p90"], frame["is_low_demand_p10"]],
        ["high_demand_p90", "low_demand_p10"],
        default="middle_demand",
    )
    return frame


def build_model_summary(
    predictions: pd.DataFrame,
    final_metrics: pd.DataFrame,
    final_runtime: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_label, group in predictions.groupby("model_label", sort=False):
        metrics_row = select_model_row(final_metrics, model_label)
        runtime_row = select_model_row(final_runtime, model_label)
        residual = pd.to_numeric(group["residual"], errors="raise")
        predicted = pd.to_numeric(group["predicted"], errors="raise")
        actual = pd.to_numeric(group["actual"], errors="raise")
        rows.append(
            {
                "model_label": model_label,
                "parameter_set_id": metrics_row.get("parameter_set_id", ""),
                "n_predictions": int(group.shape[0]),
                "mae": float(metrics_row["mae"]),
                "rmse": float(metrics_row["rmse"]),
                "mape": float(metrics_row["mape"]),
                "smape": float(metrics_row["smape"]),
                "mean_error_actual_minus_predicted": float(residual.mean()),
                "median_error_actual_minus_predicted": float(residual.median()),
                "residual_std": float(residual.std(ddof=1)),
                "overprediction_rate": float((predicted > actual).mean()),
                "underprediction_rate": float((predicted < actual).mean()),
                "negative_prediction_count": int((predicted < 0).sum()),
                "max_absolute_error": float(group["absolute_error"].max()),
                "prediction_time_seconds": float(runtime_row["prediction_time_seconds"]),
                "total_runtime_seconds": float(runtime_row["total_runtime_seconds"]),
            }
        )
    result = pd.DataFrame(rows)
    result[f"rank_by_{PRIMARY_METRIC}"] = result[PRIMARY_METRIC].rank(
        method="min",
        ascending=True,
    ).astype(int)
    return result.sort_values(
        f"rank_by_{PRIMARY_METRIC}",
        kind="mergesort",
    ).reset_index(drop=True)


def build_temporal_error_summary(
    predictions: pd.DataFrame,
    temporal_column: str,
) -> pd.DataFrame:
    if temporal_column not in predictions.columns:
        raise ValueError(f"Kolom temporal tidak ditemukan: {temporal_column}")
    rows: list[dict[str, Any]] = []
    for (model_label, value), group in predictions.groupby(
        ["model_label", temporal_column],
        sort=True,
    ):
        row = {
            "model_label": model_label,
            temporal_column: value,
            "n_predictions": int(group.shape[0]),
        }
        row.update(metric_dict_for_group(group))
        if temporal_column == "local_day_of_week":
            row["local_day_name"] = LOCAL_DAY_NAMES[int(value)]
        rows.append(row)
    return order_by_model(pd.DataFrame(rows)).drop(columns=["model_order"])


def build_weekday_weekend_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["weekday_weekend"] = np.where(
        frame["local_is_weekend"].astype(int).eq(1),
        "weekend",
        "weekday",
    )
    return build_temporal_error_summary(frame, "weekday_weekend")


def build_behavioral_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    actual_reference = predictions.drop_duplicates(subset=[TIMESTAMP_COL])
    spike_threshold = float(actual_reference["actual"].quantile(0.90))
    low_threshold = float(actual_reference["actual"].quantile(0.10))
    segments = {
        "all": pd.Series(True, index=predictions.index),
        "rush_hour_local_07_09_16_19": predictions["is_rush_hour"],
        "night_local_00_05": predictions["is_night"],
        "weekday_local": predictions["local_is_weekend"].astype(int).eq(0),
        "weekend_local": predictions["local_is_weekend"].astype(int).eq(1),
        "high_demand_spike_p90": predictions["actual"] >= spike_threshold,
        "low_demand_p10": predictions["actual"] <= low_threshold,
    }
    rows: list[dict[str, Any]] = []
    for model_label, model_df in predictions.groupby("model_label", sort=False):
        for segment_name, mask in segments.items():
            group = model_df[mask.loc[model_df.index]]
            if group.empty:
                continue
            row = {
                "model_label": model_label,
                "segment": segment_name,
                "n_predictions": int(group.shape[0]),
                "spike_threshold_p90_actual": spike_threshold
                if segment_name == "high_demand_spike_p90"
                else np.nan,
                "low_threshold_p10_actual": low_threshold
                if segment_name == "low_demand_p10"
                else np.nan,
            }
            row.update(metric_dict_for_group(group))
            rows.append(row)
    return order_by_model(pd.DataFrame(rows)).drop(columns=["model_order"])


def build_residual_distribution_summary(predictions: pd.DataFrame) -> pd.DataFrame:
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
                "residual_skew": float(residual.skew()),
                "residual_kurtosis": float(residual.kurtosis()),
                "residual_p05": float(residual.quantile(0.05)),
                "residual_p25": float(residual.quantile(0.25)),
                "residual_p75": float(residual.quantile(0.75)),
                "residual_p95": float(residual.quantile(0.95)),
                "absolute_error_p50": float(absolute_error.quantile(0.50)),
                "absolute_error_p90": float(absolute_error.quantile(0.90)),
                "absolute_error_p95": float(absolute_error.quantile(0.95)),
                "absolute_error_p99": float(absolute_error.quantile(0.99)),
                "overprediction_rate": float((predicted > actual).mean()),
                "underprediction_rate": float((predicted < actual).mean()),
                "negative_prediction_count": int((predicted < 0).sum()),
            }
        )
    return order_by_model(pd.DataFrame(rows)).drop(columns=["model_order"])


def build_residual_autocorrelation(
    predictions: pd.DataFrame,
    *,
    max_lag: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_label, group in predictions.groupby("model_label", sort=False):
        residual = (
            group.sort_values(TIMESTAMP_COL, kind="mergesort")["residual"]
            .astype(float)
            .reset_index(drop=True)
        )
        max_model_lag = min(int(max_lag), residual.shape[0] - 1)
        for lag in range(1, max_model_lag + 1):
            rows.append(
                {
                    "model_label": model_label,
                    "lag": int(lag),
                    "residual_autocorrelation": float(residual.autocorr(lag=lag)),
                    "is_hourly_lag_24": int(lag == 24),
                    "is_two_day_lag_48": int(lag == 48),
                }
            )
    return order_by_model(pd.DataFrame(rows)).drop(columns=["model_order"])


def build_horizon_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    return build_temporal_error_summary(predictions, "horizon_step")


def build_origin_block_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_label, origin_block), group in predictions.groupby(
        ["model_label", "origin_block"],
        sort=True,
    ):
        block_start = group[TIMESTAMP_COL].min()
        block_end = group[TIMESTAMP_COL].max()
        row = {
            "model_label": model_label,
            "origin_block": int(origin_block),
            "block_start": block_start.isoformat(),
            "block_end": block_end.isoformat(),
            "n_predictions": int(group.shape[0]),
        }
        row.update(metric_dict_for_group(group))
        rows.append(row)
    return order_by_model(pd.DataFrame(rows)).drop(columns=["model_order"])


def build_extreme_event_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    actual_reference = predictions.drop_duplicates(subset=[TIMESTAMP_COL])
    thresholds = {
        "low_demand_p10": (
            "<=",
            float(actual_reference["actual"].quantile(0.10)),
        ),
        "high_demand_p90": (
            ">=",
            float(actual_reference["actual"].quantile(0.90)),
        ),
        "high_demand_p95": (
            ">=",
            float(actual_reference["actual"].quantile(0.95)),
        ),
    }
    rows: list[dict[str, Any]] = []
    for model_label, model_df in predictions.groupby("model_label", sort=False):
        for event_name, (operator, threshold) in thresholds.items():
            if operator == ">=":
                group = model_df[model_df["actual"] >= threshold]
            else:
                group = model_df[model_df["actual"] <= threshold]
            row = {
                "model_label": model_label,
                "event": event_name,
                "threshold_operator": operator,
                "actual_threshold": threshold,
                "n_predictions": int(group.shape[0]),
            }
            row.update(metric_dict_for_group(group))
            rows.append(row)
    return order_by_model(pd.DataFrame(rows)).drop(columns=["model_order"])


def build_demand_tail_error_table(
    predictions: pd.DataFrame,
    *,
    tail: str,
) -> pd.DataFrame:
    if tail not in {"high", "low"}:
        raise ValueError("tail harus 'high' atau 'low'.")
    actual_reference = predictions.drop_duplicates(subset=[TIMESTAMP_COL])
    quantile = 0.90 if tail == "high" else 0.10
    threshold = float(actual_reference["actual"].quantile(quantile))
    if tail == "high":
        subset = predictions[predictions["actual"] >= threshold].copy()
    else:
        subset = predictions[predictions["actual"] <= threshold].copy()
    subset["demand_tail"] = f"{tail}_demand_p{int(quantile * 100)}"
    subset["actual_threshold"] = threshold
    columns = [
        "demand_tail",
        "actual_threshold",
        "timestamp",
        "model_label",
        "actual",
        "predicted",
        "residual",
        "absolute_error",
        "error_direction",
        "horizon_step",
        "origin_block",
        "local_hour",
        "local_day_name",
        "period_type",
    ]
    available = [column for column in columns if column in subset.columns]
    return order_by_model(subset[available]).drop(columns=["model_order"])


def build_top_absolute_errors(
    predictions: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    for model_label, group in predictions.groupby("model_label", sort=False):
        rows.append(
            group.sort_values(
                ["absolute_error", TIMESTAMP_COL],
                ascending=[False, True],
                kind="mergesort",
            ).head(int(top_n))
        )
    result = pd.concat(rows, ignore_index=True)
    result["rank_within_model"] = (
        result.groupby("model_label", sort=False)["absolute_error"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    columns = [
        "rank_within_model",
        "timestamp",
        "model_label",
        "actual",
        "predicted",
        "residual",
        "absolute_error",
        "error_direction",
        "horizon_step",
        "origin_block",
        "local_hour",
        "local_day_name",
        "period_type",
        "actual_demand_band",
    ]
    return result[columns].sort_values(
        ["model_label", "rank_within_model"],
        kind="mergesort",
    )


def build_signed_extreme_errors(
    predictions: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    frames = []
    for model_label, group in predictions.groupby("model_label", sort=False):
        under = (
            group[group["residual"] > 0]
            .sort_values(["residual", TIMESTAMP_COL], ascending=[False, True])
            .head(int(top_n))
            .copy()
        )
        over = (
            group[group["residual"] < 0]
            .assign(overprediction_magnitude=lambda df: df["residual"].abs())
            .sort_values(
                ["overprediction_magnitude", TIMESTAMP_COL],
                ascending=[False, True],
            )
            .head(int(top_n))
            .copy()
        )
        under["signed_error_type"] = "largest_underprediction"
        over["signed_error_type"] = "largest_overprediction"
        frames.extend([under, over])
    result = pd.concat(frames, ignore_index=True)
    result["rank_within_model_and_type"] = (
        result.groupby(["model_label", "signed_error_type"], sort=False)[
            "absolute_error"
        ]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    columns = [
        "signed_error_type",
        "rank_within_model_and_type",
        "timestamp",
        "model_label",
        "actual",
        "predicted",
        "residual",
        "absolute_error",
        "horizon_step",
        "origin_block",
        "local_hour",
        "local_day_name",
        "period_type",
        "actual_demand_band",
    ]
    return result[columns].sort_values(
        ["model_label", "signed_error_type", "rank_within_model_and_type"],
        kind="mergesort",
    )


def build_negative_prediction_table(predictions: pd.DataFrame) -> pd.DataFrame:
    subset = predictions[predictions["predicted"] < 0].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "model_label",
                "actual",
                "predicted",
                "residual",
                "absolute_error",
                "horizon_step",
                "origin_block",
                "local_hour",
                "local_day_name",
                "period_type",
            ]
        )
    columns = [
        "timestamp",
        "model_label",
        "actual",
        "predicted",
        "residual",
        "absolute_error",
        "horizon_step",
        "origin_block",
        "local_hour",
        "local_day_name",
        "period_type",
    ]
    return order_by_model(subset[columns]).drop(columns=["model_order"])


def build_timestamp_consensus_error(
    predictions: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for timestamp, group in predictions.groupby(TIMESTAMP_COL, sort=True):
        row: dict[str, Any] = {
            "timestamp": timestamp.isoformat(),
            "actual": float(group["actual"].iloc[0]),
            "local_hour": int(group["local_hour"].iloc[0]),
            "local_day_name": str(group["local_day_name"].iloc[0]),
            "period_type": str(group["period_type"].iloc[0]),
            "mean_absolute_error_across_models": float(group["absolute_error"].mean()),
            "max_absolute_error_across_models": float(group["absolute_error"].max()),
            "n_models": int(group["model_label"].nunique()),
        }
        for _, item in group.iterrows():
            label = str(item["model_label"]).replace("-", "_").replace(" ", "_")
            row[f"absolute_error_{label}"] = float(item["absolute_error"])
            row[f"residual_{label}"] = float(item["residual"])
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["mean_absolute_error_across_models", "timestamp"],
        ascending=[False, True],
        kind="mergesort",
    ).head(int(top_n)).reset_index(drop=True)


def metric_dict_for_group(group: pd.DataFrame) -> dict[str, Any]:
    if group.empty:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "smape": np.nan,
            "mean_error_actual_minus_predicted": np.nan,
            "median_error_actual_minus_predicted": np.nan,
            "overprediction_rate": np.nan,
            "underprediction_rate": np.nan,
            "max_absolute_error": np.nan,
        }
    metrics = compute_all_metrics(
        group["actual"],
        group["predicted"],
        include_diagnostics=False,
    )
    predicted = pd.to_numeric(group["predicted"], errors="raise")
    actual = pd.to_numeric(group["actual"], errors="raise")
    residual = pd.to_numeric(group["residual"], errors="raise")
    metrics.update(
        {
            "mean_error_actual_minus_predicted": float(residual.mean()),
            "median_error_actual_minus_predicted": float(residual.median()),
            "overprediction_rate": float((predicted > actual).mean()),
            "underprediction_rate": float((predicted < actual).mean()),
            "max_absolute_error": float(group["absolute_error"].max()),
        }
    )
    return metrics


def build_key_findings(
    *,
    model_summary: pd.DataFrame,
    behavioral_error: pd.DataFrame,
    residual_summary: pd.DataFrame,
    residual_acf: pd.DataFrame,
    high_demand_errors: pd.DataFrame,
    top_errors: pd.DataFrame,
    negative_predictions: pd.DataFrame,
) -> dict[str, Any]:
    winner = str(
        model_summary.sort_values(PRIMARY_METRIC, kind="mergesort").iloc[0][
            "model_label"
        ]
    )
    all_segments = behavioral_error[behavioral_error["segment"] != "all"].copy()
    worst_segment = all_segments.sort_values(
        "mae",
        ascending=False,
        kind="mergesort",
    ).iloc[0]
    best_segment = all_segments.sort_values("mae", kind="mergesort").iloc[0]
    high_demand_summary = (
        high_demand_errors.groupby("model_label", sort=False)["absolute_error"]
        .mean()
        .sort_values(kind="mergesort")
    )
    acf_24 = residual_acf[residual_acf["lag"] == 24].copy()
    if not acf_24.empty:
        strongest_acf_24 = acf_24.assign(
            abs_acf=lambda df: df["residual_autocorrelation"].abs()
        ).sort_values("abs_acf", ascending=False, kind="mergesort").iloc[0]
    else:
        strongest_acf_24 = pd.Series({"model_label": "", "residual_autocorrelation": np.nan})
    negative_counts = (
        negative_predictions["model_label"].value_counts().to_dict()
        if not negative_predictions.empty
        else {}
    )
    top_error_row = top_errors.sort_values(
        "absolute_error",
        ascending=False,
        kind="mergesort",
    ).iloc[0]
    return {
        "winner_by_primary_metric": winner,
        "worst_segment_overall": str(worst_segment["segment"]),
        "worst_segment_model": str(worst_segment["model_label"]),
        "worst_segment_mae": float(worst_segment["mae"]),
        "best_segment_overall": str(best_segment["segment"]),
        "best_segment_model": str(best_segment["model_label"]),
        "best_segment_mae": float(best_segment["mae"]),
        "best_high_demand_model_by_mean_abs_error": str(high_demand_summary.index[0]),
        "best_high_demand_mean_abs_error": float(high_demand_summary.iloc[0]),
        "strongest_residual_acf_lag_24_model": str(strongest_acf_24["model_label"]),
        "strongest_residual_acf_lag_24": float(
            strongest_acf_24["residual_autocorrelation"]
        ),
        "negative_prediction_counts": negative_counts,
        "largest_single_error_model": str(top_error_row["model_label"]),
        "largest_single_error_timestamp": str(top_error_row["timestamp"]),
        "largest_single_absolute_error": float(top_error_row["absolute_error"]),
    }


def save_hourly_error_plot(hourly_error: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    for model_label, group in hourly_error.groupby("model_label", sort=False):
        ax.plot(
            group["local_hour"],
            group["mae"],
            marker="o",
            linewidth=1.4,
            label=model_label,
        )
    ax.set_title("Error Pattern - MAE by Local NYC Hour")
    ax.set_xlabel("Local hour")
    ax.set_ylabel("MAE")
    ax.set_xticks(list(range(24)))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_day_of_week_error_plot(day_error: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_df = day_error.copy()
    plot_df["day_label"] = plot_df["local_day_of_week"].map(LOCAL_DAY_NAMES)
    pivot = plot_df.pivot(index="day_label", columns="model_label", values="mae")
    ordered_days = [LOCAL_DAY_NAMES[index] for index in range(7)]
    pivot = pivot.reindex(ordered_days)

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Error Pattern - MAE by Local Day of Week")
    ax.set_xlabel("Local day of week")
    ax.set_ylabel("MAE")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="")
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
        "weekday_local",
        "weekend_local",
        "high_demand_spike_p90",
        "low_demand_p10",
    ]
    plot_df = behavioral_error[behavioral_error["segment"].isin(segments)].copy()
    pivot = plot_df.pivot(index="segment", columns="model_label", values="mae")
    pivot = pivot.reindex(segments)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Error Pattern - MAE by Behavior Segment")
    ax.set_xlabel("Behavior segment")
    ax.set_ylabel("MAE")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="")
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
        ax.hist(group["residual"], bins=45, alpha=0.45, label=model_label)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_title("Error Pattern - Residual Distribution")
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_residual_acf_plot(residual_acf: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    for model_label, group in residual_acf.groupby("model_label", sort=False):
        ax.plot(
            group["lag"],
            group["residual_autocorrelation"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=model_label,
        )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.axvline(24, color="gray", linestyle="--", linewidth=0.9)
    if residual_acf["lag"].max() >= 48:
        ax.axvline(48, color="gray", linestyle=":", linewidth=0.9)
    ax.set_title("Error Pattern - Residual Autocorrelation")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Residual autocorrelation")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_high_demand_prediction_plot(
    high_demand_errors: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_df = high_demand_errors.copy()
    plot_df[TIMESTAMP_COL] = pd.to_datetime(plot_df[TIMESTAMP_COL], utc=True)
    timestamps = (
        plot_df[[TIMESTAMP_COL, "actual"]]
        .drop_duplicates(subset=[TIMESTAMP_COL])
        .sort_values(TIMESTAMP_COL, kind="mergesort")
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.scatter(
        timestamps[TIMESTAMP_COL],
        timestamps["actual"],
        label="Actual high demand",
        color="black",
        s=24,
        zorder=4,
    )
    for model_label, group in plot_df.groupby("model_label", sort=False):
        ax.scatter(
            group[TIMESTAMP_COL],
            group["predicted"],
            s=18,
            alpha=0.7,
            label=f"{model_label} predicted",
        )
    ax.set_title("Error Pattern - High Demand P90 Actual vs Predicted")
    ax.set_xlabel("UTC timestamp")
    ax.set_ylabel(TARGET_COL)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_top_error_plot(top_errors: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_df = top_errors[top_errors["rank_within_model"] <= 10].copy()
    plot_df["timestamp_label"] = pd.to_datetime(
        plot_df[TIMESTAMP_COL],
        utc=True,
    ).dt.strftime("%m-%d %H:%M")
    plot_df["label"] = plot_df["model_label"] + " | " + plot_df["timestamp_label"]
    plot_df = plot_df.sort_values("absolute_error", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["label"], plot_df["absolute_error"])
    ax.set_title("Error Pattern - Top Absolute Errors by Model")
    ax.set_xlabel("Absolute error")
    ax.set_ylabel("")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_origin_block_error_plot(
    origin_block_error: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    for model_label, group in origin_block_error.groupby("model_label", sort=False):
        ax.plot(
            group["origin_block"],
            group["mae"],
            marker="o",
            linewidth=1.4,
            label=model_label,
        )
    ax.set_title("Error Pattern - MAE by 24-hour Origin Block")
    ax.set_xlabel("Origin block")
    ax.set_ylabel("MAE")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_error_analysis_report(
    *,
    metadata: Mapping[str, Any],
    model_summary: pd.DataFrame,
    hourly_error: pd.DataFrame,
    day_error: pd.DataFrame,
    weekday_weekend: pd.DataFrame,
    behavioral_error: pd.DataFrame,
    residual_summary: pd.DataFrame,
    residual_acf: pd.DataFrame,
    horizon_error: pd.DataFrame,
    origin_block_error: pd.DataFrame,
    extreme_summary: pd.DataFrame,
    high_demand_errors: pd.DataFrame,
    low_demand_errors: pd.DataFrame,
    top_errors: pd.DataFrame,
    signed_extremes: pd.DataFrame,
    negative_predictions: pd.DataFrame,
    timestamp_consensus: pd.DataFrame,
    skip_plots: bool,
    outputs: Mapping[str, Path],
) -> str:
    key_findings = dict(metadata["key_findings"])
    winner = str(key_findings["winner_by_primary_metric"])
    worst_segment = select_worst_segment_per_model(behavioral_error)
    best_hour = select_best_or_worst_hour(hourly_error, best=True)
    worst_hour = select_best_or_worst_hour(hourly_error, best=False)
    best_day = select_best_or_worst_day(day_error, best=True)
    worst_day = select_best_or_worst_day(day_error, best=False)
    acf_selected = residual_acf[residual_acf["lag"].isin([1, 24, 48])].copy()
    high_summary = summarize_tail_details(high_demand_errors)
    low_summary = summarize_tail_details(low_demand_errors)
    top_error_view = top_errors[top_errors["rank_within_model"] <= 5].copy()
    signed_view = signed_extremes[signed_extremes["rank_within_model_and_type"] <= 3]
    origin_worst = (
        origin_block_error.sort_values(
            ["model_label", "mae"],
            ascending=[True, False],
            kind="mergesort",
        )
        .groupby("model_label", sort=False)
        .head(1)
    )
    horizon_worst = (
        horizon_error.sort_values(
            ["model_label", "mae"],
            ascending=[True, False],
            kind="mergesort",
        )
        .groupby("model_label", sort=False)
        .head(1)
    )
    negative_summary = (
        negative_predictions.groupby("model_label", sort=False)
        .size()
        .reset_index(name="negative_prediction_count")
        if not negative_predictions.empty
        else pd.DataFrame(
            [{"model_label": label, "negative_prediction_count": 0} for label in MODEL_ORDER]
        )
    )
    if not negative_summary.empty:
        negative_summary = (
            pd.DataFrame({"model_label": MODEL_ORDER})
            .merge(negative_summary, on="model_label", how="left")
            .fillna({"negative_prediction_count": 0})
        )
        negative_summary["negative_prediction_count"] = negative_summary[
            "negative_prediction_count"
        ].astype(int)

    model_view_columns = [
        f"rank_by_{PRIMARY_METRIC}",
        "model_label",
        "parameter_set_id",
        "mae",
        "rmse",
        "mape",
        "smape",
        "mean_error_actual_minus_predicted",
        "overprediction_rate",
        "underprediction_rate",
        "negative_prediction_count",
        "max_absolute_error",
    ]
    residual_view_columns = [
        "model_label",
        "mean_error_actual_minus_predicted",
        "median_error_actual_minus_predicted",
        "residual_std",
        "residual_skew",
        "residual_p05",
        "residual_p95",
        "absolute_error_p90",
        "absolute_error_p99",
    ]

    lines = [
        "# Error Pattern Analysis",
        "",
        f"Run UTC: {metadata['started_at_utc']}",
        "",
        "## Scope",
        "",
        (
            "Tahap ini melakukan analisis post-hoc atas prediksi final test "
            "dari tahap 14. Tidak ada tuning, retraining, ataupun prediksi "
            "baru; final test dipakai hanya sebagai label evaluasi dan konteks "
            "pola error."
        ),
        "",
        "## Leakage Guardrail",
        "",
        (
            "Semua baris final prediction yang dianalisis memiliki "
            "`used_actual_future_for_features=False`. Analisis ini membaca "
            "artefak final test yang sudah dibekukan, sehingga tidak mengubah "
            "ranking model atau konfigurasi hasil benchmark."
        ),
        "",
        "## Overall Error Summary",
        "",
        dataframe_to_markdown(
            model_summary[
                [column for column in model_view_columns if column in model_summary.columns]
            ],
            float_digits=6,
        ),
        "",
        "Interpretasi ringkas:",
        "",
        (
            f"Model dengan MAE final test terbaik tetap {winner}. "
            "XGBoost-Basic memiliki bias rata-rata paling kecil, sedangkan "
            "Prophet menunjukkan underprediction kuat dan prediksi negatif. "
            "XGBoost-Advanced lebih dekat dari Prophet, tetapi tail error dan "
            "RMSE-nya lebih besar daripada XGBoost-Basic."
        ),
        "",
        "## Temporal Error Analysis",
        "",
        "Worst segment per model:",
        "",
        dataframe_to_markdown(worst_segment, float_digits=6),
        "",
        "Best hour per model by MAE:",
        "",
        dataframe_to_markdown(best_hour, float_digits=6),
        "",
        "Worst hour per model by MAE:",
        "",
        dataframe_to_markdown(worst_hour, float_digits=6),
        "",
        "Best local day per model by MAE:",
        "",
        dataframe_to_markdown(best_day, float_digits=6),
        "",
        "Worst local day per model by MAE:",
        "",
        dataframe_to_markdown(worst_day, float_digits=6),
        "",
        "Weekday vs weekend:",
        "",
        dataframe_to_markdown(weekday_weekend, float_digits=6),
        "",
        (
            "Rush hour dan high-demand spike menjadi zona error yang paling "
            "kritis. Untuk XGBoost, night period relatif lebih mudah karena "
            "level demand rendah dan pola jamnya lebih stabil. Weekend memberi "
            "sinyal menarik: advanced feature set lebih kompetitif di weekend, "
            "tetapi belum cukup untuk mengalahkan basic secara keseluruhan."
        ),
        "",
        "## Residual Analysis",
        "",
        dataframe_to_markdown(
            residual_summary[
                [column for column in residual_view_columns if column in residual_summary.columns]
            ],
            float_digits=6,
        ),
        "",
        "Selected residual autocorrelation:",
        "",
        dataframe_to_markdown(acf_selected, float_digits=6),
        "",
        "Worst recursive horizon step per model:",
        "",
        dataframe_to_markdown(horizon_worst, float_digits=6),
        "",
        "Worst 24-hour origin block per model:",
        "",
        dataframe_to_markdown(origin_worst, float_digits=6),
        "",
        (
            "Residual autocorrelation yang masih tampak, terutama pada lag "
            "harian, menunjukkan bahwa sebagian pola temporal belum sepenuhnya "
            "ditangkap oleh model. Ini penting untuk dibahas sebagai limitasi "
            "forecasting recursive: error pada awal horizon dapat terbawa ke "
            "step berikutnya."
        ),
        "",
        "## Extreme Event Analysis",
        "",
        dataframe_to_markdown(extreme_summary, float_digits=6),
        "",
        "High-demand P90 summary:",
        "",
        dataframe_to_markdown(high_summary, float_digits=6),
        "",
        "Low-demand P10 summary:",
        "",
        dataframe_to_markdown(low_summary, float_digits=6),
        "",
        "Top absolute errors per model:",
        "",
        dataframe_to_markdown(top_error_view, float_digits=6),
        "",
        "Largest signed errors:",
        "",
        dataframe_to_markdown(signed_view, float_digits=6),
        "",
        "Timestamps yang sulit untuk semua model:",
        "",
        dataframe_to_markdown(timestamp_consensus.head(10), float_digits=6),
        "",
        "Negative predictions:",
        "",
        dataframe_to_markdown(negative_summary, float_digits=0),
        "",
        (
            "Spike demand cenderung menghasilkan underprediction, terutama "
            "pada Prophet. XGBoost-Basic tetap menjadi model paling kuat pada "
            "high-demand P90 secara rata-rata, tetapi masih memiliki error besar "
            "pada timestamp tertentu. XGBoost-Advanced menunjukkan beberapa "
            "kegagalan tail yang lebih ekstrem, mengindikasikan tambahan fitur "
            "belum otomatis membuat model lebih robust pada perubahan demand "
            "yang tajam."
        ),
        "",
        "## Time Cost Computing",
        "",
        (
            "Tahap 16 adalah analisis post-hoc, sehingga `train_time_seconds=0` "
            "dan `prediction_time_seconds=0`. Runtime yang dicatat adalah waktu "
            f"analisis dan penulisan output: {metadata['total_runtime_seconds']:.6f} detik."
        ),
        "",
        "## Research Interpretation",
        "",
        (
            "Pola error mendukung hasil tahap 15: XGBoost-Basic bukan hanya "
            "unggul secara rata-rata, tetapi juga lebih stabil secara residual. "
            "Advanced features memberi sinyal tambahan pada beberapa segmen "
            "seperti weekend, namun manfaatnya kalah oleh error tail yang lebih "
            "besar. Prophet menangkap pola musiman umum, tetapi kurang adaptif "
            "terhadap level demand lokal pada horizon final test dan bahkan "
            "menghasilkan prediksi negatif."
        ),
        "",
        (
            "Untuk tahap conclusion, poin utama yang perlu dibawa adalah bahwa "
            "forecasting hourly NYC Taxi sangat sensitif pada jam sibuk, spike "
            "demand, dan propagasi error recursive. Model terbaik pada penelitian "
            "ini adalah model yang paling konsisten menghadapi segmen tersebut, "
            "bukan hanya yang memiliki kapasitas fitur paling besar."
        ),
        "",
        "## Output Files",
        "",
        f"- Model summary: `{outputs['model_summary']}`",
        f"- Temporal hour error: `{outputs['hourly_error']}`",
        f"- Temporal day error: `{outputs['day_of_week_error']}`",
        f"- Behavioral error summary: `{outputs['behavioral_error']}`",
        f"- Residual distribution summary: `{outputs['residual_distribution_summary']}`",
        f"- Residual autocorrelation: `{outputs['residual_autocorrelation']}`",
        f"- Extreme event summary: `{outputs['extreme_event_summary']}`",
        f"- Top absolute errors: `{outputs['top_absolute_errors']}`",
        f"- Runtime summary: `{outputs['runtime_summary']}`",
        f"- Report mirror: `{outputs['report']}`",
    ]
    if not skip_plots:
        lines.extend(
            [
                f"- Hourly error plot: `{outputs['hourly_error_plot']}`",
                f"- Day error plot: `{outputs['day_error_plot']}`",
                f"- Behavioral error plot: `{outputs['behavioral_error_plot']}`",
                f"- Residual distribution plot: `{outputs['residual_distribution_plot']}`",
                f"- Residual ACF plot: `{outputs['residual_acf_plot']}`",
                f"- High demand prediction plot: `{outputs['high_demand_prediction_plot']}`",
                f"- Top error plot: `{outputs['top_error_plot']}`",
                f"- Origin block error plot: `{outputs['origin_block_error_plot']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def select_worst_segment_per_model(behavioral_error: pd.DataFrame) -> pd.DataFrame:
    subset = behavioral_error[behavioral_error["segment"] != "all"].copy()
    return (
        subset.sort_values(
            ["model_label", "mae"],
            ascending=[True, False],
            kind="mergesort",
        )
        .groupby("model_label", sort=False)
        .head(1)
        .reset_index(drop=True)
    )


def select_best_or_worst_hour(
    hourly_error: pd.DataFrame,
    *,
    best: bool,
) -> pd.DataFrame:
    return (
        hourly_error.sort_values(
            ["model_label", "mae"],
            ascending=[True, best],
            kind="mergesort",
        )
        .groupby("model_label", sort=False)
        .head(1)
        .reset_index(drop=True)
    )


def select_best_or_worst_day(
    day_error: pd.DataFrame,
    *,
    best: bool,
) -> pd.DataFrame:
    return (
        day_error.sort_values(
            ["model_label", "mae"],
            ascending=[True, best],
            kind="mergesort",
        )
        .groupby("model_label", sort=False)
        .head(1)
        .reset_index(drop=True)
    )


def summarize_tail_details(tail_details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_label, group in tail_details.groupby("model_label", sort=False):
        row = {
            "model_label": model_label,
            "demand_tail": str(group["demand_tail"].iloc[0]),
            "actual_threshold": float(group["actual_threshold"].iloc[0]),
            "n_predictions": int(group.shape[0]),
        }
        row.update(metric_dict_for_group(group))
        rows.append(row)
    return order_by_model(pd.DataFrame(rows)).drop(columns=["model_order"])


def select_model_row(frame: pd.DataFrame, model_label: str) -> pd.Series:
    selected = frame[frame["model_label"].astype(str) == str(model_label)]
    if selected.empty:
        raise ValueError(f"Model tidak ditemukan: {model_label}")
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


def order_by_model(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "model_label" not in result.columns:
        return result
    order_map = {label: index for index, label in enumerate(MODEL_ORDER)}
    result["model_order"] = result["model_label"].map(order_map).fillna(999).astype(int)
    sort_columns = ["model_order"]
    for candidate in [
        "segment",
        "local_hour",
        "local_day_of_week",
        "weekday_weekend",
        "horizon_step",
        "origin_block",
        "lag",
        TIMESTAMP_COL,
    ]:
        if candidate in result.columns:
            sort_columns.append(candidate)
            break
    return result.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def dataframe_to_markdown(df: pd.DataFrame, *, float_digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"

    def format_cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
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


if __name__ == "__main__":
    main()

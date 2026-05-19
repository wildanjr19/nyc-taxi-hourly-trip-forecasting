"""
Script tahap 14: Final Testing.

Scope:
- Load model hasil retraining tahap 13.
- Evaluasi Prophet, XGBoost-Basic, dan XGBoost-Advanced pada final_test.
- Gunakan blok horizon 24 jam. XGBoost memakai recursive forecasting dari
  history masa lalu dan hanya memasukkan actual final_test setelah satu blok
  24 jam selesai dievaluasi.
- Simpan metrics, predictions, plots, metadata, dan time cost computing.

Contoh:
    python -m src.experiments.final_test
    python -m src.experiments.final_test --skip-plots
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
    FINAL_TEST_PATH,
    FORECAST_HORIZON,
    LOCAL_TZ,
    METRICS,
    PRIMARY_METRIC,
    REPORTS_DIR,
    TARGET_COL,
    TIMESTAMP_COL,
    TRAIN_VAL_PATH,
    ensure_dirs,
)
from src.forecasting import (
    ACTUAL_COL,
    PREDICTION_COL,
    forecast_validation_window,
    recursive_forecast_prophet,
    validate_prediction_output,
)
from src.metrics import compute_all_metrics
from src.models.prophet_model import make_prophet_future_frame
from src.splits import load_split_timeseries
from src.tracking import (
    append_experiment_run,
    elapsed_seconds,
    log_runtime,
    make_runtime_record,
    save_experiment_metadata,
    start_timer,
    utc_now_iso,
)


EXPERIMENT_NAME = "final_test"
RETRAINING_DIR = EXPERIMENTS_DIR / "retraining"
MODEL_REGISTRY_PATH = RETRAINING_DIR / "model_registry.json"
BEST_PARAMS_USED_PATH = RETRAINING_DIR / "best_params_used.json"
REPORT_PATH = REPORTS_DIR / "final_test_report.md"
ALL_MODEL_KEYS = ("prophet", "xgb_basic", "xgb_advanced")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-time final test evaluation for retrained models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[*ALL_MODEL_KEYS, "all"],
        default=["all"],
        help="Model yang akan dievaluasi. Default: all.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip matplotlib figure generation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Izinkan overwrite output final test yang sudah ada.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    metadata = run_final_testing(
        models=args.models,
        skip_plots=args.skip_plots,
        overwrite=args.overwrite,
    )
    print("Final testing selesai.")
    print(f"Winner by {PRIMARY_METRIC}: {metadata['winner_by_primary_metric']}")
    print(f"Output dir: {metadata['outputs']['base']}")
    print(f"Metrics: {metadata['outputs']['final_metrics']}")


def run_final_testing(
    *,
    models: Optional[Sequence[str]] = None,
    skip_plots: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    selected_models = normalize_model_selection(models or ["all"])
    outputs = final_test_output_paths()
    timer = start_timer()
    started_at = utc_now_iso()
    status = "success"
    error_message = ""

    try:
        ensure_dirs()
        ensure_final_test_dirs(outputs)
        validate_output_overwrite(outputs, overwrite=overwrite)

        train_val = load_split_timeseries(TRAIN_VAL_PATH)
        final_test = load_split_timeseries(FINAL_TEST_PATH)
        validate_final_test_scope(train_val, final_test)

        registry = load_json(MODEL_REGISTRY_PATH)
        best_params_used = load_json(BEST_PARAMS_USED_PATH)
        plans = build_model_plans(
            selected_models,
            registry=registry,
            best_params_used=best_params_used,
        )

        predictions_frames: list[pd.DataFrame] = []
        metric_rows: list[dict[str, Any]] = []
        runtime_rows: list[dict[str, Any]] = []
        model_results: list[dict[str, Any]] = []

        for plan in plans:
            result = evaluate_one_model(
                plan=plan,
                train_val=train_val,
                final_test=final_test,
            )
            predictions_frames.append(result["predictions"])
            metric_rows.append(result["metrics"])
            runtime_rows.append(result["runtime"])
            model_results.append(result["result"])
            log_model_runtime(result["runtime"])
            append_experiment_run(result["result"])

        predictions = pd.concat(predictions_frames, ignore_index=True)
        validate_final_predictions(
            predictions,
            final_test=final_test,
            model_labels=[str(plan["model_label"]) for plan in plans],
        )

        final_metrics = build_final_metrics_table(metric_rows)
        final_runtime = pd.DataFrame(runtime_rows)
        residual_summary = build_residual_summary(predictions)
        horizon_error = build_horizon_error_summary(predictions)

        predictions.to_csv(outputs["final_predictions"], index=False)
        final_metrics.to_csv(outputs["final_metrics"], index=False)
        final_runtime.to_csv(outputs["final_runtime"], index=False)
        residual_summary.to_csv(outputs["residual_summary"], index=False)
        horizon_error.to_csv(outputs["horizon_error_summary"], index=False)

        if not skip_plots:
            save_actual_vs_predicted_plot(
                predictions,
                outputs["actual_vs_predicted_plot"],
            )
            save_residuals_plot(
                predictions,
                outputs["residuals_plot"],
            )

        total_runtime_seconds = elapsed_seconds(timer)
        winner = select_winner(final_metrics, metric=PRIMARY_METRIC)
        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "stage": "final_testing",
            "status": status,
            "error_message": error_message,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "total_runtime_seconds": total_runtime_seconds,
            "models_requested": selected_models,
            "models_evaluated": [result["model_label"] for result in model_results],
            "winner_by_primary_metric": winner,
            "primary_metric": PRIMARY_METRIC,
            "forecast_horizon_hours": int(FORECAST_HORIZON),
            "train_val_path": str(TRAIN_VAL_PATH),
            "final_test_path": str(FINAL_TEST_PATH),
            "train_val_summary": summarize_timeseries(train_val),
            "final_test_summary": summarize_timeseries(final_test),
            "leakage_guardrail": leakage_guardrail_notes(),
            "final_test_evaluation_count_note": (
                "This run evaluates final_test with retrained models only; no "
                "hyperparameter tuning or model adjustment is performed here."
            ),
            "model_results": model_results,
            "outputs": stringify_paths(outputs),
        }

        report_text = render_final_test_report(
            metadata=metadata,
            final_metrics=final_metrics,
            final_runtime=final_runtime,
            residual_summary=residual_summary,
            horizon_error=horizon_error,
            skip_plots=skip_plots,
        )
        outputs["summary"].write_text(report_text, encoding="utf-8")
        outputs["report"].write_text(report_text, encoding="utf-8")
        save_experiment_metadata(metadata, outputs["metadata"])
        log_runtime(
            make_runtime_record(
                experiment_name=EXPERIMENT_NAME,
                model_name="all_requested_models",
                feature_set="mixed",
                n_train_rows=int(train_val.shape[0]),
                n_prediction_rows=int(predictions.shape[0]),
                train_time_seconds=0.0,
                prediction_time_seconds=float(
                    final_runtime["prediction_time_seconds"].sum()
                ),
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
            "stage": "final_testing",
            "status": status,
            "error_message": error_message,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "total_runtime_seconds": total_runtime_seconds,
            "models_requested": selected_models,
            "outputs": stringify_paths(outputs),
        }
        save_experiment_metadata(failure_metadata, outputs["metadata"])
        log_runtime(
            make_runtime_record(
                experiment_name=EXPERIMENT_NAME,
                model_name="all_requested_models",
                feature_set="mixed",
                total_runtime_seconds=total_runtime_seconds,
                status=status,
                error_message=error_message,
            )
        )
        append_experiment_run(failure_metadata)
        raise


def normalize_model_selection(models: Sequence[str]) -> list[str]:
    normalized = [str(model).strip().lower() for model in models]
    if not normalized:
        raise ValueError("Minimal satu model harus dipilih.")
    if "all" in normalized and len(normalized) > 1:
        raise ValueError("Gunakan --models all atau daftar model spesifik.")
    if normalized == ["all"]:
        return list(ALL_MODEL_KEYS)
    invalid = sorted(set(normalized).difference(ALL_MODEL_KEYS))
    if invalid:
        raise ValueError(f"Model tidak dikenal: {invalid}")
    return list(dict.fromkeys(normalized))


def final_test_output_paths() -> dict[str, Path]:
    metrics_dir = FINAL_TEST_DIR / "metrics"
    predictions_dir = FINAL_TEST_DIR / "predictions"
    figures_dir = FINAL_TEST_DIR / "figures"
    summaries_dir = FINAL_TEST_DIR / "summaries"
    return {
        "base": FINAL_TEST_DIR,
        "metrics": metrics_dir,
        "predictions": predictions_dir,
        "figures": figures_dir,
        "summaries": summaries_dir,
        "final_metrics": metrics_dir / "final_metrics.csv",
        "final_runtime": metrics_dir / "final_runtime.csv",
        "residual_summary": metrics_dir / "residual_summary.csv",
        "horizon_error_summary": metrics_dir / "horizon_error_summary.csv",
        "final_predictions": predictions_dir / "final_predictions.csv",
        "actual_vs_predicted_plot": figures_dir / "actual_vs_predicted.png",
        "residuals_plot": figures_dir / "residuals.png",
        "metadata": FINAL_TEST_DIR / "experiment_metadata.json",
        "summary": summaries_dir / "final_test_summary.md",
        "report": REPORT_PATH,
    }


def ensure_final_test_dirs(paths: Mapping[str, Path]) -> None:
    for key in ["base", "metrics", "predictions", "figures", "summaries"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def validate_output_overwrite(
    paths: Mapping[str, Path],
    *,
    overwrite: bool,
) -> None:
    output_keys = [
        "final_metrics",
        "final_runtime",
        "residual_summary",
        "horizon_error_summary",
        "final_predictions",
        "actual_vs_predicted_plot",
        "residuals_plot",
        "metadata",
        "summary",
        "report",
    ]
    existing = [str(paths[key]) for key in output_keys if paths[key].exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output final test sudah ada. Gunakan --overwrite jika memang ingin "
            f"menulis ulang. Existing files: {existing}"
        )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON harus berupa object: {path}")
    return payload


def validate_final_test_scope(train_val: pd.DataFrame, final_test: pd.DataFrame) -> None:
    if train_val.empty:
        raise ValueError("train_val kosong.")
    if final_test.empty:
        raise ValueError("final_test kosong.")
    if train_val.index.max() >= final_test.index.min():
        raise ValueError("train_val harus berakhir sebelum final_test dimulai.")
    if train_val.index.intersection(final_test.index).size > 0:
        raise ValueError("train_val dan final_test memiliki overlap timestamp.")
    if final_test.shape[0] < FORECAST_HORIZON:
        raise ValueError("final_test harus minimal sepanjang FORECAST_HORIZON.")
    if final_test.index.has_duplicates:
        raise ValueError("final_test mengandung duplicate timestamp.")
    if not final_test.index.is_monotonic_increasing:
        raise ValueError("final_test tidak chronological.")
    if TARGET_COL not in final_test.columns:
        raise ValueError(f"Kolom target final_test hilang: {TARGET_COL}")


def build_model_plans(
    selected_models: Sequence[str],
    *,
    registry: Mapping[str, Any],
    best_params_used: Mapping[str, Any],
) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for model_key in selected_models:
        if model_key not in registry:
            raise ValueError(f"Model {model_key} tidak ditemukan di registry.")
        if model_key not in best_params_used:
            raise ValueError(f"Best params {model_key} tidak ditemukan.")

        registry_entry = dict(registry[model_key])
        params_entry = dict(best_params_used[model_key])
        model_path = Path(str(registry_entry["model_path"]))
        metadata_path = Path(str(registry_entry["metadata_path"]))
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact tidak ditemukan: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata retraining tidak ditemukan: {metadata_path}")

        retraining_metadata = load_json(metadata_path)
        if bool(retraining_metadata.get("final_test_used", True)):
            raise ValueError(
                f"Metadata retraining {model_key} menandai final_test_used=True."
            )

        plans.append(
            {
                "model_key": model_key,
                "model_label": registry_entry["model_label"],
                "model_name": registry_entry["model_name"],
                "feature_set": registry_entry["feature_set"],
                "model_path": model_path,
                "metadata_path": metadata_path,
                "artifact_format": registry_entry["artifact_format"],
                "feature_columns": list(registry_entry.get("feature_columns", [])),
                "parameter_set_id": str(params_entry["parameter_set_id"]),
                "params": dict(params_entry.get("params", {})),
                "best_cv_metrics": dict(params_entry.get("best_cv_metrics", {})),
                "retraining_metadata": retraining_metadata,
            }
        )
    return plans


def evaluate_one_model(
    *,
    plan: Mapping[str, Any],
    train_val: pd.DataFrame,
    final_test: pd.DataFrame,
) -> dict[str, Any]:
    model_timer = start_timer()
    load_timer = start_timer()
    model = load_model_from_plan(plan)
    model_load_time_seconds = elapsed_seconds(load_timer)

    prediction_timer = start_timer()
    if plan["model_key"] == "prophet":
        predictions = forecast_prophet_final_window(
            model,
            train_val,
            final_test,
            plan=plan,
            horizon=FORECAST_HORIZON,
        )
    else:
        predictions = forecast_validation_window(
            model,
            train_val,
            final_test,
            horizon=FORECAST_HORIZON,
            feature_set=str(plan["feature_set"]),
            model_name=str(plan["model_name"]),
            fold="final_test",
            parameter_set_id=str(plan["parameter_set_id"]),
            update_history_with_actuals=True,
        )
    prediction_time_seconds = elapsed_seconds(prediction_timer)
    total_runtime_seconds = elapsed_seconds(model_timer)

    predictions = prepare_final_prediction_frame(predictions, plan=plan)
    metric_row = build_metric_row(
        predictions,
        plan=plan,
        train_val=train_val,
        final_test=final_test,
        model_load_time_seconds=model_load_time_seconds,
        prediction_time_seconds=prediction_time_seconds,
        total_runtime_seconds=total_runtime_seconds,
    )
    runtime_row = build_runtime_row(
        plan=plan,
        train_val=train_val,
        final_test=final_test,
        model_load_time_seconds=model_load_time_seconds,
        prediction_time_seconds=prediction_time_seconds,
        total_runtime_seconds=total_runtime_seconds,
        predictions=predictions,
    )
    result = {
        "experiment_name": EXPERIMENT_NAME,
        "stage": "final_testing_model_evaluation",
        "status": "success",
        "model_key": plan["model_key"],
        "model_label": plan["model_label"],
        "model_name": plan["model_name"],
        "feature_set": plan["feature_set"],
        "parameter_set_id": plan["parameter_set_id"],
        "n_predictions": int(predictions.shape[0]),
        "metrics": metric_row,
        "runtime": runtime_row,
        "used_actual_future_for_features": False,
    }
    return {
        "predictions": predictions,
        "metrics": metric_row,
        "runtime": runtime_row,
        "result": result,
    }


def load_model_from_plan(plan: Mapping[str, Any]) -> Any:
    model_path = Path(plan["model_path"])
    artifact_format = str(plan["artifact_format"])
    if plan["model_key"] == "prophet":
        if artifact_format != "prophet.serialize.model_to_json":
            raise ValueError(f"Artifact Prophet tidak dikenal: {artifact_format}")
        try:
            from prophet.serialize import model_from_json
        except ImportError as exc:
            raise ImportError(
                "prophet.serialize.model_from_json tidak tersedia. "
                "Pastikan dependencies sudah terinstall."
            ) from exc
        return model_from_json(model_path.read_text(encoding="utf-8"))

    if artifact_format != "xgboost_json":
        raise ValueError(f"Artifact XGBoost tidak dikenal: {artifact_format}")
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "Package xgboost belum tersedia. Install requirements terlebih dahulu."
        ) from exc
    model = XGBRegressor()
    model.load_model(str(model_path))
    return model


def forecast_prophet_final_window(
    model: Any,
    train_history: pd.DataFrame,
    final_test: pd.DataFrame,
    *,
    plan: Mapping[str, Any],
    horizon: int,
) -> pd.DataFrame:
    if final_test.empty:
        raise ValueError("final_test Prophet kosong.")
    if train_history.index.max() >= final_test.index.min():
        raise ValueError("Train history Prophet harus berakhir sebelum final_test.")

    all_predictions: list[pd.DataFrame] = []
    current_origin = train_history.index.max()
    block_id = 0
    for start in range(0, len(final_test), horizon):
        block_id += 1
        final_block = final_test.iloc[start : start + horizon]
        future = make_prophet_future_frame(final_block.index)
        block_predictions = recursive_forecast_prophet(
            model,
            future,
            horizon=len(final_block),
            model_name=str(plan["model_name"]),
            fold="final_test",
            parameter_set_id=str(plan["parameter_set_id"]),
        )
        block_predictions[ACTUAL_COL] = pd.to_numeric(
            final_block[TARGET_COL],
            errors="raise",
        ).to_numpy(dtype=float)
        block_predictions["feature_set"] = str(plan["feature_set"])
        block_predictions["forecast_origin"] = current_origin
        block_predictions["origin_block"] = int(block_id)
        block_predictions["validation_start"] = final_block.index.min()
        block_predictions["validation_end"] = final_block.index.max()
        all_predictions.append(block_predictions)
        current_origin = final_block.index.max()

    predictions = pd.concat(all_predictions, ignore_index=True)
    validate_prediction_output(
        predictions,
        expected_index=final_test.index,
        horizon=len(final_test),
    )
    predictions.attrs["prediction_time_seconds_total"] = round(
        float(predictions["prediction_time_seconds"].sum()),
        9,
    )
    return predictions


def prepare_final_prediction_frame(
    predictions: pd.DataFrame,
    *,
    plan: Mapping[str, Any],
) -> pd.DataFrame:
    prepared = predictions.copy()
    prepared[TIMESTAMP_COL] = pd.to_datetime(
        prepared[TIMESTAMP_COL],
        utc=True,
        errors="raise",
    )
    for column in [ACTUAL_COL, PREDICTION_COL, "prediction_time_seconds"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="raise")
    prepared["horizon_step"] = pd.to_numeric(
        prepared["horizon_step"],
        errors="raise",
    ).astype(int)
    if "origin_block" in prepared.columns:
        prepared["origin_block"] = pd.to_numeric(
            prepared["origin_block"],
            errors="raise",
        ).astype(int)

    leakage_flags = coerce_bool_series(
        prepared["used_actual_future_for_features"],
        column_name="used_actual_future_for_features",
    )
    if leakage_flags.any():
        raise ValueError(f"Leakage flag aktif pada {plan['model_label']}.")
    prepared["used_actual_future_for_features"] = leakage_flags

    prepared["model_key"] = str(plan["model_key"])
    prepared["model_label"] = str(plan["model_label"])
    prepared["evaluation_split"] = "final_test"
    prepared["residual"] = prepared[ACTUAL_COL] - prepared[PREDICTION_COL]
    prepared["absolute_error"] = prepared["residual"].abs()
    prepared["squared_error"] = np.square(prepared["residual"])
    local_timestamp = prepared[TIMESTAMP_COL].dt.tz_convert(LOCAL_TZ)
    prepared["local_hour"] = local_timestamp.dt.hour
    prepared["local_day_of_week"] = local_timestamp.dt.dayofweek
    prepared["local_is_weekend"] = prepared["local_day_of_week"].isin([5, 6]).astype(int)
    prepared["local_month"] = local_timestamp.dt.month
    return prepared.sort_values(TIMESTAMP_COL, kind="mergesort").reset_index(drop=True)


def build_metric_row(
    predictions: pd.DataFrame,
    *,
    plan: Mapping[str, Any],
    train_val: pd.DataFrame,
    final_test: pd.DataFrame,
    model_load_time_seconds: float,
    prediction_time_seconds: float,
    total_runtime_seconds: float,
) -> dict[str, Any]:
    metric_values = compute_all_metrics(
        predictions[ACTUAL_COL],
        predictions[PREDICTION_COL],
    )
    retraining_metadata = dict(plan["retraining_metadata"])
    residual = pd.to_numeric(predictions["residual"], errors="raise")
    predicted = pd.to_numeric(predictions[PREDICTION_COL], errors="raise")
    actual = pd.to_numeric(predictions[ACTUAL_COL], errors="raise")
    row: dict[str, Any] = {
        "model_key": plan["model_key"],
        "model_label": plan["model_label"],
        "model_name": plan["model_name"],
        "feature_set": plan["feature_set"],
        "parameter_set_id": plan["parameter_set_id"],
        "params_json": json.dumps(plan["params"], sort_keys=True),
        "n_train_rows_retraining": int(retraining_metadata.get("n_train_rows", 0)),
        "n_train_val_history_rows": int(train_val.shape[0]),
        "n_prediction_rows": int(predictions.shape[0]),
        "final_test_start": final_test.index.min().isoformat(),
        "final_test_end": final_test.index.max().isoformat(),
        "forecast_horizon_hours": int(FORECAST_HORIZON),
        "n_origin_blocks": int(predictions["origin_block"].nunique()),
        "model_load_time_seconds": float(model_load_time_seconds),
        "prediction_time_seconds": float(prediction_time_seconds),
        "model_predict_time_seconds": float(
            pd.to_numeric(predictions["prediction_time_seconds"]).sum()
        ),
        "total_runtime_seconds": float(total_runtime_seconds),
        "retraining_train_time_seconds": float(
            retraining_metadata.get("train_time_seconds", np.nan)
        ),
        "negative_prediction_count": int((predicted < 0).sum()),
        "overprediction_rate": float((predicted > actual).mean()),
        "underprediction_rate": float((predicted < actual).mean()),
        "mean_error_actual_minus_predicted": float(residual.mean()),
        "max_absolute_error": float(predictions["absolute_error"].max()),
        "used_actual_future_for_features": False,
        "final_test_used_for_tuning": False,
    }
    row.update(metric_values)
    return row


def build_runtime_row(
    *,
    plan: Mapping[str, Any],
    train_val: pd.DataFrame,
    final_test: pd.DataFrame,
    model_load_time_seconds: float,
    prediction_time_seconds: float,
    total_runtime_seconds: float,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    retraining_metadata = dict(plan["retraining_metadata"])
    return {
        "experiment_name": EXPERIMENT_NAME,
        "model_key": plan["model_key"],
        "model_label": plan["model_label"],
        "model_name": plan["model_name"],
        "feature_set": plan["feature_set"],
        "parameter_set_id": plan["parameter_set_id"],
        "train_start": train_val.index.min().isoformat(),
        "train_end": train_val.index.max().isoformat(),
        "validation_start": final_test.index.min().isoformat(),
        "validation_end": final_test.index.max().isoformat(),
        "n_train_rows": int(train_val.shape[0]),
        "n_prediction_rows": int(predictions.shape[0]),
        "model_load_time_seconds": float(model_load_time_seconds),
        "train_time_seconds": 0.0,
        "prediction_time_seconds": float(prediction_time_seconds),
        "model_predict_time_seconds": float(
            pd.to_numeric(predictions["prediction_time_seconds"]).sum()
        ),
        "total_runtime_seconds": float(total_runtime_seconds),
        "retraining_train_time_seconds": float(
            retraining_metadata.get("train_time_seconds", np.nan)
        ),
        "status": "success",
        "error_message": "",
    }


def log_model_runtime(runtime_row: Mapping[str, Any]) -> None:
    log_runtime(
        make_runtime_record(
            experiment_name=EXPERIMENT_NAME,
            model_name=runtime_row["model_name"],
            feature_set=runtime_row["feature_set"],
            parameter_set_id=runtime_row["parameter_set_id"],
            train_start=runtime_row["train_start"],
            train_end=runtime_row["train_end"],
            validation_start=runtime_row["validation_start"],
            validation_end=runtime_row["validation_end"],
            n_train_rows=runtime_row["n_train_rows"],
            n_prediction_rows=runtime_row["n_prediction_rows"],
            train_time_seconds=0.0,
            prediction_time_seconds=runtime_row["prediction_time_seconds"],
            total_runtime_seconds=runtime_row["total_runtime_seconds"],
            status="success",
        )
    )


def validate_final_predictions(
    predictions: pd.DataFrame,
    *,
    final_test: pd.DataFrame,
    model_labels: Sequence[str],
) -> None:
    if predictions.empty:
        raise ValueError("Prediksi final test kosong.")
    required = {
        TIMESTAMP_COL,
        ACTUAL_COL,
        PREDICTION_COL,
        "model_label",
        "horizon_step",
        "origin_block",
        "used_actual_future_for_features",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Kolom prediksi final test tidak lengkap: {missing}")

    expected_index = final_test.index
    expected_actual = pd.to_numeric(final_test[TARGET_COL], errors="raise").to_numpy(
        dtype=float,
    )
    for model_label in model_labels:
        model_predictions = predictions[predictions["model_label"] == model_label].copy()
        if model_predictions.shape[0] != final_test.shape[0]:
            raise ValueError(
                f"Jumlah prediksi {model_label} tidak sama dengan final_test."
            )
        timestamps = pd.DatetimeIndex(model_predictions[TIMESTAMP_COL])
        if not timestamps.equals(expected_index):
            raise ValueError(f"Timestamp prediksi {model_label} tidak sejajar.")
        actual = pd.to_numeric(model_predictions[ACTUAL_COL], errors="raise").to_numpy(
            dtype=float,
        )
        if not np.allclose(actual, expected_actual):
            raise ValueError(f"Actual prediksi {model_label} tidak sama dengan label.")
        if model_predictions[TIMESTAMP_COL].duplicated().any():
            raise ValueError(f"Prediksi {model_label} memiliki duplicate timestamp.")
        if model_predictions["horizon_step"].min() < 1:
            raise ValueError(f"horizon_step {model_label} tidak valid.")
        if model_predictions["horizon_step"].max() > FORECAST_HORIZON:
            raise ValueError(f"horizon_step {model_label} melebihi horizon.")
        leakage_flags = coerce_bool_series(
            model_predictions["used_actual_future_for_features"],
            column_name="used_actual_future_for_features",
        )
        if leakage_flags.any():
            raise ValueError(f"Leakage flag aktif pada {model_label}.")


def build_final_metrics_table(metric_rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    metrics = pd.DataFrame(list(metric_rows))
    if metrics.empty:
        raise ValueError("Final metrics kosong.")
    metrics = metrics.sort_values(PRIMARY_METRIC, kind="mergesort").reset_index(drop=True)
    metrics.insert(0, "rank_by_primary_metric", np.arange(1, len(metrics) + 1))
    return metrics


def build_residual_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_label, group in predictions.groupby("model_label", sort=False):
        residual = pd.to_numeric(group["residual"], errors="raise")
        predicted = pd.to_numeric(group[PREDICTION_COL], errors="raise")
        actual = pd.to_numeric(group[ACTUAL_COL], errors="raise")
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


def save_actual_vs_predicted_plot(predictions: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_df = predictions.sort_values(TIMESTAMP_COL, kind="mergesort")
    actual = (
        plot_df[[TIMESTAMP_COL, ACTUAL_COL]]
        .drop_duplicates(subset=[TIMESTAMP_COL])
        .sort_values(TIMESTAMP_COL, kind="mergesort")
    )

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(actual[TIMESTAMP_COL], actual[ACTUAL_COL], label="Actual", linewidth=1.4)
    for model_label, group in plot_df.groupby("model_label", sort=False):
        ax.plot(
            group[TIMESTAMP_COL],
            group[PREDICTION_COL],
            label=model_label,
            linewidth=1.1,
            alpha=0.85,
        )
    ax.set_title("Final Test - Actual vs Predicted")
    ax.set_xlabel("UTC timestamp")
    ax.set_ylabel(TARGET_COL)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_residuals_plot(predictions: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_df = predictions.sort_values(TIMESTAMP_COL, kind="mergesort")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    for model_label, group in plot_df.groupby("model_label", sort=False):
        axes[0].plot(
            group[TIMESTAMP_COL],
            group["residual"],
            label=model_label,
            linewidth=1.0,
            alpha=0.85,
        )
        axes[1].hist(
            group["residual"],
            bins=40,
            alpha=0.5,
            label=model_label,
        )

    axes[0].axhline(0, color="black", linewidth=0.9)
    axes[0].set_title("Final Test - Residual Time Series")
    axes[0].set_xlabel("UTC timestamp")
    axes[0].set_ylabel("Residual")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend()
    axes[1].axvline(0, color="black", linewidth=0.9)
    axes[1].set_title("Final Test - Residual Distribution")
    axes[1].set_xlabel("Residual (actual - predicted)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_final_test_report(
    *,
    metadata: Mapping[str, Any],
    final_metrics: pd.DataFrame,
    final_runtime: pd.DataFrame,
    residual_summary: pd.DataFrame,
    horizon_error: pd.DataFrame,
    skip_plots: bool,
) -> str:
    winner = metadata["winner_by_primary_metric"]
    metric_view = final_metrics[
        [
            "rank_by_primary_metric",
            "model_label",
            "parameter_set_id",
            "mae",
            "rmse",
            "mape",
            "smape",
            "prediction_time_seconds",
            "retraining_train_time_seconds",
        ]
    ]
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
    outputs = metadata["outputs"]

    lines = [
        "# Final Testing",
        "",
        f"Run UTC: {metadata['started_at_utc']}",
        "",
        "## Scope",
        "",
        (
            "Tahap ini mengevaluasi model hasil retraining pada final_test yang "
            "dipisahkan sejak awal. Tidak ada tuning tambahan dan tidak ada "
            "pemilihan ulang parameter setelah melihat test metric."
        ),
        "",
        "## Methodology",
        "",
        (
            f"Forecast horizon utama adalah {FORECAST_HORIZON} jam. XGBoost "
            "diprediksi dengan recursive forecasting dalam blok 24 jam; actual "
            "final_test hanya masuk ke history setelah blok selesai dievaluasi. "
            "Prophet dievaluasi pada blok timestamp yang sama tanpa refit."
        ),
        "",
        "## Final Metric Ranking",
        "",
        dataframe_to_markdown(metric_view, float_digits=6),
        "",
        "## Residual Summary",
        "",
        dataframe_to_markdown(residual_summary, float_digits=6),
        "",
        "## Horizon Behavior",
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
        dataframe_to_markdown(final_runtime, float_digits=6),
        "",
        "## Interpretation",
        "",
        (
            f"Berdasarkan final test, model terbaik pada metric utama "
            f"{PRIMARY_METRIC} adalah {winner}. Hasil ini menjadi dasar tahap "
            "berikutnya untuk comparative evaluation dan error pattern analysis."
        ),
        "",
        "## Output Files",
        "",
        f"- Final metrics: `{outputs['final_metrics']}`",
        f"- Final runtime: `{outputs['final_runtime']}`",
        f"- Final predictions: `{outputs['final_predictions']}`",
        f"- Residual summary: `{outputs['residual_summary']}`",
        f"- Horizon error summary: `{outputs['horizon_error_summary']}`",
        f"- Experiment metadata: `{outputs['metadata']}`",
        f"- Report mirror: `{outputs['report']}`",
    ]
    if not skip_plots:
        lines.extend(
            [
                f"- Actual vs predicted plot: `{outputs['actual_vs_predicted_plot']}`",
                f"- Residuals plot: `{outputs['residuals_plot']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def summarize_timeseries(df: pd.DataFrame) -> dict[str, Any]:
    target = pd.to_numeric(df[TARGET_COL], errors="raise")
    return {
        "n_rows": int(df.shape[0]),
        "utc_start": df.index.min().isoformat(),
        "utc_end": df.index.max().isoformat(),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "target_mean": float(target.mean()),
    }


def leakage_guardrail_notes() -> list[str]:
    return [
        "Models are loaded from retraining artifacts; no tuning is performed.",
        "Final test labels are used only for metric computation.",
        "XGBoost predictions use forecast_validation_window() with recursive 24h blocks.",
        "XGBoost does not use precomputed final_test feature rows.",
        "Actual final_test values enter XGBoost history only after each 24h block.",
        "Prediction outputs must keep used_actual_future_for_features=False.",
    ]


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


def select_winner(metrics: pd.DataFrame, *, metric: str) -> str:
    if metrics.empty:
        raise ValueError("Metrics kosong.")
    return str(metrics.sort_values(metric, kind="mergesort").iloc[0]["model_label"])


def stringify_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    return {key: str(value) for key, value in paths.items()}


if __name__ == "__main__":
    main()

"""
Standalone SARIMA sanity test.

This script is intentionally separate from the main research workflow. It fits
an auto-SARIMA model on train_val, evaluates it on final_test using 24-hour
rolling-origin blocks, and writes outputs under:

    outputs/experiments/sarima_sanity_test/

It does not update the official comparative evaluation artifacts.

Example:
    python -m src.experiments.sarima_sanity_test
    python -m src.experiments.sarima_sanity_test --overwrite --skip-plots
    python -m src.experiments.sarima_sanity_test --diagnostics-only
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
    MODELING_TZ,
    PRIMARY_METRIC,
    TARGET_COL,
    TIMESTAMP_COL,
    TRAIN_VAL_PATH,
    ensure_dirs,
)
from src.forecasting import ACTUAL_COL, PREDICTION_COL, validate_prediction_output
from src.metrics import compute_all_metrics
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


EXPERIMENT_NAME = "sarima_sanity_test"
MODEL_KEY = "sarima_auto"
MODEL_LABEL = "SARIMA-Auto"
MODEL_NAME = "sarima"
OUTPUT_DIR = EXPERIMENTS_DIR / "sarima_sanity_test"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a standalone auto-SARIMA sanity test on NYC Taxi hourly data."
    )
    parser.add_argument(
        "--seasonal-period",
        type=int,
        default=24,
        help="SARIMA seasonal period. Default 24 for daily hourly seasonality.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Use only the last N train_val rows. Default 0 means full train_val.",
    )
    parser.add_argument("--max-p", type=int, default=2)
    parser.add_argument("--max-q", type=int, default=2)
    parser.add_argument("--max-P", type=int, default=1)
    parser.add_argument("--max-Q", type=int, default=1)
    parser.add_argument("--max-d", type=int, default=2)
    parser.add_argument("--max-D", type=int, default=1)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument(
        "--maxiter",
        type=int,
        default=30,
        help="Maximum optimizer iterations during auto_arima fitting.",
    )
    parser.add_argument(
        "--update-maxiter",
        type=int,
        default=0,
        help=(
            "Optimizer iterations when adding each completed 24h actual block. "
            "Default 0 updates state/history without an expensive refit."
        ),
    )
    parser.add_argument(
        "--no-seasonal",
        action="store_true",
        help="Disable seasonal terms. Kept only as a diagnostic fallback.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip matplotlib figure generation.",
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help=(
            "Build residual diagnostics from existing SARIMA predictions without "
            "refitting auto_arima."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing SARIMA sanity-test outputs.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.diagnostics_only:
        metadata = run_existing_residual_diagnostics(skip_plots=args.skip_plots)
        diagnostics = metadata.get("residual_diagnostics", {}).get("summary", {})
        print("SARIMA residual diagnostics selesai.")
        print(f"Ljung-Box white noise passed: {diagnostics.get('ljung_box_all_checked_lags_pass_white_noise')}")
        print(f"Normality passed: {diagnostics.get('normality_all_tests_pass')}")
        print(f"Summary: {metadata['outputs']['summary']}")
        return

    metadata = run_sarima_sanity_test(
        seasonal_period=args.seasonal_period,
        max_train_rows=args.max_train_rows,
        max_p=args.max_p,
        max_q=args.max_q,
        max_P=args.max_P,
        max_Q=args.max_Q,
        max_d=args.max_d,
        max_D=args.max_D,
        max_order=args.max_order,
        maxiter=args.maxiter,
        update_maxiter=args.update_maxiter,
        seasonal=not args.no_seasonal,
        skip_plots=args.skip_plots,
        overwrite=args.overwrite,
    )
    print("SARIMA sanity test selesai.")
    print(f"MAE: {metadata['metrics']['mae']:.6f}")
    print(f"RMSE: {metadata['metrics']['rmse']:.6f}")
    print(f"Output dir: {metadata['outputs']['base']}")
    print(f"Summary: {metadata['outputs']['summary']}")


def run_sarima_sanity_test(
    *,
    seasonal_period: int = 24,
    max_train_rows: int = 0,
    max_p: int = 2,
    max_q: int = 2,
    max_P: int = 1,
    max_Q: int = 1,
    max_d: int = 2,
    max_D: int = 1,
    max_order: int = 4,
    maxiter: int = 30,
    update_maxiter: int = 0,
    seasonal: bool = True,
    skip_plots: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    validate_sarima_config(
        seasonal_period=seasonal_period,
        max_train_rows=max_train_rows,
        max_p=max_p,
        max_q=max_q,
        max_P=max_P,
        max_Q=max_Q,
        max_d=max_d,
        max_D=max_D,
        max_order=max_order,
        maxiter=maxiter,
        update_maxiter=update_maxiter,
    )

    outputs = output_paths()
    timer = start_timer()
    started_at = utc_now_iso()
    status = "success"
    error_message = ""

    try:
        ensure_dirs()
        ensure_output_dirs(outputs)
        validate_overwrite(outputs, overwrite=overwrite)

        train_val_full = load_split_timeseries(TRAIN_VAL_PATH)
        final_test = load_split_timeseries(FINAL_TEST_PATH)
        validate_final_test_scope(train_val_full, final_test)
        train_used = select_training_window(train_val_full, max_train_rows=max_train_rows)

        params = build_auto_arima_params(
            seasonal_period=seasonal_period,
            max_p=max_p,
            max_q=max_q,
            max_P=max_P,
            max_Q=max_Q,
            max_d=max_d,
            max_D=max_D,
            max_order=max_order,
            maxiter=maxiter,
            seasonal=seasonal,
        )
        evaluation_config = build_evaluation_config(
            max_train_rows=max_train_rows,
            update_maxiter=update_maxiter,
            forecast_horizon=FORECAST_HORIZON,
        )
        save_experiment_metadata(
            {
                "experiment_name": EXPERIMENT_NAME,
                "model_name": MODEL_NAME,
                "model_label": MODEL_LABEL,
                "params": params,
                "evaluation_config": evaluation_config,
                "methodology_note": methodology_notes(),
            },
            outputs["params"],
        )

        fit_timer = start_timer()
        model = fit_auto_sarima(train_used[TARGET_COL], params=params)
        train_time_seconds = elapsed_seconds(fit_timer)

        model_summary_text = safe_model_summary(model)
        outputs["model_summary"].write_text(model_summary_text, encoding="utf-8")

        prediction_timer = start_timer()
        forecast_result = forecast_sarima_final_window(
            model,
            train_used,
            final_test,
            horizon=FORECAST_HORIZON,
            update_maxiter=update_maxiter,
        )
        prediction_time_seconds = elapsed_seconds(prediction_timer)
        predictions = forecast_result["predictions"]

        predictions = prepare_prediction_frame(predictions)
        validate_sarima_predictions(predictions, final_test=final_test)

        metrics = build_metric_row(
            predictions,
            train_val_full=train_val_full,
            train_used=train_used,
            final_test=final_test,
            model=model,
            params=params,
            train_time_seconds=train_time_seconds,
            prediction_time_seconds=prediction_time_seconds,
            forecast_time_seconds=forecast_result["forecast_time_seconds"],
            update_time_seconds=forecast_result["update_time_seconds"],
        )
        metrics_df = pd.DataFrame([metrics])
        residual_summary = build_residual_summary(predictions)
        horizon_error = build_horizon_error_summary(predictions)
        runtime_summary = build_runtime_summary(metrics)
        reference_comparison = build_reference_comparison(metrics)
        residual_diagnostics = build_residual_diagnostics(predictions)

        predictions.to_csv(outputs["predictions"], index=False)
        metrics_df.to_csv(outputs["metrics"], index=False)
        residual_summary.to_csv(outputs["residual_summary"], index=False)
        horizon_error.to_csv(outputs["horizon_error_summary"], index=False)
        runtime_summary.to_csv(outputs["runtime_summary"], index=False)
        reference_comparison.to_csv(outputs["reference_comparison"], index=False)
        save_residual_diagnostic_outputs(residual_diagnostics, outputs)

        if not skip_plots:
            save_actual_vs_predicted_plot(predictions, outputs["actual_vs_predicted_plot"])
            save_residuals_plot(predictions, outputs["residuals_plot"])
            save_residual_acf_pacf_plot(
                residual_diagnostics["acf_pacf"],
                outputs["residual_acf_pacf_plot"],
                confint=float(
                    residual_diagnostics["summary"].iloc[0]["acf_95_confidence_bound"]
                ),
            )
            save_residual_normality_plot(
                predictions,
                outputs["residual_normality_plot"],
            )

        total_runtime_seconds = elapsed_seconds(timer)
        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "stage": "standalone_posthoc_sarima_sanity_test",
            "status": status,
            "error_message": error_message,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "total_runtime_seconds": total_runtime_seconds,
            "model_key": MODEL_KEY,
            "model_label": MODEL_LABEL,
            "model_name": MODEL_NAME,
            "feature_set": metrics["feature_set"],
            "parameter_set_id": metrics["parameter_set_id"],
            "forecast_horizon_hours": int(FORECAST_HORIZON),
            "train_val_path": str(TRAIN_VAL_PATH),
            "final_test_path": str(FINAL_TEST_PATH),
            "train_val_full_summary": summarize_timeseries(train_val_full),
            "train_used_summary": summarize_timeseries(train_used),
            "final_test_summary": summarize_timeseries(final_test),
            "params": params,
            "evaluation_config": evaluation_config,
            "selected_order": selected_order_payload(model),
            "metrics": metrics,
            "residual_diagnostics": residual_diagnostics_metadata(residual_diagnostics),
            "time_cost_computing": {
                "train_time_seconds": float(train_time_seconds),
                "prediction_time_seconds": float(prediction_time_seconds),
                "forecast_time_seconds": float(forecast_result["forecast_time_seconds"]),
                "update_time_seconds": float(forecast_result["update_time_seconds"]),
                "total_runtime_seconds": float(total_runtime_seconds),
            },
            "methodology_note": methodology_notes(),
            "outputs": stringify_paths(outputs),
        }

        summary_text = render_summary(
            metadata=metadata,
            metrics_df=metrics_df,
            residual_summary=residual_summary,
            horizon_error=horizon_error,
            runtime_summary=runtime_summary,
            reference_comparison=reference_comparison,
            residual_diagnostic_summary=residual_diagnostics["summary"],
            ljung_box=residual_diagnostics["ljung_box"],
            normality_tests=residual_diagnostics["normality"],
            skip_plots=skip_plots,
        )
        outputs["summary"].write_text(summary_text, encoding="utf-8")
        save_experiment_metadata(metadata, outputs["metadata"])

        log_runtime(
            make_runtime_record(
                experiment_name=EXPERIMENT_NAME,
                model_name=MODEL_NAME,
                feature_set=metrics["feature_set"],
                parameter_set_id=metrics["parameter_set_id"],
                train_start=train_used.index.min(),
                train_end=train_used.index.max(),
                validation_start=final_test.index.min(),
                validation_end=final_test.index.max(),
                n_train_rows=int(train_used.shape[0]),
                n_prediction_rows=int(predictions.shape[0]),
                train_time_seconds=float(train_time_seconds),
                prediction_time_seconds=float(prediction_time_seconds),
                total_runtime_seconds=float(total_runtime_seconds),
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
            "stage": "standalone_posthoc_sarima_sanity_test",
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
                model_name=MODEL_NAME,
                feature_set="univariate_sarima",
                total_runtime_seconds=total_runtime_seconds,
                status=status,
                error_message=error_message,
            )
        )
        append_experiment_run(failure_metadata)
        raise


def run_existing_residual_diagnostics(*, skip_plots: bool = False) -> dict[str, Any]:
    """
    Rebuild residual diagnostics from existing SARIMA prediction output.

    This avoids refitting auto-SARIMA when the user only needs diagnostic tests.
    """
    outputs = output_paths()
    ensure_output_dirs(outputs)
    timer = start_timer()
    started_at = utc_now_iso()

    if not outputs["predictions"].exists():
        raise FileNotFoundError(
            f"Existing SARIMA predictions not found: {outputs['predictions']}"
        )
    if not outputs["metrics"].exists():
        raise FileNotFoundError(f"Existing SARIMA metrics not found: {outputs['metrics']}")

    predictions = load_existing_predictions(outputs["predictions"])
    metrics_df = pd.read_csv(outputs["metrics"])
    residual_summary = (
        pd.read_csv(outputs["residual_summary"])
        if outputs["residual_summary"].exists()
        else build_residual_summary(predictions)
    )
    horizon_error = (
        pd.read_csv(outputs["horizon_error_summary"])
        if outputs["horizon_error_summary"].exists()
        else build_horizon_error_summary(predictions)
    )
    runtime_summary = (
        pd.read_csv(outputs["runtime_summary"])
        if outputs["runtime_summary"].exists()
        else build_runtime_summary(metrics_df.iloc[0].to_dict())
    )
    reference_comparison = (
        pd.read_csv(outputs["reference_comparison"])
        if outputs["reference_comparison"].exists()
        else build_reference_comparison(metrics_df.iloc[0].to_dict())
    )

    residual_diagnostics = build_residual_diagnostics(predictions)
    save_residual_diagnostic_outputs(residual_diagnostics, outputs)

    if not skip_plots:
        save_residual_acf_pacf_plot(
            residual_diagnostics["acf_pacf"],
            outputs["residual_acf_pacf_plot"],
            confint=float(
                residual_diagnostics["summary"].iloc[0]["acf_95_confidence_bound"]
            ),
        )
        save_residual_normality_plot(predictions, outputs["residual_normality_plot"])

    diagnostics_runtime_seconds = elapsed_seconds(timer)
    metadata = load_json_if_exists(outputs["metadata"])
    if not metadata:
        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "stage": "standalone_posthoc_sarima_sanity_test",
            "status": "success",
            "started_at_utc": started_at,
            "model_key": MODEL_KEY,
            "model_label": MODEL_LABEL,
            "model_name": MODEL_NAME,
        }

    metadata["residual_diagnostics"] = residual_diagnostics_metadata(
        residual_diagnostics
    )
    metadata["residual_diagnostics"]["diagnostics_run_utc"] = started_at
    metadata["residual_diagnostics"]["diagnostics_runtime_seconds"] = (
        diagnostics_runtime_seconds
    )
    metadata.setdefault("time_cost_computing", {})
    metadata["time_cost_computing"]["residual_diagnostics_runtime_seconds"] = (
        diagnostics_runtime_seconds
    )
    metadata["outputs"] = stringify_paths(outputs)

    summary_text = render_summary(
        metadata=metadata,
        metrics_df=metrics_df,
        residual_summary=residual_summary,
        horizon_error=horizon_error,
        runtime_summary=runtime_summary,
        reference_comparison=reference_comparison,
        residual_diagnostic_summary=residual_diagnostics["summary"],
        ljung_box=residual_diagnostics["ljung_box"],
        normality_tests=residual_diagnostics["normality"],
        skip_plots=skip_plots,
    )
    outputs["summary"].write_text(summary_text, encoding="utf-8")
    save_experiment_metadata(metadata, outputs["metadata"])

    feature_set = (
        str(metrics_df.iloc[0].get("feature_set", "univariate_auto_sarima"))
        if not metrics_df.empty
        else "univariate_auto_sarima"
    )
    parameter_set_id = (
        str(metrics_df.iloc[0].get("parameter_set_id", "residual_diagnostics"))
        if not metrics_df.empty
        else "residual_diagnostics"
    )
    log_runtime(
        make_runtime_record(
            experiment_name=f"{EXPERIMENT_NAME}_residual_diagnostics",
            model_name=MODEL_NAME,
            feature_set=feature_set,
            parameter_set_id=parameter_set_id,
            n_prediction_rows=int(predictions.shape[0]),
            total_runtime_seconds=diagnostics_runtime_seconds,
            status="success",
        )
    )
    append_experiment_run(
        {
            "experiment_name": f"{EXPERIMENT_NAME}_residual_diagnostics",
            "model_name": MODEL_NAME,
            "model_label": MODEL_LABEL,
            "feature_set": feature_set,
            "parameter_set_id": parameter_set_id,
            "status": "success",
            "total_runtime_seconds": diagnostics_runtime_seconds,
            "residual_diagnostics": metadata["residual_diagnostics"],
        }
    )
    return metadata


def load_existing_predictions(path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(path)
    required = {TIMESTAMP_COL, ACTUAL_COL, PREDICTION_COL}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Existing predictions are missing columns: {missing}")
    predictions[TIMESTAMP_COL] = pd.to_datetime(
        predictions[TIMESTAMP_COL],
        utc=True,
        errors="raise",
    )
    predictions[ACTUAL_COL] = pd.to_numeric(predictions[ACTUAL_COL], errors="raise")
    predictions[PREDICTION_COL] = pd.to_numeric(
        predictions[PREDICTION_COL],
        errors="raise",
    )
    if "residual" not in predictions.columns:
        predictions["residual"] = predictions[ACTUAL_COL] - predictions[PREDICTION_COL]
    if "absolute_error" not in predictions.columns:
        predictions["absolute_error"] = predictions["residual"].abs()
    if "squared_error" not in predictions.columns:
        predictions["squared_error"] = np.square(predictions["residual"])
    return predictions.sort_values(TIMESTAMP_COL, kind="mergesort").reset_index(drop=True)


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def validate_sarima_config(
    *,
    seasonal_period: int,
    max_train_rows: int,
    max_p: int,
    max_q: int,
    max_P: int,
    max_Q: int,
    max_d: int,
    max_D: int,
    max_order: int,
    maxiter: int,
    update_maxiter: int,
) -> None:
    integer_args = {
        "seasonal_period": seasonal_period,
        "max_train_rows": max_train_rows,
        "max_p": max_p,
        "max_q": max_q,
        "max_P": max_P,
        "max_Q": max_Q,
        "max_d": max_d,
        "max_D": max_D,
        "max_order": max_order,
        "maxiter": maxiter,
        "update_maxiter": update_maxiter,
    }
    for name, value in integer_args.items():
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if name in {"max_train_rows", "update_maxiter"}:
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")
        elif value <= 0 and name not in {"max_p", "max_q", "max_P", "max_Q", "max_d", "max_D"}:
            raise ValueError(f"{name} must be > 0.")
        elif value < 0:
            raise ValueError(f"{name} must be >= 0.")

    if seasonal_period < 2:
        raise ValueError("seasonal_period must be >= 2 for seasonal SARIMA.")
    if max_order < 1:
        raise ValueError("max_order must be >= 1.")


def output_paths() -> dict[str, Path]:
    metrics_dir = OUTPUT_DIR / "metrics"
    predictions_dir = OUTPUT_DIR / "predictions"
    figures_dir = OUTPUT_DIR / "figures"
    summaries_dir = OUTPUT_DIR / "summaries"
    return {
        "base": OUTPUT_DIR,
        "metrics_dir": metrics_dir,
        "predictions_dir": predictions_dir,
        "figures_dir": figures_dir,
        "summaries_dir": summaries_dir,
        "params": OUTPUT_DIR / "params.json",
        "metadata": OUTPUT_DIR / "experiment_metadata.json",
        "model_summary": summaries_dir / "model_summary.txt",
        "summary": summaries_dir / "sarima_sanity_test_summary.md",
        "metrics": metrics_dir / "final_metrics.csv",
        "residual_summary": metrics_dir / "residual_summary.csv",
        "horizon_error_summary": metrics_dir / "horizon_error_summary.csv",
        "residual_diagnostic_summary": metrics_dir / "residual_diagnostic_summary.csv",
        "residual_acf_pacf": metrics_dir / "residual_acf_pacf.csv",
        "residual_ljung_box": metrics_dir / "residual_ljung_box.csv",
        "residual_normality_tests": metrics_dir / "residual_normality_tests.csv",
        "runtime_summary": metrics_dir / "runtime_summary.csv",
        "reference_comparison": metrics_dir / "reference_metric_comparison.csv",
        "predictions": predictions_dir / "final_predictions.csv",
        "actual_vs_predicted_plot": figures_dir / "actual_vs_predicted.png",
        "residuals_plot": figures_dir / "residuals.png",
        "residual_acf_pacf_plot": figures_dir / "residual_acf_pacf.png",
        "residual_normality_plot": figures_dir / "residual_normality_qq.png",
    }


def ensure_output_dirs(paths: Mapping[str, Path]) -> None:
    for key in ["base", "metrics_dir", "predictions_dir", "figures_dir", "summaries_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)


def validate_overwrite(paths: Mapping[str, Path], *, overwrite: bool) -> None:
    output_keys = [
        "params",
        "metadata",
        "model_summary",
        "summary",
        "metrics",
        "residual_summary",
        "horizon_error_summary",
        "residual_diagnostic_summary",
        "residual_acf_pacf",
        "residual_ljung_box",
        "residual_normality_tests",
        "runtime_summary",
        "reference_comparison",
        "predictions",
        "actual_vs_predicted_plot",
        "residuals_plot",
        "residual_acf_pacf_plot",
        "residual_normality_plot",
    ]
    existing = [str(paths[key]) for key in output_keys if paths[key].exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output SARIMA sanity test already exists. Use --overwrite to rerun. "
            f"Existing files: {existing}"
        )


def validate_final_test_scope(train_val: pd.DataFrame, final_test: pd.DataFrame) -> None:
    if train_val.empty:
        raise ValueError("train_val is empty.")
    if final_test.empty:
        raise ValueError("final_test is empty.")
    if train_val.index.max() >= final_test.index.min():
        raise ValueError("train_val must end before final_test starts.")
    if train_val.index.intersection(final_test.index).size > 0:
        raise ValueError("train_val and final_test overlap.")
    if final_test.shape[0] < FORECAST_HORIZON:
        raise ValueError("final_test must contain at least one full forecast horizon.")


def select_training_window(train_val: pd.DataFrame, *, max_train_rows: int) -> pd.DataFrame:
    if max_train_rows <= 0:
        return train_val.copy()
    if max_train_rows < FORECAST_HORIZON * 14:
        raise ValueError(
            "max_train_rows is too small for a meaningful hourly SARIMA sanity test."
        )
    return train_val.tail(max_train_rows).copy()


def build_auto_arima_params(
    *,
    seasonal_period: int,
    max_p: int,
    max_q: int,
    max_P: int,
    max_Q: int,
    max_d: int,
    max_D: int,
    max_order: int,
    maxiter: int,
    seasonal: bool,
) -> dict[str, Any]:
    return {
        "start_p": 0,
        "start_q": 0,
        "max_p": int(max_p),
        "max_q": int(max_q),
        "start_P": 0,
        "start_Q": 0,
        "max_P": int(max_P),
        "max_Q": int(max_Q),
        "max_d": int(max_d),
        "max_D": int(max_D),
        "max_order": int(max_order),
        "m": int(seasonal_period if seasonal else 1),
        "seasonal": bool(seasonal),
        "stepwise": True,
        "information_criterion": "aic",
        "test": "kpss",
        "seasonal_test": "ocsb",
        "suppress_warnings": True,
        "error_action": "ignore",
        "trace": False,
        "n_jobs": 1,
        "maxiter": int(maxiter),
        "with_intercept": "auto",
    }


def build_evaluation_config(
    *,
    max_train_rows: int,
    update_maxiter: int,
    forecast_horizon: int,
) -> dict[str, Any]:
    return {
        "max_train_rows": int(max_train_rows),
        "train_window_policy": (
            "full_train_val" if max_train_rows <= 0 else "tail_train_val_window"
        ),
        "forecast_horizon_hours": int(forecast_horizon),
        "rolling_origin_block_hours": int(forecast_horizon),
        "update_with_actuals_after_each_block": True,
        "update_maxiter": int(update_maxiter),
        "final_test_used_for_tuning": False,
    }


def fit_auto_sarima(y: pd.Series, *, params: Mapping[str, Any]) -> Any:
    try:
        from pmdarima.arima import auto_arima
    except ImportError as exc:
        raise ImportError(
            "pmdarima is not available. Install pmdarima before running SARIMA."
        ) from exc

    target = pd.to_numeric(y, errors="raise").astype(float)
    if target.isna().any():
        raise ValueError("SARIMA training target contains missing values.")
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("SARIMA training target contains non-finite values.")
    return auto_arima(target.to_numpy(dtype=float), **dict(params))


def forecast_sarima_final_window(
    model: Any,
    train_history: pd.DataFrame,
    final_test: pd.DataFrame,
    *,
    horizon: int,
    update_maxiter: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    forecast_time_seconds = 0.0
    update_time_seconds = 0.0
    current_origin = train_history.index.max()
    feature_set = feature_set_name(model)
    parameter_set_id = parameter_set_name(model)

    for block_id, start in enumerate(range(0, len(final_test), horizon), start=1):
        block = final_test.iloc[start : start + horizon]
        block_horizon = int(block.shape[0])

        forecast_timer = start_timer()
        predicted_values = np.asarray(model.predict(n_periods=block_horizon), dtype=float)
        block_forecast_time = elapsed_seconds(forecast_timer)
        forecast_time_seconds += block_forecast_time

        if predicted_values.shape[0] != block_horizon:
            raise ValueError("SARIMA prediction length does not match block horizon.")
        if not np.isfinite(predicted_values).all():
            raise ValueError("SARIMA produced non-finite predictions.")

        per_step_forecast_time = block_forecast_time / block_horizon
        actual_values = pd.to_numeric(block[TARGET_COL], errors="raise").to_numpy(dtype=float)
        for step, (timestamp, actual, predicted) in enumerate(
            zip(block.index, actual_values, predicted_values),
            start=1,
        ):
            records.append(
                {
                    TIMESTAMP_COL: timestamp,
                    ACTUAL_COL: float(actual),
                    PREDICTION_COL: float(predicted),
                    "model_name": MODEL_NAME,
                    "feature_set": feature_set,
                    "forecast_origin": current_origin,
                    "horizon_step": int(step),
                    "fold": "final_test",
                    "parameter_set_id": parameter_set_id,
                    "prediction_time_seconds": float(per_step_forecast_time),
                    "update_time_seconds": 0.0,
                    "origin_block": int(block_id),
                    "validation_start": block.index.min(),
                    "validation_end": block.index.max(),
                    "used_actual_future_for_features": False,
                }
            )

        update_timer = start_timer()
        model.update(actual_values, maxiter=update_maxiter)
        block_update_time = elapsed_seconds(update_timer)
        update_time_seconds += block_update_time
        if records:
            for row in records[-block_horizon:]:
                row["update_time_seconds"] = float(block_update_time / block_horizon)
        current_origin = block.index.max()

    predictions = pd.DataFrame.from_records(records)
    predictions.attrs["prediction_time_seconds_total"] = round(forecast_time_seconds, 9)
    predictions.attrs["update_time_seconds_total"] = round(update_time_seconds, 9)
    validate_prediction_output(
        predictions,
        expected_index=final_test.index,
        horizon=len(final_test),
    )
    return {
        "predictions": predictions,
        "forecast_time_seconds": round(float(forecast_time_seconds), 9),
        "update_time_seconds": round(float(update_time_seconds), 9),
    }


def prepare_prediction_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    prepared = predictions.copy()
    prepared[TIMESTAMP_COL] = pd.to_datetime(prepared[TIMESTAMP_COL], utc=True)
    prepared["forecast_origin"] = pd.to_datetime(prepared["forecast_origin"], utc=True)
    for column in [
        ACTUAL_COL,
        PREDICTION_COL,
        "prediction_time_seconds",
        "update_time_seconds",
    ]:
        prepared[column] = pd.to_numeric(prepared[column], errors="raise")
    prepared["horizon_step"] = pd.to_numeric(
        prepared["horizon_step"],
        errors="raise",
    ).astype(int)
    prepared["origin_block"] = pd.to_numeric(
        prepared["origin_block"],
        errors="raise",
    ).astype(int)
    prepared["used_actual_future_for_features"] = prepared[
        "used_actual_future_for_features"
    ].astype(bool)
    prepared["model_key"] = MODEL_KEY
    prepared["model_label"] = MODEL_LABEL
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


def validate_sarima_predictions(
    predictions: pd.DataFrame,
    *,
    final_test: pd.DataFrame,
) -> None:
    if predictions.shape[0] != final_test.shape[0]:
        raise ValueError("SARIMA predictions do not match final_test row count.")
    timestamps = pd.DatetimeIndex(predictions[TIMESTAMP_COL])
    if not timestamps.equals(final_test.index):
        raise ValueError("SARIMA prediction timestamps do not align with final_test.")
    expected_actual = pd.to_numeric(final_test[TARGET_COL], errors="raise").to_numpy(
        dtype=float,
    )
    observed_actual = pd.to_numeric(predictions[ACTUAL_COL], errors="raise").to_numpy(
        dtype=float,
    )
    if not np.allclose(expected_actual, observed_actual):
        raise ValueError("SARIMA actual values do not match final_test labels.")
    if predictions["used_actual_future_for_features"].astype(bool).any():
        raise ValueError("SARIMA leakage flag must remain False.")


def build_metric_row(
    predictions: pd.DataFrame,
    *,
    train_val_full: pd.DataFrame,
    train_used: pd.DataFrame,
    final_test: pd.DataFrame,
    model: Any,
    params: Mapping[str, Any],
    train_time_seconds: float,
    prediction_time_seconds: float,
    forecast_time_seconds: float,
    update_time_seconds: float,
) -> dict[str, Any]:
    metric_values = compute_all_metrics(
        predictions[ACTUAL_COL],
        predictions[PREDICTION_COL],
    )
    residual = pd.to_numeric(predictions["residual"], errors="raise")
    predicted = pd.to_numeric(predictions[PREDICTION_COL], errors="raise")
    actual = pd.to_numeric(predictions[ACTUAL_COL], errors="raise")
    row: dict[str, Any] = {
        "model_key": MODEL_KEY,
        "model_label": MODEL_LABEL,
        "model_name": MODEL_NAME,
        "feature_set": feature_set_name(model),
        "parameter_set_id": parameter_set_name(model),
        "params_json": json.dumps(dict(params), sort_keys=True),
        "order": str(getattr(model, "order", "")),
        "seasonal_order": str(getattr(model, "seasonal_order", "")),
        "aic": safe_float_call(model, "aic"),
        "bic": safe_float_call(model, "bic"),
        "n_train_val_full_rows": int(train_val_full.shape[0]),
        "n_train_rows_used": int(train_used.shape[0]),
        "n_prediction_rows": int(predictions.shape[0]),
        "final_test_start": final_test.index.min().isoformat(),
        "final_test_end": final_test.index.max().isoformat(),
        "forecast_horizon_hours": int(FORECAST_HORIZON),
        "n_origin_blocks": int(predictions["origin_block"].nunique()),
        "train_time_seconds": float(train_time_seconds),
        "prediction_time_seconds": float(prediction_time_seconds),
        "forecast_time_seconds": float(forecast_time_seconds),
        "update_time_seconds": float(update_time_seconds),
        "total_runtime_seconds": float(train_time_seconds + prediction_time_seconds),
        "negative_prediction_count": int((predicted < 0).sum()),
        "overprediction_rate": float((predicted > actual).mean()),
        "underprediction_rate": float((predicted < actual).mean()),
        "mean_error_actual_minus_predicted": float(residual.mean()),
        "max_absolute_error": float(predictions["absolute_error"].max()),
        "used_actual_future_for_features": False,
        "final_test_used_for_tuning": False,
        "official_workflow_member": False,
    }
    row.update(metric_values)
    return row


def build_residual_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    residual = pd.to_numeric(predictions["residual"], errors="raise")
    predicted = pd.to_numeric(predictions[PREDICTION_COL], errors="raise")
    actual = pd.to_numeric(predictions[ACTUAL_COL], errors="raise")
    return pd.DataFrame(
        [
            {
                "model_label": MODEL_LABEL,
                "n_predictions": int(predictions.shape[0]),
                "mean_error_actual_minus_predicted": float(residual.mean()),
                "median_error_actual_minus_predicted": float(residual.median()),
                "residual_std": float(residual.std(ddof=1)),
                "mean_absolute_error": float(predictions["absolute_error"].mean()),
                "rmse": float(np.sqrt(np.square(residual).mean())),
                "overprediction_rate": float((predicted > actual).mean()),
                "underprediction_rate": float((predicted < actual).mean()),
                "negative_prediction_count": int((predicted < 0).sum()),
                "max_absolute_error": float(predictions["absolute_error"].max()),
            }
        ]
    )


def build_horizon_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon_step, group in predictions.groupby("horizon_step", sort=True):
        residual = pd.to_numeric(group["residual"], errors="raise")
        rows.append(
            {
                "model_label": MODEL_LABEL,
                "horizon_step": int(horizon_step),
                "n_predictions": int(group.shape[0]),
                "mae": float(np.abs(residual).mean()),
                "rmse": float(np.sqrt(np.square(residual).mean())),
                "mean_error_actual_minus_predicted": float(residual.mean()),
            }
        )
    return pd.DataFrame(rows)


def build_residual_diagnostics(
    predictions: pd.DataFrame,
    *,
    max_lag: int = 168,
    alpha: float = 0.05,
) -> dict[str, pd.DataFrame]:
    residual = residual_series(predictions)
    if residual.shape[0] <= max_lag + 1:
        max_lag = max(1, int(residual.shape[0] // 4))

    from scipy import stats
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf, pacf

    values = residual.to_numpy(dtype=float)
    n = int(values.shape[0])
    confidence_bound = float(1.96 / np.sqrt(n))

    acf_values = acf(values, nlags=max_lag, fft=True, missing="raise")
    pacf_values = pacf(values, nlags=max_lag, method="ywm")
    acf_pacf = pd.DataFrame(
        {
            "lag": np.arange(0, max_lag + 1, dtype=int),
            "acf": acf_values,
            "pacf": pacf_values,
        }
    )
    acf_pacf["acf_abs_exceeds_95_confidence"] = (
        acf_pacf["lag"].gt(0) & acf_pacf["acf"].abs().gt(confidence_bound)
    )
    acf_pacf["pacf_abs_exceeds_95_confidence"] = (
        acf_pacf["lag"].gt(0) & acf_pacf["pacf"].abs().gt(confidence_bound)
    )

    checked_lags = [lag for lag in [24, 48, 72, 168] if lag <= max_lag]
    ljung_box = acorr_ljungbox(values, lags=checked_lags, return_df=True)
    ljung_box = ljung_box.reset_index().rename(columns={"index": "lag"})
    ljung_box["lag"] = ljung_box["lag"].astype(int)
    ljung_box["alpha"] = float(alpha)
    ljung_box["reject_white_noise_null"] = ljung_box["lb_pvalue"] < alpha
    ljung_box["interpretation"] = np.where(
        ljung_box["reject_white_noise_null"],
        "residual_autocorrelation_detected",
        "no_significant_autocorrelation_detected",
    )

    normality = build_residual_normality_tests(values, alpha=alpha)
    center_randomness = build_center_randomness_tests(values, alpha=alpha)

    acf_nonzero = acf_pacf[acf_pacf["lag"] > 0]
    summary = {
        "model_label": MODEL_LABEL,
        "n_residuals": n,
        "residual_mean": float(np.mean(values)),
        "residual_median": float(np.median(values)),
        "residual_std": float(np.std(values, ddof=1)),
        "residual_skewness": float(stats.skew(values, bias=False)),
        "residual_excess_kurtosis": float(stats.kurtosis(values, fisher=True, bias=False)),
        "acf_95_confidence_bound": confidence_bound,
        "acf_lag_1": lag_value(acf_pacf, "acf", 1),
        "acf_lag_24": lag_value(acf_pacf, "acf", 24),
        "acf_lag_168": lag_value(acf_pacf, "acf", 168),
        "pacf_lag_1": lag_value(acf_pacf, "pacf", 1),
        "pacf_lag_24": lag_value(acf_pacf, "pacf", 24),
        "pacf_lag_168": lag_value(acf_pacf, "pacf", 168),
        "significant_acf_lags_count_1_to_max_lag": int(
            acf_nonzero["acf_abs_exceeds_95_confidence"].sum()
        ),
        "significant_pacf_lags_count_1_to_max_lag": int(
            acf_nonzero["pacf_abs_exceeds_95_confidence"].sum()
        ),
        "max_abs_acf_lag_1_to_max_lag": float(acf_nonzero["acf"].abs().max()),
        "max_abs_pacf_lag_1_to_max_lag": float(acf_nonzero["pacf"].abs().max()),
        "max_lag_checked": int(max_lag),
        "ljung_box_all_checked_lags_pass_white_noise": bool(
            not ljung_box["reject_white_noise_null"].any()
        ),
        "normality_all_tests_pass": bool(
            not normality["reject_normality_null"].any()
        ),
    }
    summary.update(center_randomness)
    summary["residual_randomness_conclusion"] = residual_randomness_conclusion(
        summary,
        ljung_box=ljung_box,
        alpha=alpha,
    )
    summary["residual_normality_conclusion"] = residual_normality_conclusion(
        normality,
    )
    summary["sarima_suitability_conclusion"] = sarima_suitability_from_residuals(
        summary,
    )

    return {
        "summary": pd.DataFrame([summary]),
        "acf_pacf": acf_pacf,
        "ljung_box": ljung_box,
        "normality": normality,
    }


def residual_series(predictions: pd.DataFrame) -> pd.Series:
    if "residual" not in predictions.columns:
        raise ValueError("Predictions must contain residual column.")
    ordered = predictions.copy()
    if TIMESTAMP_COL in ordered.columns:
        ordered[TIMESTAMP_COL] = pd.to_datetime(ordered[TIMESTAMP_COL], utc=True)
        ordered = ordered.sort_values(TIMESTAMP_COL, kind="mergesort")
    residual = pd.to_numeric(ordered["residual"], errors="raise").astype(float)
    if residual.isna().any():
        raise ValueError("Residual contains missing values.")
    values = residual.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Residual contains non-finite values.")
    return residual.reset_index(drop=True)


def build_center_randomness_tests(values: np.ndarray, *, alpha: float) -> dict[str, Any]:
    from scipy import stats

    t_stat, t_pvalue = stats.ttest_1samp(values, popmean=0.0)
    positive_count = int((values > 0).sum())
    negative_count = int((values < 0).sum())
    nonzero_count = positive_count + negative_count
    if nonzero_count > 0:
        sign_pvalue = binomial_two_sided_pvalue(positive_count, nonzero_count)
    else:
        sign_pvalue = float("nan")

    runs_z, runs_pvalue = runs_test_around_zero(values)

    return {
        "mean_zero_t_statistic": float(t_stat),
        "mean_zero_p_value": float(t_pvalue),
        "mean_zero_reject_null": bool(t_pvalue < alpha),
        "positive_residual_count": positive_count,
        "negative_residual_count": negative_count,
        "positive_residual_rate": float(positive_count / values.shape[0]),
        "sign_balance_p_value": float(sign_pvalue),
        "sign_balance_reject_50_50_null": bool(sign_pvalue < alpha),
        "runs_test_z_statistic": float(runs_z),
        "runs_test_p_value": float(runs_pvalue),
        "runs_test_reject_random_order_null": bool(runs_pvalue < alpha),
    }


def binomial_two_sided_pvalue(successes: int, n_obs: int) -> float:
    from scipy import stats

    if hasattr(stats, "binomtest"):
        return float(stats.binomtest(successes, n_obs, p=0.5).pvalue)
    return float(stats.binom_test(successes, n_obs, p=0.5))


def runs_test_around_zero(values: np.ndarray) -> tuple[float, float]:
    nonzero = values[values != 0]
    if nonzero.shape[0] < 2:
        return float("nan"), float("nan")
    try:
        from statsmodels.sandbox.stats.runs import runstest_1samp

        z_stat, p_value = runstest_1samp(nonzero, cutoff=0.0, correction=True)
        return float(z_stat), float(p_value)
    except Exception:
        return manual_runs_test(nonzero)


def manual_runs_test(values: np.ndarray) -> tuple[float, float]:
    from scipy import stats

    signs = values > 0
    n_pos = int(signs.sum())
    n_neg = int((~signs).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    runs = int(1 + np.sum(signs[1:] != signs[:-1]))
    expected = 1 + (2 * n_pos * n_neg) / (n_pos + n_neg)
    variance = (
        2
        * n_pos
        * n_neg
        * (2 * n_pos * n_neg - n_pos - n_neg)
        / (((n_pos + n_neg) ** 2) * (n_pos + n_neg - 1))
    )
    z_stat = (runs - expected) / np.sqrt(variance)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return float(z_stat), float(p_value)


def build_residual_normality_tests(values: np.ndarray, *, alpha: float) -> pd.DataFrame:
    from scipy import stats

    rows: list[dict[str, Any]] = []

    jb = stats.jarque_bera(values)
    rows.append(
        {
            "test_name": "Jarque-Bera",
            "null_hypothesis": "residuals_follow_normal_distribution",
            "statistic": float(jb.statistic),
            "p_value": float(jb.pvalue),
            "alpha": float(alpha),
            "reject_normality_null": bool(jb.pvalue < alpha),
        }
    )

    if values.shape[0] >= 8:
        dagostino = stats.normaltest(values)
        rows.append(
            {
                "test_name": "D'Agostino K^2",
                "null_hypothesis": "residuals_follow_normal_distribution",
                "statistic": float(dagostino.statistic),
                "p_value": float(dagostino.pvalue),
                "alpha": float(alpha),
                "reject_normality_null": bool(dagostino.pvalue < alpha),
            }
        )

    shapiro = stats.shapiro(values)
    rows.append(
        {
            "test_name": "Shapiro-Wilk",
            "null_hypothesis": "residuals_follow_normal_distribution",
            "statistic": float(shapiro.statistic),
            "p_value": float(shapiro.pvalue),
            "alpha": float(alpha),
            "reject_normality_null": bool(shapiro.pvalue < alpha),
        }
    )

    result = pd.DataFrame(rows)
    result["interpretation"] = np.where(
        result["reject_normality_null"],
        "normality_rejected",
        "normality_not_rejected",
    )
    return result


def lag_value(acf_pacf: pd.DataFrame, column: str, lag: int) -> float:
    row = acf_pacf.loc[acf_pacf["lag"] == lag, column]
    if row.empty:
        return float("nan")
    return float(row.iloc[0])


def residual_randomness_conclusion(
    summary: Mapping[str, Any],
    *,
    ljung_box: pd.DataFrame,
    alpha: float,
) -> str:
    centered = not bool(summary["mean_zero_reject_null"])
    sign_balanced = not bool(summary["sign_balance_reject_50_50_null"])
    runs_random = not bool(summary["runs_test_reject_random_order_null"])
    white_noise = not bool(ljung_box["reject_white_noise_null"].any())

    if centered and sign_balanced and runs_random and white_noise:
        return "residuals_look_random_around_zero"
    if centered and sign_balanced and not white_noise:
        return "residuals_centered_but_autocorrelated"
    if not centered and not white_noise:
        return "residuals_biased_and_autocorrelated"
    if not runs_random:
        return "residual_sign_sequence_not_random"
    return f"mixed_residual_diagnostics_alpha_{alpha}"


def residual_normality_conclusion(normality: pd.DataFrame) -> str:
    if normality.empty:
        return "normality_not_tested"
    if normality["reject_normality_null"].any():
        return "normality_rejected"
    return "normality_not_rejected"


def sarima_suitability_from_residuals(summary: Mapping[str, Any]) -> str:
    if (
        bool(summary["ljung_box_all_checked_lags_pass_white_noise"])
        and not bool(summary["mean_zero_reject_null"])
    ):
        return "residual_diagnostics_support_sarima_fit"
    if not bool(summary["ljung_box_all_checked_lags_pass_white_noise"]):
        return "residual_autocorrelation_indicates_sarima_underfits_temporal_structure"
    return "residual_diagnostics_mixed"


def save_residual_diagnostic_outputs(
    diagnostics: Mapping[str, pd.DataFrame],
    outputs: Mapping[str, Path],
) -> None:
    diagnostics["summary"].to_csv(outputs["residual_diagnostic_summary"], index=False)
    diagnostics["acf_pacf"].to_csv(outputs["residual_acf_pacf"], index=False)
    diagnostics["ljung_box"].to_csv(outputs["residual_ljung_box"], index=False)
    diagnostics["normality"].to_csv(outputs["residual_normality_tests"], index=False)


def residual_diagnostics_metadata(
    diagnostics: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    summary = diagnostics["summary"].iloc[0].to_dict()
    return {
        "summary": summary,
        "ljung_box": diagnostics["ljung_box"].to_dict(orient="records"),
        "normality_tests": diagnostics["normality"].to_dict(orient="records"),
    }


def build_runtime_summary(metrics: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_name": EXPERIMENT_NAME,
                "model_label": MODEL_LABEL,
                "model_name": MODEL_NAME,
                "feature_set": metrics["feature_set"],
                "parameter_set_id": metrics["parameter_set_id"],
                "n_train_rows": metrics["n_train_rows_used"],
                "n_prediction_rows": metrics["n_prediction_rows"],
                "train_time_seconds": metrics["train_time_seconds"],
                "prediction_time_seconds": metrics["prediction_time_seconds"],
                "forecast_time_seconds": metrics["forecast_time_seconds"],
                "update_time_seconds": metrics["update_time_seconds"],
                "total_runtime_seconds": metrics["total_runtime_seconds"],
                "status": "success",
                "error_message": "",
            }
        ]
    )


def build_reference_comparison(metrics: Mapping[str, Any]) -> pd.DataFrame:
    sarima_row = {
        "source": "standalone_sarima_sanity_test",
        "official_workflow_member": False,
        "model_label": MODEL_LABEL,
        "parameter_set_id": metrics["parameter_set_id"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "smape": metrics["smape"],
        "prediction_time_seconds": metrics["prediction_time_seconds"],
        "train_time_seconds": metrics["train_time_seconds"],
        "negative_prediction_count": metrics["negative_prediction_count"],
    }

    reference_path = FINAL_TEST_DIR / "metrics" / "final_metrics.csv"
    if not reference_path.exists():
        comparison = pd.DataFrame([sarima_row])
    else:
        reference = pd.read_csv(reference_path)
        reference_rows = []
        for _, row in reference.iterrows():
            reference_rows.append(
                {
                    "source": "official_final_test_reference",
                    "official_workflow_member": True,
                    "model_label": row.get("model_label", ""),
                    "parameter_set_id": row.get("parameter_set_id", ""),
                    "mae": row.get("mae", np.nan),
                    "rmse": row.get("rmse", np.nan),
                    "mape": row.get("mape", np.nan),
                    "smape": row.get("smape", np.nan),
                    "prediction_time_seconds": row.get("prediction_time_seconds", np.nan),
                    "train_time_seconds": row.get(
                        "retraining_train_time_seconds",
                        np.nan,
                    ),
                    "negative_prediction_count": row.get(
                        "negative_prediction_count",
                        np.nan,
                    ),
                }
            )
        comparison = pd.DataFrame([sarima_row, *reference_rows])

    comparison["rank_by_mae_reference_only"] = (
        comparison["mae"].rank(method="first", ascending=True).astype(int)
    )
    return comparison.sort_values("rank_by_mae_reference_only", kind="mergesort")


def render_summary(
    *,
    metadata: Mapping[str, Any],
    metrics_df: pd.DataFrame,
    residual_summary: pd.DataFrame,
    horizon_error: pd.DataFrame,
    runtime_summary: pd.DataFrame,
    reference_comparison: pd.DataFrame,
    residual_diagnostic_summary: pd.DataFrame,
    ljung_box: pd.DataFrame,
    normality_tests: pd.DataFrame,
    skip_plots: bool,
) -> str:
    outputs = metadata["outputs"]
    metrics_view = metrics_df[
        [
            "model_label",
            "parameter_set_id",
            "order",
            "seasonal_order",
            "mae",
            "rmse",
            "mape",
            "smape",
            "negative_prediction_count",
        ]
    ]
    comparison_view = reference_comparison[
        [
            "rank_by_mae_reference_only",
            "model_label",
            "source",
            "official_workflow_member",
            "mae",
            "rmse",
            "mape",
            "smape",
            "prediction_time_seconds",
            "train_time_seconds",
        ]
    ]
    best_horizon = horizon_error.sort_values("mae", kind="mergesort").head(1)
    worst_horizon = horizon_error.sort_values("mae", ascending=False).head(1)
    diagnostic_columns = [
        "model_label",
        "residual_mean",
        "residual_std",
        "mean_zero_p_value",
        "sign_balance_p_value",
        "runs_test_p_value",
        "acf_lag_1",
        "acf_lag_24",
        "acf_lag_168",
        "significant_acf_lags_count_1_to_max_lag",
        "significant_pacf_lags_count_1_to_max_lag",
        "ljung_box_all_checked_lags_pass_white_noise",
        "normality_all_tests_pass",
        "residual_randomness_conclusion",
        "sarima_suitability_conclusion",
    ]
    diagnostic_view = residual_diagnostic_summary[
        [col for col in diagnostic_columns if col in residual_diagnostic_summary.columns]
    ]
    ljung_view = ljung_box[
        ["lag", "lb_stat", "lb_pvalue", "reject_white_noise_null", "interpretation"]
    ]
    normality_view = normality_tests[
        ["test_name", "statistic", "p_value", "reject_normality_null", "interpretation"]
    ]

    interpretation = build_interpretation(reference_comparison)
    diagnostic_interpretation = build_residual_diagnostic_interpretation(
        residual_diagnostic_summary,
        ljung_box,
        normality_tests,
    )
    evaluation_config = dict(metadata.get("evaluation_config", {}))
    train_window_policy = evaluation_config.get("train_window_policy", "full_train_val")
    update_maxiter = evaluation_config.get("update_maxiter", "")
    diagnostic_runtime = (
        metadata.get("time_cost_computing", {})
        .get("residual_diagnostics_runtime_seconds", "")
    )

    lines = [
        "# SARIMA Sanity Test",
        "",
        f"Run UTC: {metadata['started_at_utc']}",
        "",
        "## Scope",
        "",
        (
            "Eksperimen ini adalah post-hoc sanity test atas permintaan dosen. "
            "SARIMA tidak dimasukkan ke flow penelitian utama, tidak dipakai untuk "
            "mengubah ranking comparative evaluation, dan outputnya disimpan terpisah."
        ),
        "",
        "## Methodology",
        "",
        (
            f"Model auto-SARIMA dilatih pada train_val lalu dievaluasi pada final_test "
            f"dengan rolling-origin block {FORECAST_HORIZON} jam. Actual final_test "
            "hanya digunakan sebagai label evaluasi dan baru dimasukkan ke state model "
            "setelah block 24 jam selesai diprediksi."
        ),
        (
            f"Konfigurasi evaluasi SARIMA: train window policy `{train_window_policy}`, "
            f"`update_maxiter={update_maxiter}`, dan final_test tidak digunakan untuk tuning."
        ),
        "",
        "## SARIMA Metrics",
        "",
        dataframe_to_markdown(metrics_view, float_digits=6),
        "",
        "## Reference Comparison",
        "",
        (
            "Tabel ini hanya referensi post-hoc terhadap hasil final_test yang sudah "
            "ada. Ini bukan ranking resmi baru."
        ),
        "",
        dataframe_to_markdown(comparison_view, float_digits=6),
        "",
        "## Residual Summary",
        "",
        dataframe_to_markdown(residual_summary, float_digits=6),
        "",
        "## Residual Diagnostics",
        "",
        (
            "Diagnostik ini dipakai untuk menilai apakah residual SARIMA sudah "
            "menyerupai white noise: berpusat di sekitar nol, tidak memiliki "
            "autokorelasi tersisa, dan mendekati distribusi normal."
        ),
        "",
        dataframe_to_markdown(diagnostic_view, float_digits=6),
        "",
        "Ljung-Box test:",
        "",
        dataframe_to_markdown(ljung_view, float_digits=6),
        "",
        "Normality tests:",
        "",
        dataframe_to_markdown(normality_view, float_digits=6),
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
        dataframe_to_markdown(runtime_summary, float_digits=6),
        "",
        f"Residual diagnostics runtime seconds: `{diagnostic_runtime}`",
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        diagnostic_interpretation,
        "",
        "## Output Files",
        "",
        f"- Params: `{outputs['params']}`",
        f"- Metrics: `{outputs['metrics']}`",
        f"- Runtime summary: `{outputs['runtime_summary']}`",
        f"- Predictions: `{outputs['predictions']}`",
        f"- Residual summary: `{outputs['residual_summary']}`",
        f"- Residual diagnostic summary: `{outputs['residual_diagnostic_summary']}`",
        f"- Residual ACF/PACF values: `{outputs['residual_acf_pacf']}`",
        f"- Ljung-Box test: `{outputs['residual_ljung_box']}`",
        f"- Normality tests: `{outputs['residual_normality_tests']}`",
        f"- Horizon error summary: `{outputs['horizon_error_summary']}`",
        f"- Reference comparison: `{outputs['reference_comparison']}`",
        f"- Model summary: `{outputs['model_summary']}`",
        f"- Metadata: `{outputs['metadata']}`",
    ]
    if not skip_plots:
        lines.extend(
            [
                f"- Actual vs predicted plot: `{outputs['actual_vs_predicted_plot']}`",
                f"- Residuals plot: `{outputs['residuals_plot']}`",
                f"- Residual ACF/PACF plot: `{outputs['residual_acf_pacf_plot']}`",
                f"- Residual normality/QQ plot: `{outputs['residual_normality_plot']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def build_interpretation(reference_comparison: pd.DataFrame) -> str:
    sarima = reference_comparison[reference_comparison["model_label"] == MODEL_LABEL]
    if sarima.empty:
        return "SARIMA selesai dievaluasi, tetapi row SARIMA tidak ditemukan di tabel referensi."

    sarima_row = sarima.iloc[0]
    rank = int(sarima_row["rank_by_mae_reference_only"])
    n_models = int(reference_comparison.shape[0])

    official = reference_comparison[reference_comparison["official_workflow_member"] == True]
    best_official = None
    if not official.empty:
        best_official = official.sort_values(PRIMARY_METRIC, kind="mergesort").iloc[0]

    if best_official is None:
        return (
            "Secara teknis SARIMA dapat dijalankan pada data ini. Karena file final "
            "metric resmi tidak ditemukan, interpretasi hanya memakai metric SARIMA."
        )

    sarima_mae = float(sarima_row["mae"])
    best_mae = float(best_official["mae"])
    pct_gap = ((sarima_mae - best_mae) / best_mae) * 100.0
    return (
        "Secara teknis model konvensional SARIMA bisa digunakan pada data hourly ini. "
        f"Namun pada sanity test ini SARIMA berada di rank {rank} dari {n_models} "
        f"berdasarkan MAE referensi, dengan MAE {sarima_mae:.3f}. Dibanding model "
        f"resmi terbaik ({best_official['model_label']}, MAE {best_mae:.3f}), "
        f"gap MAE SARIMA sekitar {pct_gap:.2f}%. Jadi kesimpulan yang lebih tepat "
        "bukan 'SARIMA tidak bisa dipakai', melainkan SARIMA univariate harian ini "
        "kurang kompetitif untuk pola demand NYC Taxi dibanding model ML yang memakai "
        "lag/calendar features."
    )


def build_residual_diagnostic_interpretation(
    residual_diagnostic_summary: pd.DataFrame,
    ljung_box: pd.DataFrame,
    normality_tests: pd.DataFrame,
) -> str:
    if residual_diagnostic_summary.empty:
        return "Residual diagnostics tidak tersedia."

    row = residual_diagnostic_summary.iloc[0]
    mean_p = float(row.get("mean_zero_p_value", np.nan))
    runs_p = float(row.get("runs_test_p_value", np.nan))
    acf24 = float(row.get("acf_lag_24", np.nan))
    acf168 = float(row.get("acf_lag_168", np.nan))
    lb_reject_count = int(ljung_box["reject_white_noise_null"].sum())
    lb_total = int(ljung_box.shape[0])
    normality_reject_count = int(normality_tests["reject_normality_null"].sum())
    normality_total = int(normality_tests.shape[0])

    centered_text = (
        "Residual tidak menunjukkan bias mean yang kuat"
        if not bool(row.get("mean_zero_reject_null", True))
        else "Residual masih menunjukkan indikasi bias mean"
    )
    runs_text = (
        "urutan tanda residual tidak ditolak sebagai acak"
        if not bool(row.get("runs_test_reject_random_order_null", True))
        else "urutan tanda residual belum terlihat acak"
    )
    white_noise_text = (
        "Ljung-Box tidak menolak white-noise pada lag yang dicek"
        if lb_reject_count == 0
        else f"Ljung-Box menolak white-noise pada {lb_reject_count} dari {lb_total} lag yang dicek"
    )
    normality_text = (
        "normalitas residual tidak ditolak"
        if normality_reject_count == 0
        else f"normalitas residual ditolak oleh {normality_reject_count} dari {normality_total} uji"
    )

    return (
        f"Diagnostik residual menunjukkan: {centered_text} "
        f"(t-test p={mean_p:.4g}) dan {runs_text} (runs test p={runs_p:.4g}), "
        f"tetapi {white_noise_text}. ACF residual masih penting untuk dibaca, "
        f"terutama ACF lag-24={acf24:.4f} dan lag-168={acf168:.4f}. Selain itu, "
        f"{normality_text}. Jadi untuk pertanyaan kecocokan model konvensional, "
        "auto-SARIMA univariate ini dapat dipakai sebagai baseline, tetapi residualnya "
        "belum ideal sebagai white noise; masih ada struktur temporal/distribusional "
        "yang belum ditangkap model."
    )


def save_actual_vs_predicted_plot(predictions: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_df = predictions.sort_values(TIMESTAMP_COL, kind="mergesort")
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(plot_df[TIMESTAMP_COL], plot_df[ACTUAL_COL], label="Actual", linewidth=1.3)
    ax.plot(
        plot_df[TIMESTAMP_COL],
        plot_df[PREDICTION_COL],
        label=MODEL_LABEL,
        linewidth=1.1,
        alpha=0.85,
    )
    ax.set_title("SARIMA Sanity Test - Actual vs Predicted")
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
    axes[0].plot(
        plot_df[TIMESTAMP_COL],
        plot_df["residual"],
        label=MODEL_LABEL,
        linewidth=1.0,
        alpha=0.85,
    )
    axes[0].axhline(0, color="black", linewidth=0.9)
    axes[0].set_title("SARIMA Sanity Test - Residual Time Series")
    axes[0].set_xlabel("UTC timestamp")
    axes[0].set_ylabel("Residual")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend()

    axes[1].hist(plot_df["residual"], bins=40, alpha=0.65, label=MODEL_LABEL)
    axes[1].axvline(0, color="black", linewidth=0.9)
    axes[1].set_title("SARIMA Sanity Test - Residual Distribution")
    axes[1].set_xlabel("Residual (actual - predicted)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_residual_acf_pacf_plot(
    acf_pacf: pd.DataFrame,
    output_path: Path,
    *,
    confint: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_df = acf_pacf[acf_pacf["lag"] > 0].copy()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].bar(plot_df["lag"], plot_df["acf"], width=0.85, color="#3b6ea8")
    axes[0].axhline(confint, color="black", linestyle="--", linewidth=0.9)
    axes[0].axhline(-confint, color="black", linestyle="--", linewidth=0.9)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("SARIMA Residual ACF")
    axes[0].set_ylabel("ACF")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(plot_df["lag"], plot_df["pacf"], width=0.85, color="#8f5d3f")
    axes[1].axhline(confint, color="black", linestyle="--", linewidth=0.9)
    axes[1].axhline(-confint, color="black", linestyle="--", linewidth=0.9)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("SARIMA Residual PACF")
    axes[1].set_xlabel("Lag (hours)")
    axes[1].set_ylabel("PACF")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_residual_normality_plot(predictions: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from scipy import stats

    residual = residual_series(predictions)
    values = residual.to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].hist(values, bins=40, alpha=0.7, color="#4d7c6f", density=True)
    x = np.linspace(values.min(), values.max(), 300)
    normal_pdf = stats.norm.pdf(x, loc=values.mean(), scale=values.std(ddof=1))
    axes[0].plot(x, normal_pdf, color="black", linewidth=1.2, label="Normal fit")
    axes[0].axvline(0, color="black", linewidth=0.9, linestyle="--")
    axes[0].set_title("Residual Distribution vs Normal Fit")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend()

    stats.probplot(values, dist="norm", plot=axes[1])
    axes[1].set_title("Residual Q-Q Plot")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def feature_set_name(model: Any) -> str:
    seasonal_order = str(getattr(model, "seasonal_order", ""))
    m = ""
    if seasonal_order:
        parts = seasonal_order.strip("()").split(",")
        if len(parts) == 4:
            m = parts[-1].strip()
    suffix = f"_m{m}" if m else ""
    return f"univariate_auto_sarima{suffix}"


def parameter_set_name(model: Any) -> str:
    order = str(getattr(model, "order", "")).replace(" ", "")
    seasonal_order = str(getattr(model, "seasonal_order", "")).replace(" ", "")
    return f"auto_order_{order}_seasonal_{seasonal_order}"


def selected_order_payload(model: Any) -> dict[str, Any]:
    return {
        "order": str(getattr(model, "order", "")),
        "seasonal_order": str(getattr(model, "seasonal_order", "")),
        "aic": safe_float_call(model, "aic"),
        "bic": safe_float_call(model, "bic"),
    }


def safe_float_call(model: Any, method_name: str) -> Optional[float]:
    method = getattr(model, method_name, None)
    if not callable(method):
        return None
    try:
        value = method()
    except Exception:
        return None
    if value is None:
        return None
    return float(value)


def safe_model_summary(model: Any) -> str:
    try:
        summary = model.summary()
        if hasattr(summary, "as_text"):
            return str(summary.as_text())
        return str(summary)
    except Exception as exc:
        return f"Model summary unavailable: {exc}\n"


def summarize_timeseries(df: pd.DataFrame) -> dict[str, Any]:
    target = pd.to_numeric(df[TARGET_COL], errors="raise")
    return {
        "n_rows": int(df.shape[0]),
        "utc_start": df.index.min().isoformat(),
        "utc_end": df.index.max().isoformat(),
        "timezone": str(df.index.tz),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "target_mean": float(target.mean()),
    }


def methodology_notes() -> list[str]:
    return [
        "Standalone post-hoc sanity test; not part of the official model comparison flow.",
        "Training uses train_val only; final_test is never used during auto_arima fitting.",
        "Final_test labels are used only for metric computation and for state update after each completed 24h block.",
        "The default seasonal period is m=24, so this test captures daily hourly seasonality but not explicit weekly m=168 seasonality.",
        "No exogenous regressors, lag feature engineering, or calendar features are used.",
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


def stringify_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    return {key: str(value) for key, value in paths.items()}


if __name__ == "__main__":
    main()

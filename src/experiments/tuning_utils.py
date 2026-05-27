"""
tuning_utils.py

Utilitas bersama untuk tahap 10 Hyperparameter Tuning.

Guardrail utama:
- CV hanya memakai train_val.
- XGBoost dilatih dengan feature rows training fold saja.
- XGBoost validation diprediksi dengan forecast_validation_window(), bukan
  model.predict() pada feature matrix validation.
- Runtime training dan prediction dicatat lewat src.tracking.py.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.config import (
    CV_GAP_HOURS,
    CV_N_FOLDS,
    CV_VAL_HORIZON_HOURS,
    FORECAST_HORIZON,
    METRICS,
    PRIMARY_METRIC,
    PROPHET_OUTPUT_DIR,
    PROPHET_REGRESSOR_BASIC_OUTPUT_DIR,
    PROPHET_SEARCH_SPACE,
    SECONDARY_METRIC,
    TARGET_COL,
    TIMESTAMP_COL,
    TRAIN_VAL_PATH,
    XGB_ADVANCED_FEATURES_PATH,
    XGB_ADVANCED_OUTPUT_DIR,
    XGB_BASIC_FEATURES_PATH,
    XGB_BASIC_OUTPUT_DIR,
    XGB_SEARCH_SPACE,
    ensure_dirs,
)
from src.features import ADVANCED_FEATURE_SET, BASIC_FEATURE_SET, get_feature_columns
from src.forecasting import (
    ACTUAL_COL,
    PREDICTION_COL,
    forecast_validation_window_prophet_regressor,
    forecast_validation_window,
    recursive_forecast_prophet,
    validate_prediction_output,
)
from src.metrics import compute_all_metrics, summarize_cv_metrics
from src.models.prophet_model import (
    PROPHET_BASIC_REGRESSORS,
    PROPHET_FEATURE_SET,
    PROPHET_REGRESSOR_BASIC_FEATURE_SET,
    fit_prophet_model,
    make_prophet_regressor_frame,
    make_prophet_regressor_model,
    make_prophet_future_frame,
)
from src.models.xgboost_model import fit_xgb_model
from src.splits import load_split_timeseries, make_expanding_window_splits
from src.tracking import (
    append_experiment_run,
    elapsed_seconds,
    log_runtime,
    make_runtime_record,
    save_experiment_metadata,
    start_timer,
    utc_now_iso,
)


PathLike = Union[str, Path]


def run_prophet_tuning(
    *,
    input_path: PathLike = TRAIN_VAL_PATH,
    output_dir: PathLike = PROPHET_OUTPUT_DIR,
    search_space: Mapping[str, Sequence[Any]] = PROPHET_SEARCH_SPACE,
    n_folds: int = CV_N_FOLDS,
    val_horizon: int = CV_VAL_HORIZON_HOURS,
    gap: int = CV_GAP_HOURS,
    forecast_horizon: int = FORECAST_HORIZON,
    max_parameter_sets: Optional[int] = None,
    primary_metric: str = PRIMARY_METRIC,
    secondary_metric: str = SECONDARY_METRIC,
    skip_plots: bool = False,
) -> dict[str, Any]:
    """
    Jalankan tuning Prophet dengan expanding-window CV.
    """
    experiment_name = "tune_prophet"
    model_name = "prophet"
    feature_set = PROPHET_FEATURE_SET
    paths = prepare_tuning_output_paths(output_dir)
    reset_tuning_outputs(paths)

    experiment_timer = start_timer()
    started_at = utc_now_iso()
    status = "success"
    error_message = ""
    predictions_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []

    try:
        ensure_dirs()
        train_val = load_split_timeseries(input_path)
        splits = make_expanding_window_splits(
            train_val,
            n_folds=n_folds,
            val_horizon=val_horizon,
            gap=gap,
            forecast_horizon=forecast_horizon,
        )
        parameter_grid = list(
            iter_parameter_grid(
                search_space,
                prefix="prophet",
                max_parameter_sets=max_parameter_sets,
            )
        )
        save_tuning_params(
            paths["params"],
            experiment_name=experiment_name,
            model_name=model_name,
            feature_set=feature_set,
            search_space=search_space,
            parameter_grid=parameter_grid,
            max_parameter_sets=max_parameter_sets,
            cv_config=_cv_config_dict(n_folds, val_horizon, gap, forecast_horizon),
            primary_metric=primary_metric,
            secondary_metric=secondary_metric,
        )

        for parameter_set_id, params in parameter_grid:
            for split in splits:
                fold_result = _run_one_prophet_fold(
                    split=split,
                    params=params,
                    parameter_set_id=parameter_set_id,
                    experiment_name=experiment_name,
                    model_name=model_name,
                    feature_set=feature_set,
                    forecast_horizon=forecast_horizon,
                )
                predictions_frames.append(fold_result["predictions"])
                metrics_rows.append(fold_result["metrics"])
                _append_dataframe_csv(fold_result["predictions"], paths["predictions"])
                _append_dataframe_csv(pd.DataFrame([fold_result["metrics"]]), paths["metrics"])

        final_outputs = finalize_tuning_outputs(
            predictions_frames=predictions_frames,
            metrics_rows=metrics_rows,
            paths=paths,
            search_space=search_space,
            parameter_grid=parameter_grid,
            experiment_name=experiment_name,
            model_name=model_name,
            feature_set=feature_set,
            started_at=started_at,
            input_path=input_path,
            primary_metric=primary_metric,
            secondary_metric=secondary_metric,
            cv_config=_cv_config_dict(n_folds, val_horizon, gap, forecast_horizon),
            skip_plots=skip_plots,
        )
        return final_outputs
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        append_experiment_run(
            {
                "experiment_name": experiment_name,
                "model_name": model_name,
                "feature_set": feature_set,
                "status": status,
                "error_message": error_message,
                "total_runtime_seconds": elapsed_seconds(experiment_timer),
            }
        )
        raise
    finally:
        total_runtime_seconds = elapsed_seconds(experiment_timer)
        save_experiment_metadata(
            {
                "experiment_name": experiment_name,
                "model_name": model_name,
                "feature_set": feature_set,
                "status": status,
                "error_message": error_message,
                "started_at_utc": started_at,
                "finished_at_utc": utc_now_iso(),
                "total_runtime_seconds": total_runtime_seconds,
                "output_dir": str(output_dir),
            },
            paths["latest_run_metadata"],
        )


def run_prophet_regressor_basic_tuning(
    *,
    input_path: PathLike = TRAIN_VAL_PATH,
    output_dir: PathLike = PROPHET_REGRESSOR_BASIC_OUTPUT_DIR,
    search_space: Mapping[str, Sequence[Any]] = PROPHET_SEARCH_SPACE,
    n_folds: int = CV_N_FOLDS,
    val_horizon: int = CV_VAL_HORIZON_HOURS,
    gap: int = CV_GAP_HOURS,
    forecast_horizon: int = FORECAST_HORIZON,
    max_parameter_sets: Optional[int] = None,
    primary_metric: str = PRIMARY_METRIC,
    secondary_metric: str = SECONDARY_METRIC,
    skip_plots: bool = False,
) -> dict[str, Any]:
    """
    Jalankan tuning Prophet dengan regressor XGBoost-Basic.
    """
    experiment_name = "tune_prophet_regressor_basic"
    model_name = "prophet_regressor_basic"
    feature_set = PROPHET_REGRESSOR_BASIC_FEATURE_SET
    paths = prepare_tuning_output_paths(output_dir)
    reset_tuning_outputs(paths)

    experiment_timer = start_timer()
    started_at = utc_now_iso()
    status = "success"
    error_message = ""
    predictions_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []

    try:
        ensure_dirs()
        train_val = load_split_timeseries(input_path)
        splits = make_expanding_window_splits(
            train_val,
            n_folds=n_folds,
            val_horizon=val_horizon,
            gap=gap,
            forecast_horizon=forecast_horizon,
        )
        parameter_grid = list(
            iter_parameter_grid(
                search_space,
                prefix="prophet_regressor_basic",
                max_parameter_sets=max_parameter_sets,
            )
        )
        save_tuning_params(
            paths["params"],
            experiment_name=experiment_name,
            model_name=model_name,
            feature_set=feature_set,
            search_space=search_space,
            parameter_grid=parameter_grid,
            max_parameter_sets=max_parameter_sets,
            cv_config=_cv_config_dict(n_folds, val_horizon, gap, forecast_horizon),
            primary_metric=primary_metric,
            secondary_metric=secondary_metric,
            extra_metadata={
                "regressors": list(PROPHET_BASIC_REGRESSORS),
                "leakage_guardrail": (
                    "Training regressors use training history only; validation "
                    "regressor lags are built recursively from past history and "
                    "previous predictions."
                ),
            },
        )

        for parameter_set_id, params in parameter_grid:
            for split in splits:
                fold_result = _run_one_prophet_regressor_basic_fold(
                    split=split,
                    params=params,
                    parameter_set_id=parameter_set_id,
                    experiment_name=experiment_name,
                    model_name=model_name,
                    feature_set=feature_set,
                    forecast_horizon=forecast_horizon,
                )
                predictions_frames.append(fold_result["predictions"])
                metrics_rows.append(fold_result["metrics"])
                _append_dataframe_csv(fold_result["predictions"], paths["predictions"])
                _append_dataframe_csv(pd.DataFrame([fold_result["metrics"]]), paths["metrics"])

        final_outputs = finalize_tuning_outputs(
            predictions_frames=predictions_frames,
            metrics_rows=metrics_rows,
            paths=paths,
            search_space=search_space,
            parameter_grid=parameter_grid,
            experiment_name=experiment_name,
            model_name=model_name,
            feature_set=feature_set,
            started_at=started_at,
            input_path=input_path,
            primary_metric=primary_metric,
            secondary_metric=secondary_metric,
            cv_config=_cv_config_dict(n_folds, val_horizon, gap, forecast_horizon),
            skip_plots=skip_plots,
            extra_metadata={
                "regressors": list(PROPHET_BASIC_REGRESSORS),
                "validation_predicted_with_recursive_forecasting": True,
            },
        )
        return final_outputs
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        append_experiment_run(
            {
                "experiment_name": experiment_name,
                "model_name": model_name,
                "feature_set": feature_set,
                "status": status,
                "error_message": error_message,
                "total_runtime_seconds": elapsed_seconds(experiment_timer),
            }
        )
        raise
    finally:
        total_runtime_seconds = elapsed_seconds(experiment_timer)
        save_experiment_metadata(
            {
                "experiment_name": experiment_name,
                "model_name": model_name,
                "feature_set": feature_set,
                "status": status,
                "error_message": error_message,
                "started_at_utc": started_at,
                "finished_at_utc": utc_now_iso(),
                "total_runtime_seconds": total_runtime_seconds,
                "output_dir": str(output_dir),
            },
            paths["latest_run_metadata"],
        )


def run_xgb_tuning(
    *,
    feature_set: str,
    input_path: PathLike = TRAIN_VAL_PATH,
    feature_path: Optional[PathLike] = None,
    output_dir: Optional[PathLike] = None,
    search_space: Mapping[str, Sequence[Any]] = XGB_SEARCH_SPACE,
    n_folds: int = CV_N_FOLDS,
    val_horizon: int = CV_VAL_HORIZON_HOURS,
    gap: int = CV_GAP_HOURS,
    forecast_horizon: int = FORECAST_HORIZON,
    max_parameter_sets: Optional[int] = None,
    primary_metric: str = PRIMARY_METRIC,
    secondary_metric: str = SECONDARY_METRIC,
    skip_plots: bool = False,
) -> dict[str, Any]:
    """
    Jalankan tuning XGBoost Basic/Advanced dengan recursive validation.
    """
    normalized_feature_set = normalize_xgb_feature_set(feature_set)
    model_name = "xgboost"
    experiment_name = f"tune_{normalized_feature_set}"
    feature_path = feature_path or default_feature_path(normalized_feature_set)
    output_dir = output_dir or default_output_dir(normalized_feature_set)
    paths = prepare_tuning_output_paths(output_dir)
    reset_tuning_outputs(paths)

    experiment_timer = start_timer()
    started_at = utc_now_iso()
    status = "success"
    error_message = ""
    predictions_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []

    try:
        ensure_dirs()
        train_val = load_split_timeseries(input_path)
        feature_matrix = load_xgb_feature_matrix(feature_path, normalized_feature_set)
        splits = make_expanding_window_splits(
            train_val,
            n_folds=n_folds,
            val_horizon=val_horizon,
            gap=gap,
            forecast_horizon=forecast_horizon,
        )
        parameter_grid = list(
            iter_parameter_grid(
                search_space,
                prefix=normalized_feature_set,
                max_parameter_sets=max_parameter_sets,
            )
        )
        save_tuning_params(
            paths["params"],
            experiment_name=experiment_name,
            model_name=model_name,
            feature_set=normalized_feature_set,
            search_space=search_space,
            parameter_grid=parameter_grid,
            max_parameter_sets=max_parameter_sets,
            cv_config=_cv_config_dict(n_folds, val_horizon, gap, forecast_horizon),
            primary_metric=primary_metric,
            secondary_metric=secondary_metric,
            extra_metadata={
                "feature_path": str(feature_path),
                "leakage_guardrail": (
                    "Training uses precomputed feature rows only for training "
                    "timestamps; validation uses forecast_validation_window()."
                ),
            },
        )

        for parameter_set_id, params in parameter_grid:
            for split in splits:
                fold_result = _run_one_xgb_fold(
                    split=split,
                    feature_matrix=feature_matrix,
                    params=params,
                    parameter_set_id=parameter_set_id,
                    experiment_name=experiment_name,
                    model_name=model_name,
                    feature_set=normalized_feature_set,
                    forecast_horizon=forecast_horizon,
                )
                predictions_frames.append(fold_result["predictions"])
                metrics_rows.append(fold_result["metrics"])
                _append_dataframe_csv(fold_result["predictions"], paths["predictions"])
                _append_dataframe_csv(pd.DataFrame([fold_result["metrics"]]), paths["metrics"])

        final_outputs = finalize_tuning_outputs(
            predictions_frames=predictions_frames,
            metrics_rows=metrics_rows,
            paths=paths,
            search_space=search_space,
            parameter_grid=parameter_grid,
            experiment_name=experiment_name,
            model_name=model_name,
            feature_set=normalized_feature_set,
            started_at=started_at,
            input_path=input_path,
            primary_metric=primary_metric,
            secondary_metric=secondary_metric,
            cv_config=_cv_config_dict(n_folds, val_horizon, gap, forecast_horizon),
            skip_plots=skip_plots,
            extra_metadata={"feature_path": str(feature_path)},
        )
        return final_outputs
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        append_experiment_run(
            {
                "experiment_name": experiment_name,
                "model_name": model_name,
                "feature_set": normalized_feature_set,
                "status": status,
                "error_message": error_message,
                "total_runtime_seconds": elapsed_seconds(experiment_timer),
            }
        )
        raise
    finally:
        total_runtime_seconds = elapsed_seconds(experiment_timer)
        save_experiment_metadata(
            {
                "experiment_name": experiment_name,
                "model_name": model_name,
                "feature_set": normalized_feature_set,
                "status": status,
                "error_message": error_message,
                "started_at_utc": started_at,
                "finished_at_utc": utc_now_iso(),
                "total_runtime_seconds": total_runtime_seconds,
                "output_dir": str(output_dir),
            },
            paths["latest_run_metadata"],
        )


def iter_parameter_grid(
    search_space: Mapping[str, Sequence[Any]],
    *,
    prefix: str,
    max_parameter_sets: Optional[int] = None,
) -> Iterable[tuple[str, dict[str, Any]]]:
    """
    Iterasi grid search secara deterministik.
    """
    if not search_space:
        raise ValueError("search_space tidak boleh kosong.")
    if max_parameter_sets is not None and int(max_parameter_sets) <= 0:
        raise ValueError("max_parameter_sets harus > 0 jika diisi.")

    keys = list(search_space.keys())
    values: list[list[Any]] = []
    for key in keys:
        candidates = list(search_space[key])
        if not candidates:
            raise ValueError(f"Search space untuk {key} kosong.")
        values.append(candidates)

    limit = None if max_parameter_sets is None else int(max_parameter_sets)
    for position, combo in enumerate(product(*values), start=1):
        if limit is not None and position > limit:
            break
        parameter_set_id = f"{prefix}_{position:03d}"
        yield parameter_set_id, dict(zip(keys, combo))


def prepare_tuning_output_paths(output_dir: PathLike) -> dict[str, Path]:
    """
    Buat path output standar tahap tuning.
    """
    base = Path(output_dir)
    figures = base / "figures"
    summaries = base / "summaries"
    for directory in [base, figures, summaries]:
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "base": base,
        "figures": figures,
        "summaries": summaries,
        "params": base / "params.json",
        "predictions": base / "cv_predictions.csv",
        "metrics": base / "cv_metrics.csv",
        "metrics_summary": base / "cv_metrics_summary.csv",
        "runtime_summary": base / "runtime_summary.csv",
        "best_params": base / "best_params.json",
        "experiment_metadata": base / "experiment_metadata.json",
        "latest_run_metadata": base / "latest_run_metadata.json",
        "summary": summaries / "tuning_summary.txt",
        "best_plot": figures / "best_actual_vs_predicted.png",
    }


def reset_tuning_outputs(paths: Mapping[str, Path]) -> None:
    """
    Hapus output tuning lama yang akan ditulis ulang oleh run baru.
    """
    for key in [
        "predictions",
        "metrics",
        "metrics_summary",
        "runtime_summary",
        "best_params",
        "experiment_metadata",
        "summary",
        "best_plot",
    ]:
        path = paths[key]
        if path.exists():
            path.unlink()


def save_tuning_params(
    output_path: PathLike,
    *,
    experiment_name: str,
    model_name: str,
    feature_set: str,
    search_space: Mapping[str, Sequence[Any]],
    parameter_grid: Sequence[tuple[str, Mapping[str, Any]]],
    max_parameter_sets: Optional[int],
    cv_config: Mapping[str, Any],
    primary_metric: str,
    secondary_metric: str,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Simpan search space dan daftar parameter yang benar-benar akan dijalankan.
    """
    payload: dict[str, Any] = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "feature_set": feature_set,
        "search_space": {key: list(value) for key, value in search_space.items()},
        "max_parameter_sets": max_parameter_sets,
        "n_parameter_sets_planned": len(parameter_grid),
        "parameter_sets": [
            {"parameter_set_id": parameter_set_id, "params": dict(params)}
            for parameter_set_id, params in parameter_grid
        ],
        "cv_config": dict(cv_config),
        "primary_metric": primary_metric,
        "secondary_metric": secondary_metric,
    }
    if extra_metadata:
        payload.update(dict(extra_metadata))
    save_experiment_metadata(payload, output_path)


def load_xgb_feature_matrix(
    path: PathLike,
    feature_set: str,
    *,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Load feature matrix train_val untuk training rows XGBoost.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Feature matrix XGBoost tidak ditemukan: {source}. "
            "Jalankan tahap feature engineering terlebih dahulu."
        )

    df = pd.read_csv(source)
    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"Kolom timestamp tidak ditemukan pada feature matrix: {source}")
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True, errors="raise")
    df = df.sort_values(TIMESTAMP_COL, kind="mergesort").set_index(TIMESTAMP_COL)

    feature_columns = get_feature_columns(feature_set)
    missing_columns = sorted({target_col, *feature_columns}.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Kolom feature matrix tidak lengkap: {missing_columns}")
    if df.empty:
        raise ValueError("Feature matrix XGBoost kosong.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Feature matrix XGBoost tidak chronological.")
    if df.index.has_duplicates:
        raise ValueError("Feature matrix XGBoost mengandung duplicate timestamp.")

    numeric = df[[target_col, *feature_columns]].apply(pd.to_numeric, errors="raise")
    if numeric.isna().any().any():
        raise ValueError("Feature matrix XGBoost mengandung missing value.")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Feature matrix XGBoost mengandung nilai non-finite.")

    return df


def finalize_tuning_outputs(
    *,
    predictions_frames: Sequence[pd.DataFrame],
    metrics_rows: Sequence[Mapping[str, Any]],
    paths: Mapping[str, Path],
    search_space: Mapping[str, Sequence[Any]],
    parameter_grid: Sequence[tuple[str, Mapping[str, Any]]],
    experiment_name: str,
    model_name: str,
    feature_set: str,
    started_at: str,
    input_path: PathLike,
    primary_metric: str,
    secondary_metric: str,
    cv_config: Mapping[str, Any],
    skip_plots: bool,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """
    Buat summary, best_params, metadata, dan plot setelah tuning selesai.
    """
    if not predictions_frames:
        raise ValueError("Tidak ada prediksi tuning yang dihasilkan.")
    if not metrics_rows:
        raise ValueError("Tidak ada metrics tuning yang dihasilkan.")

    predictions = pd.concat(predictions_frames, ignore_index=True)
    metrics_df = pd.DataFrame(list(metrics_rows))
    metrics_summary = build_metrics_summary(
        metrics_df,
        parameter_grid=parameter_grid,
        primary_metric=primary_metric,
        secondary_metric=secondary_metric,
    )
    runtime_summary = build_runtime_summary(metrics_df)
    best_payload = build_best_params_payload(
        metrics_summary,
        parameter_grid=parameter_grid,
        primary_metric=primary_metric,
        secondary_metric=secondary_metric,
    )

    metrics_summary.to_csv(paths["metrics_summary"], index=False)
    runtime_summary.to_csv(paths["runtime_summary"], index=False)
    save_experiment_metadata(best_payload, paths["best_params"])

    best_predictions = predictions[
        predictions["parameter_set_id"].astype(str)
        == str(best_payload["parameter_set_id"])
    ].copy()
    if not skip_plots:
        save_actual_vs_predicted_plot(
            best_predictions,
            paths["best_plot"],
            title=f"{experiment_name} best CV predictions",
        )

    total_runtime_seconds = float(pd.to_numeric(metrics_df["total_runtime_seconds"]).sum())
    metadata: dict[str, Any] = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "feature_set": feature_set,
        "status": "success",
        "input_path": str(input_path),
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "cv_config": dict(cv_config),
        "search_space": {key: list(value) for key, value in search_space.items()},
        "n_parameter_sets_planned": len(parameter_grid),
        "n_metric_rows": int(metrics_df.shape[0]),
        "n_prediction_rows": int(predictions.shape[0]),
        "primary_metric": primary_metric,
        "secondary_metric": secondary_metric,
        "best": best_payload,
        "outputs": {key: str(value) for key, value in paths.items()},
        "time_cost_computing": {
            "sum_fold_runtime_seconds": total_runtime_seconds,
            "sum_train_time_seconds": float(
                pd.to_numeric(metrics_df["train_time_seconds"]).sum()
            ),
            "sum_prediction_time_seconds": float(
                pd.to_numeric(metrics_df["prediction_time_seconds"]).sum()
            ),
        },
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    save_experiment_metadata(metadata, paths["experiment_metadata"])
    paths["summary"].write_text(render_tuning_summary(metadata), encoding="utf-8")

    append_experiment_run(
        {
            "experiment_name": experiment_name,
            "model_name": model_name,
            "feature_set": feature_set,
            "status": "success",
            "best_parameter_set_id": best_payload["parameter_set_id"],
            "best_params": best_payload["params"],
            "best_metrics": best_payload["metrics"],
            "time_cost_computing": metadata["time_cost_computing"],
        }
    )

    return metadata


def build_metrics_summary(
    metrics_df: pd.DataFrame,
    *,
    parameter_grid: Sequence[tuple[str, Mapping[str, Any]]],
    primary_metric: str,
    secondary_metric: str,
) -> pd.DataFrame:
    """
    Ringkas metrics per parameter_set_id dan tambahkan parameter columns.
    """
    group_cols = ["experiment_name", "model_name", "feature_set", "parameter_set_id"]
    summary = summarize_cv_metrics(
        metrics_df,
        group_cols=group_cols,
        metric_cols=METRICS,
    )
    param_frame = _parameter_frame(parameter_grid)
    summary = summary.merge(param_frame, on="parameter_set_id", how="left")

    primary_col = f"{primary_metric}_mean"
    secondary_col = f"{secondary_metric}_mean"
    missing_sort_columns = [
        col for col in [primary_col, secondary_col] if col not in summary.columns
    ]
    if missing_sort_columns:
        raise ValueError(f"Kolom sort best params hilang: {missing_sort_columns}")
    return summary.sort_values(
        by=[primary_col, secondary_col, "parameter_set_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def build_runtime_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ringkas TIME COST COMPUTING per parameter_set_id.
    """
    group_cols = ["experiment_name", "model_name", "feature_set", "parameter_set_id"]
    runtime_cols = [
        "train_time_seconds",
        "prediction_time_seconds",
        "total_runtime_seconds",
    ]
    grouped = metrics_df.groupby(group_cols, dropna=False, sort=False)
    rows: list[dict[str, Any]] = []
    for group_key, group_df in grouped:
        group_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(group_cols, group_values))
        row["n_folds"] = int(group_df["fold"].nunique(dropna=False))
        for col in runtime_cols:
            values = pd.to_numeric(group_df[col], errors="coerce")
            row[f"{col}_sum"] = float(values.sum())
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def build_best_params_payload(
    metrics_summary: pd.DataFrame,
    *,
    parameter_grid: Sequence[tuple[str, Mapping[str, Any]]],
    primary_metric: str,
    secondary_metric: str,
) -> dict[str, Any]:
    """
    Ambil konfigurasi terbaik berdasarkan primary metric mean.
    """
    if metrics_summary.empty:
        raise ValueError("metrics_summary kosong.")
    best_row = metrics_summary.iloc[0].to_dict()
    parameter_lookup = {
        parameter_set_id: dict(params) for parameter_set_id, params in parameter_grid
    }
    parameter_set_id = str(best_row["parameter_set_id"])
    params = parameter_lookup[parameter_set_id]

    return {
        "selection_rule": (
            f"minimize {primary_metric}_mean; tie-breaker {secondary_metric}_mean"
        ),
        "parameter_set_id": parameter_set_id,
        "params": params,
        "metrics": {
            key: value
            for key, value in best_row.items()
            if key.endswith("_mean") or key.endswith("_std") or key in ["n_rows", "n_folds"]
        },
    }


def save_actual_vs_predicted_plot(
    predictions: pd.DataFrame,
    output_path: PathLike,
    *,
    title: str,
) -> None:
    """
    Simpan plot actual vs predicted untuk parameter terbaik.
    """
    if predictions.empty:
        raise ValueError("Prediksi untuk plot kosong.")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib belum tersedia; gunakan --skip-plots atau install requirements."
        ) from exc

    plot_df = predictions.copy()
    plot_df[TIMESTAMP_COL] = pd.to_datetime(plot_df[TIMESTAMP_COL], utc=True)
    plot_df = plot_df.sort_values(TIMESTAMP_COL, kind="mergesort")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(plot_df[TIMESTAMP_COL], plot_df[ACTUAL_COL], label="Actual", linewidth=1.3)
    ax.plot(
        plot_df[TIMESTAMP_COL],
        plot_df[PREDICTION_COL],
        label="Predicted",
        linewidth=1.1,
        alpha=0.85,
    )
    ax.set_title(title)
    ax.set_xlabel("UTC timestamp")
    ax.set_ylabel(TARGET_COL)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def normalize_xgb_feature_set(feature_set: str) -> str:
    normalized = str(feature_set).strip().lower().replace("-", "_")
    aliases = {
        "basic": BASIC_FEATURE_SET,
        "xgb_basic": BASIC_FEATURE_SET,
        "xgboost_basic": BASIC_FEATURE_SET,
        "advanced": ADVANCED_FEATURE_SET,
        "xgb_advanced": ADVANCED_FEATURE_SET,
        "xgboost_advanced": ADVANCED_FEATURE_SET,
    }
    if normalized not in aliases:
        raise ValueError(f"Feature set XGBoost tidak dikenal: {feature_set}")
    return aliases[normalized]


def default_feature_path(feature_set: str) -> Path:
    normalized = normalize_xgb_feature_set(feature_set)
    if normalized == BASIC_FEATURE_SET:
        return XGB_BASIC_FEATURES_PATH
    if normalized == ADVANCED_FEATURE_SET:
        return XGB_ADVANCED_FEATURES_PATH
    raise ValueError(f"Feature set XGBoost tidak dikenal: {feature_set}")


def default_output_dir(feature_set: str) -> Path:
    normalized = normalize_xgb_feature_set(feature_set)
    if normalized == BASIC_FEATURE_SET:
        return XGB_BASIC_OUTPUT_DIR
    if normalized == ADVANCED_FEATURE_SET:
        return XGB_ADVANCED_OUTPUT_DIR
    raise ValueError(f"Feature set XGBoost tidak dikenal: {feature_set}")


def render_tuning_summary(metadata: Mapping[str, Any]) -> str:
    best = metadata["best"]
    time_cost = metadata["time_cost_computing"]
    outputs = metadata["outputs"]

    lines = [
        "=" * 70,
        f"HYPERPARAMETER TUNING SUMMARY - {metadata['experiment_name']}",
        "=" * 70,
        "",
        "1. OUTPUT STATUS",
        f"   - Status: {metadata['status']}",
        f"   - Model: {metadata['model_name']}",
        f"   - Feature set: {metadata['feature_set']}",
        f"   - Input path: {metadata['input_path']}",
        "",
        "2. CV CONFIGURATION",
        f"   - Folds: {metadata['cv_config']['n_folds']}",
        f"   - Validation horizon: {metadata['cv_config']['val_horizon_hours']} hours",
        f"   - Forecast horizon: {metadata['cv_config']['forecast_horizon_hours']} hours",
        f"   - Gap: {metadata['cv_config']['gap_hours']} hours",
        "",
        "3. BEST CONFIGURATION",
        f"   - Parameter set: {best['parameter_set_id']}",
        f"   - Selection rule: {best['selection_rule']}",
        f"   - Params: {json.dumps(best['params'], sort_keys=True)}",
        "",
        "4. TIME COST COMPUTING",
        "   - Sum fold runtime seconds: "
        f"{time_cost['sum_fold_runtime_seconds']:.6f}",
        "   - Sum train time seconds: "
        f"{time_cost['sum_train_time_seconds']:.6f}",
        "   - Sum prediction time seconds: "
        f"{time_cost['sum_prediction_time_seconds']:.6f}",
        "",
        "5. OUTPUT FILES",
        f"   - Params: {outputs['params']}",
        f"   - CV metrics: {outputs['metrics']}",
        f"   - CV metrics summary: {outputs['metrics_summary']}",
        f"   - CV predictions: {outputs['predictions']}",
        f"   - Best params: {outputs['best_params']}",
        f"   - Runtime summary: {outputs['runtime_summary']}",
        "",
    ]
    return "\n".join(lines)


def _run_one_prophet_fold(
    *,
    split: Mapping[str, Any],
    params: Mapping[str, Any],
    parameter_set_id: str,
    experiment_name: str,
    model_name: str,
    feature_set: str,
    forecast_horizon: int,
) -> dict[str, Any]:
    fold = int(split["fold"])
    train = split["train"]
    validation = split["validation"]
    fold_timer = start_timer()
    train_time: Optional[float] = None
    prediction_time: Optional[float] = None
    status = "success"
    error_message = ""

    try:
        train_timer = start_timer()
        model = fit_prophet_model(train, params=params)
        train_time = elapsed_seconds(train_timer)

        prediction_timer = start_timer()
        predictions = forecast_prophet_validation_window(
            model,
            train,
            validation,
            horizon=forecast_horizon,
            model_name=model_name,
            feature_set=feature_set,
            fold=fold,
            parameter_set_id=parameter_set_id,
        )
        prediction_time = elapsed_seconds(prediction_timer)
        total_runtime = elapsed_seconds(fold_timer)
        metrics = _build_fold_metrics_row(
            predictions,
            params=params,
            experiment_name=experiment_name,
            model_name=model_name,
            feature_set=feature_set,
            fold=fold,
            parameter_set_id=parameter_set_id,
            train=train,
            validation=validation,
            train_time_seconds=train_time,
            prediction_time_seconds=prediction_time,
            total_runtime_seconds=total_runtime,
        )
        _log_fold_success(
            metrics,
            experiment_name=experiment_name,
            model_name=model_name,
            feature_set=feature_set,
        )
        return {"predictions": predictions, "metrics": metrics}
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        total_runtime = elapsed_seconds(fold_timer)
        _log_fold_failure(
            experiment_name=experiment_name,
            model_name=model_name,
            feature_set=feature_set,
            fold=fold,
            parameter_set_id=parameter_set_id,
            train=train,
            validation=validation,
            train_time_seconds=train_time,
            prediction_time_seconds=prediction_time,
            total_runtime_seconds=total_runtime,
            status=status,
            error_message=error_message,
        )
        raise


def _run_one_prophet_regressor_basic_fold(
    *,
    split: Mapping[str, Any],
    params: Mapping[str, Any],
    parameter_set_id: str,
    experiment_name: str,
    model_name: str,
    feature_set: str,
    forecast_horizon: int,
) -> dict[str, Any]:
    fold = int(split["fold"])
    train = split["train"]
    validation = split["validation"]
    fold_timer = start_timer()
    train_time: Optional[float] = None
    prediction_time: Optional[float] = None
    status = "success"
    error_message = ""

    try:
        train_timer = start_timer()
        prophet_train = make_prophet_regressor_frame(
            train,
            include_y=True,
            drop_na=True,
        )
        model = make_prophet_regressor_model(
            params,
            regressors=PROPHET_BASIC_REGRESSORS,
        )
        model.fit(prophet_train)
        train_time = elapsed_seconds(train_timer)

        prediction_timer = start_timer()
        predictions = forecast_validation_window_prophet_regressor(
            model,
            train,
            validation,
            horizon=forecast_horizon,
            model_name=model_name,
            fold=fold,
            parameter_set_id=parameter_set_id,
            update_history_with_actuals=True,
        )
        prediction_time = elapsed_seconds(prediction_timer)
        total_runtime = elapsed_seconds(fold_timer)
        metrics = _build_fold_metrics_row(
            predictions,
            params=params,
            experiment_name=experiment_name,
            model_name=model_name,
            feature_set=feature_set,
            fold=fold,
            parameter_set_id=parameter_set_id,
            train=train,
            validation=validation,
            train_time_seconds=train_time,
            prediction_time_seconds=prediction_time,
            total_runtime_seconds=total_runtime,
        )
        metrics["source_train_history_rows"] = int(train.shape[0])
        metrics["regressor_training_rows"] = int(prophet_train.shape[0])
        metrics["regressors"] = "|".join(PROPHET_BASIC_REGRESSORS)
        metrics["validation_predicted_with_recursive_forecasting"] = True
        _log_fold_success(
            metrics,
            experiment_name=experiment_name,
            model_name=model_name,
            feature_set=feature_set,
        )
        return {"predictions": predictions, "metrics": metrics}
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        total_runtime = elapsed_seconds(fold_timer)
        _log_fold_failure(
            experiment_name=experiment_name,
            model_name=model_name,
            feature_set=feature_set,
            fold=fold,
            parameter_set_id=parameter_set_id,
            train=train,
            validation=validation,
            train_time_seconds=train_time,
            prediction_time_seconds=prediction_time,
            total_runtime_seconds=total_runtime,
            status=status,
            error_message=error_message,
        )
        raise


def _run_one_xgb_fold(
    *,
    split: Mapping[str, Any],
    feature_matrix: pd.DataFrame,
    params: Mapping[str, Any],
    parameter_set_id: str,
    experiment_name: str,
    model_name: str,
    feature_set: str,
    forecast_horizon: int,
) -> dict[str, Any]:
    fold = int(split["fold"])
    train = split["train"]
    validation = split["validation"]
    feature_columns = get_feature_columns(feature_set)
    fold_timer = start_timer()
    train_time: Optional[float] = None
    prediction_time: Optional[float] = None
    status = "success"
    error_message = ""

    try:
        train_features = select_xgb_training_rows(
            feature_matrix,
            train_index=train.index,
            validation_index=validation.index,
        )
        train_timer = start_timer()
        model = fit_xgb_model(
            train_features,
            feature_columns=feature_columns,
            params=params,
        )
        train_time = elapsed_seconds(train_timer)

        prediction_timer = start_timer()
        predictions = forecast_validation_window(
            model,
            train,
            validation,
            horizon=forecast_horizon,
            feature_set=feature_set,
            model_name=model_name,
            fold=fold,
            parameter_set_id=parameter_set_id,
            update_history_with_actuals=True,
        )
        prediction_time = elapsed_seconds(prediction_timer)
        total_runtime = elapsed_seconds(fold_timer)
        metrics = _build_fold_metrics_row(
            predictions,
            params=params,
            experiment_name=experiment_name,
            model_name=model_name,
            feature_set=feature_set,
            fold=fold,
            parameter_set_id=parameter_set_id,
            train=train_features,
            validation=validation,
            train_time_seconds=train_time,
            prediction_time_seconds=prediction_time,
            total_runtime_seconds=total_runtime,
        )
        metrics["source_train_history_rows"] = int(train.shape[0])
        metrics["feature_training_rows"] = int(train_features.shape[0])
        metrics["validation_predicted_with_recursive_forecasting"] = True
        _log_fold_success(
            metrics,
            experiment_name=experiment_name,
            model_name=model_name,
            feature_set=feature_set,
        )
        return {"predictions": predictions, "metrics": metrics}
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        total_runtime = elapsed_seconds(fold_timer)
        _log_fold_failure(
            experiment_name=experiment_name,
            model_name=model_name,
            feature_set=feature_set,
            fold=fold,
            parameter_set_id=parameter_set_id,
            train=train,
            validation=validation,
            train_time_seconds=train_time,
            prediction_time_seconds=prediction_time,
            total_runtime_seconds=total_runtime,
            status=status,
            error_message=error_message,
        )
        raise


def forecast_prophet_validation_window(
    model: Any,
    train_history: pd.DataFrame,
    validation_data: pd.DataFrame,
    *,
    horizon: int,
    model_name: str,
    feature_set: str,
    fold: int,
    parameter_set_id: str,
) -> pd.DataFrame:
    """
    Prediksi Prophet dalam rolling-origin chunks 24 jam untuk konsistensi CV.
    """
    if validation_data.empty:
        raise ValueError("validation_data Prophet kosong.")
    if TARGET_COL not in validation_data.columns:
        raise ValueError(f"Kolom target validation Prophet hilang: {TARGET_COL}")
    if train_history.index.max() >= validation_data.index.min():
        raise ValueError("Train history Prophet harus berakhir sebelum validation.")

    all_predictions: list[pd.DataFrame] = []
    current_origin = train_history.index.max()
    block_id = 0

    for start in range(0, len(validation_data), horizon):
        block_id += 1
        validation_block = validation_data.iloc[start : start + horizon]
        future = make_prophet_future_frame(validation_block.index)
        block_predictions = recursive_forecast_prophet(
            model,
            future,
            horizon=len(validation_block),
            model_name=model_name,
            fold=fold,
            parameter_set_id=parameter_set_id,
        )
        block_predictions[ACTUAL_COL] = pd.to_numeric(
            validation_block[TARGET_COL],
            errors="raise",
        ).to_numpy(dtype=float)
        block_predictions["feature_set"] = feature_set
        block_predictions["forecast_origin"] = current_origin
        block_predictions["origin_block"] = int(block_id)
        block_predictions["validation_start"] = validation_block.index.min()
        block_predictions["validation_end"] = validation_block.index.max()
        all_predictions.append(block_predictions)
        current_origin = validation_block.index.max()

    predictions = pd.concat(all_predictions, ignore_index=True)
    validate_prediction_output(
        predictions,
        expected_index=validation_data.index,
        horizon=len(validation_data),
    )
    predictions.attrs["prediction_time_seconds_total"] = round(
        float(predictions["prediction_time_seconds"].sum()),
        9,
    )
    return predictions


def select_xgb_training_rows(
    feature_matrix: pd.DataFrame,
    *,
    train_index: pd.DatetimeIndex,
    validation_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Ambil hanya feature rows yang timestamp-nya berada pada training fold.
    """
    selected_index = feature_matrix.index.intersection(train_index)
    train_features = feature_matrix.loc[selected_index].copy()
    if train_features.empty:
        raise ValueError("Tidak ada feature rows XGBoost untuk training fold.")
    if train_features.index.max() > train_index.max():
        raise ValueError("Feature training XGBoost melewati train_end fold.")
    if train_features.index.intersection(validation_index).size > 0:
        raise ValueError("Feature training XGBoost overlap dengan validation fold.")
    return train_features


def _build_fold_metrics_row(
    predictions: pd.DataFrame,
    *,
    params: Mapping[str, Any],
    experiment_name: str,
    model_name: str,
    feature_set: str,
    fold: int,
    parameter_set_id: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_time_seconds: float,
    prediction_time_seconds: float,
    total_runtime_seconds: float,
) -> dict[str, Any]:
    metric_values = compute_all_metrics(
        predictions[ACTUAL_COL],
        predictions[PREDICTION_COL],
    )
    row: dict[str, Any] = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "feature_set": feature_set,
        "parameter_set_id": parameter_set_id,
        "fold": int(fold),
        "train_start": train.index.min().isoformat(),
        "train_end": train.index.max().isoformat(),
        "validation_start": validation.index.min().isoformat(),
        "validation_end": validation.index.max().isoformat(),
        "n_train_rows": int(train.shape[0]),
        "n_prediction_rows": int(predictions.shape[0]),
        "train_time_seconds": float(train_time_seconds),
        "prediction_time_seconds": float(prediction_time_seconds),
        "model_predict_time_seconds": float(
            pd.to_numeric(predictions["prediction_time_seconds"]).sum()
        ),
        "total_runtime_seconds": float(total_runtime_seconds),
        "used_actual_future_for_features": bool(
            predictions["used_actual_future_for_features"].astype(bool).any()
        ),
    }
    row.update(metric_values)
    for key, value in params.items():
        row[f"param_{key}"] = value
    return row


def _log_fold_success(
    metrics: Mapping[str, Any],
    *,
    experiment_name: str,
    model_name: str,
    feature_set: str,
) -> None:
    runtime_record = make_runtime_record(
        experiment_name=experiment_name,
        model_name=model_name,
        feature_set=feature_set,
        fold=metrics["fold"],
        parameter_set_id=metrics["parameter_set_id"],
        train_start=metrics["train_start"],
        train_end=metrics["train_end"],
        validation_start=metrics["validation_start"],
        validation_end=metrics["validation_end"],
        n_train_rows=metrics["n_train_rows"],
        n_prediction_rows=metrics["n_prediction_rows"],
        train_time_seconds=metrics["train_time_seconds"],
        prediction_time_seconds=metrics["prediction_time_seconds"],
        total_runtime_seconds=metrics["total_runtime_seconds"],
        status="success",
    )
    log_runtime(runtime_record)
    append_experiment_run(
        {
            **dict(metrics),
            "experiment_name": experiment_name,
            "model_name": model_name,
            "feature_set": feature_set,
            "status": "success",
        }
    )


def _log_fold_failure(
    *,
    experiment_name: str,
    model_name: str,
    feature_set: str,
    fold: int,
    parameter_set_id: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_time_seconds: Optional[float],
    prediction_time_seconds: Optional[float],
    total_runtime_seconds: float,
    status: str,
    error_message: str,
) -> None:
    runtime_record = make_runtime_record(
        experiment_name=experiment_name,
        model_name=model_name,
        feature_set=feature_set,
        fold=fold,
        parameter_set_id=parameter_set_id,
        train_start=train.index.min(),
        train_end=train.index.max(),
        validation_start=validation.index.min(),
        validation_end=validation.index.max(),
        n_train_rows=train.shape[0],
        n_prediction_rows=validation.shape[0],
        train_time_seconds=train_time_seconds,
        prediction_time_seconds=prediction_time_seconds,
        total_runtime_seconds=total_runtime_seconds,
        status=status,
        error_message=error_message,
    )
    log_runtime(runtime_record)
    append_experiment_run(
        {
            "experiment_name": experiment_name,
            "model_name": model_name,
            "feature_set": feature_set,
            "fold": fold,
            "parameter_set_id": parameter_set_id,
            "status": status,
            "error_message": error_message,
            "total_runtime_seconds": total_runtime_seconds,
        }
    )


def _parameter_frame(
    parameter_grid: Sequence[tuple[str, Mapping[str, Any]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for parameter_set_id, params in parameter_grid:
        row = {"parameter_set_id": parameter_set_id}
        for key, value in params.items():
            row[f"param_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _append_dataframe_csv(df: pd.DataFrame, output_path: PathLike) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_header = not destination.exists() or destination.stat().st_size == 0
    df.to_csv(destination, mode="a", header=write_header, index=False)


def _cv_config_dict(
    n_folds: int,
    val_horizon: int,
    gap: int,
    forecast_horizon: int,
) -> dict[str, int]:
    return {
        "n_folds": int(n_folds),
        "val_horizon_hours": int(val_horizon),
        "gap_hours": int(gap),
        "forecast_horizon_hours": int(forecast_horizon),
    }


__all__ = [
    "build_best_params_payload",
    "build_metrics_summary",
    "build_runtime_summary",
    "default_feature_path",
    "default_output_dir",
    "finalize_tuning_outputs",
    "forecast_prophet_validation_window",
    "iter_parameter_grid",
    "load_xgb_feature_matrix",
    "normalize_xgb_feature_set",
    "prepare_tuning_output_paths",
    "render_tuning_summary",
    "run_prophet_regressor_basic_tuning",
    "run_prophet_tuning",
    "run_xgb_tuning",
    "select_xgb_training_rows",
]

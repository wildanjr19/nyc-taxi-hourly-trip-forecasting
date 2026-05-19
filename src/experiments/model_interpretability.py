"""
Script tahap 15b: Model Interpretability Add-ons.

Scope:
- Load model artifact final hasil tahap 13 retraining.
- Buat Prophet component values/plots untuk train_val dan optional timestamp
  final test sebagai post-hoc view.
- Hitung SHAP XGBoost pada feature matrix train_val yang sudah dibuat pada
  tahap feature engineering.
- Tidak melakukan tuning ulang, retraining ulang, atau perubahan final metrics,
  final predictions, dan ranking model.

Contoh:
    python -m src.experiments.model_interpretability
    python -m src.experiments.model_interpretability --skip-plots
    python -m src.experiments.model_interpretability --shap-sample-size 1000
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
    FINAL_TEST_PATH,
    LOCAL_TZ,
    MODELING_TZ,
    OUTPUT_DIR,
    PROJECT_ROOT,
    PROPHET_OUTPUT_DIR,
    REPORTS_DIR,
    TARGET_COL,
    TIMESTAMP_COL,
    TRAIN_VAL_PATH,
    XGB_ADVANCED_FEATURES_PATH,
    XGB_ADVANCED_OUTPUT_DIR,
    XGB_BASIC_FEATURES_PATH,
    XGB_BASIC_OUTPUT_DIR,
    ensure_dirs,
)
from src.experiments.tuning_utils import load_xgb_feature_matrix
from src.features import ADVANCED_FEATURE_SET, BASIC_FEATURE_SET, get_feature_columns
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


EXPERIMENT_NAME = "model_interpretability"
RETRAINING_DIR = EXPERIMENTS_DIR / "retraining"
MODEL_REGISTRY_PATH = RETRAINING_DIR / "model_registry.json"
OUTPUT_DIR_STAGE = EXPERIMENTS_DIR / "model_interpretability"
REPORT_PATH = REPORTS_DIR / "model_interpretability.md"

PROPHET_MODEL_PATH = RETRAINING_DIR / "models" / "prophet_retrained.json"
XGB_BASIC_MODEL_PATH = RETRAINING_DIR / "models" / "xgb_basic_retrained.json"
XGB_ADVANCED_MODEL_PATH = RETRAINING_DIR / "models" / "xgb_advanced_retrained.json"

DEFAULT_SHAP_SAMPLE_SIZE = 2000
SHAP_SAMPLE_STRATEGY = "chronological_even_spacing"
PROPHET_COMPONENT_COLUMNS = [
    "trend",
    "trend_lower",
    "trend_upper",
    "weekly",
    "weekly_lower",
    "weekly_upper",
    "daily",
    "daily_lower",
    "daily_upper",
    "additive_terms",
    "additive_terms_lower",
    "additive_terms_upper",
    "multiplicative_terms",
    "multiplicative_terms_lower",
    "multiplicative_terms_upper",
    "yhat",
    "yhat_lower",
    "yhat_upper",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model interpretability add-ons for final retrained models."
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip matplotlib/SHAP figure generation.",
    )
    parser.add_argument(
        "--shap-sample-size",
        type=int,
        default=DEFAULT_SHAP_SAMPLE_SIZE,
        help=(
            "Maximum train_val feature rows used for SHAP. "
            f"Default: {DEFAULT_SHAP_SAMPLE_SIZE}."
        ),
    )
    parser.add_argument(
        "--skip-final-test-components",
        action="store_true",
        help=(
            "Skip optional Prophet component view on final-test timestamps. "
            "This view uses timestamps only and never labels."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    metadata = run_model_interpretability(
        shap_sample_size=args.shap_sample_size,
        skip_plots=args.skip_plots,
        include_final_test_components=not args.skip_final_test_components,
    )
    print("Model interpretability selesai.")
    print(f"Status: {metadata['status']}")
    print(f"Output dir: {metadata['outputs']['base']}")
    print(f"Report: {metadata['outputs']['report']}")


def run_model_interpretability(
    *,
    shap_sample_size: int = DEFAULT_SHAP_SAMPLE_SIZE,
    skip_plots: bool = False,
    include_final_test_components: bool = True,
) -> dict[str, Any]:
    if int(shap_sample_size) <= 0:
        raise ValueError("shap_sample_size harus > 0.")

    timer = start_timer()
    started_at = utc_now_iso()
    status = "success"
    error_message = ""
    outputs = interpretability_output_paths()

    try:
        ensure_dirs()
        ensure_interpretability_dirs(outputs)
        registry = load_model_registry(MODEL_REGISTRY_PATH)
        validate_model_artifacts(registry)

        train_val = load_split_timeseries(TRAIN_VAL_PATH)
        prophet_result = run_prophet_interpretability(
            train_val=train_val,
            outputs=outputs,
            skip_plots=skip_plots,
            include_final_test_components=include_final_test_components,
        )

        xgb_basic_result = run_xgb_shap_interpretability(
            model_key="xgb_basic",
            model_label="XGBoost-Basic",
            feature_set=BASIC_FEATURE_SET,
            model_path=XGB_BASIC_MODEL_PATH,
            feature_path=XGB_BASIC_FEATURES_PATH,
            output_dir=outputs["xgb_basic_shap"],
            sample_size=shap_sample_size,
            skip_plots=skip_plots,
            dependence_features=["lag_1", "lag_24", "lag_168"],
        )
        xgb_advanced_result = run_xgb_shap_interpretability(
            model_key="xgb_advanced",
            model_label="XGBoost-Advanced",
            feature_set=ADVANCED_FEATURE_SET,
            model_path=XGB_ADVANCED_MODEL_PATH,
            feature_path=XGB_ADVANCED_FEATURES_PATH,
            output_dir=outputs["xgb_advanced_shap"],
            sample_size=shap_sample_size,
            skip_plots=skip_plots,
            dependence_features=[
                "lag_1",
                "lag_24",
                "lag_168",
                "rolling_mean_24",
                "rolling_mean_168",
                "rolling_std_24",
            ],
        )

        runtime_summary = build_runtime_summary(
            prophet_result,
            xgb_basic_result,
            xgb_advanced_result,
        )
        runtime_summary.to_csv(outputs["runtime_summary"], index=False)

        total_runtime_seconds = elapsed_seconds(timer)
        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "stage": "model_interpretability_addons",
            "status": status,
            "error_message": error_message,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "total_runtime_seconds": total_runtime_seconds,
            "skip_plots": bool(skip_plots),
            "include_final_test_components": bool(include_final_test_components),
            "shap_sample_size_requested": int(shap_sample_size),
            "sampling_strategy": SHAP_SAMPLE_STRATEGY,
            "input_artifacts": {
                "model_registry": str(MODEL_REGISTRY_PATH),
                "prophet_model": str(PROPHET_MODEL_PATH),
                "xgb_basic_model": str(XGB_BASIC_MODEL_PATH),
                "xgb_advanced_model": str(XGB_ADVANCED_MODEL_PATH),
                "train_val": str(TRAIN_VAL_PATH),
                "xgb_basic_train_val_features": str(XGB_BASIC_FEATURES_PATH),
                "xgb_advanced_train_val_features": str(XGB_ADVANCED_FEATURES_PATH),
                "final_test_timestamps_only": str(FINAL_TEST_PATH)
                if include_final_test_components
                else None,
            },
            "leakage_guardrail": [
                "Tidak ada tuning ulang.",
                "Tidak ada retraining ulang.",
                "Tidak ada perubahan pada final predictions, final metrics, atau ranking.",
                "Prophet components memakai model retrained tahap 13.",
                "SHAP XGBoost memakai feature matrix train_val saja.",
                "SHAP final test tidak dihitung dari precomputed feature matrix final test.",
                "Optional Prophet final-test component view memakai timestamp saja, bukan label.",
            ],
            "prophet": prophet_result,
            "xgb_basic": xgb_basic_result,
            "xgb_advanced": xgb_advanced_result,
            "outputs": stringify_paths(outputs),
        }

        report_text = render_interpretability_report(
            metadata=metadata,
            runtime_summary=runtime_summary,
            prophet_result=prophet_result,
            xgb_basic_result=xgb_basic_result,
            xgb_advanced_result=xgb_advanced_result,
        )
        write_text(outputs["summary"], report_text)
        write_text(outputs["report"], report_text)
        save_experiment_metadata(metadata, outputs["metadata"])
        log_runtime(
            make_runtime_record(
                experiment_name=EXPERIMENT_NAME,
                model_name="all_models",
                feature_set="interpretability",
                n_train_rows=int(train_val.shape[0]),
                n_prediction_rows=int(
                    xgb_basic_result["sample_rows"]
                    + xgb_advanced_result["sample_rows"]
                ),
                train_time_seconds=0.0,
                prediction_time_seconds=float(
                    xgb_basic_result["shap_compute_time_seconds"]
                    + xgb_advanced_result["shap_compute_time_seconds"]
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
            "stage": "model_interpretability_addons",
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
                model_name="all_models",
                feature_set="interpretability",
                total_runtime_seconds=total_runtime_seconds,
                status=status,
                error_message=error_message,
            )
        )
        append_experiment_run(failure_metadata)
        raise


def interpretability_output_paths() -> dict[str, Path]:
    prophet_components = PROPHET_OUTPUT_DIR / "components"
    prophet_figures = PROPHET_OUTPUT_DIR / "figures"
    xgb_basic_shap = XGB_BASIC_OUTPUT_DIR / "shap"
    xgb_advanced_shap = XGB_ADVANCED_OUTPUT_DIR / "shap"
    metrics_dir = OUTPUT_DIR_STAGE / "metrics"
    summaries_dir = OUTPUT_DIR_STAGE / "summaries"
    return {
        "base": OUTPUT_DIR_STAGE,
        "metrics": metrics_dir,
        "summaries": summaries_dir,
        "prophet_components": prophet_components,
        "prophet_figures": prophet_figures,
        "xgb_basic_shap": xgb_basic_shap,
        "xgb_advanced_shap": xgb_advanced_shap,
        "prophet_components_train_val": prophet_components
        / "prophet_components_train_val.csv",
        "prophet_components_final_test": prophet_components
        / "prophet_components_final_test.csv",
        "prophet_components_plot": prophet_figures / "prophet_components_train_val.png",
        "prophet_trend_plot": prophet_figures / "prophet_trend_component.png",
        "prophet_daily_plot": prophet_figures / "prophet_daily_component.png",
        "prophet_weekly_plot": prophet_figures / "prophet_weekly_component.png",
        "runtime_summary": metrics_dir / "interpretability_runtime_summary.csv",
        "metadata": OUTPUT_DIR_STAGE / "experiment_metadata.json",
        "summary": summaries_dir / "model_interpretability_summary.md",
        "report": REPORT_PATH,
    }


def ensure_interpretability_dirs(paths: Mapping[str, Path]) -> None:
    for key in [
        "base",
        "metrics",
        "summaries",
        "prophet_components",
        "prophet_figures",
        "xgb_basic_shap",
        "xgb_advanced_shap",
    ]:
        paths[key].mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model registry retraining tidak ditemukan: {path}")
    payload = load_json(path)
    for key in ["prophet", "xgb_basic", "xgb_advanced"]:
        if key not in payload:
            raise ValueError(f"Model registry tidak memiliki entry: {key}")
    return payload


def validate_model_artifacts(registry: Mapping[str, Any]) -> None:
    expected_paths = {
        "prophet": PROPHET_MODEL_PATH,
        "xgb_basic": XGB_BASIC_MODEL_PATH,
        "xgb_advanced": XGB_ADVANCED_MODEL_PATH,
    }
    for key, expected_path in expected_paths.items():
        if not expected_path.exists():
            raise FileNotFoundError(f"Model artifact {key} tidak ditemukan: {expected_path}")
        registry_path = Path(str(registry[key]["model_path"]))
        if registry_path.resolve() != expected_path.resolve():
            raise ValueError(
                f"Model registry {key} tidak sesuai. "
                f"Expected={expected_path}, found={registry_path}"
            )


def run_prophet_interpretability(
    *,
    train_val: pd.DataFrame,
    outputs: Mapping[str, Path],
    skip_plots: bool,
    include_final_test_components: bool,
) -> dict[str, Any]:
    model_load_timer = start_timer()
    model = load_prophet_model(PROPHET_MODEL_PATH)
    model_load_time = elapsed_seconds(model_load_timer)

    train_component_timer = start_timer()
    train_components = build_prophet_components(model, train_val.index)
    train_component_time = elapsed_seconds(train_component_timer)
    train_components.to_csv(outputs["prophet_components_train_val"], index=False)

    final_components_time = 0.0
    final_component_rows = 0
    final_component_period = None
    if include_final_test_components:
        final_index = load_timestamp_index_only(FINAL_TEST_PATH)
        final_component_timer = start_timer()
        final_components = build_prophet_components(model, final_index)
        final_components_time = elapsed_seconds(final_component_timer)
        final_components.to_csv(outputs["prophet_components_final_test"], index=False)
        final_component_rows = int(final_components.shape[0])
        final_component_period = summarize_timestamp_column(final_components)

    plot_time = 0.0
    plot_outputs: list[str] = []
    if not skip_plots:
        plot_timer = start_timer()
        save_prophet_component_plots(
            model=model,
            train_components=train_components,
            outputs=outputs,
        )
        plot_time = elapsed_seconds(plot_timer)
        plot_outputs = [
            str(outputs["prophet_components_plot"]),
            str(outputs["prophet_trend_plot"]),
            str(outputs["prophet_daily_plot"]),
            str(outputs["prophet_weekly_plot"]),
        ]

    total_runtime = round(
        model_load_time + train_component_time + final_components_time + plot_time,
        6,
    )
    components_present = [
        column for column in ["trend", "daily", "weekly"] if column in train_components
    ]
    components_absent = [
        name for name in ["yearly", "monthly"] if name not in train_components.columns
    ]
    result = {
        "model_key": "prophet",
        "model_label": "Prophet",
        "model_name": "prophet",
        "feature_set": "prophet_internal",
        "model_path": str(PROPHET_MODEL_PATH),
        "train_val_component_rows": int(train_components.shape[0]),
        "train_val_period": summarize_timestamp_column(train_components),
        "final_test_component_rows": final_component_rows,
        "final_test_period": final_component_period,
        "components_present": components_present,
        "components_absent": components_absent,
        "yearly_seasonality": bool(getattr(model, "yearly_seasonality", False)),
        "weekly_seasonality": bool(getattr(model, "weekly_seasonality", False)),
        "daily_seasonality": bool(getattr(model, "daily_seasonality", False)),
        "monthly_custom_seasonality_present": "monthly" in getattr(
            model,
            "seasonalities",
            {},
        ),
        "model_load_time_seconds": model_load_time,
        "component_compute_train_val_seconds": train_component_time,
        "component_compute_final_test_seconds": final_components_time,
        "plot_time_seconds": plot_time,
        "total_runtime_seconds": total_runtime,
        "outputs": {
            "train_val_components": str(outputs["prophet_components_train_val"]),
            "final_test_components": str(outputs["prophet_components_final_test"])
            if include_final_test_components
            else None,
            "plots": plot_outputs,
        },
    }
    log_runtime(
        make_runtime_record(
            experiment_name=EXPERIMENT_NAME,
            model_name="prophet",
            feature_set="prophet_internal",
            n_train_rows=int(train_components.shape[0]),
            n_prediction_rows=final_component_rows,
            train_time_seconds=0.0,
            prediction_time_seconds=train_component_time + final_components_time,
            total_runtime_seconds=total_runtime,
            status="success",
        )
    )
    return result


def load_prophet_model(path: Path) -> Any:
    try:
        from prophet.serialize import model_from_json
    except ImportError as exc:
        raise ImportError(
            "prophet.serialize.model_from_json tidak tersedia. "
            "Install dependencies dari requirements.txt."
        ) from exc
    return model_from_json(path.read_text(encoding="utf-8"))


def build_prophet_components(model: Any, index: pd.DatetimeIndex) -> pd.DataFrame:
    future = make_prophet_future_frame(index)
    forecast = model.predict(future)
    available_columns = [
        column for column in PROPHET_COMPONENT_COLUMNS if column in forecast.columns
    ]
    components = forecast[["ds", *available_columns]].copy()
    components.insert(
        0,
        TIMESTAMP_COL,
        pd.to_datetime(components["ds"], errors="raise").dt.tz_localize(MODELING_TZ),
    )
    local_ts = components[TIMESTAMP_COL].dt.tz_convert(LOCAL_TZ)
    components["local_hour"] = local_ts.dt.hour.astype(int)
    components["local_day_of_week"] = local_ts.dt.dayofweek.astype(int)
    components["local_is_weekend"] = components["local_day_of_week"].isin([5, 6])
    return components


def load_timestamp_index_only(path: Path) -> pd.DatetimeIndex:
    if not path.exists():
        raise FileNotFoundError(f"File timestamp tidak ditemukan: {path}")
    frame = pd.read_csv(path, usecols=[TIMESTAMP_COL])
    timestamps = pd.to_datetime(frame[TIMESTAMP_COL], utc=True, errors="raise")
    index = pd.DatetimeIndex(timestamps).sort_values()
    if index.empty:
        raise ValueError(f"Timestamp kosong: {path}")
    if index.has_duplicates:
        raise ValueError(f"Timestamp duplicate: {path}")
    if not index.is_monotonic_increasing:
        raise ValueError(f"Timestamp tidak chronological: {path}")
    return index


def save_prophet_component_plots(
    *,
    model: Any,
    train_components: pd.DataFrame,
    outputs: Mapping[str, Path],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    component_forecast = train_components.drop(
        columns=[TIMESTAMP_COL, "local_hour", "local_day_of_week", "local_is_weekend"],
        errors="ignore",
    ).copy()
    fig_components = model.plot_components(component_forecast)
    fig_components.suptitle("Prophet Components - Train/Validation Window", y=1.01)
    fig_components.tight_layout()
    fig_components.savefig(outputs["prophet_components_plot"], dpi=150)
    plt.close(fig_components)

    trend_df = train_components.sort_values(TIMESTAMP_COL, kind="mergesort")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(trend_df[TIMESTAMP_COL], trend_df["trend"], linewidth=1.2)
    ax.set_title("Prophet Trend Component - Train/Validation Window")
    ax.set_xlabel("UTC timestamp")
    ax.set_ylabel("Trend component")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outputs["prophet_trend_plot"], dpi=150)
    plt.close(fig)

    daily = (
        train_components.groupby("local_hour", as_index=False)["daily"]
        .mean()
        .sort_values("local_hour", kind="mergesort")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(daily["local_hour"], daily["daily"], marker="o", linewidth=1.4)
    ax.set_title("Prophet Daily Seasonality Component by NYC Local Hour")
    ax.set_xlabel("NYC local hour")
    ax.set_ylabel("Daily component")
    ax.set_xticks(list(range(24)))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outputs["prophet_daily_plot"], dpi=150)
    plt.close(fig)

    weekly = (
        train_components.groupby("local_day_of_week", as_index=False)["weekly"]
        .mean()
        .sort_values("local_day_of_week", kind="mergesort")
    )
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(weekly["local_day_of_week"], weekly["weekly"], alpha=0.85)
    ax.set_title("Prophet Weekly Seasonality Component by NYC Local Day")
    ax.set_xlabel("NYC local day")
    ax.set_ylabel("Weekly component")
    ax.set_xticks(list(range(7)))
    ax.set_xticklabels(day_names)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outputs["prophet_weekly_plot"], dpi=150)
    plt.close(fig)


def run_xgb_shap_interpretability(
    *,
    model_key: str,
    model_label: str,
    feature_set: str,
    model_path: Path,
    feature_path: Path,
    output_dir: Path,
    sample_size: int,
    skip_plots: bool,
    dependence_features: Sequence[str],
) -> dict[str, Any]:
    output_paths = xgb_shap_output_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_matrix = load_xgb_feature_matrix(feature_path, feature_set)
    feature_columns = get_feature_columns(feature_set)
    validate_shap_feature_matrix(feature_matrix, feature_columns)
    sample = sample_feature_matrix(feature_matrix, sample_size=sample_size)
    x_sample = sample[feature_columns].astype(float).copy()

    model_load_timer = start_timer()
    model = load_xgb_model(model_path)
    model_load_time = elapsed_seconds(model_load_timer)
    validate_xgb_feature_names(model, feature_columns)

    shap_timer = start_timer()
    shap_values, expected_value = compute_tree_shap_values(model, x_sample)
    shap_compute_time = elapsed_seconds(shap_timer)

    importance = build_shap_importance(shap_values, feature_columns)
    group_summary = build_shap_group_summary(importance)
    shap_sample_values = build_shap_values_sample(sample, x_sample, shap_values)

    importance.to_csv(output_paths["importance"], index=False)
    group_summary.to_csv(output_paths["group_summary"], index=False)
    shap_sample_values.to_csv(output_paths["values_sample"], index=False)

    plot_time = 0.0
    plot_outputs: list[str] = []
    if not skip_plots:
        plot_timer = start_timer()
        save_shap_plots(
            shap_values=shap_values,
            x_sample=x_sample,
            importance=importance,
            output_paths=output_paths,
            dependence_features=dependence_features,
        )
        plot_time = elapsed_seconds(plot_timer)
        plot_outputs = [
            str(output_paths["summary_bar"]),
            str(output_paths["beeswarm"]),
            *[
                str(output_paths["dependence"](feature))
                for feature in dependence_features
                if feature in feature_columns
            ],
        ]

    total_runtime = round(model_load_time + shap_compute_time + plot_time, 6)
    sample_period = summarize_index(sample.index)
    result = {
        "model_key": model_key,
        "model_label": model_label,
        "model_name": "xgboost",
        "feature_set": feature_set,
        "model_path": str(model_path),
        "feature_path": str(feature_path),
        "feature_rows_total": int(feature_matrix.shape[0]),
        "sample_rows": int(sample.shape[0]),
        "sample_size_requested": int(sample_size),
        "sample_strategy": SHAP_SAMPLE_STRATEGY,
        "sample_period": sample_period,
        "feature_columns": feature_columns,
        "top_features": importance.head(10).to_dict(orient="records"),
        "group_importance": group_summary.to_dict(orient="records"),
        "expected_value": to_jsonable_scalar(expected_value),
        "model_load_time_seconds": model_load_time,
        "shap_compute_time_seconds": shap_compute_time,
        "plot_time_seconds": plot_time,
        "total_runtime_seconds": total_runtime,
        "outputs": {
            "importance": str(output_paths["importance"]),
            "group_summary": str(output_paths["group_summary"]),
            "values_sample": str(output_paths["values_sample"]),
            "plots": plot_outputs,
        },
    }
    log_runtime(
        make_runtime_record(
            experiment_name=EXPERIMENT_NAME,
            model_name="xgboost",
            feature_set=feature_set,
            n_train_rows=int(feature_matrix.shape[0]),
            n_prediction_rows=int(sample.shape[0]),
            train_time_seconds=0.0,
            prediction_time_seconds=shap_compute_time,
            total_runtime_seconds=total_runtime,
            status="success",
        )
    )
    return result


def xgb_shap_output_paths(output_dir: Path) -> dict[str, Any]:
    return {
        "importance": output_dir / "shap_importance.csv",
        "group_summary": output_dir / "shap_group_importance.csv",
        "values_sample": output_dir / "shap_values_sample.csv",
        "summary_bar": output_dir / "shap_summary_bar.png",
        "beeswarm": output_dir / "shap_beeswarm.png",
        "dependence": lambda feature: output_dir / f"shap_dependence_{feature}.png",
    }


def validate_shap_feature_matrix(
    feature_matrix: pd.DataFrame,
    feature_columns: Sequence[str],
) -> None:
    if TARGET_COL in feature_columns:
        raise ValueError("Target tidak boleh masuk dalam feature_columns SHAP.")
    missing = sorted(set(feature_columns).difference(feature_matrix.columns))
    if missing:
        raise ValueError(f"Feature matrix SHAP tidak lengkap: {missing}")
    if TARGET_COL not in feature_matrix.columns:
        raise ValueError(f"Feature matrix SHAP tidak memiliki target audit: {TARGET_COL}")


def sample_feature_matrix(
    feature_matrix: pd.DataFrame,
    *,
    sample_size: int,
) -> pd.DataFrame:
    if int(sample_size) >= feature_matrix.shape[0]:
        return feature_matrix.copy()
    positions = np.linspace(
        0,
        feature_matrix.shape[0] - 1,
        num=int(sample_size),
        dtype=int,
    )
    positions = np.unique(positions)
    return feature_matrix.iloc[positions].copy()


def load_xgb_model(path: Path) -> Any:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "xgboost belum tersedia. Install dependencies dari requirements.txt."
        ) from exc
    model = XGBRegressor()
    model.load_model(str(path))
    return model


def validate_xgb_feature_names(model: Any, feature_columns: Sequence[str]) -> None:
    booster_names = list(model.get_booster().feature_names or [])
    if booster_names and booster_names != list(feature_columns):
        raise ValueError(
            "Feature names pada model XGBoost tidak sesuai dengan feature matrix. "
            f"model={booster_names}, matrix={list(feature_columns)}"
        )


def compute_tree_shap_values(model: Any, x_sample: pd.DataFrame) -> tuple[np.ndarray, Any]:
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "Package shap belum tersedia. Tambahkan/instal dependency shap."
        ) from exc

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    if isinstance(shap_values, list):
        if len(shap_values) != 1:
            raise ValueError("SHAP multi-output tidak didukung untuk regresi ini.")
        shap_values = shap_values[0]
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values
    shap_array = np.asarray(shap_values, dtype=float)
    if shap_array.ndim != 2:
        raise ValueError(f"Dimensi SHAP tidak valid: {shap_array.shape}")
    if shap_array.shape != x_sample.shape:
        raise ValueError(
            f"Bentuk SHAP tidak sesuai. shap={shap_array.shape}, X={x_sample.shape}"
        )
    return shap_array, explainer.expected_value


def build_shap_importance(
    shap_values: np.ndarray,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    std_abs = np.abs(shap_values).std(axis=0)
    frame = pd.DataFrame(
        {
            "feature": list(feature_columns),
            "feature_group": [feature_group_name(feature) for feature in feature_columns],
            "mean_abs_shap": mean_abs,
            "mean_shap": mean_signed,
            "std_abs_shap": std_abs,
        }
    )
    total = float(frame["mean_abs_shap"].sum())
    frame["share_of_total_mean_abs_shap"] = (
        frame["mean_abs_shap"] / total if total > 0 else 0.0
    )
    frame = frame.sort_values(
        ["mean_abs_shap", "feature"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame


def build_shap_group_summary(importance: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        importance.groupby("feature_group", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    total = float(grouped["mean_abs_shap"].sum())
    grouped["share_of_total_mean_abs_shap"] = (
        grouped["mean_abs_shap"] / total if total > 0 else 0.0
    )
    grouped.insert(0, "rank", np.arange(1, len(grouped) + 1))
    return grouped


def feature_group_name(feature: str) -> str:
    if feature.startswith("lag_"):
        return "lag"
    if feature.startswith("rolling_mean_"):
        return "rolling_mean"
    if feature.startswith("rolling_std_"):
        return "rolling_std"
    return "calendar"


def build_shap_values_sample(
    sample: pd.DataFrame,
    x_sample: pd.DataFrame,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    output = pd.DataFrame({TIMESTAMP_COL: sample.index.astype(str)})
    for column in x_sample.columns:
        output[f"feature__{column}"] = x_sample[column].to_numpy()
    for position, column in enumerate(x_sample.columns):
        output[f"shap__{column}"] = shap_values[:, position]
    return output


def save_shap_plots(
    *,
    shap_values: np.ndarray,
    x_sample: pd.DataFrame,
    importance: pd.DataFrame,
    output_paths: Mapping[str, Any],
    dependence_features: Sequence[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import shap

    max_display = min(20, x_sample.shape[1])
    shap.summary_plot(
        shap_values,
        x_sample,
        plot_type="bar",
        show=False,
        max_display=max_display,
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(output_paths["summary_bar"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    shap.summary_plot(
        shap_values,
        x_sample,
        show=False,
        max_display=max_display,
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(output_paths["beeswarm"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    available = set(x_sample.columns)
    for feature in dependence_features:
        if feature not in available:
            continue
        shap.dependence_plot(
            feature,
            shap_values,
            x_sample,
            interaction_index=None,
            show=False,
        )
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(output_paths["dependence"](feature), dpi=150, bbox_inches="tight")
        plt.close(fig)

    save_importance_plot(importance, output_paths["summary_bar"])


def save_importance_plot(importance: pd.DataFrame, output_path: Path) -> None:
    """
    Keep SHAP summary_bar as the canonical bar plot.

    This helper intentionally does not overwrite the file; it exists so future
    extensions can add a local Matplotlib fallback without changing callers.
    """
    if not output_path.exists():
        raise FileNotFoundError(f"SHAP summary bar plot gagal dibuat: {output_path}")


def build_runtime_summary(*results: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "model_key": result["model_key"],
                "model_label": result["model_label"],
                "model_name": result["model_name"],
                "feature_set": result["feature_set"],
                "model_load_time_seconds": result.get("model_load_time_seconds", 0.0),
                "component_compute_train_val_seconds": result.get(
                    "component_compute_train_val_seconds",
                    0.0,
                ),
                "component_compute_final_test_seconds": result.get(
                    "component_compute_final_test_seconds",
                    0.0,
                ),
                "shap_compute_time_seconds": result.get(
                    "shap_compute_time_seconds",
                    0.0,
                ),
                "plot_time_seconds": result.get("plot_time_seconds", 0.0),
                "total_runtime_seconds": result.get("total_runtime_seconds", 0.0),
                "sample_rows": result.get("sample_rows", ""),
            }
        )
    total_row = {
        "model_key": "total",
        "model_label": "Total",
        "model_name": "all_models",
        "feature_set": "interpretability",
        "model_load_time_seconds": sum(
            float(row["model_load_time_seconds"]) for row in rows
        ),
        "component_compute_train_val_seconds": sum(
            float(row["component_compute_train_val_seconds"]) for row in rows
        ),
        "component_compute_final_test_seconds": sum(
            float(row["component_compute_final_test_seconds"]) for row in rows
        ),
        "shap_compute_time_seconds": sum(
            float(row["shap_compute_time_seconds"]) for row in rows
        ),
        "plot_time_seconds": sum(float(row["plot_time_seconds"]) for row in rows),
        "total_runtime_seconds": sum(float(row["total_runtime_seconds"]) for row in rows),
        "sample_rows": sum(
            int(row["sample_rows"]) for row in rows if str(row["sample_rows"]).strip()
        ),
    }
    return pd.DataFrame([*rows, total_row])


def render_interpretability_report(
    *,
    metadata: Mapping[str, Any],
    runtime_summary: pd.DataFrame,
    prophet_result: Mapping[str, Any],
    xgb_basic_result: Mapping[str, Any],
    xgb_advanced_result: Mapping[str, Any],
) -> str:
    final_ranking = load_optional_final_ranking()
    basic_importance = pd.DataFrame(xgb_basic_result["top_features"])
    advanced_importance = pd.DataFrame(xgb_advanced_result["top_features"])
    basic_groups = pd.DataFrame(xgb_basic_result["group_importance"])
    advanced_groups = pd.DataFrame(xgb_advanced_result["group_importance"])

    best_model_sentence = ""
    if final_ranking is not None and not final_ranking.empty:
        winner = str(final_ranking.sort_values("mae", kind="mergesort").iloc[0]["model_label"])
        winner_mae = float(final_ranking.sort_values("mae", kind="mergesort").iloc[0]["mae"])
        best_model_sentence = (
            f"Pada comparative evaluation, model terbaik final test berdasarkan MAE "
            f"adalah {winner} (MAE {winner_mae:.6f}). "
        )

    rolling_share = select_group_share(advanced_groups, ["rolling_mean", "rolling_std"])
    lag_share_advanced = select_group_share(advanced_groups, ["lag"])
    lag_share_basic = select_group_share(basic_groups, ["lag"])

    lines = [
        "# Model Interpretability",
        "",
        f"Run UTC: {metadata['started_at_utc']}",
        "",
        "## Scope",
        "",
        (
            "Tahap 15b ini bersifat post-hoc interpretability. Script hanya membaca "
            "model retrained tahap 13 dan feature matrix train_val. Tidak ada tuning "
            "ulang, retraining ulang, perubahan final predictions, perubahan final "
            "metrics, atau perubahan ranking model."
        ),
        "",
        "## Leakage Guardrail",
        "",
        (
            "SHAP untuk XGBoost dihitung pada feature matrix train_val, bukan pada "
            "precomputed feature matrix final test. Final-test SHAP recursive belum "
            "dihitung pada tahap ini. Prophet final-test component view, jika ada, "
            "hanya memakai timestamp sebagai post-hoc inspection dan tidak memakai "
            "label aktual."
        ),
        "",
        "## Prophet Components",
        "",
        (
            "Komponen Prophet yang valid untuk konfigurasi saat ini adalah trend, "
            "daily seasonality, dan weekly seasonality. Yearly seasonality tidak "
            "diaktifkan (`yearly_seasonality=False`), dan monthly seasonality tidak "
            "ditambahkan sebagai custom seasonality sehingga tidak ada klaim learned "
            "monthly component."
        ),
        "",
        "Ringkasan Prophet:",
        "",
        dataframe_to_markdown(
            pd.DataFrame(
                [
                    {
                        "component_rows_train_val": prophet_result[
                            "train_val_component_rows"
                        ],
                        "component_rows_final_test": prophet_result[
                            "final_test_component_rows"
                        ],
                        "components_present": ", ".join(
                            prophet_result["components_present"]
                        ),
                        "components_absent": ", ".join(
                            prophet_result["components_absent"]
                        ),
                    }
                ]
            ),
            float_digits=6,
        ),
        "",
        "## XGBoost-Basic SHAP",
        "",
        (
            f"SHAP XGBoost-Basic memakai {xgb_basic_result['sample_rows']} sample "
            f"dari {xgb_basic_result['feature_rows_total']} feature rows train_val "
            f"dengan strategi {xgb_basic_result['sample_strategy']}. "
            f"Kontribusi grup lag mencakup {lag_share_basic * 100:.2f}% dari total "
            "mean absolute SHAP."
        ),
        "",
        dataframe_to_markdown(
            basic_importance[
                [
                    "rank",
                    "feature",
                    "feature_group",
                    "mean_abs_shap",
                    "share_of_total_mean_abs_shap",
                ]
            ],
            float_digits=6,
        ),
        "",
        "## XGBoost-Advanced SHAP",
        "",
        (
            f"SHAP XGBoost-Advanced memakai {xgb_advanced_result['sample_rows']} "
            f"sample dari {xgb_advanced_result['feature_rows_total']} feature rows "
            f"train_val dengan strategi {xgb_advanced_result['sample_strategy']}. "
            f"Kontribusi grup lag adalah {lag_share_advanced * 100:.2f}%, sedangkan "
            f"rolling statistics adalah {rolling_share * 100:.2f}% dari total mean "
            "absolute SHAP."
        ),
        "",
        dataframe_to_markdown(
            advanced_importance[
                [
                    "rank",
                    "feature",
                    "feature_group",
                    "mean_abs_shap",
                    "share_of_total_mean_abs_shap",
                ]
            ],
            float_digits=6,
        ),
        "",
        "## Link to Comparative Evaluation",
        "",
        (
            best_model_sentence
            + "Hasil SHAP membantu menjelaskan mengapa XGBoost-Basic dapat tetap "
            "robust walaupun fiturnya lebih sederhana: sinyal dominan berada pada "
            "lag utama, terutama lag jangka pendek dan pola harian/mingguan. Pada "
            "advanced model, tambahan rolling statistics memang dipakai, tetapi "
            "kontribusinya perlu dibaca sebagai kompleksitas tambahan yang tidak "
            "otomatis meningkatkan generalisasi final test."
        ),
        "",
        "## Time Cost Computing",
        "",
        dataframe_to_markdown(runtime_summary, float_digits=6),
        "",
        "## Output Files",
        "",
        f"- Prophet components train_val: `{prophet_result['outputs']['train_val_components']}`",
        f"- Prophet components final-test view: `{prophet_result['outputs']['final_test_components']}`",
        f"- XGBoost-Basic SHAP importance: `{xgb_basic_result['outputs']['importance']}`",
        f"- XGBoost-Advanced SHAP importance: `{xgb_advanced_result['outputs']['importance']}`",
        f"- Runtime summary: `{metadata['outputs']['runtime_summary']}`",
        f"- Report mirror: `{metadata['outputs']['report']}`",
    ]
    return "\n".join(lines) + "\n"


def load_optional_final_ranking() -> Optional[pd.DataFrame]:
    path = OUTPUT_DIR / "experiments" / "comparative_evaluation" / "metrics" / "final_metric_ranking.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def select_group_share(group_summary: pd.DataFrame, groups: Sequence[str]) -> float:
    if group_summary.empty:
        return 0.0
    selected = group_summary[group_summary["feature_group"].isin(groups)]
    if selected.empty:
        return 0.0
    return float(selected["share_of_total_mean_abs_shap"].sum())


def summarize_timestamp_column(frame: pd.DataFrame) -> dict[str, Any]:
    timestamps = pd.to_datetime(frame[TIMESTAMP_COL], utc=True, errors="raise")
    return {
        "n_rows": int(frame.shape[0]),
        "utc_start": timestamps.min().isoformat(),
        "utc_end": timestamps.max().isoformat(),
    }


def summarize_index(index: pd.DatetimeIndex) -> dict[str, Any]:
    return {
        "n_rows": int(len(index)),
        "utc_start": index.min().isoformat(),
        "utc_end": index.max().isoformat(),
    }


def to_jsonable_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


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
            "| " + " | ".join(format_cell(row[column]) for column in df.columns) + " |"
        )
    return "\n".join(lines)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON harus berupa object: {path}")
    return payload


def stringify_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    return {key: str(value) for key, value in paths.items()}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

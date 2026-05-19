"""
Script tahap 13: Retraining Best Configurations.

Scope:
- Load best params hasil tuning untuk Prophet, XGBoost-Basic, dan
  XGBoost-Advanced.
- Latih ulang model pada seluruh data train_val.
- Simpan model artifact, metadata, best params yang dipakai, dan time cost
  computing.

Catatan leakage:
- Script ini tidak membaca final_test.
- XGBoost dilatih hanya pada feature matrix train_val yang sudah dibuat dari
  train_val.
- Tidak ada tuning tambahan dan tidak ada pemilihan parameter baru.

Contoh:
    python -m src.experiments.retrain_best_models --dry-run
    python -m src.experiments.retrain_best_models
    python -m src.experiments.retrain_best_models --models xgb_basic xgb_advanced
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
    PROPHET_OUTPUT_DIR,
    TARGET_COL,
    TRAIN_VAL_PATH,
    XGB_ADVANCED_FEATURES_PATH,
    XGB_ADVANCED_OUTPUT_DIR,
    XGB_BASIC_FEATURES_PATH,
    XGB_BASIC_OUTPUT_DIR,
    ensure_dirs,
)
from src.experiments.tuning_utils import load_xgb_feature_matrix
from src.features import ADVANCED_FEATURE_SET, BASIC_FEATURE_SET, get_feature_columns
from src.models.prophet_model import PROPHET_FEATURE_SET, fit_prophet_model
from src.models.xgboost_model import fit_xgb_model
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


EXPERIMENT_NAME = "retrain_best_models"
OUTPUT_DIR = EXPERIMENTS_DIR / "retraining"
ALL_MODEL_KEYS = ("prophet", "xgb_basic", "xgb_advanced")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain best Prophet and XGBoost configurations on train_val."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[*ALL_MODEL_KEYS, "all"],
        default=["all"],
        help="Model yang akan dilatih ulang. Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validasi input dan rencana output tanpa training atau menulis artifact.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    metadata = run_retraining_best_models(
        models=args.models,
        dry_run=args.dry_run,
    )

    if metadata["dry_run"]:
        print("Dry run retraining selesai.")
        print(f"Models planned: {', '.join(metadata['models_requested'])}")
        print(f"Train rows: {metadata['train_val_summary']['n_rows']}")
        print(f"Output dir planned: {metadata['outputs']['base']}")
        return

    print("Retraining best configurations selesai.")
    print(f"Models retrained: {', '.join(metadata['models_requested'])}")
    print(f"Output dir: {metadata['outputs']['base']}")
    print(f"Summary: {metadata['outputs']['summary']}")


def run_retraining_best_models(
    *,
    models: Optional[Sequence[str]] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    selected_models = normalize_model_selection(models or ["all"])
    outputs = retraining_output_paths()
    timer = start_timer()
    started_at = utc_now_iso()
    status = "success"
    error_message = ""

    try:
        ensure_dirs()
        ensure_retraining_dirs(outputs)
        train_val = load_split_timeseries(TRAIN_VAL_PATH)
        train_val_summary = summarize_timeseries(train_val)
        plans = build_model_plans(selected_models, train_val, outputs)

        if dry_run:
            return {
                "experiment_name": EXPERIMENT_NAME,
                "status": status,
                "error_message": error_message,
                "dry_run": True,
                "started_at_utc": started_at,
                "finished_at_utc": utc_now_iso(),
                "total_runtime_seconds": elapsed_seconds(timer),
                "models_requested": selected_models,
                "train_val_path": str(TRAIN_VAL_PATH),
                "train_val_summary": train_val_summary,
                "model_plans": plans,
                "methodology_note": methodology_notes(),
                "outputs": stringify_paths(outputs),
            }

        results: list[dict[str, Any]] = []
        for plan in plans:
            model_key = str(plan["model_key"])
            try:
                if model_key == "prophet":
                    result = retrain_prophet(train_val, plan)
                else:
                    result = retrain_xgb(train_val, plan)
                results.append(result)
            except Exception as exc:
                log_model_failure(plan, train_val, str(exc))
                raise

        total_runtime_seconds = elapsed_seconds(timer)
        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "status": status,
            "error_message": error_message,
            "dry_run": False,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "total_runtime_seconds": total_runtime_seconds,
            "models_requested": selected_models,
            "train_val_path": str(TRAIN_VAL_PATH),
            "train_val_summary": train_val_summary,
            "methodology_note": methodology_notes(),
            "model_results": results,
            "outputs": stringify_paths(outputs),
        }

        save_retraining_outputs(metadata, outputs)
        log_runtime(
            make_runtime_record(
                experiment_name=EXPERIMENT_NAME,
                model_name="all_requested_models",
                feature_set="mixed",
                n_train_rows=train_val.shape[0],
                n_prediction_rows=0,
                train_time_seconds=sum(
                    float(result["train_time_seconds"]) for result in results
                ),
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
            "dry_run": bool(dry_run),
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "total_runtime_seconds": total_runtime_seconds,
            "models_requested": selected_models,
            "outputs": stringify_paths(outputs),
        }
        if not dry_run:
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
        raise ValueError("Gunakan --models all atau daftar model spesifik, bukan keduanya.")
    if normalized == ["all"]:
        return list(ALL_MODEL_KEYS)

    invalid = sorted(set(normalized).difference(ALL_MODEL_KEYS))
    if invalid:
        raise ValueError(f"Model tidak dikenal: {invalid}")
    return list(dict.fromkeys(normalized))


def retraining_output_paths() -> dict[str, Path]:
    base = OUTPUT_DIR
    return {
        "base": base,
        "models": base / "models",
        "metadata_dir": base / "metadata",
        "summaries": base / "summaries",
        "training_summary": base / "training_summary.csv",
        "runtime_summary": base / "runtime_summary.csv",
        "best_params_used": base / "best_params_used.json",
        "model_registry": base / "model_registry.json",
        "metadata": base / "experiment_metadata.json",
        "summary": base / "summaries" / "retraining_summary.md",
    }


def ensure_retraining_dirs(paths: Mapping[str, Path]) -> None:
    for key in ["base", "models", "metadata_dir", "summaries"]:
        paths[key].mkdir(parents=True, exist_ok=True)


def build_model_plans(
    selected_models: Sequence[str],
    train_val: pd.DataFrame,
    outputs: Mapping[str, Path],
) -> list[dict[str, Any]]:
    specs = model_specs(outputs)
    plans: list[dict[str, Any]] = []

    for model_key in selected_models:
        spec = specs[model_key]
        best_payload = load_best_params(spec["best_params_path"], model_key=model_key)
        plan: dict[str, Any] = {
            **spec,
            "best_params": best_payload,
            "parameter_set_id": best_payload["parameter_set_id"],
            "params": dict(best_payload["params"]),
            "best_cv_metrics": dict(best_payload.get("metrics", {})),
            "train_val_rows": int(train_val.shape[0]),
            "train_start": train_val.index.min().isoformat(),
            "train_end": train_val.index.max().isoformat(),
            "final_test_used": False,
        }

        if model_key == "prophet":
            validate_prophet_plan(train_val, plan)
            plan["training_rows"] = int(train_val.shape[0])
        else:
            feature_matrix = load_xgb_feature_matrix(
                spec["feature_path"],
                spec["feature_set"],
            )
            validate_xgb_plan(train_val, feature_matrix, plan)
            plan["training_rows"] = int(feature_matrix.shape[0])
            plan["feature_columns"] = get_feature_columns(spec["feature_set"])
            plan["feature_rows_dropped_due_to_history"] = int(
                train_val.shape[0] - feature_matrix.shape[0]
            )

        plans.append(plan)

    return plans


def model_specs(outputs: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        "prophet": {
            "model_key": "prophet",
            "model_label": "Prophet",
            "model_name": "prophet",
            "feature_set": PROPHET_FEATURE_SET,
            "best_params_path": PROPHET_OUTPUT_DIR / "best_params.json",
            "model_path": outputs["models"] / "prophet_retrained.json",
            "metadata_path": outputs["metadata_dir"] / "prophet_metadata.json",
        },
        "xgb_basic": {
            "model_key": "xgb_basic",
            "model_label": "XGBoost-Basic",
            "model_name": "xgboost",
            "feature_set": BASIC_FEATURE_SET,
            "best_params_path": XGB_BASIC_OUTPUT_DIR / "best_params.json",
            "feature_path": XGB_BASIC_FEATURES_PATH,
            "model_path": outputs["models"] / "xgb_basic_retrained.json",
            "metadata_path": outputs["metadata_dir"] / "xgb_basic_metadata.json",
        },
        "xgb_advanced": {
            "model_key": "xgb_advanced",
            "model_label": "XGBoost-Advanced",
            "model_name": "xgboost",
            "feature_set": ADVANCED_FEATURE_SET,
            "best_params_path": XGB_ADVANCED_OUTPUT_DIR / "best_params.json",
            "feature_path": XGB_ADVANCED_FEATURES_PATH,
            "model_path": outputs["models"] / "xgb_advanced_retrained.json",
            "metadata_path": outputs["metadata_dir"] / "xgb_advanced_metadata.json",
        },
    }


def load_best_params(path: Path, *, model_key: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Best params untuk {model_key} tidak ditemukan: {path}. "
            "Jalankan tahap tuning terlebih dahulu."
        )

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Best params harus JSON object: {path}")

    required = {"parameter_set_id", "params"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Best params {model_key} tidak lengkap: {missing}")
    if not isinstance(payload["params"], dict):
        raise ValueError(f"Field params {model_key} harus object.")
    if not str(payload["parameter_set_id"]).strip():
        raise ValueError(f"parameter_set_id {model_key} kosong.")

    payload["parameter_set_id"] = str(payload["parameter_set_id"])
    return payload


def validate_prophet_plan(train_val: pd.DataFrame, plan: Mapping[str, Any]) -> None:
    if train_val.empty:
        raise ValueError("train_val kosong untuk retraining Prophet.")
    if TARGET_COL not in train_val.columns:
        raise ValueError(f"Kolom target hilang untuk Prophet: {TARGET_COL}")
    target = pd.to_numeric(train_val[TARGET_COL], errors="raise")
    if target.isna().any() or not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("Target train_val Prophet mengandung nilai invalid.")
    if str(plan["feature_set"]) != PROPHET_FEATURE_SET:
        raise ValueError("Feature set Prophet tidak sesuai.")


def validate_xgb_plan(
    train_val: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    plan: Mapping[str, Any],
) -> None:
    if feature_matrix.empty:
        raise ValueError(f"Feature matrix kosong untuk {plan['model_key']}.")
    if feature_matrix.index.difference(train_val.index).size > 0:
        raise ValueError(
            f"Feature matrix {plan['model_key']} memiliki timestamp di luar train_val."
        )
    if feature_matrix.index.max() != train_val.index.max():
        raise ValueError(
            f"Feature matrix {plan['model_key']} tidak memakai akhir train_val penuh."
        )
    if feature_matrix.index.min() < train_val.index.min():
        raise ValueError(
            f"Feature matrix {plan['model_key']} dimulai sebelum train_val."
        )

    feature_columns = get_feature_columns(str(plan["feature_set"]))
    missing = sorted({TARGET_COL, *feature_columns}.difference(feature_matrix.columns))
    if missing:
        raise ValueError(f"Kolom training {plan['model_key']} hilang: {missing}")


def retrain_prophet(train_val: pd.DataFrame, plan: Mapping[str, Any]) -> dict[str, Any]:
    model_timer = start_timer()
    train_timer = start_timer()
    model = fit_prophet_model(train_val, params=plan["params"])
    train_time_seconds = elapsed_seconds(train_timer)
    save_prophet_model(model, Path(plan["model_path"]))
    total_runtime_seconds = elapsed_seconds(model_timer)

    result = model_result_payload(
        plan,
        train_rows=train_val.shape[0],
        train_time_seconds=train_time_seconds,
        total_runtime_seconds=total_runtime_seconds,
        extra={
            "artifact_format": "prophet.serialize.model_to_json",
            "feature_columns": [],
        },
    )
    save_experiment_metadata(result, plan["metadata_path"])
    log_model_success(result)
    append_experiment_run(result)
    return result


def retrain_xgb(train_val: pd.DataFrame, plan: Mapping[str, Any]) -> dict[str, Any]:
    feature_matrix = load_xgb_feature_matrix(plan["feature_path"], plan["feature_set"])
    validate_xgb_plan(train_val, feature_matrix, plan)
    feature_columns = get_feature_columns(str(plan["feature_set"]))

    model_timer = start_timer()
    train_timer = start_timer()
    model = fit_xgb_model(
        feature_matrix,
        feature_columns=feature_columns,
        params=plan["params"],
    )
    train_time_seconds = elapsed_seconds(train_timer)
    Path(plan["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(plan["model_path"]))
    total_runtime_seconds = elapsed_seconds(model_timer)

    result = model_result_payload(
        plan,
        train_rows=feature_matrix.shape[0],
        train_time_seconds=train_time_seconds,
        total_runtime_seconds=total_runtime_seconds,
        extra={
            "artifact_format": "xgboost_json",
            "feature_path": str(plan["feature_path"]),
            "feature_columns": feature_columns,
            "source_train_val_rows": int(train_val.shape[0]),
            "feature_rows_dropped_due_to_history": int(
                train_val.shape[0] - feature_matrix.shape[0]
            ),
        },
    )
    save_experiment_metadata(result, plan["metadata_path"])
    log_model_success(result)
    append_experiment_run(result)
    return result


def save_prophet_model(model: Any, output_path: Path) -> None:
    try:
        from prophet.serialize import model_to_json
    except ImportError as exc:
        raise ImportError(
            "prophet.serialize.model_to_json tidak tersedia. "
            "Pastikan package prophet sudah terinstall dari requirements.txt."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(model_to_json(model), encoding="utf-8")


def model_result_payload(
    plan: Mapping[str, Any],
    *,
    train_rows: int,
    train_time_seconds: float,
    total_runtime_seconds: float,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment_name": EXPERIMENT_NAME,
        "stage": "retraining_best_configurations",
        "status": "success",
        "model_key": plan["model_key"],
        "model_label": plan["model_label"],
        "model_name": plan["model_name"],
        "feature_set": plan["feature_set"],
        "parameter_set_id": plan["parameter_set_id"],
        "params": dict(plan["params"]),
        "best_cv_metrics": dict(plan.get("best_cv_metrics", {})),
        "train_val_path": str(TRAIN_VAL_PATH),
        "best_params_path": str(plan["best_params_path"]),
        "model_path": str(plan["model_path"]),
        "metadata_path": str(plan["metadata_path"]),
        "train_start": plan["train_start"],
        "train_end": plan["train_end"],
        "n_train_rows": int(train_rows),
        "forecast_horizon_hours": int(FORECAST_HORIZON),
        "train_time_seconds": float(train_time_seconds),
        "prediction_time_seconds": 0.0,
        "total_runtime_seconds": float(total_runtime_seconds),
        "final_test_used": False,
        "tuning_added": False,
    }
    if extra:
        payload.update(dict(extra))
    return payload


def log_model_success(result: Mapping[str, Any]) -> None:
    log_runtime(
        make_runtime_record(
            experiment_name=EXPERIMENT_NAME,
            model_name=result["model_name"],
            feature_set=result["feature_set"],
            parameter_set_id=result["parameter_set_id"],
            train_start=result["train_start"],
            train_end=result["train_end"],
            n_train_rows=result["n_train_rows"],
            n_prediction_rows=0,
            train_time_seconds=result["train_time_seconds"],
            prediction_time_seconds=0.0,
            total_runtime_seconds=result["total_runtime_seconds"],
            status="success",
        )
    )


def log_model_failure(
    plan: Mapping[str, Any],
    train_val: pd.DataFrame,
    error_message: str,
) -> None:
    failure = {
        "experiment_name": EXPERIMENT_NAME,
        "stage": "retraining_best_configurations",
        "status": "failed",
        "error_message": error_message,
        "model_key": plan["model_key"],
        "model_label": plan["model_label"],
        "model_name": plan["model_name"],
        "feature_set": plan["feature_set"],
        "parameter_set_id": plan["parameter_set_id"],
        "train_start": train_val.index.min().isoformat(),
        "train_end": train_val.index.max().isoformat(),
        "n_train_rows": int(train_val.shape[0]),
    }
    log_runtime(
        make_runtime_record(
            experiment_name=EXPERIMENT_NAME,
            model_name=plan["model_name"],
            feature_set=plan["feature_set"],
            parameter_set_id=plan["parameter_set_id"],
            train_start=failure["train_start"],
            train_end=failure["train_end"],
            n_train_rows=failure["n_train_rows"],
            n_prediction_rows=0,
            status="failed",
            error_message=error_message,
        )
    )
    append_experiment_run(failure)


def save_retraining_outputs(
    metadata: Mapping[str, Any],
    outputs: Mapping[str, Path],
) -> None:
    results = list(metadata["model_results"])
    training_summary = pd.DataFrame(
        [
            {
                "model_key": result["model_key"],
                "model_label": result["model_label"],
                "feature_set": result["feature_set"],
                "parameter_set_id": result["parameter_set_id"],
                "n_train_rows": result["n_train_rows"],
                "train_start": result["train_start"],
                "train_end": result["train_end"],
                "train_time_seconds": result["train_time_seconds"],
                "prediction_time_seconds": result["prediction_time_seconds"],
                "total_runtime_seconds": result["total_runtime_seconds"],
                "model_path": result["model_path"],
                "metadata_path": result["metadata_path"],
            }
            for result in results
        ]
    )
    training_summary.to_csv(outputs["training_summary"], index=False)

    runtime_summary = training_summary[
        [
            "model_key",
            "model_label",
            "feature_set",
            "train_time_seconds",
            "prediction_time_seconds",
            "total_runtime_seconds",
        ]
    ].copy()
    runtime_summary.loc["total"] = {
        "model_key": "total",
        "model_label": "Total",
        "feature_set": "mixed",
        "train_time_seconds": float(training_summary["train_time_seconds"].sum()),
        "prediction_time_seconds": 0.0,
        "total_runtime_seconds": float(training_summary["total_runtime_seconds"].sum()),
    }
    runtime_summary.to_csv(outputs["runtime_summary"], index=False)

    save_experiment_metadata(
        {
            result["model_key"]: {
                "parameter_set_id": result["parameter_set_id"],
                "params": result["params"],
                "best_cv_metrics": result["best_cv_metrics"],
                "source_best_params_path": result["best_params_path"],
            }
            for result in results
        },
        outputs["best_params_used"],
    )
    save_experiment_metadata(
        {
            result["model_key"]: {
                "model_label": result["model_label"],
                "model_name": result["model_name"],
                "feature_set": result["feature_set"],
                "model_path": result["model_path"],
                "metadata_path": result["metadata_path"],
                "artifact_format": result["artifact_format"],
                "feature_columns": result.get("feature_columns", []),
            }
            for result in results
        },
        outputs["model_registry"],
    )
    save_experiment_metadata(metadata, outputs["metadata"])
    outputs["summary"].write_text(render_retraining_summary(metadata), encoding="utf-8")


def render_retraining_summary(metadata: Mapping[str, Any]) -> str:
    lines = [
        "# Retraining Best Configurations",
        "",
        f"Run UTC: {metadata['started_at_utc']}",
        "",
        "## Scope",
        "",
        (
            "Model dilatih ulang memakai seluruh `train_val.csv` dan best params "
            "hasil tuning. Script ini tidak membaca `final_test.csv`, tidak "
            "melakukan tuning tambahan, dan tidak memilih ulang parameter."
        ),
        "",
        "## Training Summary",
        "",
        "| Model | Feature set | Parameter set | Train rows | Train time seconds | Model path |",
        "|---|---|---:|---:|---:|---|",
    ]
    for result in metadata["model_results"]:
        lines.append(
            "| {model} | {feature_set} | {param_id} | {rows} | {train_time:.6f} | `{path}` |".format(
                model=result["model_label"],
                feature_set=result["feature_set"],
                param_id=result["parameter_set_id"],
                rows=result["n_train_rows"],
                train_time=float(result["train_time_seconds"]),
                path=result["model_path"],
            )
        )

    lines.extend(
        [
            "",
            "## Time Cost Computing",
            "",
            f"- Total retraining runtime seconds: {metadata['total_runtime_seconds']:.6f}",
            "- Prediction time seconds: 0.000000 karena tahap ini hanya training.",
            "",
            "## Output Files",
            "",
            f"- Training summary: `{metadata['outputs']['training_summary']}`",
            f"- Runtime summary: `{metadata['outputs']['runtime_summary']}`",
            f"- Best params used: `{metadata['outputs']['best_params_used']}`",
            f"- Model registry: `{metadata['outputs']['model_registry']}`",
            f"- Experiment metadata: `{metadata['outputs']['metadata']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize_timeseries(df: pd.DataFrame) -> dict[str, Any]:
    target = pd.to_numeric(df[TARGET_COL], errors="raise")
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "utc_start": df.index.min().isoformat(),
        "utc_end": df.index.max().isoformat(),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "target_mean": float(target.mean()),
    }


def methodology_notes() -> list[str]:
    return [
        "Best params berasal dari output tuning/CV, bukan dari final test.",
        "Retraining memakai seluruh train_val sebelum final testing.",
        "Final test tidak dibaca atau digunakan oleh script retraining.",
        "XGBoost memakai feature matrix train_val hanya untuk training rows.",
        "Tidak ada tuning tambahan pada tahap retraining.",
    ]


def stringify_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    return {key: str(value) for key, value in paths.items()}


if __name__ == "__main__":
    main()

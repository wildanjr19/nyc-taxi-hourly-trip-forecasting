"""
Script Revisi 1.6: Retraining Prophet-Regressor-Basic.

Scope:
- Load best params Prophet-Regressor-Basic dari tuning revisi.
- Latih ulang model pada seluruh train_val.
- Gunakan hanya rows training yang memiliki regressor lengkap setelah history
  awal 168 jam.
- Simpan model, metadata, summary, dan time cost computing.

Catatan leakage:
- Script ini tidak membaca final_test.
- Lag regressor untuk training dibuat dari actual train_val masa lalu saja.
- Tidak ada tuning tambahan dan tidak ada pemilihan parameter baru.

Contoh:
    python -m src.experiments.retrain_prophet_regressor_basic --dry-run
    python -m src.experiments.retrain_prophet_regressor_basic
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
    MODELING_TZ,
    PROPHET_REGRESSOR_BASIC_OUTPUT_DIR,
    TARGET_COL,
    TRAIN_VAL_PATH,
    ensure_dirs,
)
from src.models.prophet_model import (
    PROPHET_BASIC_REGRESSORS,
    PROPHET_REGRESSOR_BASIC_FEATURE_SET,
    make_prophet_regressor_frame,
    make_prophet_regressor_model,
)
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


EXPERIMENT_NAME = "retrain_prophet_regressor_basic"
MODEL_KEY = "prophet_regressor_basic"
MODEL_LABEL = "Prophet-Regressor-Basic"
OUTPUT_DIR = EXPERIMENTS_DIR / "retraining"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain Prophet-Regressor-Basic on full train_val."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validasi input dan rencana output tanpa training atau menulis artifact.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    metadata = run_retraining_prophet_regressor_basic(dry_run=args.dry_run)

    if metadata["dry_run"]:
        print("Dry run retraining Prophet-Regressor-Basic selesai.")
        print(f"Training rows after history drop: {metadata['training_frame']['n_rows']}")
        print(f"Output model planned: {metadata['outputs']['model']}")
        return

    print("Retraining Prophet-Regressor-Basic selesai.")
    print(f"Parameter set: {metadata['parameter_set_id']}")
    print(f"Training rows: {metadata['training_frame']['n_rows']}")
    print(f"Model artifact: {metadata['outputs']['model']}")
    print(f"Summary: {metadata['outputs']['summary']}")


def run_retraining_prophet_regressor_basic(*, dry_run: bool = False) -> dict[str, Any]:
    outputs = output_paths()
    timer = start_timer()
    started_at = utc_now_iso()
    status = "success"
    error_message = ""

    try:
        ensure_dirs()
        ensure_output_dirs(outputs)

        best_params = load_best_params(
            PROPHET_REGRESSOR_BASIC_OUTPUT_DIR / "best_params.json"
        )
        train_val = load_split_timeseries(TRAIN_VAL_PATH)
        training_frame = make_prophet_regressor_frame(
            train_val,
            target_col=TARGET_COL,
            include_y=True,
            drop_na=True,
        )
        validate_training_frame(training_frame, train_val)

        frame_summary = summarize_training_frame(training_frame, train_val)
        base_metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "stage": "revisi_1_6_retraining",
            "status": status,
            "error_message": error_message,
            "dry_run": bool(dry_run),
            "started_at_utc": started_at,
            "model_key": MODEL_KEY,
            "model_label": MODEL_LABEL,
            "model_name": MODEL_KEY,
            "feature_set": PROPHET_REGRESSOR_BASIC_FEATURE_SET,
            "regressors": list(PROPHET_BASIC_REGRESSORS),
            "parameter_set_id": best_params["parameter_set_id"],
            "params": dict(best_params["params"]),
            "best_cv_metrics": dict(best_params.get("metrics", {})),
            "train_val_path": str(TRAIN_VAL_PATH),
            "best_params_path": str(
                PROPHET_REGRESSOR_BASIC_OUTPUT_DIR / "best_params.json"
            ),
            "train_val": summarize_train_val(train_val),
            "training_frame": frame_summary,
            "forecast_horizon_hours": int(FORECAST_HORIZON),
            "final_test_used": False,
            "tuning_added": False,
            "methodology_note": methodology_notes(),
            "outputs": stringify_paths(outputs),
        }

        if dry_run:
            return {
                **base_metadata,
                "finished_at_utc": utc_now_iso(),
                "train_time_seconds": 0.0,
                "prediction_time_seconds": 0.0,
                "total_runtime_seconds": elapsed_seconds(timer),
            }

        train_timer = start_timer()
        model = make_prophet_regressor_model(
            best_params["params"],
            regressors=PROPHET_BASIC_REGRESSORS,
        )
        model.fit(training_frame)
        train_time_seconds = elapsed_seconds(train_timer)
        save_prophet_model(model, outputs["model"])
        total_runtime_seconds = elapsed_seconds(timer)

        metadata = {
            **base_metadata,
            "finished_at_utc": utc_now_iso(),
            "train_time_seconds": train_time_seconds,
            "prediction_time_seconds": 0.0,
            "total_runtime_seconds": total_runtime_seconds,
            "artifact_format": "prophet.serialize.model_to_json",
            "model_path": str(outputs["model"]),
        }

        save_outputs(metadata, outputs)
        log_success(metadata)
        append_experiment_run(metadata)
        return metadata
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        total_runtime_seconds = elapsed_seconds(timer)
        failure_metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "stage": "revisi_1_6_retraining",
            "status": status,
            "error_message": error_message,
            "dry_run": bool(dry_run),
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "total_runtime_seconds": total_runtime_seconds,
            "model_key": MODEL_KEY,
            "model_label": MODEL_LABEL,
            "model_name": MODEL_KEY,
            "feature_set": PROPHET_REGRESSOR_BASIC_FEATURE_SET,
            "outputs": stringify_paths(outputs),
        }
        if not dry_run:
            save_experiment_metadata(failure_metadata, outputs["metadata"])
        log_runtime(
            make_runtime_record(
                experiment_name=EXPERIMENT_NAME,
                model_name=MODEL_KEY,
                feature_set=PROPHET_REGRESSOR_BASIC_FEATURE_SET,
                total_runtime_seconds=total_runtime_seconds,
                status=status,
                error_message=error_message,
            )
        )
        append_experiment_run(failure_metadata)
        raise


def output_paths() -> dict[str, Path]:
    return {
        "base": OUTPUT_DIR,
        "models": OUTPUT_DIR / "models",
        "metadata_dir": OUTPUT_DIR / "metadata",
        "summaries": OUTPUT_DIR / "summaries",
        "model": OUTPUT_DIR / "models" / "prophet_regressor_basic_retrained.json",
        "metadata": OUTPUT_DIR / "metadata" / "prophet_regressor_basic_metadata.json",
        "training_summary": OUTPUT_DIR / "prophet_regressor_basic_training_summary.csv",
        "runtime_summary": OUTPUT_DIR / "prophet_regressor_basic_runtime_summary.csv",
        "best_params_used": OUTPUT_DIR / "prophet_regressor_basic_best_params_used.json",
        "summary": (
            OUTPUT_DIR
            / "summaries"
            / "prophet_regressor_basic_retraining_summary.md"
        ),
    }


def ensure_output_dirs(paths: Mapping[str, Path]) -> None:
    for key in ["base", "models", "metadata_dir", "summaries"]:
        paths[key].mkdir(parents=True, exist_ok=True)


def load_best_params(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            "Best params Prophet-Regressor-Basic tidak ditemukan: "
            f"{path}. Jalankan tuning revisi terlebih dahulu."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Best params harus JSON object: {path}")

    required = {"parameter_set_id", "params"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Best params Prophet-Regressor-Basic tidak lengkap: {missing}")
    if not isinstance(payload["params"], dict):
        raise ValueError("Field params Prophet-Regressor-Basic harus object.")
    if not str(payload["parameter_set_id"]).strip():
        raise ValueError("parameter_set_id Prophet-Regressor-Basic kosong.")

    payload["parameter_set_id"] = str(payload["parameter_set_id"])
    return payload


def validate_training_frame(
    training_frame: pd.DataFrame,
    train_val: pd.DataFrame,
) -> None:
    required_columns = {"ds", "y", *PROPHET_BASIC_REGRESSORS}
    missing = sorted(required_columns.difference(training_frame.columns))
    if missing:
        raise ValueError(f"Kolom training Prophet-Regressor-Basic hilang: {missing}")
    if training_frame.empty:
        raise ValueError("Training frame Prophet-Regressor-Basic kosong.")
    if training_frame.shape[0] >= train_val.shape[0]:
        raise ValueError(
            "Training frame Prophet-Regressor-Basic seharusnya drop history awal."
        )

    numeric_columns = ["y", *PROPHET_BASIC_REGRESSORS]
    numeric = training_frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    if numeric.isna().any().any():
        raise ValueError("Training frame mengandung missing numeric values.")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Training frame mengandung non-finite numeric values.")

    ds = pd.to_datetime(training_frame["ds"], errors="raise")
    if ds.dt.tz is not None:
        raise ValueError("Kolom ds Prophet harus UTC-naive.")
    if not ds.is_monotonic_increasing:
        raise ValueError("Kolom ds training tidak chronological.")
    if ds.duplicated().any():
        raise ValueError("Kolom ds training mengandung duplicate timestamp.")


def summarize_train_val(train_val: pd.DataFrame) -> dict[str, Any]:
    target = pd.to_numeric(train_val[TARGET_COL], errors="raise")
    return {
        "n_rows": int(train_val.shape[0]),
        "n_columns": int(train_val.shape[1]),
        "utc_start": train_val.index.min().isoformat(),
        "utc_end": train_val.index.max().isoformat(),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "target_mean": float(target.mean()),
    }


def summarize_training_frame(
    training_frame: pd.DataFrame,
    train_val: pd.DataFrame,
) -> dict[str, Any]:
    ds = pd.to_datetime(training_frame["ds"], errors="raise")
    ds_utc = pd.DatetimeIndex(ds).tz_localize(MODELING_TZ)
    target = pd.to_numeric(training_frame["y"], errors="raise")
    return {
        "n_rows": int(training_frame.shape[0]),
        "n_columns": int(training_frame.shape[1]),
        "utc_start": ds_utc.min().isoformat(),
        "utc_end": ds_utc.max().isoformat(),
        "ds_is_utc_naive": True,
        "history_rows_dropped": int(train_val.shape[0] - training_frame.shape[0]),
        "regressors": list(PROPHET_BASIC_REGRESSORS),
        "n_regressors": int(len(PROPHET_BASIC_REGRESSORS)),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "target_mean": float(target.mean()),
    }


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


def save_outputs(metadata: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    summary_row = {
        "model_key": MODEL_KEY,
        "model_label": MODEL_LABEL,
        "feature_set": PROPHET_REGRESSOR_BASIC_FEATURE_SET,
        "parameter_set_id": metadata["parameter_set_id"],
        "source_train_val_rows": metadata["train_val"]["n_rows"],
        "n_train_rows": metadata["training_frame"]["n_rows"],
        "history_rows_dropped": metadata["training_frame"]["history_rows_dropped"],
        "train_start": metadata["training_frame"]["utc_start"],
        "train_end": metadata["training_frame"]["utc_end"],
        "train_time_seconds": metadata["train_time_seconds"],
        "prediction_time_seconds": metadata["prediction_time_seconds"],
        "total_runtime_seconds": metadata["total_runtime_seconds"],
        "model_path": metadata["model_path"],
        "metadata_path": str(outputs["metadata"]),
    }
    pd.DataFrame([summary_row]).to_csv(outputs["training_summary"], index=False)
    pd.DataFrame(
        [
            {
                "model_key": MODEL_KEY,
                "model_label": MODEL_LABEL,
                "feature_set": PROPHET_REGRESSOR_BASIC_FEATURE_SET,
                "train_time_seconds": metadata["train_time_seconds"],
                "prediction_time_seconds": 0.0,
                "total_runtime_seconds": metadata["total_runtime_seconds"],
            }
        ]
    ).to_csv(outputs["runtime_summary"], index=False)
    save_experiment_metadata(
        {
            "parameter_set_id": metadata["parameter_set_id"],
            "params": metadata["params"],
            "best_cv_metrics": metadata["best_cv_metrics"],
            "source_best_params_path": metadata["best_params_path"],
        },
        outputs["best_params_used"],
    )
    save_experiment_metadata(metadata, outputs["metadata"])
    outputs["summary"].write_text(render_summary(metadata), encoding="utf-8")


def log_success(metadata: Mapping[str, Any]) -> None:
    log_runtime(
        make_runtime_record(
            experiment_name=EXPERIMENT_NAME,
            model_name=MODEL_KEY,
            feature_set=PROPHET_REGRESSOR_BASIC_FEATURE_SET,
            parameter_set_id=metadata["parameter_set_id"],
            train_start=metadata["training_frame"]["utc_start"],
            train_end=metadata["training_frame"]["utc_end"],
            n_train_rows=metadata["training_frame"]["n_rows"],
            n_prediction_rows=0,
            train_time_seconds=metadata["train_time_seconds"],
            prediction_time_seconds=0.0,
            total_runtime_seconds=metadata["total_runtime_seconds"],
            status="success",
        )
    )


def render_summary(metadata: Mapping[str, Any]) -> str:
    frame = metadata["training_frame"]
    outputs = metadata["outputs"]
    lines = [
        "# Revisi 1.6 - Retraining Prophet-Regressor-Basic",
        "",
        f"Run UTC: {metadata['started_at_utc']}",
        "",
        "## Scope",
        "",
        (
            "Model Prophet-Regressor-Basic dilatih ulang memakai seluruh "
            "`train_val.csv` dengan best params dari tuning revisi. Final test "
            "tidak dibaca atau digunakan pada tahap ini."
        ),
        "",
        "## Training Data",
        "",
        f"- Source train_val rows: {metadata['train_val']['n_rows']}",
        f"- Training rows after history drop: {frame['n_rows']}",
        f"- History rows dropped: {frame['history_rows_dropped']}",
        f"- Training period UTC: {frame['utc_start']} to {frame['utc_end']}",
        f"- Regressors: {', '.join(frame['regressors'])}",
        "",
        "## Best Configuration",
        "",
        f"- Parameter set: {metadata['parameter_set_id']}",
        f"- Params: {json.dumps(metadata['params'], sort_keys=True)}",
        "",
        "## Time Cost Computing",
        "",
        f"- Training time seconds: {metadata['train_time_seconds']:.6f}",
        "- Prediction time seconds: 0.000000 karena tahap ini hanya training.",
        f"- Total runtime seconds: {metadata['total_runtime_seconds']:.6f}",
        "",
        "## Output Files",
        "",
        f"- Model artifact: `{outputs['model']}`",
        f"- Metadata: `{outputs['metadata']}`",
        f"- Training summary: `{outputs['training_summary']}`",
        f"- Runtime summary: `{outputs['runtime_summary']}`",
        f"- Best params used: `{outputs['best_params_used']}`",
        "",
        "## Leakage Note",
        "",
        (
            "Lag regressor training dibuat dari actual train_val masa lalu saja. "
            "Final testing nantinya tetap wajib memakai recursive forecasting "
            "untuk membangun lag regressor masa depan."
        ),
    ]
    return "\n".join(lines) + "\n"


def methodology_notes() -> list[str]:
    return [
        "Best params berasal dari tuning Prophet-Regressor-Basic revisi, bukan final test.",
        "Retraining memakai seluruh train_val sebelum final testing revisi.",
        "Training rows awal yang belum memiliki lag_168 lengkap di-drop.",
        "Final test tidak dibaca atau digunakan oleh script retraining.",
        "Tidak ada tuning tambahan pada tahap retraining.",
    ]


def stringify_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    return {key: str(value) for key, value in paths.items()}


if __name__ == "__main__":
    main()

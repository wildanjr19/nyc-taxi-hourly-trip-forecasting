"""
tracking.py

Runtime logging dan experiment tracking untuk pipeline penelitian NYC Taxi.

Modul ini menjadi satu pintu resmi untuk mencatat TIME COST COMPUTING:
- durasi training,
- durasi prediction,
- total runtime,
- status sukses/gagal,
- metadata eksperimen dalam JSON/JSONL.

Durasi diukur dengan time.perf_counter() agar stabil untuk elapsed time.
Timestamp run tetap memakai UTC wall-clock time untuk audit eksperimen.
"""

from __future__ import annotations

import csv
import json
import math
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from src.config import EXPERIMENT_RUNS_JSONL, RUNTIME_LOG_COLUMNS, RUNTIME_LOG_CSV


PathLike = Union[str, Path]


def utc_now_iso() -> str:
    """
    Timestamp UTC untuk audit run.
    """
    return datetime.now(timezone.utc).isoformat()


def start_timer() -> float:
    """
    Mulai timer monotonic untuk mengukur elapsed runtime.
    """
    return time.perf_counter()


def elapsed_seconds(start_time: float, *, ndigits: int = 6) -> float:
    """
    Hitung elapsed seconds dari start_time time.perf_counter().
    """
    elapsed = time.perf_counter() - float(start_time)
    if elapsed < 0:
        raise ValueError("Elapsed time negatif; periksa nilai start_time.")
    return round(elapsed, ndigits)


def make_runtime_record(
    *,
    experiment_name: str,
    model_name: str = "none",
    feature_set: str = "none",
    fold: Optional[Union[int, str]] = None,
    parameter_set_id: Optional[Union[int, str]] = None,
    train_start: Optional[Union[str, datetime]] = None,
    train_end: Optional[Union[str, datetime]] = None,
    validation_start: Optional[Union[str, datetime]] = None,
    validation_end: Optional[Union[str, datetime]] = None,
    n_train_rows: Optional[Union[int, str]] = None,
    n_prediction_rows: Optional[Union[int, str]] = None,
    train_time_seconds: Optional[float] = None,
    prediction_time_seconds: Optional[float] = None,
    total_runtime_seconds: Optional[float] = None,
    status: str = "success",
    error_message: str = "",
    timestamp_run: Optional[str] = None,
) -> dict[str, Any]:
    """
    Buat record runtime sesuai schema standar di src.config.
    """
    return normalize_runtime_record(
        {
            "timestamp_run": timestamp_run or utc_now_iso(),
            "experiment_name": experiment_name,
            "model_name": model_name,
            "feature_set": feature_set,
            "fold": fold,
            "parameter_set_id": parameter_set_id,
            "train_start": _format_datetime_like(train_start),
            "train_end": _format_datetime_like(train_end),
            "validation_start": _format_datetime_like(validation_start),
            "validation_end": _format_datetime_like(validation_end),
            "n_train_rows": n_train_rows,
            "n_prediction_rows": n_prediction_rows,
            "train_time_seconds": train_time_seconds,
            "prediction_time_seconds": prediction_time_seconds,
            "total_runtime_seconds": total_runtime_seconds,
            "status": status,
            "error_message": error_message,
        }
    )


def normalize_runtime_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalisasi record runtime agar selalu mengikuti RUNTIME_LOG_COLUMNS.
    """
    normalized = {column: "" for column in RUNTIME_LOG_COLUMNS}
    for column in RUNTIME_LOG_COLUMNS:
        if column in record:
            normalized[column] = _to_csv_value(record[column])

    if not normalized["timestamp_run"]:
        normalized["timestamp_run"] = utc_now_iso()
    if not normalized["experiment_name"]:
        raise ValueError("runtime record wajib memiliki experiment_name.")
    if not normalized["model_name"]:
        normalized["model_name"] = "none"
    if not normalized["feature_set"]:
        normalized["feature_set"] = "none"
    if not normalized["status"]:
        normalized["status"] = "success"

    return normalized


def log_runtime(
    record: Mapping[str, Any],
    output_path: PathLike = RUNTIME_LOG_CSV,
) -> dict[str, Any]:
    """
    Append runtime record ke CSV log standar.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_runtime_record(record)

    file_has_content = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNTIME_LOG_COLUMNS)
        if not file_has_content:
            writer.writeheader()
        writer.writerow(normalized)

    return normalized


def save_experiment_metadata(
    metadata: Mapping[str, Any],
    output_path: PathLike,
) -> Path:
    """
    Simpan metadata eksperimen ke JSON dengan serializer yang aman untuk
    Path, datetime, pandas/numpy scalar, dan NaN.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _to_jsonable(dict(metadata))
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def append_experiment_run(
    run_record: Mapping[str, Any],
    output_path: PathLike = EXPERIMENT_RUNS_JSONL,
) -> dict[str, Any]:
    """
    Append satu run eksperimen ke JSONL.

    JSONL dipakai agar tuning panjang bisa menyimpan hasil incremental tanpa
    menimpa run sebelumnya.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    enriched = dict(run_record)
    enriched.setdefault("timestamp_run", utc_now_iso())
    enriched.setdefault("status", "success")
    serializable = _to_jsonable(enriched)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(serializable, ensure_ascii=False))
        handle.write("\n")

    return serializable


def _format_datetime_like(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            return value
    return value


def _to_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    value = _format_datetime_like(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return round(value, 9)
    return value


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _to_jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return _to_jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


__all__ = [
    "utc_now_iso",
    "start_timer",
    "elapsed_seconds",
    "make_runtime_record",
    "normalize_runtime_record",
    "log_runtime",
    "save_experiment_metadata",
    "append_experiment_run",
]

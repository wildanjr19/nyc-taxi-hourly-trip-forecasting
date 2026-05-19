"""
splits.py

Tahap split time-based untuk dataset NYC Taxi hourly.

Modul ini memisahkan final test set sejak awal agar benar-benar unseen
untuk tuning, model selection, dan feature engineering berikutnya. Modul
ini juga menyediakan expanding-window time series cross validation untuk
tuning di train_val saja.

Output utama hold-out:
- data/processed/train_val.csv
- data/processed/final_test.csv
- outputs/splits/summaries/holdout_split_summary.txt
- outputs/splits/summaries/holdout_split_metadata.json

Output utama CV:
- outputs/splits/summaries/time_series_cv_folds.csv
- outputs/splits/summaries/time_series_cv_summary.txt
- outputs/splits/summaries/time_series_cv_metadata.json
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from src.config import (
    CV_GAP_HOURS,
    CV_N_FOLDS,
    CV_VAL_HORIZON_HOURS,
    FINAL_TEST_DAYS,
    FINAL_TEST_PATH,
    FORECAST_HORIZON,
    HOLDOUT_SPLIT_METADATA_PATH,
    HOLDOUT_SPLIT_SUMMARY_PATH,
    LOCAL_TZ,
    MODELING_TZ,
    TARGET_COL,
    TIMESTAMP_COL,
    TIME_SERIES_CV_FOLDS_PATH,
    TIME_SERIES_CV_METADATA_PATH,
    TIME_SERIES_CV_SUMMARY_PATH,
    TRAIN_VAL_PATH,
    ensure_dirs,
)
from src.preprocessing import (
    DST_AMBIGUOUS_COL,
    LOCAL_NAIVE_TIMESTAMP_COL,
    LOCAL_TIMESTAMP_COL,
    load_preprocessed_timeseries,
    set_datetime_index,
    sort_by_timestamp,
)
from src.tracking import log_runtime


def make_holdout_split(
    df: pd.DataFrame,
    test_days: int = FINAL_TEST_DAYS,
    *,
    timestamp_col: str = TIMESTAMP_COL,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Pisahkan data menjadi train_val dan final_test secara kronologis.

    Policy split:
    - final_test berisi timestamp pada 30 hari terakhir berdasarkan index UTC
      modeling, yaitu timestamp > (timestamp_terakhir - 30 hari).
    - train_val berisi seluruh timestamp sebelum atau sama dengan cutoff.
    - Tidak ada shuffle dan tidak ada akses test untuk tuning.
    """
    if test_days <= 0:
        raise ValueError("test_days harus > 0")

    timeseries = _ensure_datetime_index(
        df,
        timestamp_col=timestamp_col,
        target_col=target_col,
    )

    split_cutoff = timeseries.index.max() - pd.Timedelta(days=test_days)
    train_val = timeseries.loc[timeseries.index <= split_cutoff].copy()
    final_test = timeseries.loc[timeseries.index > split_cutoff].copy()

    validation = validate_split_order(
        train_val,
        final_test,
        source_df=timeseries,
        expected_test_days=test_days,
        target_col=target_col,
    )

    summary = {
        "stage": "holdout_time_based_split",
        "status": "success",
        "split_policy": "final_test timestamps > max_timestamp - test_days",
        "test_days": int(test_days),
        "expected_final_test_hours": int(test_days * 24),
        "split_cutoff_utc": _format_one_timestamp(split_cutoff),
        "source": _summarize_period(timeseries),
        "train_val": _summarize_period(train_val),
        "final_test": _summarize_period(final_test),
        "target_col": target_col,
        "columns": list(timeseries.columns),
        "validation": validation,
        "methodology_note": [
            "Split dilakukan setelah time-based preprocessing dan sebelum tuning.",
            "Final test disimpan terpisah dan tidak boleh dipakai untuk CV/tuning.",
            "Urutan timestamp dipertahankan; tidak ada random split atau shuffle.",
        ],
    }
    return train_val, final_test, summary


def validate_split_order(
    train_val: pd.DataFrame,
    final_test: pd.DataFrame,
    *,
    source_df: Optional[pd.DataFrame] = None,
    expected_test_days: int = FINAL_TEST_DAYS,
    target_col: str = TARGET_COL,
) -> dict[str, Any]:
    """
    Validasi bahwa split chronological, tidak overlap, dan siap dipakai.
    """
    if train_val.empty:
        raise ValueError("train_val kosong setelah split.")
    if final_test.empty:
        raise ValueError("final_test kosong setelah split.")

    _validate_timeseries_frame(train_val, frame_name="train_val", target_col=target_col)
    _validate_timeseries_frame(final_test, frame_name="final_test", target_col=target_col)

    if train_val.index.max() >= final_test.index.min():
        raise ValueError(
            "Split tidak chronological: timestamp akhir train_val harus lebih kecil "
            "dari timestamp awal final_test."
        )

    overlap = train_val.index.intersection(final_test.index)
    if len(overlap) > 0:
        raise ValueError(
            f"Terdapat overlap timestamp antara train_val dan final_test: "
            f"{_format_timestamp_examples(overlap)}"
        )

    expected_test_hours = expected_test_days * 24
    observed_test_rows = int(final_test.shape[0])
    first_gap_hours = (
        final_test.index.min() - train_val.index.max()
    ) / pd.Timedelta(hours=1)

    validation = {
        "chronological_order": True,
        "overlap_count": int(len(overlap)),
        "train_val_max_before_final_test_min": bool(train_val.index.max() < final_test.index.min()),
        "gap_hours_between_train_and_test": float(first_gap_hours),
        "expected_final_test_hours": int(expected_test_hours),
        "observed_final_test_rows": observed_test_rows,
        "observed_final_test_equals_expected_hours": bool(
            observed_test_rows == expected_test_hours
        ),
        "final_test_has_at_least_forecast_horizon": bool(observed_test_rows >= 24),
    }

    if source_df is not None:
        _validate_timeseries_frame(source_df, frame_name="source_df", target_col=target_col)
        combined_index = train_val.index.append(final_test.index)
        validation["row_count_preserved"] = bool(
            train_val.shape[0] + final_test.shape[0] == source_df.shape[0]
        )
        validation["index_coverage_preserved"] = bool(combined_index.equals(source_df.index))

        if not validation["row_count_preserved"]:
            raise ValueError("Jumlah baris train_val + final_test tidak sama dengan source_df.")
        if not validation["index_coverage_preserved"]:
            raise ValueError("Index gabungan split tidak sama dengan index source_df.")

    if not validation["final_test_has_at_least_forecast_horizon"]:
        raise ValueError("final_test harus minimal memiliki 24 baris untuk horizon forecast.")

    return validation


def make_expanding_window_splits(
    df: pd.DataFrame,
    n_folds: int = CV_N_FOLDS,
    val_horizon: int = CV_VAL_HORIZON_HOURS,
    gap: int = CV_GAP_HOURS,
    *,
    min_train_size: int = FORECAST_HORIZON,
    forecast_horizon: int = FORECAST_HORIZON,
    timestamp_col: str = TIMESTAMP_COL,
    target_col: str = TARGET_COL,
) -> list[dict[str, Any]]:
    """
    Buat expanding-window cross validation split dari train_val.

    Validation window tiap fold memakai blok berurutan di bagian akhir
    train_val. Dengan konfigurasi default:
    - 5 fold,
    - 168 baris/jam validasi per fold,
    - gap 0,
    - validation window dapat dievaluasi sebagai 7 blok recursive 24 jam.

    Return berupa list dict agar script tuning berikutnya bisa langsung
    mengambil `split["train"]`, `split["validation"]`, dan metadata fold.
    """
    n_folds = _coerce_integer(n_folds, argument_name="n_folds", min_value=2)
    val_horizon = _coerce_integer(
        val_horizon,
        argument_name="val_horizon",
        min_value=1,
    )
    gap = _coerce_integer(gap, argument_name="gap", min_value=0)
    min_train_size = _coerce_integer(
        min_train_size,
        argument_name="min_train_size",
        min_value=1,
    )
    forecast_horizon = _coerce_integer(
        forecast_horizon,
        argument_name="forecast_horizon",
        min_value=1,
    )

    if val_horizon < forecast_horizon:
        raise ValueError(
            "val_horizon harus >= forecast_horizon agar validasi memuat "
            f"minimal satu horizon penuh. val_horizon={val_horizon}, "
            f"forecast_horizon={forecast_horizon}"
        )

    timeseries = _ensure_datetime_index(
        df,
        timestamp_col=timestamp_col,
        target_col=target_col,
    )

    total_validation_rows = n_folds * val_horizon
    minimum_required_rows = min_train_size + gap + total_validation_rows
    if len(timeseries) < minimum_required_rows:
        raise ValueError(
            "Data tidak cukup untuk expanding-window CV. "
            f"required_rows={minimum_required_rows}, observed_rows={len(timeseries)}"
        )

    first_validation_start_pos = len(timeseries) - total_validation_rows
    splits: list[dict[str, Any]] = []

    for fold_number in range(1, n_folds + 1):
        validation_start_pos = first_validation_start_pos + (
            (fold_number - 1) * val_horizon
        )
        validation_end_pos = validation_start_pos + val_horizon
        train_end_pos = validation_start_pos - gap

        train = timeseries.iloc[:train_end_pos].copy()
        validation = timeseries.iloc[validation_start_pos:validation_end_pos].copy()

        metadata = _build_cv_fold_metadata(
            fold=fold_number,
            train=train,
            validation=validation,
            train_start_pos=0,
            train_end_pos=train_end_pos,
            validation_start_pos=validation_start_pos,
            validation_end_pos=validation_end_pos,
            gap=gap,
            val_horizon=val_horizon,
            forecast_horizon=forecast_horizon,
        )
        splits.append(
            {
                "fold": fold_number,
                "train": train,
                "validation": validation,
                "metadata": metadata,
            }
        )

    validate_cv_splits(
        splits,
        source_df=timeseries,
        n_folds=n_folds,
        val_horizon=val_horizon,
        gap=gap,
        forecast_horizon=forecast_horizon,
        target_col=target_col,
    )
    return splits


def validate_cv_splits(
    splits: list[dict[str, Any]],
    *,
    source_df: Optional[pd.DataFrame] = None,
    n_folds: Optional[int] = None,
    val_horizon: Optional[int] = None,
    gap: Optional[int] = None,
    forecast_horizon: int = FORECAST_HORIZON,
    target_col: str = TARGET_COL,
) -> dict[str, Any]:
    """
    Validasi fold CV agar chronological, tidak overlap, dan siap tuning.
    """
    if not splits:
        raise ValueError("Daftar CV split kosong.")

    expected_n_folds = (
        _coerce_integer(n_folds, argument_name="n_folds", min_value=1)
        if n_folds is not None
        else len(splits)
    )
    expected_val_horizon = (
        _coerce_integer(val_horizon, argument_name="val_horizon", min_value=1)
        if val_horizon is not None
        else None
    )
    expected_gap = (
        _coerce_integer(gap, argument_name="gap", min_value=0)
        if gap is not None
        else None
    )
    forecast_horizon = _coerce_integer(
        forecast_horizon,
        argument_name="forecast_horizon",
        min_value=1,
    )

    if len(splits) != expected_n_folds:
        raise ValueError(
            f"Jumlah fold tidak sesuai: observed={len(splits)}, "
            f"expected={expected_n_folds}"
        )

    fold_numbers: list[int] = []
    validation_indexes: list[pd.DatetimeIndex] = []
    previous_train_end: Optional[pd.Timestamp] = None
    previous_validation_end: Optional[pd.Timestamp] = None
    previous_train_rows: Optional[int] = None

    for expected_fold, split in enumerate(splits, start=1):
        missing_keys = {"fold", "train", "validation", "metadata"}.difference(split)
        if missing_keys:
            raise ValueError(f"Split fold kehilangan key wajib: {sorted(missing_keys)}")

        fold = _coerce_integer(split["fold"], argument_name="fold", min_value=1)
        if fold != expected_fold:
            raise ValueError(
                f"Nomor fold harus berurutan mulai dari 1. "
                f"observed={fold}, expected={expected_fold}"
            )
        fold_numbers.append(fold)

        train = split["train"]
        validation = split["validation"]
        metadata = split["metadata"]
        if not isinstance(train, pd.DataFrame):
            raise TypeError("split['train'] harus berupa pandas DataFrame.")
        if not isinstance(validation, pd.DataFrame):
            raise TypeError("split['validation'] harus berupa pandas DataFrame.")

        _validate_timeseries_frame(train, frame_name=f"cv_train_fold_{fold}", target_col=target_col)
        _validate_timeseries_frame(
            validation,
            frame_name=f"cv_validation_fold_{fold}",
            target_col=target_col,
        )

        if expected_val_horizon is not None and len(validation) != expected_val_horizon:
            raise ValueError(
                f"Validation horizon fold {fold} tidak sesuai: "
                f"observed={len(validation)}, expected={expected_val_horizon}"
            )
        if train.index.max() >= validation.index.min():
            raise ValueError(
                f"Fold {fold} tidak chronological: train_end >= validation_start."
            )
        if train.index.intersection(validation.index).size > 0:
            raise ValueError(f"Fold {fold} memiliki overlap train dan validation.")

        if previous_train_end is not None and train.index.max() <= previous_train_end:
            raise ValueError("Train window tidak expanding antar fold.")
        if previous_validation_end is not None and validation.index.min() <= previous_validation_end:
            raise ValueError("Validation window antar fold overlap atau tidak chronological.")
        if previous_train_rows is not None and len(train) <= previous_train_rows:
            raise ValueError("Jumlah baris train tidak meningkat pada fold berikutnya.")

        if expected_gap is not None and int(metadata.get("gap_rows", -1)) != expected_gap:
            raise ValueError(
                f"Metadata gap fold {fold} tidak sesuai expected_gap={expected_gap}."
            )
        if int(metadata.get("forecast_horizon_hours", -1)) != forecast_horizon:
            raise ValueError(
                f"Metadata forecast horizon fold {fold} tidak sesuai "
                f"forecast_horizon={forecast_horizon}."
            )
        if bool(metadata.get("train_end_before_validation_start")) is not True:
            raise ValueError(f"Metadata fold {fold} gagal validasi chronological.")

        validation_indexes.append(validation.index)
        previous_train_end = train.index.max()
        previous_validation_end = validation.index.max()
        previous_train_rows = len(train)

    combined_validation_index = validation_indexes[0].append(validation_indexes[1:])
    if combined_validation_index.has_duplicates:
        raise ValueError("Validation index antar fold mengandung duplicate timestamp.")
    if not combined_validation_index.is_monotonic_increasing:
        raise ValueError("Validation index gabungan tidak chronological.")

    validation = {
        "passed": True,
        "n_folds": int(len(splits)),
        "fold_numbers": fold_numbers,
        "total_validation_rows": int(len(combined_validation_index)),
        "validation_start_utc": _format_one_timestamp(combined_validation_index.min()),
        "validation_end_utc": _format_one_timestamp(combined_validation_index.max()),
        "validation_overlap_count": 0,
        "train_windows_expanding": True,
        "train_before_validation_all_folds": True,
        "forecast_horizon_hours": int(forecast_horizon),
        "validation_horizon_multiple_of_forecast_horizon": bool(
            expected_val_horizon is not None
            and expected_val_horizon % forecast_horizon == 0
        ),
    }

    if source_df is not None:
        source = _ensure_datetime_index(
            source_df,
            timestamp_col=TIMESTAMP_COL,
            target_col=target_col,
        )
        expected_tail = source.index[-len(combined_validation_index) :]
        validation["validation_windows_use_tail_of_train_val"] = bool(
            combined_validation_index.equals(expected_tail)
        )
        validation["source_rows"] = int(source.shape[0])
        if not validation["validation_windows_use_tail_of_train_val"]:
            raise ValueError(
                "Validation windows CV tidak sama dengan blok terakhir train_val."
            )

    return validation


def build_cv_fold_metadata_frame(splits: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Ubah metadata fold menjadi tabel CSV-friendly.
    """
    rows: list[dict[str, Any]] = []
    for split in splits:
        metadata = dict(split["metadata"])
        metadata.pop("rolling_origin_blocks", None)
        rows.append(metadata)

    if not rows:
        raise ValueError("Tidak ada metadata fold untuk disimpan.")
    return pd.DataFrame(rows)


def save_time_series_cv_outputs(
    splits: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    folds_path: Union[str, Path] = TIME_SERIES_CV_FOLDS_PATH,
    summary_path: Union[str, Path] = TIME_SERIES_CV_SUMMARY_PATH,
    metadata_path: Union[str, Path] = TIME_SERIES_CV_METADATA_PATH,
) -> None:
    """
    Simpan fold metadata, summary teks, dan metadata JSON CV.
    """
    fold_destination = Path(folds_path)
    fold_destination.parent.mkdir(parents=True, exist_ok=True)
    build_cv_fold_metadata_frame(splits).to_csv(fold_destination, index=False)

    summary_destination = Path(summary_path)
    summary_destination.parent.mkdir(parents=True, exist_ok=True)
    summary_destination.write_text(_render_cv_summary_text(summary), encoding="utf-8")

    metadata_destination = Path(metadata_path)
    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_destination.write_text(
        json.dumps(_to_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_time_series_cv_split(
    *,
    input_path: Union[str, Path] = TRAIN_VAL_PATH,
    n_folds: int = CV_N_FOLDS,
    val_horizon: int = CV_VAL_HORIZON_HOURS,
    gap: int = CV_GAP_HOURS,
    forecast_horizon: int = FORECAST_HORIZON,
    folds_path: Union[str, Path] = TIME_SERIES_CV_FOLDS_PATH,
    summary_path: Union[str, Path] = TIME_SERIES_CV_SUMMARY_PATH,
    metadata_path: Union[str, Path] = TIME_SERIES_CV_METADATA_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Entry point tahap 9 untuk membuat metadata CV dari train_val saja.
    """
    ensure_dirs()
    start_time = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    status = "success"
    error_message = ""
    source_rows: Union[int, str] = ""
    validation_rows: Union[int, str] = ""

    try:
        train_val = load_split_timeseries(input_path)
        source_rows = int(train_val.shape[0])

        splits = make_expanding_window_splits(
            train_val,
            n_folds=n_folds,
            val_horizon=val_horizon,
            gap=gap,
            forecast_horizon=forecast_horizon,
        )
        validation = validate_cv_splits(
            splits,
            source_df=train_val,
            n_folds=n_folds,
            val_horizon=val_horizon,
            gap=gap,
            forecast_horizon=forecast_horizon,
        )
        validation_rows = int(validation["total_validation_rows"])

        elapsed_seconds = time.perf_counter() - start_time
        summary = _build_cv_summary(
            source_df=train_val,
            splits=splits,
            validation=validation,
            input_path=input_path,
            folds_path=folds_path,
            summary_path=summary_path,
            metadata_path=metadata_path,
            runtime_seconds=round(elapsed_seconds, 6),
            timestamp_run_utc=started_at,
            n_folds=n_folds,
            val_horizon=val_horizon,
            gap=gap,
            forecast_horizon=forecast_horizon,
        )

        save_time_series_cv_outputs(
            splits,
            summary,
            folds_path=folds_path,
            summary_path=summary_path,
            metadata_path=metadata_path,
        )
        return splits, summary
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        raise
    finally:
        total_runtime_seconds = round(time.perf_counter() - start_time, 6)
        _append_cv_runtime_log(
            total_runtime_seconds=total_runtime_seconds,
            timestamp_run=started_at,
            status=status,
            error_message=error_message,
            n_train_rows=source_rows,
            n_prediction_rows=validation_rows,
        )


def save_holdout_split(
    train_val: pd.DataFrame,
    final_test: pd.DataFrame,
    *,
    train_val_path: Union[str, Path] = TRAIN_VAL_PATH,
    final_test_path: Union[str, Path] = FINAL_TEST_PATH,
) -> None:
    """
    Simpan train_val dan final_test dengan UTC timestamp sebagai kolom pertama.
    """
    _save_timeseries_csv(train_val, train_val_path)
    _save_timeseries_csv(final_test, final_test_path)


def save_holdout_summary(
    summary: dict[str, Any],
    summary_path: Union[str, Path] = HOLDOUT_SPLIT_SUMMARY_PATH,
) -> None:
    """
    Simpan ringkasan split dalam format teks yang mudah diaudit.
    """
    destination = Path(summary_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_render_summary_text(summary), encoding="utf-8")


def save_holdout_metadata(
    summary: dict[str, Any],
    metadata_path: Union[str, Path] = HOLDOUT_SPLIT_METADATA_PATH,
) -> None:
    """
    Simpan metadata split lengkap dalam JSON untuk reproducibility.
    """
    destination = Path(metadata_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_to_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_split_timeseries(
    path: Union[str, Path],
    *,
    local_tz: str = LOCAL_TZ,
    modeling_tz: str = MODELING_TZ,
) -> pd.DataFrame:
    """
    Load train_val/final_test dengan dtype timestamp yang stabil.

    Fungsi ini sengaja mirror dengan loader preprocessing karena
    `timestamp_local` pada CSV dapat memiliki offset DST campuran.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"File split tidak ditemukan: {source}")

    df = pd.read_csv(source)
    required_columns = {
        TIMESTAMP_COL,
        TARGET_COL,
        LOCAL_NAIVE_TIMESTAMP_COL,
        LOCAL_TIMESTAMP_COL,
        DST_AMBIGUOUS_COL,
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Kolom wajib tidak ditemukan pada data split: {missing_columns}")

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True, errors="raise")
    df[LOCAL_NAIVE_TIMESTAMP_COL] = pd.to_datetime(
        df[LOCAL_NAIVE_TIMESTAMP_COL],
        errors="raise",
    )
    df[DST_AMBIGUOUS_COL] = df[DST_AMBIGUOUS_COL].astype(bool)
    df[LOCAL_TIMESTAMP_COL] = df[LOCAL_NAIVE_TIMESTAMP_COL].dt.tz_localize(
        local_tz,
        ambiguous=False,
        nonexistent="raise",
    )

    expected_utc = df[LOCAL_TIMESTAMP_COL].dt.tz_convert(modeling_tz)
    if not expected_utc.equals(df[TIMESTAMP_COL]):
        mismatch_count = int((expected_utc != df[TIMESTAMP_COL]).sum())
        raise ValueError(
            "Timestamp UTC tidak konsisten dengan timestamp lokal hasil "
            f"rekonstruksi. mismatch_count={mismatch_count}"
        )

    loaded = set_datetime_index(sort_by_timestamp(df, timestamp_col=TIMESTAMP_COL))
    _validate_timeseries_frame(loaded, frame_name=str(source), target_col=TARGET_COL)
    return loaded


def run_holdout_split(
    *,
    test_days: int = FINAL_TEST_DAYS,
    train_val_path: Union[str, Path] = TRAIN_VAL_PATH,
    final_test_path: Union[str, Path] = FINAL_TEST_PATH,
    summary_path: Union[str, Path] = HOLDOUT_SPLIT_SUMMARY_PATH,
    metadata_path: Union[str, Path] = HOLDOUT_SPLIT_METADATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Entry point utama untuk menjalankan tahap 5 dari full_preprocessed.csv.
    """
    ensure_dirs()
    start_time = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    status = "success"
    error_message = ""
    train_rows = ""
    test_rows = ""

    try:
        df = load_preprocessed_timeseries()
        train_val, final_test, summary = make_holdout_split(df, test_days=test_days)

        elapsed_seconds = time.perf_counter() - start_time
        summary["runtime_seconds"] = round(elapsed_seconds, 6)
        summary["timestamp_run_utc"] = started_at

        save_holdout_split(
            train_val,
            final_test,
            train_val_path=train_val_path,
            final_test_path=final_test_path,
        )
        save_holdout_summary(summary, summary_path=summary_path)
        save_holdout_metadata(summary, metadata_path=metadata_path)

        train_rows = int(train_val.shape[0])
        test_rows = int(final_test.shape[0])
        return train_val, final_test, summary
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        raise
    finally:
        total_runtime_seconds = round(time.perf_counter() - start_time, 6)
        _append_split_runtime_log(
            total_runtime_seconds=total_runtime_seconds,
            timestamp_run=started_at,
            status=status,
            error_message=error_message,
            n_train_rows=train_rows,
            n_prediction_rows=test_rows,
        )


def _ensure_datetime_index(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    target_col: str,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Data input untuk split kosong.")
    if target_col not in df.columns:
        raise ValueError(f"Kolom target tidak ditemukan: {target_col}")

    if isinstance(df.index, pd.DatetimeIndex):
        indexed = df.copy()
    elif timestamp_col in df.columns:
        indexed = set_datetime_index(
            sort_by_timestamp(df.copy(), timestamp_col=timestamp_col),
            timestamp_col=timestamp_col,
        )
    else:
        raise ValueError(
            f"Data harus memiliki DatetimeIndex atau kolom timestamp '{timestamp_col}'."
        )

    if not indexed.index.is_monotonic_increasing:
        indexed = indexed.sort_index(kind="mergesort")

    _validate_timeseries_frame(indexed, frame_name="input", target_col=target_col)
    return indexed


def _validate_timeseries_frame(
    df: pd.DataFrame,
    *,
    frame_name: str,
    target_col: str,
) -> None:
    if df.empty:
        raise ValueError(f"{frame_name} kosong.")
    if target_col not in df.columns:
        raise ValueError(f"Kolom target hilang pada {frame_name}: {target_col}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Index {frame_name} harus berupa pandas DatetimeIndex.")
    if str(df.index.tz) != MODELING_TZ:
        raise ValueError(
            f"Index {frame_name} harus timezone {MODELING_TZ}, ditemukan {df.index.tz}"
        )
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"Index {frame_name} tidak chronological.")
    if df.index.has_duplicates:
        raise ValueError(f"Index {frame_name} mengandung duplicate timestamp.")
    if df[target_col].isna().any():
        raise ValueError(f"Kolom target {frame_name} mengandung missing value.")


def _save_timeseries_csv(df: pd.DataFrame, output_path: Union[str, Path]) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=True)


def _summarize_period(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "n_rows": 0,
            "n_columns": 0,
            "utc_start": None,
            "utc_end": None,
            "local_start": None,
            "local_end": None,
            "target_min": None,
            "target_max": None,
            "target_mean": None,
        }

    local_start = None
    local_end = None
    if LOCAL_TIMESTAMP_COL in df.columns:
        local_start = _format_one_timestamp(df[LOCAL_TIMESTAMP_COL].min())
        local_end = _format_one_timestamp(df[LOCAL_TIMESTAMP_COL].max())

    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "utc_start": _format_one_timestamp(df.index.min()),
        "utc_end": _format_one_timestamp(df.index.max()),
        "local_start": local_start,
        "local_end": local_end,
        "target_min": float(df[TARGET_COL].min()),
        "target_max": float(df[TARGET_COL].max()),
        "target_mean": float(df[TARGET_COL].mean()),
    }


def _append_split_runtime_log(
    *,
    total_runtime_seconds: float,
    timestamp_run: str,
    status: str,
    error_message: str,
    n_train_rows: Union[int, str],
    n_prediction_rows: Union[int, str],
) -> None:
    record = {
        "timestamp_run": timestamp_run,
        "experiment_name": "holdout_time_based_split",
        "model_name": "none",
        "feature_set": "none",
        "fold": "",
        "parameter_set_id": "",
        "train_start": "",
        "train_end": "",
        "validation_start": "",
        "validation_end": "",
        "n_train_rows": n_train_rows,
        "n_prediction_rows": n_prediction_rows,
        "train_time_seconds": "",
        "prediction_time_seconds": "",
        "total_runtime_seconds": total_runtime_seconds,
        "status": status,
        "error_message": error_message,
    }
    log_runtime(record)


def _build_cv_fold_metadata(
    *,
    fold: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_start_pos: int,
    train_end_pos: int,
    validation_start_pos: int,
    validation_end_pos: int,
    gap: int,
    val_horizon: int,
    forecast_horizon: int,
) -> dict[str, Any]:
    train_spacing = _summarize_time_spacing(train.index)
    validation_spacing = _summarize_time_spacing(validation.index)
    boundary_delta_hours = (
        validation.index.min() - train.index.max()
    ) / pd.Timedelta(hours=1)
    observed_gap_hours = max(0.0, float(boundary_delta_hours) - 1.0)
    rolling_origin_blocks = _build_rolling_origin_blocks(
        validation.index,
        initial_origin=train.index.max(),
        forecast_horizon=forecast_horizon,
    )

    return {
        "fold": int(fold),
        "strategy": "expanding_window",
        "train_start_utc": _format_one_timestamp(train.index.min()),
        "train_end_utc": _format_one_timestamp(train.index.max()),
        "validation_start_utc": _format_one_timestamp(validation.index.min()),
        "validation_end_utc": _format_one_timestamp(validation.index.max()),
        "train_start_pos": int(train_start_pos),
        "train_end_pos_exclusive": int(train_end_pos),
        "validation_start_pos": int(validation_start_pos),
        "validation_end_pos_exclusive": int(validation_end_pos),
        "n_train_rows": int(train.shape[0]),
        "n_validation_rows": int(validation.shape[0]),
        "gap_rows": int(gap),
        "observed_boundary_delta_hours": float(boundary_delta_hours),
        "observed_gap_hours_between_train_and_validation": observed_gap_hours,
        "validation_horizon_hours": int(val_horizon),
        "forecast_horizon_hours": int(forecast_horizon),
        "n_origin_blocks": int(len(rolling_origin_blocks)),
        "origin_block_size_hours": int(forecast_horizon),
        "last_origin_block_rows": int(rolling_origin_blocks[-1]["n_prediction_rows"]),
        "validation_horizon_multiple_of_forecast_horizon": bool(
            validation.shape[0] % forecast_horizon == 0
        ),
        "train_end_before_validation_start": bool(
            train.index.max() < validation.index.min()
        ),
        "train_validation_overlap_count": int(train.index.intersection(validation.index).size),
        "train_non_hourly_delta_count": int(train_spacing["non_hourly_delta_count"]),
        "validation_non_hourly_delta_count": int(
            validation_spacing["non_hourly_delta_count"]
        ),
        "rolling_origin_blocks": rolling_origin_blocks,
    }


def _build_rolling_origin_blocks(
    validation_index: pd.DatetimeIndex,
    *,
    initial_origin: pd.Timestamp,
    forecast_horizon: int,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for start in range(0, len(validation_index), forecast_horizon):
        block_index = validation_index[start : start + forecast_horizon]
        origin = initial_origin if start == 0 else validation_index[start - 1]
        blocks.append(
            {
                "origin_block": int(len(blocks) + 1),
                "forecast_origin_utc": _format_one_timestamp(origin),
                "prediction_start_utc": _format_one_timestamp(block_index.min()),
                "prediction_end_utc": _format_one_timestamp(block_index.max()),
                "validation_position_start": int(start),
                "validation_position_end_exclusive": int(start + len(block_index)),
                "n_prediction_rows": int(len(block_index)),
                "horizon_step_start": 1,
                "horizon_step_end": int(len(block_index)),
            }
        )
    return blocks


def _build_cv_summary(
    *,
    source_df: pd.DataFrame,
    splits: list[dict[str, Any]],
    validation: dict[str, Any],
    input_path: Union[str, Path],
    folds_path: Union[str, Path],
    summary_path: Union[str, Path],
    metadata_path: Union[str, Path],
    runtime_seconds: float,
    timestamp_run_utc: str,
    n_folds: int,
    val_horizon: int,
    gap: int,
    forecast_horizon: int,
) -> dict[str, Any]:
    return {
        "stage": "time_series_cross_validation",
        "status": "success",
        "input_path": str(input_path),
        "outputs": {
            "fold_metadata_csv": str(folds_path),
            "summary": str(summary_path),
            "metadata": str(metadata_path),
        },
        "cv_config": {
            "strategy": "expanding_window",
            "n_folds": int(n_folds),
            "validation_horizon_hours": int(val_horizon),
            "gap_hours": int(gap),
            "forecast_horizon_hours": int(forecast_horizon),
            "rolling_origin_blocks_per_full_fold": int(
                np.ceil(val_horizon / forecast_horizon)
            ),
        },
        "source": _summarize_period(source_df),
        "source_time_spacing": _summarize_time_spacing(source_df.index),
        "folds": [split["metadata"] for split in splits],
        "validation": validation,
        "methodology_note": [
            "CV dibuat hanya dari train_val; final_test tidak diload atau digunakan.",
            "Setiap fold memakai expanding train window dan validation window setelah train.",
            "Tidak ada shuffle dan tidak ada overlap train-validation dalam fold.",
            "Validation horizon 168 jam dievaluasi sebagai rolling-origin block 24 jam.",
            "Actual validation boleh dipakai hanya setelah satu block selesai dievaluasi.",
        ],
        "runtime_seconds": runtime_seconds,
        "timestamp_run_utc": timestamp_run_utc,
    }


def _append_cv_runtime_log(
    *,
    total_runtime_seconds: float,
    timestamp_run: str,
    status: str,
    error_message: str,
    n_train_rows: Union[int, str],
    n_prediction_rows: Union[int, str],
) -> None:
    record = {
        "timestamp_run": timestamp_run,
        "experiment_name": "time_series_cross_validation_split",
        "model_name": "none",
        "feature_set": "none",
        "fold": "",
        "parameter_set_id": "",
        "train_start": "",
        "train_end": "",
        "validation_start": "",
        "validation_end": "",
        "n_train_rows": n_train_rows,
        "n_prediction_rows": n_prediction_rows,
        "train_time_seconds": "",
        "prediction_time_seconds": "",
        "total_runtime_seconds": total_runtime_seconds,
        "status": status,
        "error_message": error_message,
    }
    log_runtime(record)


def _render_cv_summary_text(summary: dict[str, Any]) -> str:
    source = summary["source"]
    source_spacing = summary["source_time_spacing"]
    config = summary["cv_config"]
    validation = summary["validation"]

    lines = [
        "=" * 70,
        "TIME SERIES CROSS VALIDATION SUMMARY - NYC Taxi Hourly Trip Count",
        "=" * 70,
        "",
        "1. OUTPUT STATUS",
        f"   - Status: {summary['status']}",
        f"   - Input path: {summary['input_path']}",
        f"   - Fold metadata CSV: {summary['outputs']['fold_metadata_csv']}",
        f"   - Metadata JSON: {summary['outputs']['metadata']}",
        "",
        "2. CV CONFIGURATION",
        f"   - Strategy: {config['strategy']}",
        f"   - Number of folds: {config['n_folds']}",
        f"   - Validation horizon per fold: {config['validation_horizon_hours']} hours",
        f"   - Forecast horizon: {config['forecast_horizon_hours']} hours",
        f"   - Gap: {config['gap_hours']} hours",
        "   - Rolling-origin blocks per full fold: "
        f"{config['rolling_origin_blocks_per_full_fold']}",
        "",
        "3. SOURCE TRAIN_VAL PERIOD",
        f"   - Rows: {source['n_rows']}",
        f"   - UTC start: {source['utc_start']}",
        f"   - UTC end: {source['utc_end']}",
        f"   - Local NYC start: {source['local_start']}",
        f"   - Local NYC end: {source['local_end']}",
        f"   - Non-hourly UTC gaps: {source_spacing['non_hourly_delta_count']}",
        "",
        "4. FOLD PERIODS",
    ]

    for fold in summary["folds"]:
        lines.append(
            "   - Fold {fold}: train rows={train_rows}, "
            "{train_start} to {train_end}; validation rows={val_rows}, "
            "{val_start} to {val_end}; origin blocks={origin_blocks}".format(
                fold=fold["fold"],
                train_rows=fold["n_train_rows"],
                train_start=fold["train_start_utc"],
                train_end=fold["train_end_utc"],
                val_rows=fold["n_validation_rows"],
                val_start=fold["validation_start_utc"],
                val_end=fold["validation_end_utc"],
                origin_blocks=fold["n_origin_blocks"],
            )
        )

    lines.extend(
        [
            "",
            "5. VALIDATION CHECKS",
            f"   - Passed: {validation['passed']}",
            f"   - Total validation rows: {validation['total_validation_rows']}",
            "   - Validation windows use tail of train_val: "
            f"{validation.get('validation_windows_use_tail_of_train_val', '')}",
            f"   - Train windows expanding: {validation['train_windows_expanding']}",
            "   - Train before validation all folds: "
            f"{validation['train_before_validation_all_folds']}",
            "   - Validation horizon multiple of forecast horizon: "
            f"{validation['validation_horizon_multiple_of_forecast_horizon']}",
            "",
            "6. METHODOLOGY NOTE",
            "   - Final test tetap untouched; CV hanya memakai train_val.",
            "   - Fold tidak dirandom dan urutan waktu dipertahankan.",
            "   - Validation window 168 jam siap dievaluasi dengan blok recursive 24 jam.",
            "",
            "7. TIME COST COMPUTING",
            f"   - CV split runtime seconds: {summary['runtime_seconds']}",
            f"   - Timestamp run UTC: {summary['timestamp_run_utc']}",
            "",
        ]
    )
    return "\n".join(lines)


def _summarize_time_spacing(index: pd.DatetimeIndex) -> dict[str, Any]:
    if len(index) <= 1:
        return {
            "expected_hourly_delta": "0 days 01:00:00",
            "non_hourly_delta_count": 0,
            "non_positive_delta_count": 0,
            "non_hourly_delta_examples": [],
        }

    deltas = index.to_series().diff().dropna()
    expected_delta = pd.Timedelta(hours=1)
    non_hourly_mask = deltas != expected_delta
    non_positive_mask = deltas <= pd.Timedelta(0)
    examples = []
    for timestamp, delta in deltas[non_hourly_mask].head(10).items():
        examples.append(
            {
                "timestamp": _format_one_timestamp(timestamp),
                "delta": str(delta),
            }
        )

    return {
        "expected_hourly_delta": str(expected_delta),
        "non_hourly_delta_count": int(non_hourly_mask.sum()),
        "non_positive_delta_count": int(non_positive_mask.sum()),
        "non_hourly_delta_examples": examples,
    }


def _coerce_integer(value: Any, *, argument_name: str, min_value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{argument_name} harus integer, bukan boolean.")
    if isinstance(value, (int, np.integer)):
        coerced = int(value)
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{argument_name} harus integer: {value}")
        coerced = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{argument_name} tidak boleh kosong.")
        coerced = int(stripped)
    else:
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{argument_name} harus integer: {value}") from exc

    if coerced < min_value:
        raise ValueError(f"{argument_name} harus >= {min_value}, ditemukan {coerced}.")
    return coerced


def _render_summary_text(summary: dict[str, Any]) -> str:
    source = summary["source"]
    train_val = summary["train_val"]
    final_test = summary["final_test"]
    validation = summary["validation"]

    lines = [
        "=" * 70,
        "HOLD-OUT TIME-BASED SPLIT SUMMARY - NYC Taxi Hourly Trip Count",
        "=" * 70,
        "",
        "1. OUTPUT STATUS",
        f"   - Status: {summary['status']}",
        f"   - Split policy: {summary['split_policy']}",
        f"   - Test days: {summary['test_days']}",
        f"   - Split cutoff UTC: {summary['split_cutoff_utc']}",
        "",
        "2. SOURCE DATA PERIOD",
        f"   - Rows: {source['n_rows']}",
        f"   - UTC start: {source['utc_start']}",
        f"   - UTC end: {source['utc_end']}",
        f"   - Local NYC start: {source['local_start']}",
        f"   - Local NYC end: {source['local_end']}",
        "",
        "3. TRAIN + VALIDATION PERIOD",
        f"   - Rows: {train_val['n_rows']}",
        f"   - UTC start: {train_val['utc_start']}",
        f"   - UTC end: {train_val['utc_end']}",
        f"   - Local NYC start: {train_val['local_start']}",
        f"   - Local NYC end: {train_val['local_end']}",
        f"   - Target mean: {train_val['target_mean']:.6f}",
        "",
        "4. FINAL TEST PERIOD",
        f"   - Rows: {final_test['n_rows']}",
        f"   - Expected rows for 30 days hourly: {summary['expected_final_test_hours']}",
        f"   - UTC start: {final_test['utc_start']}",
        f"   - UTC end: {final_test['utc_end']}",
        f"   - Local NYC start: {final_test['local_start']}",
        f"   - Local NYC end: {final_test['local_end']}",
        f"   - Target mean: {final_test['target_mean']:.6f}",
        "",
        "5. VALIDATION CHECKS",
        f"   - Chronological order: {validation['chronological_order']}",
        f"   - Overlap count: {validation['overlap_count']}",
        "   - Train max before final test min: "
        f"{validation['train_val_max_before_final_test_min']}",
        "   - Gap hours between train and final test: "
        f"{validation['gap_hours_between_train_and_test']}",
        "   - Final test rows equals expected hours: "
        f"{validation['observed_final_test_equals_expected_hours']}",
        f"   - Row count preserved: {validation.get('row_count_preserved', '')}",
        f"   - Index coverage preserved: {validation.get('index_coverage_preserved', '')}",
        "",
        "6. METHODOLOGY NOTE",
        "   - Final test dipisahkan sebelum tuning/model selection.",
        "   - Final test tidak boleh digunakan untuk hyperparameter tuning.",
        "   - Split berbasis UTC modeling index dan tetap menyimpan timestamp lokal NYC.",
        "",
        "7. TIME COST COMPUTING",
        f"   - Split runtime seconds: {summary['runtime_seconds']}",
        f"   - Timestamp run UTC: {summary['timestamp_run_utc']}",
        "",
    ]
    return "\n".join(lines)


def _format_timestamp_examples(
    timestamps: Union[pd.Series, pd.DatetimeIndex],
    limit: int = 10,
) -> list[str]:
    return [_format_one_timestamp(value) for value in list(timestamps)[:limit]]


def _format_one_timestamp(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    train_val, final_test, summary = run_holdout_split()
    print("Hold-out time-based split selesai.")
    print(f"Train+validation rows: {train_val.shape[0]}")
    print(f"Final test rows: {final_test.shape[0]}")
    print(f"Train+validation output: {TRAIN_VAL_PATH}")
    print(f"Final test output: {FINAL_TEST_PATH}")
    print(f"Summary: {HOLDOUT_SPLIT_SUMMARY_PATH}")
    print(f"Runtime seconds: {summary['runtime_seconds']}")


if __name__ == "__main__":
    main()


__all__ = [
    "build_cv_fold_metadata_frame",
    "load_split_timeseries",
    "make_expanding_window_splits",
    "make_holdout_split",
    "run_holdout_split",
    "run_time_series_cv_split",
    "save_holdout_metadata",
    "save_holdout_split",
    "save_holdout_summary",
    "save_time_series_cv_outputs",
    "validate_cv_splits",
    "validate_split_order",
]

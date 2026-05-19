"""
preprocessing.py

Tahap time-based preprocessing untuk dataset NYC Taxi hourly.

Modul ini menyiapkan data time series tanpa membuat fitur, tanpa split,
dan tanpa menyentuh final test. Fokusnya adalah parsing timestamp,
sorting kronologis, audit missing/duplicate hour, serta konversi timezone
dari waktu lokal NYC ke UTC untuk modeling.

Output utama:
- data/processed/full_preprocessed.csv
- outputs/preprocessing/summaries/preprocessing_summary.txt
- outputs/preprocessing/summaries/preprocessing_metadata.json
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd

from src.config import (
    FULL_PREPROCESSED_PATH,
    LOCAL_TZ,
    MODELING_TZ,
    PREPROCESSING_METADATA_PATH,
    PREPROCESSING_SUMMARY_PATH,
    RAW_DATA_PATH,
    TARGET_COL,
    TIMESTAMP_COL,
    ensure_dirs,
)
from src.data_loading import load_hourly_taxi_data
from src.tracking import log_runtime


LOCAL_NAIVE_TIMESTAMP_COL = "timestamp_local_naive"
LOCAL_TIMESTAMP_COL = "timestamp_local"
DST_AMBIGUOUS_COL = "is_dst_ambiguous_local_hour"


def parse_timestamp_column(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
) -> pd.DataFrame:
    """
    Parse kolom timestamp menjadi pandas datetime.

    Fungsi ini sengaja hanya melakukan parsing, bukan timezone conversion,
    agar error format timestamp bisa diketahui sebelum aturan timezone
    diterapkan.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Kolom timestamp tidak ditemukan: {timestamp_col}")

    parsed = df.copy()
    parsed_timestamp = pd.to_datetime(parsed[timestamp_col], errors="coerce")
    invalid_mask = parsed_timestamp.isna()

    if invalid_mask.any():
        sample_values = parsed.loc[invalid_mask, timestamp_col].head(5).tolist()
        raise ValueError(
            f"Terdapat {int(invalid_mask.sum())} timestamp yang gagal diparse. "
            f"Contoh nilai bermasalah: {sample_values}"
        )

    parsed[timestamp_col] = parsed_timestamp
    return parsed


def sort_by_timestamp(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
) -> pd.DataFrame:
    """
    Urutkan data secara kronologis dengan stable sort.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Kolom timestamp tidak ditemukan: {timestamp_col}")

    return df.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)


def check_duplicate_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
) -> dict[str, Any]:
    """
    Audit duplicate timestamp tanpa menghapus baris.
    """
    if timestamp_col in df.columns:
        timestamps = pd.Series(df[timestamp_col])
    elif df.index.name == timestamp_col:
        timestamps = pd.Series(df.index)
    else:
        raise ValueError(f"Kolom/index timestamp tidak ditemukan: {timestamp_col}")

    duplicate_mask = timestamps.duplicated(keep=False)
    duplicate_values = timestamps[duplicate_mask].drop_duplicates()

    return {
        "duplicate_count": int(duplicate_mask.sum()),
        "duplicate_unique_count": int(duplicate_values.shape[0]),
        "duplicate_examples": _format_timestamp_examples(duplicate_values),
    }


def check_missing_hours(
    timestamps: Union[pd.Series, pd.DatetimeIndex, Iterable[pd.Timestamp]],
    *,
    freq: str = "h",
) -> dict[str, Any]:
    """
    Audit missing hourly timestamps pada rentang min-max timestamp.

    Fungsi ini tidak melakukan reindex/imputation karena tahap preprocessing
    hanya boleh mengaudit missing terlebih dahulu.
    """
    timestamp_index = pd.DatetimeIndex(pd.Series(timestamps).dropna())

    if timestamp_index.empty:
        raise ValueError("Tidak ada timestamp valid untuk missing-hour check.")

    if timestamp_index.has_duplicates:
        unique_index = timestamp_index.drop_duplicates()
    else:
        unique_index = timestamp_index

    expected_index = pd.date_range(
        start=unique_index.min(),
        end=unique_index.max(),
        freq=freq,
        tz=unique_index.tz,
    )
    missing_index = expected_index.difference(unique_index)

    return {
        "expected_count": int(len(expected_index)),
        "observed_count": int(len(unique_index)),
        "missing_count": int(len(missing_index)),
        "missing_examples": _format_timestamp_examples(missing_index),
    }


def handle_timezone(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
    local_tz: str = LOCAL_TZ,
    modeling_tz: str = MODELING_TZ,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Buat timestamp lokal NYC dan timestamp UTC untuk modeling.

    Dataset mentah saat ini memakai naive timestamp yang merepresentasikan
    waktu lokal NYC. Untuk jam ambiguous saat DST fall-back, data hanya
    memiliki satu baris 01:00, sehingga `ambiguous='infer'` tidak cukup.
    Dalam kondisi itu, timestamp ambiguous dilokalisasi sebagai standard
    time (`ambiguous=False`) dan dicatat eksplisit dalam metadata.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Kolom timestamp tidak ditemukan: {timestamp_col}")

    converted = df.copy()
    timestamp_series = pd.Series(converted[timestamp_col])
    timezone_info = getattr(timestamp_series.dt, "tz", None)

    if timezone_info is None:
        local_naive = timestamp_series
        ambiguous_mask = _detect_ambiguous_local_hours(local_naive, local_tz)
        nonexistent_mask = _detect_nonexistent_local_hours(local_naive, local_tz)

        if nonexistent_mask.any():
            examples = _format_timestamp_examples(local_naive[nonexistent_mask])
            raise ValueError(
                "Data mengandung timestamp lokal NYC yang nonexistent akibat DST "
                f"spring-forward. Contoh: {examples}. Periksa agregasi data mentah."
            )

        local_timestamp = local_naive.dt.tz_localize(
            local_tz,
            ambiguous=False,
            nonexistent="raise",
        )
        timezone_strategy = "localized_naive_nyc_to_utc"
        ambiguous_policy = "ambiguous fall-back hours localized as standard time"
    else:
        local_timestamp = timestamp_series.dt.tz_convert(local_tz)
        local_naive = local_timestamp.dt.tz_localize(None)
        ambiguous_mask = pd.Series(False, index=converted.index)
        nonexistent_mask = pd.Series(False, index=converted.index)
        timezone_strategy = "converted_existing_timezone_to_nyc_and_utc"
        ambiguous_policy = "input timestamp already timezone-aware"

    utc_timestamp = local_timestamp.dt.tz_convert(modeling_tz)

    converted[LOCAL_NAIVE_TIMESTAMP_COL] = local_naive.to_numpy()
    converted[LOCAL_TIMESTAMP_COL] = local_timestamp
    converted[DST_AMBIGUOUS_COL] = ambiguous_mask.to_numpy(dtype=bool)
    converted[timestamp_col] = utc_timestamp

    metadata = {
        "timezone_strategy": timezone_strategy,
        "local_timezone": local_tz,
        "modeling_timezone": modeling_tz,
        "ambiguous_policy": ambiguous_policy,
        "ambiguous_local_hour_count": int(ambiguous_mask.sum()),
        "ambiguous_local_hour_examples": _format_timestamp_examples(
            local_naive[ambiguous_mask]
        ),
        "nonexistent_local_hour_count": int(nonexistent_mask.sum()),
        "nonexistent_local_hour_examples": _format_timestamp_examples(
            local_naive[nonexistent_mask]
        ),
    }
    return converted, metadata


def set_datetime_index(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
) -> pd.DataFrame:
    """
    Set timestamp UTC sebagai DatetimeIndex modeling.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Kolom timestamp tidak ditemukan: {timestamp_col}")

    indexed = df.set_index(timestamp_col, drop=True)
    indexed.index.name = timestamp_col

    if not isinstance(indexed.index, pd.DatetimeIndex):
        raise TypeError("Index modeling harus berupa pandas DatetimeIndex.")

    if not indexed.index.is_monotonic_increasing:
        raise ValueError("Index modeling belum monotonic increasing.")

    if indexed.index.has_duplicates:
        raise ValueError("Index modeling mengandung duplicate timestamp.")

    return indexed


def prepare_modeling_timeseries(
    df: pd.DataFrame,
    *,
    timestamp_col: str = TIMESTAMP_COL,
    target_col: str = TARGET_COL,
    local_tz: str = LOCAL_TZ,
    modeling_tz: str = MODELING_TZ,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Jalankan parsing, sorting, timezone conversion, dan audit time index.
    """
    parsed = parse_timestamp_column(df, timestamp_col=timestamp_col)
    parsed_duplicate_info = check_duplicate_timestamps(parsed, timestamp_col=timestamp_col)

    sorted_df = sort_by_timestamp(parsed, timestamp_col=timestamp_col)
    local_missing_info = check_missing_hours(sorted_df[timestamp_col], freq="h")

    converted, timezone_info = handle_timezone(
        sorted_df,
        timestamp_col=timestamp_col,
        local_tz=local_tz,
        modeling_tz=modeling_tz,
    )
    converted = sort_by_timestamp(converted, timestamp_col=timestamp_col)
    utc_missing_info = check_missing_hours(converted[timestamp_col], freq="h")
    utc_duplicate_info = check_duplicate_timestamps(converted, timestamp_col=timestamp_col)

    prepared = set_datetime_index(converted, timestamp_col=timestamp_col)

    target_missing_count = int(prepared[target_col].isna().sum())
    total_missing_values = int(prepared.isna().sum().sum())

    summary = {
        "stage": "time_based_preprocessing",
        "status": "success",
        "n_rows": int(prepared.shape[0]),
        "n_columns": int(prepared.shape[1]),
        "target_col": target_col,
        "target_missing_count": target_missing_count,
        "total_missing_values": total_missing_values,
        "time_range_local_start": _format_one_timestamp(prepared[LOCAL_TIMESTAMP_COL].min()),
        "time_range_local_end": _format_one_timestamp(prepared[LOCAL_TIMESTAMP_COL].max()),
        "time_range_utc_start": _format_one_timestamp(prepared.index.min()),
        "time_range_utc_end": _format_one_timestamp(prepared.index.max()),
        "local_duplicate_info": parsed_duplicate_info,
        "utc_duplicate_info": utc_duplicate_info,
        "local_missing_hour_info": local_missing_info,
        "utc_missing_hour_info": utc_missing_info,
        "timezone_info": timezone_info,
        "chronological_order": bool(prepared.index.is_monotonic_increasing),
        "modeling_index_timezone": str(prepared.index.tz),
        "output_columns": list(prepared.columns),
    }

    _validate_preprocessed_dataframe(prepared, summary)
    return prepared, summary


def run_time_based_preprocessing(
    input_path: Union[str, Path] = RAW_DATA_PATH,
    output_path: Union[str, Path] = FULL_PREPROCESSED_PATH,
    summary_path: Union[str, Path] = PREPROCESSING_SUMMARY_PATH,
    metadata_path: Union[str, Path] = PREPROCESSING_METADATA_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Entry point utama untuk menjalankan tahap 4 dari file mentah ke output.
    """
    ensure_dirs()
    start_time = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    status = "success"
    error_message = ""

    try:
        raw_df = load_hourly_taxi_data(input_path)
        prepared_df, summary = prepare_modeling_timeseries(raw_df)
        elapsed_seconds = time.perf_counter() - start_time
        summary["runtime_seconds"] = round(elapsed_seconds, 6)
        summary["timestamp_run_utc"] = started_at

        save_preprocessed_data(prepared_df, output_path)
        save_preprocessing_summary(summary, summary_path)
        save_preprocessing_metadata(summary, metadata_path)
        return prepared_df, summary
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        raise
    finally:
        total_runtime_seconds = round(time.perf_counter() - start_time, 6)
        _append_preprocessing_runtime_log(
            total_runtime_seconds=total_runtime_seconds,
            timestamp_run=started_at,
            status=status,
            error_message=error_message,
        )


def save_preprocessed_data(
    df: pd.DataFrame,
    output_path: Union[str, Path] = FULL_PREPROCESSED_PATH,
) -> None:
    """
    Simpan dataframe preprocessed dengan UTC timestamp sebagai kolom pertama.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=True)


def save_preprocessing_summary(
    summary: dict[str, Any],
    summary_path: Union[str, Path] = PREPROCESSING_SUMMARY_PATH,
) -> None:
    """
    Simpan ringkasan preprocessing dalam format teks yang mudah dibaca.
    """
    destination = Path(summary_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_render_summary_text(summary), encoding="utf-8")


def save_preprocessing_metadata(
    summary: dict[str, Any],
    metadata_path: Union[str, Path] = PREPROCESSING_METADATA_PATH,
) -> None:
    """
    Simpan metadata lengkap dalam JSON untuk audit/reproducibility.
    """
    destination = Path(metadata_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_to_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_preprocessed_timeseries(
    path: Union[str, Path] = FULL_PREPROCESSED_PATH,
    *,
    local_tz: str = LOCAL_TZ,
    modeling_tz: str = MODELING_TZ,
) -> pd.DataFrame:
    """
    Load output preprocessing dengan dtype timestamp yang stabil.

    Catatan: `timestamp_local` di CSV memiliki offset DST campuran (-05:00
    dan -04:00). Karena itu, kolom lokal direkonstruksi dari
    `timestamp_local_naive` agar kembali menjadi dtype
    datetime64[ns, America/New_York].
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"File preprocessed tidak ditemukan: {source}")

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
        raise ValueError(
            f"Kolom wajib tidak ditemukan pada data preprocessed: {missing_columns}"
        )

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
    if not loaded.index.is_monotonic_increasing:
        raise ValueError("Data preprocessed yang diload tidak kronologis.")
    if loaded.index.has_duplicates:
        raise ValueError("Data preprocessed yang diload mengandung duplicate timestamp.")
    if str(loaded.index.tz) != modeling_tz:
        raise ValueError(
            f"Index preprocessed harus timezone {modeling_tz}, ditemukan {loaded.index.tz}"
        )

    return loaded


def _validate_preprocessed_dataframe(
    df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """
    Validasi integrasi dasar sebelum output dianggap siap dipakai tahap lanjut.
    """
    if df.empty:
        raise ValueError("Data preprocessed kosong.")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Kolom target hilang setelah preprocessing: {TARGET_COL}")

    if LOCAL_TIMESTAMP_COL not in df.columns:
        raise ValueError(f"Kolom timezone lokal hilang: {LOCAL_TIMESTAMP_COL}")

    if LOCAL_NAIVE_TIMESTAMP_COL not in df.columns:
        raise ValueError(f"Kolom timestamp lokal naive hilang: {LOCAL_NAIVE_TIMESTAMP_COL}")

    if str(df.index.tz) != MODELING_TZ:
        raise ValueError(
            f"Index modeling harus timezone {MODELING_TZ}, ditemukan {df.index.tz}"
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError("Data preprocessed tidak terurut kronologis.")

    if df.index.has_duplicates:
        raise ValueError("Data preprocessed masih mengandung duplicate UTC timestamp.")

    if int(summary["utc_duplicate_info"]["duplicate_count"]) != 0:
        raise ValueError("Audit menemukan duplicate UTC timestamp.")


def _detect_ambiguous_local_hours(
    timestamps: pd.Series,
    local_tz: str,
) -> pd.Series:
    """
    Deteksi jam lokal ambiguous akibat DST fall-back.
    """
    localized = timestamps.dt.tz_localize(
        local_tz,
        ambiguous="NaT",
        nonexistent="shift_forward",
    )
    return localized.isna()


def _detect_nonexistent_local_hours(
    timestamps: pd.Series,
    local_tz: str,
) -> pd.Series:
    """
    Deteksi jam lokal nonexistent akibat DST spring-forward.
    """
    localized = timestamps.dt.tz_localize(
        local_tz,
        ambiguous=False,
        nonexistent="NaT",
    )
    return localized.isna()


def _append_preprocessing_runtime_log(
    *,
    total_runtime_seconds: float,
    timestamp_run: str,
    status: str,
    error_message: str,
) -> None:
    """
    Catat runtime tahap preprocessing ke CSV log umum.
    """
    record = {
        "timestamp_run": timestamp_run,
        "experiment_name": "time_based_preprocessing",
        "model_name": "none",
        "feature_set": "none",
        "fold": "",
        "parameter_set_id": "",
        "train_start": "",
        "train_end": "",
        "validation_start": "",
        "validation_end": "",
        "n_train_rows": "",
        "n_prediction_rows": "",
        "train_time_seconds": "",
        "prediction_time_seconds": "",
        "total_runtime_seconds": total_runtime_seconds,
        "status": status,
        "error_message": error_message,
    }
    log_runtime(record)


def _render_summary_text(summary: dict[str, Any]) -> str:
    local_missing = summary["local_missing_hour_info"]
    utc_missing = summary["utc_missing_hour_info"]
    local_duplicates = summary["local_duplicate_info"]
    utc_duplicates = summary["utc_duplicate_info"]
    timezone_info = summary["timezone_info"]

    lines = [
        "=" * 70,
        "TIME-BASED PREPROCESSING SUMMARY - NYC Taxi Hourly Trip Count",
        "=" * 70,
        "",
        "1. OUTPUT STATUS",
        f"   - Status: {summary['status']}",
        f"   - Rows: {summary['n_rows']}",
        f"   - Columns: {summary['n_columns']}",
        f"   - Target column: {summary['target_col']}",
        f"   - Target missing count: {summary['target_missing_count']}",
        f"   - Total missing values after preprocessing: {summary['total_missing_values']}",
        "",
        "2. TIME RANGE",
        f"   - Local NYC start: {summary['time_range_local_start']}",
        f"   - Local NYC end: {summary['time_range_local_end']}",
        f"   - UTC modeling start: {summary['time_range_utc_start']}",
        f"   - UTC modeling end: {summary['time_range_utc_end']}",
        f"   - Modeling index timezone: {summary['modeling_index_timezone']}",
        f"   - Chronological order: {summary['chronological_order']}",
        "",
        "3. TIMEZONE HANDLING",
        f"   - Strategy: {timezone_info['timezone_strategy']}",
        f"   - Local timezone: {timezone_info['local_timezone']}",
        f"   - Modeling timezone: {timezone_info['modeling_timezone']}",
        f"   - Ambiguous policy: {timezone_info['ambiguous_policy']}",
        f"   - Ambiguous local hour count: {timezone_info['ambiguous_local_hour_count']}",
        "   - Ambiguous local hour examples: "
        f"{timezone_info['ambiguous_local_hour_examples']}",
        f"   - Nonexistent local hour count: {timezone_info['nonexistent_local_hour_count']}",
        "",
        "4. DUPLICATE TIMESTAMP AUDIT",
        f"   - Local duplicate rows: {local_duplicates['duplicate_count']}",
        f"   - UTC duplicate rows: {utc_duplicates['duplicate_count']}",
        "",
        "5. MISSING HOUR AUDIT",
        "   - Local NYC hourly axis:",
        f"     observed={local_missing['observed_count']}, "
        f"expected={local_missing['expected_count']}, "
        f"missing={local_missing['missing_count']}",
        f"     examples={local_missing['missing_examples']}",
        "   - UTC modeling hourly axis:",
        f"     observed={utc_missing['observed_count']}, "
        f"expected={utc_missing['expected_count']}, "
        f"missing={utc_missing['missing_count']}",
        f"     examples={utc_missing['missing_examples']}",
        "",
        "6. METHODOLOGY NOTE",
        "   - Data mentah memakai timestamp lokal NYC dan aman menjadi input tahap 4.",
        "   - Output tetap menyimpan timestamp lokal untuk calendar features.",
        "   - Index modeling memakai UTC sesuai timezone policy proyek.",
        "   - Missing/ambiguous DST dicatat, tidak diimputasi pada tahap ini.",
        "",
        "7. TIME COST COMPUTING",
        f"   - Preprocessing runtime seconds: {summary['runtime_seconds']}",
        f"   - Timestamp run UTC: {summary['timestamp_run_utc']}",
        "",
    ]
    return "\n".join(lines)


def _format_timestamp_examples(
    timestamps: Union[pd.Series, pd.DatetimeIndex, Iterable[Any]],
    limit: int = 10,
) -> list[str]:
    timestamp_list = list(timestamps)[:limit]
    return [_format_one_timestamp(value) for value in timestamp_list]


def _format_one_timestamp(value: Any) -> str:
    if pd.isna(value):
        return "NaT"
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
    """
    Jalankan preprocessing saat modul dipanggil sebagai script.
    """
    prepared_df, summary = run_time_based_preprocessing()
    print("Time-based preprocessing selesai.")
    print(f"Rows: {prepared_df.shape[0]}")
    print(f"Output: {FULL_PREPROCESSED_PATH}")
    print(f"Summary: {PREPROCESSING_SUMMARY_PATH}")
    print(f"Runtime seconds: {summary['runtime_seconds']}")


if __name__ == "__main__":
    main()


__all__ = [
    "DST_AMBIGUOUS_COL",
    "LOCAL_NAIVE_TIMESTAMP_COL",
    "LOCAL_TIMESTAMP_COL",
    "check_duplicate_timestamps",
    "check_missing_hours",
    "handle_timezone",
    "load_preprocessed_timeseries",
    "parse_timestamp_column",
    "prepare_modeling_timeseries",
    "run_time_based_preprocessing",
    "save_preprocessed_data",
    "set_datetime_index",
    "sort_by_timestamp",
]

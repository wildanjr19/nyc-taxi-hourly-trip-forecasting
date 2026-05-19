"""
features.py

Tahap feature engineering untuk XGBoost pada penelitian NYC Taxi hourly.

Modul ini membuat dua feature set sesuai FEATURE_SET.md:
- XGBoost-Basic: lag minimal dan calendar feature sederhana.
- XGBoost-Advanced: lag lebih kaya, shifted rolling statistics, dan
  calendar feature tambahan.

Catatan metodologi:
- Tidak ada shuffle.
- Feature dibuat dari train_val, bukan final_test.
- Lag memakai target masa lalu melalui shift.
- Rolling feature selalu memakai target.shift(1) sebelum rolling.
- Calendar feature dibuat dari timestamp lokal NYC.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS_PATH,
    FEATURE_ENGINEERING_METADATA_PATH,
    FEATURE_ENGINEERING_SUMMARY_PATH,
    FORECAST_HORIZON,
    LOCAL_TZ,
    MODELING_TZ,
    TARGET_COL,
    TIMESTAMP_COL,
    TRAIN_VAL_PATH,
    XGB_ADVANCED_CALENDAR,
    XGB_ADVANCED_FEATURES_PATH,
    XGB_ADVANCED_LAGS,
    XGB_ADVANCED_ROLLING_STD_WINDOWS,
    XGB_ADVANCED_ROLLING_WINDOWS,
    XGB_BASIC_CALENDAR,
    XGB_BASIC_FEATURES_PATH,
    XGB_BASIC_LAGS,
    ensure_dirs,
)
from src.preprocessing import LOCAL_NAIVE_TIMESTAMP_COL, LOCAL_TIMESTAMP_COL
from src.splits import load_split_timeseries
from src.tracking import log_runtime


BASIC_FEATURE_SET = "xgb_basic"
ADVANCED_FEATURE_SET = "xgb_advanced"


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    lags: Sequence[int] = XGB_BASIC_LAGS,
) -> pd.DataFrame:
    """
    Tambahkan lag features berbasis target masa lalu.
    """
    _validate_positive_integers(lags, argument_name="lags")
    _validate_target_column(df, target_col=target_col)

    engineered = df.copy()
    target = pd.to_numeric(engineered[target_col], errors="raise")

    for lag in lags:
        engineered[f"lag_{lag}"] = target.shift(lag)

    return engineered


def add_shifted_rolling_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    mean_windows: Sequence[int] = XGB_ADVANCED_ROLLING_WINDOWS,
    std_windows: Sequence[int] = XGB_ADVANCED_ROLLING_STD_WINDOWS,
) -> pd.DataFrame:
    """
    Tambahkan rolling statistics tanpa leakage.

    Pola yang digunakan:
        shifted = target.shift(1)
        rolling_feature = shifted.rolling(window).mean()

    Dengan pola ini, feature pada timestamp t hanya memakai target sebelum t.
    """
    _validate_positive_integers(mean_windows, argument_name="mean_windows")
    _validate_positive_integers(std_windows, argument_name="std_windows")
    _validate_target_column(df, target_col=target_col)

    engineered = df.copy()
    target = pd.to_numeric(engineered[target_col], errors="raise")
    shifted = target.shift(1)

    for window in mean_windows:
        engineered[f"rolling_mean_{window}"] = shifted.rolling(
            window=window,
            min_periods=window,
        ).mean()

    for window in std_windows:
        engineered[f"rolling_std_{window}"] = shifted.rolling(
            window=window,
            min_periods=window,
        ).std()

    return engineered


def add_calendar_features(
    df: pd.DataFrame,
    calendar_features: Sequence[str],
    *,
    local_timezone: str = LOCAL_TZ,
) -> pd.DataFrame:
    """
    Tambahkan calendar features berdasarkan waktu lokal NYC.
    """
    supported_features = {"hour", "day_of_week", "is_weekend", "month"}
    requested_features = set(calendar_features)
    unsupported = sorted(requested_features.difference(supported_features))
    if unsupported:
        raise ValueError(f"Calendar features tidak didukung: {unsupported}")

    engineered = df.copy()
    local_naive = _get_local_naive_timestamps(engineered, local_timezone=local_timezone)

    if "hour" in requested_features:
        engineered["hour"] = local_naive.dt.hour.astype("int16")
    if "day_of_week" in requested_features:
        engineered["day_of_week"] = local_naive.dt.dayofweek.astype("int16")
    if "is_weekend" in requested_features:
        engineered["is_weekend"] = local_naive.dt.dayofweek.isin([5, 6]).astype("int8")
    if "month" in requested_features:
        engineered["month"] = local_naive.dt.month.astype("int16")

    return engineered


def make_xgb_basic_features(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Buat feature matrix XGBoost-Basic.
    """
    _validate_source_timeseries(df, target_col=target_col)

    engineered = add_lag_features(df, target_col=target_col, lags=XGB_BASIC_LAGS)
    engineered = add_calendar_features(engineered, XGB_BASIC_CALENDAR)

    feature_columns = get_feature_columns(BASIC_FEATURE_SET)
    leakage_validation = validate_no_future_leakage(
        engineered,
        feature_set=BASIC_FEATURE_SET,
        target_col=target_col,
    )
    if not leakage_validation["passed"]:
        raise ValueError(f"Validasi leakage gagal untuk {BASIC_FEATURE_SET}.")

    return _finalize_feature_matrix(
        engineered,
        feature_columns=feature_columns,
        target_col=target_col,
        drop_na=drop_na,
        feature_set=BASIC_FEATURE_SET,
    )


def make_xgb_advanced_features(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Buat feature matrix XGBoost-Advanced.
    """
    _validate_source_timeseries(df, target_col=target_col)

    engineered = add_lag_features(df, target_col=target_col, lags=XGB_ADVANCED_LAGS)
    engineered = add_shifted_rolling_features(
        engineered,
        target_col=target_col,
        mean_windows=XGB_ADVANCED_ROLLING_WINDOWS,
        std_windows=XGB_ADVANCED_ROLLING_STD_WINDOWS,
    )
    engineered = add_calendar_features(engineered, XGB_ADVANCED_CALENDAR)

    feature_columns = get_feature_columns(ADVANCED_FEATURE_SET)
    leakage_validation = validate_no_future_leakage(
        engineered,
        feature_set=ADVANCED_FEATURE_SET,
        target_col=target_col,
    )
    if not leakage_validation["passed"]:
        raise ValueError(f"Validasi leakage gagal untuk {ADVANCED_FEATURE_SET}.")

    return _finalize_feature_matrix(
        engineered,
        feature_columns=feature_columns,
        target_col=target_col,
        drop_na=drop_na,
        feature_set=ADVANCED_FEATURE_SET,
    )


def drop_rows_with_feature_na(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Drop baris yang belum memiliki cukup history untuk lag/rolling feature.
    """
    required_columns = [target_col, *feature_columns]
    missing_columns = sorted(set(required_columns).difference(df.columns))
    if missing_columns:
        raise ValueError(f"Kolom feature wajib tidak ditemukan: {missing_columns}")

    cleaned = df.dropna(subset=required_columns).copy()
    if cleaned.empty:
        raise ValueError(
            "Feature matrix kosong setelah drop NA. "
            "Periksa panjang data, lag, dan rolling window."
        )
    return cleaned


def get_feature_columns(feature_set: str) -> list[str]:
    """
    Ambil daftar kolom fitur yang sah untuk feature set tertentu.
    """
    normalized = _normalize_feature_set(feature_set)

    if normalized == BASIC_FEATURE_SET:
        return [
            *(f"lag_{lag}" for lag in XGB_BASIC_LAGS),
            *XGB_BASIC_CALENDAR,
        ]

    if normalized == ADVANCED_FEATURE_SET:
        return [
            *(f"lag_{lag}" for lag in XGB_ADVANCED_LAGS),
            *(f"rolling_mean_{window}" for window in XGB_ADVANCED_ROLLING_WINDOWS),
            *(f"rolling_std_{window}" for window in XGB_ADVANCED_ROLLING_STD_WINDOWS),
            *XGB_ADVANCED_CALENDAR,
        ]

    raise ValueError(f"Feature set tidak dikenal: {feature_set}")


def validate_no_future_leakage(
    df: pd.DataFrame,
    *,
    feature_set: str,
    target_col: str = TARGET_COL,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """
    Validasi bahwa lag dan rolling feature memakai target masa lalu saja.
    """
    normalized = _normalize_feature_set(feature_set)
    if normalized == BASIC_FEATURE_SET:
        lags = XGB_BASIC_LAGS
        mean_windows: Sequence[int] = []
        std_windows: Sequence[int] = []
    elif normalized == ADVANCED_FEATURE_SET:
        lags = XGB_ADVANCED_LAGS
        mean_windows = XGB_ADVANCED_ROLLING_WINDOWS
        std_windows = XGB_ADVANCED_ROLLING_STD_WINDOWS
    else:
        raise ValueError(f"Feature set tidak dikenal: {feature_set}")

    _validate_target_column(df, target_col=target_col)
    target = pd.to_numeric(df[target_col], errors="raise")
    checks: dict[str, Any] = {}

    for lag in lags:
        column = f"lag_{lag}"
        if column not in df.columns:
            raise ValueError(f"Kolom lag tidak ditemukan: {column}")
        expected = target.shift(lag)
        checks[column] = _compare_series_with_nan(
            df[column],
            expected,
            rtol=rtol,
            atol=atol,
        )

    shifted = target.shift(1)
    for window in mean_windows:
        column = f"rolling_mean_{window}"
        if column not in df.columns:
            raise ValueError(f"Kolom rolling mean tidak ditemukan: {column}")
        expected = shifted.rolling(window=window, min_periods=window).mean()
        checks[column] = _compare_series_with_nan(
            df[column],
            expected,
            rtol=rtol,
            atol=atol,
        )

    for window in std_windows:
        column = f"rolling_std_{window}"
        if column not in df.columns:
            raise ValueError(f"Kolom rolling std tidak ditemukan: {column}")
        expected = shifted.rolling(window=window, min_periods=window).std()
        checks[column] = _compare_series_with_nan(
            df[column],
            expected,
            rtol=rtol,
            atol=atol,
        )

    failed_checks = [
        name for name, result in checks.items() if not result["passed"]
    ]

    return {
        "feature_set": normalized,
        "passed": len(failed_checks) == 0,
        "failed_checks": failed_checks,
        "checks": checks,
        "rolling_rule": "target.shift(1).rolling(window)",
    }


def run_feature_engineering(
    *,
    input_path: Union[str, Path] = TRAIN_VAL_PATH,
    basic_output_path: Union[str, Path] = XGB_BASIC_FEATURES_PATH,
    advanced_output_path: Union[str, Path] = XGB_ADVANCED_FEATURES_PATH,
    summary_path: Union[str, Path] = FEATURE_ENGINEERING_SUMMARY_PATH,
    metadata_path: Union[str, Path] = FEATURE_ENGINEERING_METADATA_PATH,
    feature_columns_path: Union[str, Path] = FEATURE_COLUMNS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Entry point tahap 6 untuk membuat feature matrix dari train_val saja.
    """
    ensure_dirs()
    start_time = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    status = "success"
    error_message = ""
    source_rows: Union[int, str] = ""
    output_rows: Union[int, str] = ""

    try:
        train_val = load_split_timeseries(input_path)
        source_rows = int(train_val.shape[0])

        basic = make_xgb_basic_features(train_val)
        advanced = make_xgb_advanced_features(train_val)
        output_rows = int(advanced.shape[0])

        basic_leakage_validation = validate_no_future_leakage(
            add_calendar_features(
                add_lag_features(train_val, lags=XGB_BASIC_LAGS),
                XGB_BASIC_CALENDAR,
            ),
            feature_set=BASIC_FEATURE_SET,
        )
        advanced_leakage_validation = validate_no_future_leakage(
            add_calendar_features(
                add_shifted_rolling_features(
                    add_lag_features(train_val, lags=XGB_ADVANCED_LAGS),
                    mean_windows=XGB_ADVANCED_ROLLING_WINDOWS,
                    std_windows=XGB_ADVANCED_ROLLING_STD_WINDOWS,
                ),
                XGB_ADVANCED_CALENDAR,
            ),
            feature_set=ADVANCED_FEATURE_SET,
        )

        elapsed_seconds = time.perf_counter() - start_time
        summary = _build_feature_summary(
            source_df=train_val,
            basic_df=basic,
            advanced_df=advanced,
            basic_leakage_validation=basic_leakage_validation,
            advanced_leakage_validation=advanced_leakage_validation,
            runtime_seconds=round(elapsed_seconds, 6),
            timestamp_run_utc=started_at,
            input_path=input_path,
            basic_output_path=basic_output_path,
            advanced_output_path=advanced_output_path,
        )

        save_feature_matrix(basic, basic_output_path)
        save_feature_matrix(advanced, advanced_output_path)
        save_feature_columns(feature_columns_path)
        save_feature_engineering_summary(summary, summary_path)
        save_feature_engineering_metadata(summary, metadata_path)

        return basic, advanced, summary
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        raise
    finally:
        total_runtime_seconds = round(time.perf_counter() - start_time, 6)
        _append_feature_runtime_log(
            total_runtime_seconds=total_runtime_seconds,
            timestamp_run=started_at,
            status=status,
            error_message=error_message,
            n_train_rows=source_rows,
            n_prediction_rows=output_rows,
        )


def save_feature_matrix(
    df: pd.DataFrame,
    output_path: Union[str, Path],
) -> None:
    """
    Simpan feature matrix dengan UTC timestamp sebagai index/kolom pertama.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=True)


def save_feature_columns(
    output_path: Union[str, Path] = FEATURE_COLUMNS_PATH,
) -> None:
    """
    Simpan daftar feature columns agar tuning memakai kolom yang eksplisit.
    """
    payload = {
        BASIC_FEATURE_SET: get_feature_columns(BASIC_FEATURE_SET),
        ADVANCED_FEATURE_SET: get_feature_columns(ADVANCED_FEATURE_SET),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_feature_engineering_summary(
    summary: dict[str, Any],
    summary_path: Union[str, Path] = FEATURE_ENGINEERING_SUMMARY_PATH,
) -> None:
    """
    Simpan ringkasan feature engineering dalam format teks.
    """
    destination = Path(summary_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_render_summary_text(summary), encoding="utf-8")


def save_feature_engineering_metadata(
    summary: dict[str, Any],
    metadata_path: Union[str, Path] = FEATURE_ENGINEERING_METADATA_PATH,
) -> None:
    """
    Simpan metadata lengkap feature engineering dalam JSON.
    """
    destination = Path(metadata_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_to_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _finalize_feature_matrix(
    engineered: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_col: str,
    drop_na: bool,
    feature_set: str,
) -> pd.DataFrame:
    missing_columns = sorted(set(feature_columns).difference(engineered.columns))
    if missing_columns:
        raise ValueError(f"Kolom fitur belum dibuat untuk {feature_set}: {missing_columns}")

    selected = engineered[[target_col, *feature_columns]].copy()
    if drop_na:
        selected = drop_rows_with_feature_na(
            selected,
            feature_columns,
            target_col=target_col,
        )

    _validate_feature_matrix(
        selected,
        feature_columns=feature_columns,
        target_col=target_col,
        feature_set=feature_set,
    )
    return selected


def _validate_source_timeseries(
    df: pd.DataFrame,
    *,
    target_col: str,
) -> None:
    if df.empty:
        raise ValueError("Data input feature engineering kosong.")
    _validate_target_column(df, target_col=target_col)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Feature engineering membutuhkan DatetimeIndex.")
    if str(df.index.tz) != MODELING_TZ:
        raise ValueError(
            f"Index feature engineering harus timezone {MODELING_TZ}, "
            f"ditemukan {df.index.tz}"
        )
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index feature engineering tidak chronological.")
    if df.index.has_duplicates:
        raise ValueError("Index feature engineering mengandung duplicate timestamp.")
    if df[target_col].isna().any():
        raise ValueError(f"Kolom target {target_col} mengandung missing value.")

    deltas = df.index.to_series().diff().dropna()
    non_positive_delta_count = int((deltas <= pd.Timedelta(0)).sum())
    if non_positive_delta_count > 0:
        raise ValueError(
            "Data input memiliki timestamp yang tidak bergerak maju. "
            f"non_positive_delta_count={non_positive_delta_count}"
        )


def _validate_feature_matrix(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_col: str,
    feature_set: str,
) -> None:
    _validate_source_timeseries(df, target_col=target_col)
    missing_columns = sorted(set(feature_columns).difference(df.columns))
    if missing_columns:
        raise ValueError(f"Kolom fitur hilang pada {feature_set}: {missing_columns}")
    missing_feature_values = int(df[list(feature_columns)].isna().sum().sum())
    if missing_feature_values > 0:
        raise ValueError(
            f"Feature matrix {feature_set} masih memiliki missing feature values: "
            f"{missing_feature_values}"
        )


def _validate_target_column(df: pd.DataFrame, *, target_col: str) -> None:
    if target_col not in df.columns:
        raise ValueError(f"Kolom target tidak ditemukan: {target_col}")
    pd.to_numeric(df[target_col], errors="raise")


def _validate_positive_integers(
    values: Sequence[int],
    *,
    argument_name: str,
) -> None:
    if not values:
        raise ValueError(f"{argument_name} tidak boleh kosong.")
    invalid_values = [value for value in values if int(value) <= 0]
    if invalid_values:
        raise ValueError(f"{argument_name} harus berisi integer positif: {invalid_values}")


def _get_local_naive_timestamps(
    df: pd.DataFrame,
    *,
    local_timezone: str,
) -> pd.Series:
    if LOCAL_NAIVE_TIMESTAMP_COL in df.columns:
        local_naive = pd.to_datetime(
            df[LOCAL_NAIVE_TIMESTAMP_COL],
            errors="raise",
        )
        return pd.Series(local_naive, index=df.index)

    if LOCAL_TIMESTAMP_COL in df.columns:
        local_utc = pd.to_datetime(
            df[LOCAL_TIMESTAMP_COL],
            utc=True,
            errors="raise",
        )
        local_naive = local_utc.dt.tz_convert(local_timezone).dt.tz_localize(None)
        return pd.Series(local_naive, index=df.index)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "Calendar features membutuhkan timestamp lokal atau DatetimeIndex."
        )
    if df.index.tz is None:
        raise ValueError("DatetimeIndex harus timezone-aware untuk calendar features.")

    local_index = df.index.tz_convert(local_timezone).tz_localize(None)
    return pd.Series(local_index, index=df.index)


def _compare_series_with_nan(
    actual: pd.Series,
    expected: pd.Series,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    actual_values = pd.to_numeric(actual, errors="raise").to_numpy(dtype=float)
    expected_values = pd.to_numeric(expected, errors="raise").to_numpy(dtype=float)
    is_same = np.isclose(
        actual_values,
        expected_values,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    mismatch_count = int((~is_same).sum())
    first_mismatch_position: Optional[int] = None
    if mismatch_count > 0:
        first_mismatch_position = int(np.flatnonzero(~is_same)[0])

    return {
        "passed": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "first_mismatch_position": first_mismatch_position,
    }


def _normalize_feature_set(feature_set: str) -> str:
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
        raise ValueError(f"Feature set tidak dikenal: {feature_set}")
    return aliases[normalized]


def _build_feature_summary(
    *,
    source_df: pd.DataFrame,
    basic_df: pd.DataFrame,
    advanced_df: pd.DataFrame,
    basic_leakage_validation: dict[str, Any],
    advanced_leakage_validation: dict[str, Any],
    runtime_seconds: float,
    timestamp_run_utc: str,
    input_path: Union[str, Path],
    basic_output_path: Union[str, Path],
    advanced_output_path: Union[str, Path],
) -> dict[str, Any]:
    basic_feature_columns = get_feature_columns(BASIC_FEATURE_SET)
    advanced_feature_columns = get_feature_columns(ADVANCED_FEATURE_SET)

    return {
        "stage": "feature_engineering",
        "status": "success",
        "input_path": str(input_path),
        "outputs": {
            BASIC_FEATURE_SET: str(basic_output_path),
            ADVANCED_FEATURE_SET: str(advanced_output_path),
            "feature_columns": str(FEATURE_COLUMNS_PATH),
        },
        "source": _summarize_feature_period(source_df),
        "source_time_spacing": _summarize_time_spacing(source_df.index),
        BASIC_FEATURE_SET: {
            "n_rows": int(basic_df.shape[0]),
            "n_columns": int(basic_df.shape[1]),
            "dropped_rows_due_to_history": int(source_df.shape[0] - basic_df.shape[0]),
            "feature_columns": basic_feature_columns,
            "n_feature_columns": int(len(basic_feature_columns)),
            "period": _summarize_feature_period(basic_df),
            "time_spacing": _summarize_time_spacing(basic_df.index),
            "leakage_validation": basic_leakage_validation,
        },
        ADVANCED_FEATURE_SET: {
            "n_rows": int(advanced_df.shape[0]),
            "n_columns": int(advanced_df.shape[1]),
            "dropped_rows_due_to_history": int(source_df.shape[0] - advanced_df.shape[0]),
            "feature_columns": advanced_feature_columns,
            "n_feature_columns": int(len(advanced_feature_columns)),
            "period": _summarize_feature_period(advanced_df),
            "time_spacing": _summarize_time_spacing(advanced_df.index),
            "leakage_validation": advanced_leakage_validation,
        },
        "methodology_note": [
            "Feature matrix dibuat dari train_val saja.",
            "Final test tidak diload atau digunakan pada tahap feature engineering ini.",
            "Lag features memakai target.shift(lag).",
            "Rolling features memakai target.shift(1).rolling(window).",
            "Calendar features dibuat dari timestamp lokal NYC.",
            "Output hanya menyimpan target dan kolom fitur eksplisit.",
        ],
        "forecast_horizon_hours": int(FORECAST_HORIZON),
        "runtime_seconds": runtime_seconds,
        "timestamp_run_utc": timestamp_run_utc,
    }


def _summarize_feature_period(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "n_rows": 0,
            "utc_start": None,
            "utc_end": None,
            "target_min": None,
            "target_max": None,
            "target_mean": None,
        }

    return {
        "n_rows": int(df.shape[0]),
        "utc_start": _format_one_timestamp(df.index.min()),
        "utc_end": _format_one_timestamp(df.index.max()),
        "target_min": float(df[TARGET_COL].min()),
        "target_max": float(df[TARGET_COL].max()),
        "target_mean": float(df[TARGET_COL].mean()),
    }


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


def _append_feature_runtime_log(
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
        "experiment_name": "feature_engineering",
        "model_name": "xgboost",
        "feature_set": "basic_and_advanced",
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


def _render_summary_text(summary: dict[str, Any]) -> str:
    source = summary["source"]
    source_spacing = summary["source_time_spacing"]
    basic = summary[BASIC_FEATURE_SET]
    advanced = summary[ADVANCED_FEATURE_SET]

    lines = [
        "=" * 70,
        "FEATURE ENGINEERING SUMMARY - NYC Taxi Hourly Trip Count",
        "=" * 70,
        "",
        "1. OUTPUT STATUS",
        f"   - Status: {summary['status']}",
        f"   - Input path: {summary['input_path']}",
        f"   - Basic output: {summary['outputs'][BASIC_FEATURE_SET]}",
        f"   - Advanced output: {summary['outputs'][ADVANCED_FEATURE_SET]}",
        f"   - Feature columns: {summary['outputs']['feature_columns']}",
        "",
        "2. SOURCE TRAIN_VAL PERIOD",
        f"   - Rows: {source['n_rows']}",
        f"   - UTC start: {source['utc_start']}",
        f"   - UTC end: {source['utc_end']}",
        f"   - Target mean: {source['target_mean']:.6f}",
        f"   - Non-hourly UTC gaps: {source_spacing['non_hourly_delta_count']}",
        "",
        "3. XGBOOST-BASIC FEATURES",
        f"   - Rows: {basic['n_rows']}",
        f"   - Columns: {basic['n_columns']}",
        f"   - Feature columns: {basic['n_feature_columns']}",
        f"   - Dropped rows due to history: {basic['dropped_rows_due_to_history']}",
        f"   - UTC start after drop: {basic['period']['utc_start']}",
        f"   - UTC end after drop: {basic['period']['utc_end']}",
        f"   - Non-hourly UTC gaps: {basic['time_spacing']['non_hourly_delta_count']}",
        f"   - Leakage validation passed: {basic['leakage_validation']['passed']}",
        f"   - Features: {', '.join(basic['feature_columns'])}",
        "",
        "4. XGBOOST-ADVANCED FEATURES",
        f"   - Rows: {advanced['n_rows']}",
        f"   - Columns: {advanced['n_columns']}",
        f"   - Feature columns: {advanced['n_feature_columns']}",
        f"   - Dropped rows due to history: {advanced['dropped_rows_due_to_history']}",
        f"   - UTC start after drop: {advanced['period']['utc_start']}",
        f"   - UTC end after drop: {advanced['period']['utc_end']}",
        "   - Non-hourly UTC gaps: "
        f"{advanced['time_spacing']['non_hourly_delta_count']}",
        f"   - Leakage validation passed: {advanced['leakage_validation']['passed']}",
        f"   - Features: {', '.join(advanced['feature_columns'])}",
        "",
        "5. LEAKAGE PREVENTION NOTE",
        "   - Lag features dibuat dengan target.shift(lag).",
        "   - Rolling features dibuat dengan target.shift(1) sebelum rolling.",
        "   - Calendar features memakai waktu lokal NYC, bukan UTC.",
        "   - Final test tidak digunakan pada tahap ini.",
        "   - Gap UTC non-hourly dicatat sebagai audit DST/preprocessing, bukan dipakai",
        "     untuk mengakses nilai masa depan.",
        "",
        "6. TIME COST COMPUTING",
        f"   - Feature engineering runtime seconds: {summary['runtime_seconds']}",
        f"   - Timestamp run UTC: {summary['timestamp_run_utc']}",
        "",
    ]
    return "\n".join(lines)


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
    basic, advanced, summary = run_feature_engineering()
    print("Feature engineering selesai.")
    print(f"XGBoost-Basic rows: {basic.shape[0]}")
    print(f"XGBoost-Advanced rows: {advanced.shape[0]}")
    print(f"Basic output: {XGB_BASIC_FEATURES_PATH}")
    print(f"Advanced output: {XGB_ADVANCED_FEATURES_PATH}")
    print(f"Summary: {FEATURE_ENGINEERING_SUMMARY_PATH}")
    print(f"Runtime seconds: {summary['runtime_seconds']}")


if __name__ == "__main__":
    main()


__all__ = [
    "ADVANCED_FEATURE_SET",
    "BASIC_FEATURE_SET",
    "add_calendar_features",
    "add_lag_features",
    "add_shifted_rolling_features",
    "drop_rows_with_feature_na",
    "get_feature_columns",
    "make_xgb_advanced_features",
    "make_xgb_basic_features",
    "run_feature_engineering",
    "save_feature_columns",
    "save_feature_engineering_metadata",
    "save_feature_engineering_summary",
    "save_feature_matrix",
    "validate_no_future_leakage",
]

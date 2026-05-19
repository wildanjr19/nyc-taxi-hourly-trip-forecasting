"""
forecasting.py

Mekanisme forecasting reusable untuk penelitian NYC Taxi hourly.

Fokus tahap ini adalah recursive forecasting untuk model machine learning
berbasis lag/rolling feature seperti XGBoost. Modul ini sengaja tidak
melatih model; ia hanya membangun fitur next-step dari history masa lalu,
memanggil model.predict(), lalu memasukkan prediksi ke history sementara
untuk step berikutnya.

Prinsip leakage prevention:
- History input harus chronological dan timezone UTC.
- Fitur lag/rolling hanya dibuat dari target pada history yang sudah ada.
- Aktual validation/test pada horizon yang sedang diprediksi tidak pernah
  dipakai untuk membentuk fitur di dalam blok recursive.
- Calendar features dibuat dari timestamp lokal NYC yang diturunkan dari
  timestamp UTC target prediksi.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.config import (
    FORECAST_HORIZON,
    LOCAL_TZ,
    MODELING_TZ,
    TARGET_COL,
    TIMESTAMP_COL,
    XGB_ADVANCED_CALENDAR,
    XGB_ADVANCED_LAGS,
    XGB_ADVANCED_ROLLING_STD_WINDOWS,
    XGB_ADVANCED_ROLLING_WINDOWS,
    XGB_BASIC_CALENDAR,
    XGB_BASIC_LAGS,
)
from src.features import ADVANCED_FEATURE_SET, BASIC_FEATURE_SET, get_feature_columns


PREDICTION_COL = "predicted"
ACTUAL_COL = "actual"


def build_next_step_features(
    history: pd.DataFrame,
    timestamp: Union[str, pd.Timestamp],
    feature_set: str = BASIC_FEATURE_SET,
    *,
    target_col: str = TARGET_COL,
    local_timezone: str = LOCAL_TZ,
) -> pd.DataFrame:
    """
    Bangun feature row untuk satu timestamp prediksi berikutnya.

    Parameter `history` hanya perlu berisi target masa lalu. Jika history juga
    memiliki kolom lain, kolom tersebut diabaikan agar fitur next-step tetap
    konsisten dengan definisi di `src.features`.
    """
    normalized = normalize_feature_set(feature_set)
    feature_columns = get_feature_columns(normalized)
    forecast_timestamp = _coerce_utc_timestamp(timestamp)
    target_history = _validate_history(history, target_col=target_col)

    if forecast_timestamp <= target_history.index.max():
        raise ValueError(
            "Timestamp prediksi harus berada setelah timestamp terakhir history. "
            f"timestamp={forecast_timestamp}, history_end={target_history.index.max()}"
        )

    feature_values: dict[str, Any] = {}
    lags, mean_windows, std_windows, calendar_features = _feature_spec(normalized)

    values = target_history[target_col]
    for lag in lags:
        feature_values[f"lag_{lag}"] = _last_value_at_lag(values, lag)

    for window in mean_windows:
        feature_values[f"rolling_mean_{window}"] = _rolling_mean(values, window)

    for window in std_windows:
        feature_values[f"rolling_std_{window}"] = _rolling_std(values, window)

    feature_values.update(
        _calendar_feature_values(
            forecast_timestamp,
            calendar_features=calendar_features,
            local_timezone=local_timezone,
        )
    )

    missing_columns = sorted(set(feature_columns).difference(feature_values))
    if missing_columns:
        raise ValueError(
            f"Fitur next-step belum lengkap untuk {normalized}: {missing_columns}"
        )

    feature_row = pd.DataFrame(
        [[feature_values[column] for column in feature_columns]],
        index=pd.DatetimeIndex([forecast_timestamp], name=TIMESTAMP_COL),
        columns=feature_columns,
    )
    _validate_feature_row(feature_row, feature_set=normalized)
    return feature_row


def recursive_forecast_xgb(
    model: Any,
    history_df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
    feature_set: str = BASIC_FEATURE_SET,
    *,
    future_index: Optional[Sequence[Union[str, pd.Timestamp]]] = None,
    target_col: str = TARGET_COL,
    model_name: str = "xgboost",
    forecast_origin: Optional[Union[str, pd.Timestamp]] = None,
    fold: Optional[Union[int, str]] = None,
    parameter_set_id: Optional[Union[int, str]] = None,
) -> pd.DataFrame:
    """
    Prediksi multi-step dengan recursive forecasting untuk XGBoost.

    Untuk setiap step, prediksi sebelumnya dimasukkan ke history sementara.
    Dengan begitu, step ke-2 sampai ke-h memakai prediksi internal, bukan
    aktual masa depan dari validation/test.
    """
    if horizon <= 0:
        raise ValueError("horizon harus > 0")

    normalized = normalize_feature_set(feature_set)
    working_history = _validate_history(history_df, target_col=target_col)
    _validate_predict_model(model)

    if future_index is None:
        target_index = _make_default_future_index(working_history.index, horizon)
    else:
        target_index = _coerce_future_index(future_index)
        if len(target_index) != horizon:
            raise ValueError(
                "Panjang future_index harus sama dengan horizon. "
                f"len(future_index)={len(target_index)}, horizon={horizon}"
            )

    if target_index.min() <= working_history.index.max():
        raise ValueError(
            "future_index harus seluruhnya berada setelah timestamp akhir history."
        )
    if not target_index.is_monotonic_increasing:
        raise ValueError("future_index harus chronological.")
    if target_index.has_duplicates:
        raise ValueError("future_index mengandung duplicate timestamp.")

    origin = (
        _coerce_utc_timestamp(forecast_origin)
        if forecast_origin is not None
        else working_history.index.max()
    )

    records: list[dict[str, Any]] = []
    total_prediction_start = time.perf_counter()

    for step, forecast_timestamp in enumerate(target_index, start=1):
        feature_row = build_next_step_features(
            working_history,
            forecast_timestamp,
            normalized,
            target_col=target_col,
        )

        step_start = time.perf_counter()
        predicted_value = _predict_one(model, feature_row)
        step_prediction_time = time.perf_counter() - step_start

        working_history = _append_target_row(
            working_history,
            timestamp=forecast_timestamp,
            value=predicted_value,
            target_col=target_col,
        )

        records.append(
            {
                TIMESTAMP_COL: forecast_timestamp,
                ACTUAL_COL: np.nan,
                PREDICTION_COL: predicted_value,
                "model_name": model_name,
                "feature_set": normalized,
                "forecast_origin": origin,
                "horizon_step": int(step),
                "fold": "" if fold is None else fold,
                "parameter_set_id": "" if parameter_set_id is None else parameter_set_id,
                "prediction_time_seconds": round(step_prediction_time, 9),
                "used_actual_future_for_features": False,
            }
        )

    predictions = pd.DataFrame.from_records(records)
    predictions.attrs["prediction_time_seconds_total"] = round(
        time.perf_counter() - total_prediction_start,
        9,
    )
    validate_prediction_output(
        predictions,
        expected_index=target_index,
        horizon=horizon,
    )
    return predictions


def forecast_validation_window(
    model: Any,
    train_history: pd.DataFrame,
    validation_data: Union[pd.DataFrame, pd.Series, pd.DatetimeIndex, Sequence[Any]],
    horizon: int = FORECAST_HORIZON,
    feature_set: str = BASIC_FEATURE_SET,
    *,
    target_col: str = TARGET_COL,
    model_name: str = "xgboost",
    fold: Optional[Union[int, str]] = None,
    parameter_set_id: Optional[Union[int, str]] = None,
    update_history_with_actuals: bool = True,
) -> pd.DataFrame:
    """
    Prediksi validation window dengan rolling-origin chunks.

    Validation window dapat lebih panjang dari horizon utama. Fungsi ini
    membagi validation window menjadi blok `horizon` jam, menjalankan
    recursive forecast pada setiap blok, lalu memperbarui origin.

    Jika `update_history_with_actuals=True`, aktual dari blok yang sudah
    selesai dievaluasi ditambahkan ke history untuk origin berikutnya. Ini
    tidak memakai aktual masa depan di dalam blok 24 jam, tetapi mensimulasikan
    rolling-origin ketika observasi masa lalu sudah tersedia.
    """
    if horizon <= 0:
        raise ValueError("horizon harus > 0")

    validation_frame = _coerce_validation_frame(
        validation_data,
        target_col=target_col,
    )
    if validation_frame.empty:
        raise ValueError("validation_data kosong.")

    current_history = _validate_history(train_history, target_col=target_col)
    if validation_frame.index.min() <= current_history.index.max():
        raise ValueError(
            "Validation window harus dimulai setelah timestamp akhir train_history."
        )

    all_predictions: list[pd.DataFrame] = []
    block_id = 0

    for start in range(0, len(validation_frame), horizon):
        block_id += 1
        validation_block = validation_frame.iloc[start : start + horizon]
        block_horizon = int(len(validation_block))
        origin = current_history.index.max()

        block_predictions = recursive_forecast_xgb(
            model,
            current_history,
            horizon=block_horizon,
            feature_set=feature_set,
            future_index=validation_block.index,
            target_col=target_col,
            model_name=model_name,
            forecast_origin=origin,
            fold=fold,
            parameter_set_id=parameter_set_id,
        )
        block_predictions["origin_block"] = int(block_id)
        block_predictions["validation_start"] = validation_block.index.min()
        block_predictions["validation_end"] = validation_block.index.max()

        has_actual = target_col in validation_block.columns
        if has_actual:
            block_predictions[ACTUAL_COL] = validation_block[target_col].to_numpy(
                dtype=float,
            )

        all_predictions.append(block_predictions)

        if update_history_with_actuals and has_actual:
            current_history = _append_actual_block(
                current_history,
                validation_block,
                target_col=target_col,
            )
        else:
            current_history = _append_prediction_block(
                current_history,
                block_predictions,
                target_col=target_col,
            )

    predictions = pd.concat(all_predictions, ignore_index=True)
    validate_prediction_output(
        predictions,
        expected_index=validation_frame.index,
        horizon=len(validation_frame),
    )
    predictions.attrs["prediction_time_seconds_total"] = round(
        float(predictions["prediction_time_seconds"].sum()),
        9,
    )
    predictions.attrs["update_history_with_actuals"] = bool(update_history_with_actuals)
    return predictions


def recursive_forecast_prophet(
    model: Any,
    future_df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
    *,
    model_name: str = "prophet",
    fold: Optional[Union[int, str]] = None,
    parameter_set_id: Optional[Union[int, str]] = None,
) -> pd.DataFrame:
    """
    Wrapper prediksi Prophet dengan format output yang sama.

    Prophet tidak membutuhkan recursive lag features seperti XGBoost, tetapi
    wrapper ini disediakan agar output CV/final test tetap konsisten.
    """
    if horizon <= 0:
        raise ValueError("horizon harus > 0")
    if "ds" not in future_df.columns:
        raise ValueError("future_df untuk Prophet harus memiliki kolom 'ds'.")
    _validate_predict_model(model)

    future = future_df.head(horizon).copy()
    if future.shape[0] != horizon:
        raise ValueError(
            f"future_df tidak cukup panjang untuk horizon={horizon}."
        )

    prediction_start = time.perf_counter()
    forecast = model.predict(future)
    prediction_time = time.perf_counter() - prediction_start

    if "yhat" not in forecast.columns:
        raise ValueError("Output Prophet tidak memiliki kolom 'yhat'.")

    timestamps = _coerce_future_index(future["ds"], allow_naive_as_utc=True)
    predictions = pd.DataFrame(
        {
            TIMESTAMP_COL: timestamps,
            ACTUAL_COL: np.nan,
            PREDICTION_COL: pd.to_numeric(forecast["yhat"], errors="raise").to_numpy(
                dtype=float,
            ),
            "model_name": model_name,
            "feature_set": "prophet_internal",
            "forecast_origin": pd.NaT,
            "horizon_step": np.arange(1, horizon + 1, dtype=int),
            "fold": "" if fold is None else fold,
            "parameter_set_id": "" if parameter_set_id is None else parameter_set_id,
            "prediction_time_seconds": round(prediction_time / horizon, 9),
            "used_actual_future_for_features": False,
        }
    )
    predictions.attrs["prediction_time_seconds_total"] = round(prediction_time, 9)
    validate_prediction_output(
        predictions,
        expected_index=timestamps,
        horizon=horizon,
    )
    return predictions


def validate_prediction_output(
    predictions: pd.DataFrame,
    *,
    expected_index: Optional[pd.DatetimeIndex] = None,
    horizon: Optional[int] = None,
) -> dict[str, Any]:
    """
    Validasi format output prediksi agar siap dipakai metrics dan tracking.
    """
    required_columns = {
        TIMESTAMP_COL,
        ACTUAL_COL,
        PREDICTION_COL,
        "model_name",
        "feature_set",
        "forecast_origin",
        "horizon_step",
        "prediction_time_seconds",
        "used_actual_future_for_features",
    }
    missing_columns = sorted(required_columns.difference(predictions.columns))
    if missing_columns:
        raise ValueError(f"Kolom output prediksi tidak lengkap: {missing_columns}")

    if predictions.empty:
        raise ValueError("Output prediksi kosong.")

    timestamps = _coerce_future_index(predictions[TIMESTAMP_COL])
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamp output prediksi tidak chronological.")
    if timestamps.has_duplicates:
        raise ValueError("Timestamp output prediksi mengandung duplicate.")

    predicted = pd.to_numeric(predictions[PREDICTION_COL], errors="raise")
    if not np.isfinite(predicted.to_numpy(dtype=float)).all():
        raise ValueError("Output prediksi mengandung nilai non-finite.")

    if horizon is not None and int(horizon) != len(predictions):
        raise ValueError(
            f"Jumlah output prediksi tidak sesuai horizon: {len(predictions)} != {horizon}"
        )

    if expected_index is not None and not timestamps.equals(expected_index):
        raise ValueError("Timestamp output prediksi tidak sama dengan expected_index.")

    leakage_flags = predictions["used_actual_future_for_features"].astype(bool)
    if leakage_flags.any():
        raise ValueError("Output menandai adanya penggunaan actual future untuk fitur.")

    return {
        "n_predictions": int(len(predictions)),
        "timestamp_start": timestamps.min().isoformat(),
        "timestamp_end": timestamps.max().isoformat(),
        "prediction_time_seconds_total": float(
            predictions["prediction_time_seconds"].sum()
        ),
        "used_actual_future_for_features": False,
    }


def normalize_feature_set(feature_set: str) -> str:
    """
    Normalisasi alias feature set agar modul forecasting konsisten.
    """
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


def _feature_spec(
    feature_set: str,
) -> tuple[list[int], list[int], list[int], list[str]]:
    if feature_set == BASIC_FEATURE_SET:
        return (
            list(XGB_BASIC_LAGS),
            [],
            [],
            list(XGB_BASIC_CALENDAR),
        )
    if feature_set == ADVANCED_FEATURE_SET:
        return (
            list(XGB_ADVANCED_LAGS),
            list(XGB_ADVANCED_ROLLING_WINDOWS),
            list(XGB_ADVANCED_ROLLING_STD_WINDOWS),
            list(XGB_ADVANCED_CALENDAR),
        )
    raise ValueError(f"Feature set tidak dikenal: {feature_set}")


def _validate_history(
    history: pd.DataFrame,
    *,
    target_col: str,
) -> pd.DataFrame:
    if history.empty:
        raise ValueError("History untuk forecasting kosong.")
    if target_col not in history.columns:
        raise ValueError(f"Kolom target tidak ditemukan pada history: {target_col}")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise TypeError("History forecasting harus memakai DatetimeIndex.")
    if history.index.tz is None:
        raise ValueError("History forecasting harus timezone-aware.")

    prepared = history[[target_col]].copy()
    prepared.index = prepared.index.tz_convert(MODELING_TZ)

    if not prepared.index.is_monotonic_increasing:
        raise ValueError("History forecasting tidak chronological.")
    if prepared.index.has_duplicates:
        raise ValueError("History forecasting mengandung duplicate timestamp.")

    target = pd.to_numeric(prepared[target_col], errors="raise")
    if target.isna().any():
        raise ValueError("Target history mengandung missing value.")
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("Target history mengandung nilai non-finite.")

    max_required_history = _max_required_history()
    if len(prepared) < max_required_history:
        raise ValueError(
            "History forecasting belum cukup panjang untuk lag/rolling maksimum. "
            f"required={max_required_history}, observed={len(prepared)}"
        )

    prepared[target_col] = target.astype(float)
    return prepared


def _max_required_history() -> int:
    candidates = [
        *XGB_BASIC_LAGS,
        *XGB_ADVANCED_LAGS,
        *XGB_ADVANCED_ROLLING_WINDOWS,
        *XGB_ADVANCED_ROLLING_STD_WINDOWS,
    ]
    return int(max(candidates))


def _last_value_at_lag(values: pd.Series, lag: int) -> float:
    if len(values) < lag:
        raise ValueError(f"History tidak cukup untuk lag_{lag}.")
    return float(values.iloc[-lag])


def _rolling_mean(values: pd.Series, window: int) -> float:
    if len(values) < window:
        raise ValueError(f"History tidak cukup untuk rolling_mean_{window}.")
    return float(values.iloc[-window:].mean())


def _rolling_std(values: pd.Series, window: int) -> float:
    if len(values) < window:
        raise ValueError(f"History tidak cukup untuk rolling_std_{window}.")
    return float(values.iloc[-window:].std())


def _calendar_feature_values(
    timestamp: pd.Timestamp,
    *,
    calendar_features: Sequence[str],
    local_timezone: str,
) -> dict[str, Any]:
    local_timestamp = timestamp.tz_convert(local_timezone)
    values: dict[str, Any] = {}

    if "hour" in calendar_features:
        values["hour"] = int(local_timestamp.hour)
    if "day_of_week" in calendar_features:
        values["day_of_week"] = int(local_timestamp.dayofweek)
    if "is_weekend" in calendar_features:
        values["is_weekend"] = int(local_timestamp.dayofweek in [5, 6])
    if "month" in calendar_features:
        values["month"] = int(local_timestamp.month)

    unsupported = sorted(set(calendar_features).difference(values))
    if unsupported:
        raise ValueError(f"Calendar features tidak didukung: {unsupported}")
    return values


def _validate_feature_row(
    feature_row: pd.DataFrame,
    *,
    feature_set: str,
) -> None:
    feature_columns = get_feature_columns(feature_set)
    if list(feature_row.columns) != feature_columns:
        raise ValueError("Urutan kolom fitur next-step tidak sesuai feature_columns.")
    if feature_row.isna().any().any():
        raise ValueError("Feature row next-step masih memiliki missing value.")
    numeric = feature_row.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Feature row next-step memiliki nilai non-finite.")


def _validate_predict_model(model: Any) -> None:
    if not hasattr(model, "predict") or not callable(model.predict):
        raise TypeError("Model harus memiliki method callable predict().")


def _predict_one(model: Any, feature_row: pd.DataFrame) -> float:
    prediction = model.predict(feature_row)
    values = np.asarray(prediction).reshape(-1)
    if values.size != 1:
        raise ValueError(
            "model.predict(feature_row) harus menghasilkan tepat satu prediksi."
        )
    value = float(values[0])
    if not np.isfinite(value):
        raise ValueError("Prediksi model menghasilkan nilai non-finite.")
    return value


def _append_target_row(
    history: pd.DataFrame,
    *,
    timestamp: pd.Timestamp,
    value: float,
    target_col: str,
) -> pd.DataFrame:
    new_row = pd.DataFrame(
        {target_col: [float(value)]},
        index=pd.DatetimeIndex([timestamp], name=history.index.name or TIMESTAMP_COL),
    )
    updated = pd.concat([history[[target_col]], new_row], axis=0)
    if not updated.index.is_monotonic_increasing:
        raise ValueError("History hasil update tidak chronological.")
    if updated.index.has_duplicates:
        raise ValueError("History hasil update mengandung duplicate timestamp.")
    return updated


def _append_actual_block(
    history: pd.DataFrame,
    validation_block: pd.DataFrame,
    *,
    target_col: str,
) -> pd.DataFrame:
    actual_block = validation_block[[target_col]].copy()
    actual_block[target_col] = pd.to_numeric(
        actual_block[target_col],
        errors="raise",
    ).astype(float)
    if not np.isfinite(actual_block[target_col].to_numpy(dtype=float)).all():
        raise ValueError("Actual validation block mengandung nilai non-finite.")
    updated = pd.concat([history[[target_col]], actual_block], axis=0)
    if not updated.index.is_monotonic_increasing:
        raise ValueError("History + actual validation block tidak chronological.")
    if updated.index.has_duplicates:
        raise ValueError("History + actual validation block mengandung duplicate.")
    return updated


def _append_prediction_block(
    history: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target_col: str,
) -> pd.DataFrame:
    predicted_index = _coerce_future_index(predictions[TIMESTAMP_COL])
    predicted_values = pd.to_numeric(
        predictions[PREDICTION_COL],
        errors="raise",
    ).to_numpy(dtype=float)
    prediction_block = pd.DataFrame(
        {target_col: predicted_values},
        index=predicted_index,
    )
    updated = pd.concat([history[[target_col]], prediction_block], axis=0)
    if not updated.index.is_monotonic_increasing:
        raise ValueError("History + prediction block tidak chronological.")
    if updated.index.has_duplicates:
        raise ValueError("History + prediction block mengandung duplicate.")
    return updated


def _coerce_validation_frame(
    validation_data: Union[pd.DataFrame, pd.Series, pd.DatetimeIndex, Sequence[Any]],
    *,
    target_col: str,
) -> pd.DataFrame:
    if isinstance(validation_data, pd.DataFrame):
        frame = validation_data.copy()
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError("validation_data DataFrame harus memakai DatetimeIndex.")
        frame.index = frame.index.tz_convert(MODELING_TZ)
        _validate_time_index(frame.index, frame_name="validation_data")
        if target_col in frame.columns:
            frame[target_col] = pd.to_numeric(frame[target_col], errors="raise").astype(
                float,
            )
            if not np.isfinite(frame[target_col].to_numpy(dtype=float)).all():
                raise ValueError("validation_data mengandung actual non-finite.")
        return frame

    if isinstance(validation_data, pd.Series):
        if not isinstance(validation_data.index, pd.DatetimeIndex):
            raise TypeError("validation_data Series harus memakai DatetimeIndex.")
        index = validation_data.index.tz_convert(MODELING_TZ)
        _validate_time_index(index, frame_name="validation_data")
        target = pd.to_numeric(validation_data, errors="raise").astype(float)
        if not np.isfinite(target.to_numpy(dtype=float)).all():
            raise ValueError("validation_data mengandung actual non-finite.")
        return pd.DataFrame(
            {target_col: target},
            index=index,
        )

    index = _coerce_future_index(validation_data)
    _validate_time_index(index, frame_name="validation_data")
    return pd.DataFrame(index=index)


def _coerce_utc_timestamp(timestamp: Union[str, pd.Timestamp]) -> pd.Timestamp:
    coerced = pd.Timestamp(timestamp)
    if coerced.tzinfo is None:
        raise ValueError(
            "Timestamp forecasting harus timezone-aware UTC atau dapat dikonversi ke UTC."
        )
    return coerced.tz_convert(MODELING_TZ)


def _coerce_future_index(
    values: Union[pd.DatetimeIndex, pd.Series, Sequence[Any]],
    *,
    allow_naive_as_utc: bool = False,
) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(values)
    if index.empty:
        raise ValueError("Future index kosong.")
    if index.tz is None:
        if allow_naive_as_utc:
            index = index.tz_localize(MODELING_TZ)
        else:
            raise ValueError("Future index harus timezone-aware.")
    index = index.tz_convert(MODELING_TZ)
    index.name = TIMESTAMP_COL
    if index.has_duplicates:
        raise ValueError("Future index mengandung duplicate timestamp.")
    return index


def _validate_time_index(index: pd.DatetimeIndex, *, frame_name: str) -> None:
    if index.tz is None:
        raise ValueError(f"Index {frame_name} harus timezone-aware.")
    if str(index.tz) != MODELING_TZ:
        raise ValueError(
            f"Index {frame_name} harus timezone {MODELING_TZ}, ditemukan {index.tz}"
        )
    if not index.is_monotonic_increasing:
        raise ValueError(f"Index {frame_name} tidak chronological.")
    if index.has_duplicates:
        raise ValueError(f"Index {frame_name} mengandung duplicate timestamp.")


def _make_default_future_index(
    history_index: pd.DatetimeIndex,
    horizon: int,
) -> pd.DatetimeIndex:
    start = history_index.max() + pd.Timedelta(hours=1)
    return pd.date_range(
        start=start,
        periods=horizon,
        freq="h",
        tz=MODELING_TZ,
        name=TIMESTAMP_COL,
    )


__all__ = [
    "ACTUAL_COL",
    "PREDICTION_COL",
    "build_next_step_features",
    "forecast_validation_window",
    "normalize_feature_set",
    "recursive_forecast_prophet",
    "recursive_forecast_xgb",
    "validate_prediction_output",
]

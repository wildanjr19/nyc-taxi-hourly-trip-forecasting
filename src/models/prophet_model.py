"""
prophet_model.py

Wrapper kecil untuk Prophet agar script eksperimen tetap bersih.

Catatan timezone:
- Pipeline modeling memakai UTC timezone-aware index.
- Prophet tidak menerima kolom ds timezone-aware pada banyak versi.
- Karena itu ds dikonversi menjadi UTC-naive. Artinya nilai jam tetap UTC,
  hanya metadata timezone yang dilepas untuk kompatibilitas Prophet.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from src.config import MODELING_TZ, TARGET_COL, TIMESTAMP_COL


PROPHET_FEATURE_SET = "prophet_internal"

DEFAULT_PROPHET_PARAMS: dict[str, Any] = {
    "weekly_seasonality": True,
    "daily_seasonality": True,
    "yearly_seasonality": False,
}

SUPPORTED_PROPHET_PARAMS = {
    "changepoint_prior_scale",
    "seasonality_prior_scale",
    "seasonality_mode",
    "weekly_seasonality",
    "daily_seasonality",
    "yearly_seasonality",
}


def make_prophet_model(params: Optional[Mapping[str, Any]] = None) -> Any:
    """
    Buat instance Prophet dengan parameter yang tervalidasi.
    """
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError(
            "Package prophet belum tersedia. Install dependencies dengan "
            "`pip install -r requirements.txt` sebelum menjalankan tuning Prophet."
        ) from exc

    model_params = dict(DEFAULT_PROPHET_PARAMS)
    if params is not None:
        unknown = sorted(set(params).difference(SUPPORTED_PROPHET_PARAMS))
        if unknown:
            raise ValueError(f"Parameter Prophet tidak didukung: {unknown}")
        model_params.update(dict(params))

    return Prophet(**model_params)


def make_prophet_frame(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    include_y: bool = True,
) -> pd.DataFrame:
    """
    Ubah DataFrame time series menjadi format Prophet: ds dan y.
    """
    _validate_time_indexed_frame(df, target_col=target_col, require_target=include_y)

    ds = df.index.tz_convert(MODELING_TZ).tz_localize(None)
    frame = pd.DataFrame({"ds": ds})

    if include_y:
        y = pd.to_numeric(df[target_col], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Target training Prophet mengandung nilai non-finite.")
        frame["y"] = y

    return frame


def make_prophet_future_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Buat future dataframe Prophet dari UTC DatetimeIndex.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index future Prophet harus berupa pandas DatetimeIndex.")
    if index.empty:
        raise ValueError("index future Prophet kosong.")
    if index.tz is None:
        raise ValueError("index future Prophet harus timezone-aware.")
    if not index.is_monotonic_increasing:
        raise ValueError("index future Prophet tidak chronological.")
    if index.has_duplicates:
        raise ValueError("index future Prophet mengandung duplicate timestamp.")

    ds = index.tz_convert(MODELING_TZ).tz_localize(None)
    return pd.DataFrame({"ds": ds})


def fit_prophet_model(
    train_df: pd.DataFrame,
    *,
    params: Optional[Mapping[str, Any]] = None,
    target_col: str = TARGET_COL,
) -> Any:
    """
    Fit Prophet pada training fold.
    """
    model = make_prophet_model(params)
    prophet_train = make_prophet_frame(
        train_df,
        target_col=target_col,
        include_y=True,
    )
    model.fit(prophet_train)
    return model


def _validate_time_indexed_frame(
    df: pd.DataFrame,
    *,
    target_col: str,
    require_target: bool,
) -> None:
    if df.empty:
        raise ValueError("Data Prophet kosong.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Data Prophet harus memakai DatetimeIndex.")
    if df.index.tz is None:
        raise ValueError("Index Prophet harus timezone-aware.")
    if str(df.index.tz) != MODELING_TZ:
        raise ValueError(
            f"Index Prophet harus timezone {MODELING_TZ}, ditemukan {df.index.tz}."
        )
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index Prophet tidak chronological.")
    if df.index.has_duplicates:
        raise ValueError("Index Prophet mengandung duplicate timestamp.")
    if require_target and target_col not in df.columns:
        raise ValueError(f"Kolom target Prophet tidak ditemukan: {target_col}")


__all__ = [
    "DEFAULT_PROPHET_PARAMS",
    "PROPHET_FEATURE_SET",
    "SUPPORTED_PROPHET_PARAMS",
    "fit_prophet_model",
    "make_prophet_frame",
    "make_prophet_future_frame",
    "make_prophet_model",
]

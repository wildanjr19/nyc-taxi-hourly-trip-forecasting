"""
xgboost_model.py

Wrapper XGBoost untuk eksperimen tuning. Modul ini hanya bertanggung jawab
untuk membuat dan melatih estimator; prediksi validation/test tetap dilakukan
oleh src.forecasting agar recursive forecasting dan leakage guardrail konsisten.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, TARGET_COL


DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "tree_method": "hist",
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


def make_xgb_regressor(params: Optional[Mapping[str, Any]] = None) -> Any:
    """
    Buat instance XGBRegressor dengan default yang reproducible.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "Package xgboost belum tersedia. Install dependencies dengan "
            "`pip install -r requirements.txt` sebelum menjalankan tuning XGBoost."
        ) from exc

    model_params = dict(DEFAULT_XGB_PARAMS)
    if params is not None:
        model_params.update(dict(params))
    return XGBRegressor(**model_params)


def fit_xgb_model(
    train_features: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    params: Optional[Mapping[str, Any]] = None,
    target_col: str = TARGET_COL,
) -> Any:
    """
    Fit XGBoost pada feature rows training fold saja.
    """
    _validate_feature_training_frame(
        train_features,
        feature_columns=feature_columns,
        target_col=target_col,
    )

    model = make_xgb_regressor(params)
    x_train = train_features[list(feature_columns)].copy()
    y_train = pd.to_numeric(train_features[target_col], errors="raise").astype(float)
    model.fit(x_train, y_train)
    return model


def _validate_feature_training_frame(
    train_features: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_col: str,
) -> None:
    if train_features.empty:
        raise ValueError("Feature training XGBoost kosong.")
    if not isinstance(train_features.index, pd.DatetimeIndex):
        raise TypeError("Feature training XGBoost harus memakai DatetimeIndex.")
    if not train_features.index.is_monotonic_increasing:
        raise ValueError("Feature training XGBoost tidak chronological.")
    if train_features.index.has_duplicates:
        raise ValueError("Feature training XGBoost mengandung duplicate timestamp.")

    required_columns = {target_col, *feature_columns}
    missing_columns = sorted(required_columns.difference(train_features.columns))
    if missing_columns:
        raise ValueError(f"Kolom feature training XGBoost hilang: {missing_columns}")

    numeric = train_features[[target_col, *feature_columns]].apply(
        pd.to_numeric,
        errors="raise",
    )
    if numeric.isna().any().any():
        raise ValueError("Feature training XGBoost mengandung missing value.")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Feature training XGBoost mengandung nilai non-finite.")


__all__ = [
    "DEFAULT_XGB_PARAMS",
    "fit_xgb_model",
    "make_xgb_regressor",
]

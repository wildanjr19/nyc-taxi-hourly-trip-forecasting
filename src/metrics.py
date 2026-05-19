"""
metrics.py

Metric evaluasi reusable untuk penelitian NYC Taxi hourly forecasting.

Modul ini menyediakan perhitungan metric yang konsisten untuk Prophet,
XGBoost-Basic, dan XGBoost-Advanced. Semua fungsi menjaga bentuk evaluasi
yang sama untuk CV, tuning, retraining check, dan final testing.

Catatan metodologi:
- MAE dan RMSE berada pada satuan trip_count.
- MAPE dan sMAPE dikembalikan dalam persen.
- MAPE mengecualikan baris dengan actual == 0 karena pembagian nol tidak
  terdefinisi; jumlah baris tersebut dicatat oleh compute_all_metrics().
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.config import METRICS


ACTUAL_COL = "actual"
PREDICTED_COL = "predicted"

DEFAULT_GROUP_COLUMNS = ["model_name", "feature_set", "parameter_set_id"]
DIAGNOSTIC_COLUMNS = [
    "n_obs",
    "n_valid_obs",
    "n_invalid_obs",
    "mape_zero_actual_count",
    "mape_valid_count",
    "smape_zero_denominator_count",
]


ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]


def mae(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    drop_invalid: bool = True,
) -> float:
    """
    Hitung Mean Absolute Error.
    """
    true, pred, _ = _coerce_metric_inputs(
        y_true,
        y_pred,
        drop_invalid=drop_invalid,
    )
    return float(np.mean(np.abs(true - pred)))


def rmse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    drop_invalid: bool = True,
) -> float:
    """
    Hitung Root Mean Squared Error.
    """
    true, pred, _ = _coerce_metric_inputs(
        y_true,
        y_pred,
        drop_invalid=drop_invalid,
    )
    return float(np.sqrt(np.mean(np.square(true - pred))))


def mape(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    drop_invalid: bool = True,
) -> float:
    """
    Hitung Mean Absolute Percentage Error dalam persen.

    Baris dengan actual == 0 dikeluarkan dari perhitungan karena MAPE tidak
    terdefinisi pada titik tersebut. Jika semua actual bernilai nol, fungsi
    mengembalikan NaN.
    """
    true, pred, _ = _coerce_metric_inputs(
        y_true,
        y_pred,
        drop_invalid=drop_invalid,
    )
    valid_mask = true != 0
    if not valid_mask.any():
        return float("nan")

    percentage_errors = np.abs((true[valid_mask] - pred[valid_mask]) / true[valid_mask])
    return float(np.mean(percentage_errors) * 100.0)


def smape(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    drop_invalid: bool = True,
) -> float:
    """
    Hitung symmetric MAPE dalam persen.

    Formula yang dipakai:
        100 * mean(2 * abs(actual - predicted) / (abs(actual) + abs(predicted)))

    Jika actual dan predicted sama-sama nol, kontribusi titik tersebut
    didefinisikan sebagai 0 karena tidak ada error aktual.
    """
    true, pred, _ = _coerce_metric_inputs(
        y_true,
        y_pred,
        drop_invalid=drop_invalid,
    )
    denominator = np.abs(true) + np.abs(pred)
    ratio = np.zeros_like(true, dtype=float)
    non_zero_denominator = denominator != 0
    ratio[non_zero_denominator] = (
        2.0
        * np.abs(true[non_zero_denominator] - pred[non_zero_denominator])
        / denominator[non_zero_denominator]
    )
    return float(np.mean(ratio) * 100.0)


def compute_all_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    drop_invalid: bool = True,
    include_diagnostics: bool = True,
) -> dict[str, Union[float, int]]:
    """
    Hitung semua metric wajib dan, secara default, metadata diagnostik.

    Metadata diagnostik penting untuk audit hasil penelitian, terutama untuk
    menjelaskan berapa baris yang dikeluarkan dari MAPE karena actual == 0.
    """
    true, pred, diagnostics = _coerce_metric_inputs(
        y_true,
        y_pred,
        drop_invalid=drop_invalid,
    )

    zero_actual_mask = true == 0
    smape_zero_denominator_mask = (np.abs(true) + np.abs(pred)) == 0

    result: dict[str, Union[float, int]] = {
        "mae": mae(true, pred, drop_invalid=False),
        "rmse": rmse(true, pred, drop_invalid=False),
        "mape": mape(true, pred, drop_invalid=False),
        "smape": smape(true, pred, drop_invalid=False),
    }

    if include_diagnostics:
        result.update(
            {
                "n_obs": diagnostics["n_obs"],
                "n_valid_obs": diagnostics["n_valid_obs"],
                "n_invalid_obs": diagnostics["n_invalid_obs"],
                "mape_zero_actual_count": int(zero_actual_mask.sum()),
                "mape_valid_count": int((~zero_actual_mask).sum()),
                "smape_zero_denominator_count": int(smape_zero_denominator_mask.sum()),
            }
        )

    return result


def compute_metrics_from_predictions(
    predictions: pd.DataFrame,
    *,
    actual_col: str = ACTUAL_COL,
    predicted_col: str = PREDICTED_COL,
    group_cols: Optional[Sequence[str]] = None,
    include_diagnostics: bool = True,
) -> pd.DataFrame:
    """
    Hitung metric dari DataFrame prediksi standar.

    Jika group_cols diisi, metric dihitung per grup. Ini berguna untuk
    menghitung metric per fold, model, feature_set, atau parameter_set_id.
    """
    missing_cols = [
        col for col in [actual_col, predicted_col] if col not in predictions.columns
    ]
    if missing_cols:
        raise ValueError(f"Kolom prediksi wajib tidak ditemukan: {missing_cols}")

    if predictions.empty:
        raise ValueError("DataFrame prediksi kosong; metric tidak bisa dihitung.")

    grouping = list(group_cols or [])
    missing_group_cols = [col for col in grouping if col not in predictions.columns]
    if missing_group_cols:
        raise ValueError(f"Kolom group tidak ditemukan: {missing_group_cols}")

    if not grouping:
        metrics = compute_all_metrics(
            predictions[actual_col],
            predictions[predicted_col],
            include_diagnostics=include_diagnostics,
        )
        return pd.DataFrame([metrics])

    rows: list[dict[str, Any]] = []
    grouped = predictions.groupby(grouping, dropna=False, sort=False)
    for group_key, group_df in grouped:
        group_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(grouping, group_values))
        row.update(
            compute_all_metrics(
                group_df[actual_col],
                group_df[predicted_col],
                include_diagnostics=include_diagnostics,
            )
        )
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_cv_metrics(
    metrics_df: pd.DataFrame,
    *,
    group_cols: Optional[Sequence[str]] = None,
    metric_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Ringkas metric CV menjadi mean, std, min, dan max.

    Default grouping mengikuti kolom eksperimen yang umum tersedia:
    model_name, feature_set, dan parameter_set_id. Kolom yang tidak ada
    otomatis diabaikan.
    """
    if metrics_df.empty:
        raise ValueError("metrics_df kosong; summary CV tidak bisa dibuat.")

    metrics_to_summarize = list(metric_cols or METRICS)
    missing_metrics = [
        metric for metric in metrics_to_summarize if metric not in metrics_df.columns
    ]
    if missing_metrics:
        raise ValueError(f"Kolom metric tidak ditemukan: {missing_metrics}")

    grouping = list(group_cols) if group_cols is not None else [
        col for col in DEFAULT_GROUP_COLUMNS if col in metrics_df.columns
    ]
    missing_group_cols = [col for col in grouping if col not in metrics_df.columns]
    if missing_group_cols:
        raise ValueError(f"Kolom group tidak ditemukan: {missing_group_cols}")

    if not grouping:
        return pd.DataFrame([_summarize_metric_frame(metrics_df, metrics_to_summarize)])

    rows: list[dict[str, Any]] = []
    grouped = metrics_df.groupby(grouping, dropna=False, sort=False)
    for group_key, group_df in grouped:
        group_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(grouping, group_values))
        row.update(_summarize_metric_frame(group_df, metrics_to_summarize))
        rows.append(row)

    return pd.DataFrame(rows)


def _summarize_metric_frame(
    metrics_df: pd.DataFrame,
    metric_cols: Sequence[str],
) -> dict[str, Union[float, int]]:
    summary: dict[str, Union[float, int]] = {
        "n_rows": int(metrics_df.shape[0]),
    }
    if "fold" in metrics_df.columns:
        summary["n_folds"] = int(metrics_df["fold"].nunique(dropna=False))

    for metric in metric_cols:
        values = pd.to_numeric(metrics_df[metric], errors="coerce").dropna()
        if values.empty:
            summary[f"{metric}_mean"] = float("nan")
            summary[f"{metric}_std"] = float("nan")
            summary[f"{metric}_min"] = float("nan")
            summary[f"{metric}_max"] = float("nan")
            continue

        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = (
            float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0
        )
        summary[f"{metric}_min"] = float(values.min())
        summary[f"{metric}_max"] = float(values.max())

    return summary


def _coerce_metric_inputs(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    drop_invalid: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    true = _coerce_numeric_series(y_true, name="y_true")
    pred = _coerce_numeric_series(y_pred, name="y_pred")

    if true.shape[0] != pred.shape[0]:
        raise ValueError(
            "Panjang y_true dan y_pred harus sama. "
            f"y_true={true.shape[0]}, y_pred={pred.shape[0]}"
        )

    true_values = true.to_numpy(dtype=float)
    pred_values = pred.to_numpy(dtype=float)
    valid_mask = np.isfinite(true_values) & np.isfinite(pred_values)

    if not valid_mask.all() and not drop_invalid:
        invalid_count = int((~valid_mask).sum())
        raise ValueError(
            f"Terdapat {invalid_count} pasangan y_true/y_pred yang NaN atau non-finite."
        )

    if drop_invalid:
        true_values = true_values[valid_mask]
        pred_values = pred_values[valid_mask]

    if true_values.size == 0:
        raise ValueError("Tidak ada pasangan y_true/y_pred valid untuk evaluasi metric.")

    diagnostics = {
        "n_obs": int(true.shape[0]),
        "n_valid_obs": int(valid_mask.sum()),
        "n_invalid_obs": int((~valid_mask).sum()),
    }
    return true_values, pred_values, diagnostics


def _coerce_numeric_series(values: ArrayLike, *, name: str) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        raise TypeError(f"{name} harus satu dimensi, bukan DataFrame.")

    series = values.reset_index(drop=True) if isinstance(values, pd.Series) else pd.Series(values)
    if series.ndim != 1:
        raise TypeError(f"{name} harus satu dimensi.")

    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.reset_index(drop=True)


__all__ = [
    "ACTUAL_COL",
    "PREDICTED_COL",
    "mae",
    "rmse",
    "mape",
    "smape",
    "compute_all_metrics",
    "compute_metrics_from_predictions",
    "summarize_cv_metrics",
]

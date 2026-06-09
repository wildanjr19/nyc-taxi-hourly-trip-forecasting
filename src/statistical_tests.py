"""
statistical_tests.py

Utility statistik untuk membandingkan akurasi forecast pada artefak final test.

Fokus utama modul ini adalah Diebold-Mariano test. Implementasi dibuat reusable
agar bisa dipakai pada level blok 24 jam, per horizon, atau timestamp-level
tanpa menyentuh proses tuning/retraining.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd


Alternative = Literal["two_sided", "less", "greater"]
VarianceEstimator = Literal["newey_west", "acf"]
PValueDistribution = Literal["t", "normal"]


@dataclass(frozen=True)
class DieboldMarianoResult:
    statistic: float
    p_value: float
    mean_loss_difference: float
    variance_loss_difference: float
    long_run_variance: float
    n_obs: int
    forecast_horizon: int
    hac_lags: int
    alternative: str
    variance_estimator: str
    pvalue_distribution: str
    hln_correction_applied: bool


def diebold_mariano_test(
    loss_model_a: Sequence[float] | np.ndarray | pd.Series,
    loss_model_b: Sequence[float] | np.ndarray | pd.Series,
    *,
    forecast_horizon: int = 1,
    hac_lags: int | None = None,
    alternative: Alternative = "two_sided",
    variance_estimator: VarianceEstimator = "newey_west",
    apply_hln_correction: bool = True,
) -> DieboldMarianoResult:
    """
    Jalankan Diebold-Mariano test dari dua deret loss yang sudah sejajar.

    Loss differential d_t = loss_model_a - loss_model_b.
    Nilai mean_loss_difference positif berarti model_b memiliki loss lebih
    kecil daripada model_a secara rata-rata.
    """
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon harus >= 1.")
    if alternative not in {"two_sided", "less", "greater"}:
        raise ValueError(f"alternative tidak valid: {alternative}")
    if variance_estimator not in {"newey_west", "acf"}:
        raise ValueError(f"variance_estimator tidak valid: {variance_estimator}")

    loss_a = _coerce_loss_array(loss_model_a, name="loss_model_a")
    loss_b = _coerce_loss_array(loss_model_b, name="loss_model_b")
    if loss_a.shape[0] != loss_b.shape[0]:
        raise ValueError(
            "Panjang loss_model_a dan loss_model_b harus sama. "
            f"a={loss_a.shape[0]}, b={loss_b.shape[0]}"
        )

    differential = loss_a - loss_b
    valid = np.isfinite(differential)
    differential = differential[valid]
    n_obs = int(differential.shape[0])
    if n_obs < 3:
        raise ValueError("DM test membutuhkan minimal 3 observasi valid.")

    max_lag = n_obs - 1
    if hac_lags is None:
        hac_lags = min(max(forecast_horizon - 1, 0), max_lag)
    hac_lags = int(hac_lags)
    if hac_lags < 0:
        raise ValueError("hac_lags harus >= 0.")
    if hac_lags > max_lag:
        raise ValueError(f"hac_lags terlalu besar untuk n_obs={n_obs}: {hac_lags}")

    mean_diff = float(np.mean(differential))
    centered = differential - mean_diff
    variance_diff = float(np.sum(np.square(centered)) / n_obs)
    long_run_variance = _long_run_variance(
        centered,
        hac_lags=hac_lags,
        variance_estimator=variance_estimator,
    )
    if not np.isfinite(long_run_variance) or long_run_variance <= 0:
        raise ValueError(
            "Long-run variance loss differential tidak positif. "
            f"Nilai={long_run_variance}"
        )

    dm_stat = mean_diff / math.sqrt(long_run_variance / n_obs)
    correction_applied = False
    if apply_hln_correction:
        correction = _harvey_leybourne_newbold_correction(
            n_obs=n_obs,
            forecast_horizon=forecast_horizon,
        )
        dm_stat *= correction
        correction_applied = True

    p_value, distribution = _p_value(
        dm_stat,
        n_obs=n_obs,
        alternative=alternative,
    )
    return DieboldMarianoResult(
        statistic=float(dm_stat),
        p_value=float(p_value),
        mean_loss_difference=mean_diff,
        variance_loss_difference=variance_diff,
        long_run_variance=float(long_run_variance),
        n_obs=n_obs,
        forecast_horizon=int(forecast_horizon),
        hac_lags=hac_lags,
        alternative=alternative,
        variance_estimator=variance_estimator,
        pvalue_distribution=distribution,
        hln_correction_applied=correction_applied,
    )


def loss_from_predictions(
    predictions: pd.DataFrame,
    *,
    loss_type: Literal["squared_error", "absolute_error"],
    actual_col: str = "actual",
    predicted_col: str = "predicted",
) -> pd.Series:
    """
    Hitung loss per baris dari DataFrame prediksi standar.
    """
    if actual_col not in predictions.columns or predicted_col not in predictions.columns:
        raise ValueError(f"Kolom {actual_col}/{predicted_col} tidak ditemukan.")
    actual = pd.to_numeric(predictions[actual_col], errors="raise")
    predicted = pd.to_numeric(predictions[predicted_col], errors="raise")
    residual = actual - predicted
    if loss_type == "squared_error":
        return pd.Series(np.square(residual), index=predictions.index)
    if loss_type == "absolute_error":
        return pd.Series(np.abs(residual), index=predictions.index)
    raise ValueError(f"loss_type tidak valid: {loss_type}")


def adjust_pvalues(
    p_values: Sequence[float] | pd.Series,
    *,
    method: Literal["holm", "benjamini_hochberg"],
) -> list[float]:
    """
    Koreksi multiple testing untuk deret p-value.
    """
    p = pd.to_numeric(pd.Series(p_values), errors="coerce").to_numpy(dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    valid_mask = np.isfinite(p)
    valid = p[valid_mask]
    if valid.size == 0:
        return adjusted.tolist()
    if method == "holm":
        adjusted[valid_mask] = _holm_adjust(valid)
    elif method == "benjamini_hochberg":
        adjusted[valid_mask] = _benjamini_hochberg_adjust(valid)
    else:
        raise ValueError(f"method tidak valid: {method}")
    return adjusted.tolist()


def _coerce_loss_array(
    values: Sequence[float] | np.ndarray | pd.Series,
    *,
    name: str,
) -> np.ndarray:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if numeric.ndim != 1:
        raise ValueError(f"{name} harus satu dimensi.")
    return numeric


def _long_run_variance(
    centered: np.ndarray,
    *,
    hac_lags: int,
    variance_estimator: VarianceEstimator,
) -> float:
    n_obs = centered.shape[0]
    gamma_0 = float(np.sum(centered * centered) / n_obs)
    lrv = gamma_0
    for lag in range(1, hac_lags + 1):
        gamma_lag = float(np.sum(centered[lag:] * centered[:-lag]) / n_obs)
        weight = 1.0
        if variance_estimator == "newey_west":
            weight = 1.0 - lag / (hac_lags + 1.0)
        lrv += 2.0 * weight * gamma_lag
    return float(lrv)


def _harvey_leybourne_newbold_correction(
    *,
    n_obs: int,
    forecast_horizon: int,
) -> float:
    numerator = (
        n_obs
        + 1
        - 2 * forecast_horizon
        + forecast_horizon * (forecast_horizon - 1) / n_obs
    )
    correction = numerator / n_obs
    if correction <= 0:
        return 1.0
    return math.sqrt(correction)


def _p_value(
    statistic: float,
    *,
    n_obs: int,
    alternative: Alternative,
) -> tuple[float, PValueDistribution]:
    try:
        from scipy import stats  # type: ignore

        df = max(n_obs - 1, 1)
        if alternative == "two_sided":
            return float(2.0 * stats.t.sf(abs(statistic), df=df)), "t"
        if alternative == "greater":
            return float(stats.t.sf(statistic, df=df)), "t"
        return float(stats.t.cdf(statistic, df=df)), "t"
    except Exception:
        if alternative == "two_sided":
            return float(math.erfc(abs(statistic) / math.sqrt(2.0))), "normal"
        if alternative == "greater":
            return float(0.5 * math.erfc(statistic / math.sqrt(2.0))), "normal"
        return float(0.5 * math.erfc(-statistic / math.sqrt(2.0))), "normal"


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    m = p_values.shape[0]
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for idx, value in enumerate(sorted_p):
        adjusted_value = min((m - idx) * value, 1.0)
        running_max = max(running_max, adjusted_value)
        adjusted_sorted[idx] = running_max
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def _benjamini_hochberg_adjust(p_values: np.ndarray) -> np.ndarray:
    m = p_values.shape[0]
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted_sorted = np.empty(m, dtype=float)
    running_min = 1.0
    for reverse_idx in range(m - 1, -1, -1):
        rank = reverse_idx + 1
        adjusted_value = min(sorted_p[reverse_idx] * m / rank, 1.0)
        running_min = min(running_min, adjusted_value)
        adjusted_sorted[reverse_idx] = running_min
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


__all__ = [
    "DieboldMarianoResult",
    "adjust_pvalues",
    "diebold_mariano_test",
    "loss_from_predictions",
]

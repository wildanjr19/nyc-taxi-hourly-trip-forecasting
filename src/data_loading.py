"""
data_loading.py

Script ini digunakan untuk membaca dataset mentah NYC Taxi hourly dan
melakukan validasi awal sebelum masuk ke tahap EDA/preprocessing.

Kegunaan utama:
- Membaca file CSV dataset hourly dari path yang dikonfigurasi.
- Menstandarkan nama kolom agar konsisten dengan `src.config`.
- Memastikan kolom timestamp dan target `trip_count` tersedia.
- Mendeteksi nilai target yang tidak valid seperti missing, non-numeric,
  negatif, tidak finite, atau bukan bilangan hitung.
- Menyediakan ringkasan loading awal tanpa mengubah urutan waktu data.

Catatan metodologi:
- Script ini tidak melakukan shuffle.
- Script ini tidak melakukan split data.
- Script ini tidak membuat fitur dan tidak memakai informasi masa depan.
- Parsing timezone dan chronological sorting dikerjakan pada tahap
  preprocessing agar tanggung jawab modul tetap jelas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    TARGET_COL,
    TIMESTAMP_COL,
    TIMESTAMP_COL_ALIASES,
)


DEFAULT_COLUMN_ALIASES: Mapping[str, Sequence[str]] = {
    TIMESTAMP_COL: tuple(TIMESTAMP_COL_ALIASES),
}


def _normalize_column_name(column_name: object) -> str:
    """
    Mengubah nama kolom menjadi format snake_case sederhana.
    """
    return str(column_name).strip().lower().replace(" ", "_")


def standardize_column_names(
    df: pd.DataFrame,
    column_aliases: Optional[Mapping[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """
    Menstandarkan nama kolom dataset.

    Proses:
    1. Trim whitespace.
    2. Ubah ke lowercase.
    3. Ganti spasi dengan underscore.
    4. Rename alias kolom penting ke nama standar dari `src.config`.

    Contoh:
    - `pickup_hour` -> `timestamp`

    Returns
    -------
    pd.DataFrame
        Copy dataframe dengan nama kolom standar.
    """
    standardized = df.copy()
    standardized.columns = [_normalize_column_name(col) for col in standardized.columns]

    if standardized.columns.duplicated().any():
        duplicated = standardized.columns[standardized.columns.duplicated()].tolist()
        raise ValueError(
            "Terdapat duplikasi nama kolom setelah standardisasi: "
            f"{duplicated}. Periksa nama kolom dataset mentah."
        )

    aliases = column_aliases or DEFAULT_COLUMN_ALIASES
    rename_map: dict[str, str] = {}

    for standard_name, candidate_aliases in aliases.items():
        normalized_standard = _normalize_column_name(standard_name)
        normalized_aliases = [_normalize_column_name(alias) for alias in candidate_aliases]

        if normalized_standard in standardized.columns:
            continue

        matched_aliases = [alias for alias in normalized_aliases if alias in standardized.columns]
        if len(matched_aliases) > 1:
            raise ValueError(
                f"Kolom alias untuk '{normalized_standard}' ambigu: {matched_aliases}. "
                "Sisakan satu kolom waktu yang valid sebelum loading."
            )

        if matched_aliases:
            rename_map[matched_aliases[0]] = normalized_standard

    if rename_map:
        standardized = standardized.rename(columns=rename_map)

    return standardized


def validate_required_columns(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
    target_col: str = TARGET_COL,
) -> None:
    """
    Memastikan kolom timestamp dan target tersedia.

    Raises
    ------
    ValueError
        Jika salah satu kolom wajib tidak ditemukan.
    """
    required_columns = [timestamp_col, target_col]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        available_columns = list(df.columns)
        raise ValueError(
            "Kolom wajib tidak ditemukan. "
            f"Missing={missing_columns}; available={available_columns}"
        )


def validate_target_values(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    require_integer_like: bool = True,
) -> None:
    """
    Memvalidasi nilai target `trip_count`.

    Target dianggap tidak valid jika:
    - missing,
    - tidak bisa dikonversi ke numerik,
    - bernilai infinite,
    - negatif,
    - bukan integer-like saat `require_integer_like=True`.

    Fungsi ini hanya memvalidasi dan tidak mengubah dataframe input.
    """
    if target_col not in df.columns:
        raise ValueError(f"Kolom target tidak ditemukan: {target_col}")

    raw_target = df[target_col]
    numeric_target = pd.to_numeric(raw_target, errors="coerce")

    missing_count = int(raw_target.isna().sum())
    non_numeric_count = int(numeric_target.isna().sum() - missing_count)
    invalid_messages: list[str] = []

    if missing_count > 0:
        invalid_messages.append(f"missing={missing_count}")

    if non_numeric_count > 0:
        invalid_messages.append(f"non_numeric={non_numeric_count}")

    finite_mask = np.isfinite(numeric_target.to_numpy(dtype=float, na_value=np.nan))
    non_finite_count = int((~finite_mask & numeric_target.notna().to_numpy()).sum())
    if non_finite_count > 0:
        invalid_messages.append(f"non_finite={non_finite_count}")

    valid_numeric = numeric_target[finite_mask]
    negative_count = int((valid_numeric < 0).sum())
    if negative_count > 0:
        invalid_messages.append(f"negative={negative_count}")

    if require_integer_like:
        fractional_count = int(((valid_numeric % 1) != 0).sum())
        if fractional_count > 0:
            invalid_messages.append(f"fractional={fractional_count}")

    if invalid_messages:
        raise ValueError(
            f"Nilai target '{target_col}' tidak valid: "
            + ", ".join(invalid_messages)
        )


def summarize_loaded_data(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
    target_col: str = TARGET_COL,
) -> dict[str, Any]:
    """
    Membuat ringkasan awal dataframe hasil loading.

    Ringkasan ini berguna untuk EDA dan audit awal tanpa menjalankan
    preprocessing atau menyentuh final test.
    """
    validate_required_columns(df, timestamp_col=timestamp_col, target_col=target_col)

    numeric_target = pd.to_numeric(df[target_col], errors="coerce")

    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "timestamp_col": timestamp_col,
        "target_col": target_col,
        "target_missing_count": int(df[target_col].isna().sum()),
        "target_non_numeric_count": int(numeric_target.isna().sum() - df[target_col].isna().sum()),
        "target_negative_count": int((numeric_target.dropna() < 0).sum()),
        "target_min": None if numeric_target.dropna().empty else float(numeric_target.min()),
        "target_max": None if numeric_target.dropna().empty else float(numeric_target.max()),
    }


def load_hourly_taxi_data(
    path: Union[str, Path] = RAW_DATA_PATH,
    *,
    standardize_columns: bool = True,
    validate: bool = True,
    read_csv_kwargs: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """
    Membaca dataset hourly NYC Taxi dari file CSV.

    Parameters
    ----------
    path
        Lokasi file CSV. Default memakai `RAW_DATA_PATH` dari `src.config`.
    standardize_columns
        Jika True, nama kolom distandarkan dan alias timestamp direname
        menjadi `TIMESTAMP_COL`.
    validate
        Jika True, validasi kolom wajib dan nilai target dijalankan.
    read_csv_kwargs
        Opsi tambahan untuk `pandas.read_csv`.

    Returns
    -------
    pd.DataFrame
        DataFrame mentah yang sudah memiliki nama kolom standar dan siap
        masuk ke EDA atau preprocessing.
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path}")

    kwargs = dict(read_csv_kwargs or {})
    df = pd.read_csv(data_path, **kwargs)

    if df.empty:
        raise ValueError(f"Dataset kosong: {data_path}")

    if standardize_columns:
        df = standardize_column_names(df)

    if validate:
        validate_required_columns(df)
        validate_target_values(df)

    return df


__all__ = [
    "DEFAULT_COLUMN_ALIASES",
    "load_hourly_taxi_data",
    "standardize_column_names",
    "summarize_loaded_data",
    "validate_required_columns",
    "validate_target_values",
]

"""
config.py

Konfigurasi global untuk proyek penelitian forecasting NYC Taxi hourly trip_count.

Script ini menjadi fondasi seluruh pipeline agar eksperimen bersifat:
- Modular: semua path, parameter, dan konstanta di satu tempat.
- Reproducible: random seed, timezone policy, dan naming convention konsisten.
- Bebas leakage: aturan split, horizon, dan CV didefinisikan secara eksplisit.

Cara penggunaan:
    from src.config import DATA_DIR, TARGET_COL, FORECAST_HORIZON
    # atau
    from src.config import ensure_dirs

Potensi eror yang diatasi:
1. Path relatif yang pecah saat script dijalankan dari direktori berbeda.
   -> Diatasi dengan Path(__file__).resolve() untuk mendeteksi PROJECT_ROOT.
2. Folder output belum ada saat eksperimen pertama kali menulis file.
   -> Fungsi ensure_dirs() tersedia untuk membuat semua direktori output.
3. Perbedaan timezone antara waktu lokal NYC dan index modeling UTC.
   -> Konstan LOCAL_TZ dan MODELING_TZ mendokumentasikan kebijakan ini.
4. Inkonsistensi nama kolom atau horizon antar script.
   -> Semua nama kolom, horizon, dan seed disentralisasi di sini.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. ROOT DAN DIREKTORI
# ---------------------------------------------------------------------------
# PROJECT_ROOT dihitung dari lokasi file config.py (src/config.py).
# Struktur yang diharapkan: PROJECT_ROOT/src/config.py
CONFIG_FILE = Path(__file__).resolve()
PROJECT_ROOT = CONFIG_FILE.parent.parent

# Direktori data
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Direktori output eksperimen
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
PREPROCESSING_OUTPUT_DIR = OUTPUT_DIR / "preprocessing"
SPLIT_OUTPUT_DIR = OUTPUT_DIR / "splits"
FEATURE_OUTPUT_DIR = OUTPUT_DIR / "features"
EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"
LOGS_DIR = OUTPUT_DIR / "logs"
FINAL_TEST_DIR = OUTPUT_DIR / "final_test"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Subfolder eksperimen per model
PROPHET_OUTPUT_DIR = EXPERIMENTS_DIR / "prophet"
PROPHET_REGRESSOR_BASIC_OUTPUT_DIR = EXPERIMENTS_DIR / "prophet_regressor_basic"
XGB_BASIC_OUTPUT_DIR = EXPERIMENTS_DIR / "xgb_basic"
XGB_ADVANCED_OUTPUT_DIR = EXPERIMENTS_DIR / "xgb_advanced"

# Direktori notebook
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ---------------------------------------------------------------------------
# 2. FILE DATA
# ---------------------------------------------------------------------------
RAW_DATA_FILENAME = "yellow_taxi_hourly_2025_2026_clean.csv"
TRAIN_VAL_FILENAME = "train_val.csv"
FINAL_TEST_FILENAME = "final_test.csv"
FULL_PREPROCESSED_FILENAME = "full_preprocessed.csv"

RAW_DATA_PATH = DATA_DIR / RAW_DATA_FILENAME
TRAIN_VAL_PATH = PROCESSED_DIR / TRAIN_VAL_FILENAME
FINAL_TEST_PATH = PROCESSED_DIR / FINAL_TEST_FILENAME
FULL_PREPROCESSED_PATH = PROCESSED_DIR / FULL_PREPROCESSED_FILENAME
FEATURES_DIR = PROCESSED_DIR / "features"
XGB_BASIC_FEATURES_PATH = FEATURES_DIR / "xgb_basic_train_val_features.csv"
XGB_ADVANCED_FEATURES_PATH = FEATURES_DIR / "xgb_advanced_train_val_features.csv"
PREPROCESSING_SUMMARY_PATH = PREPROCESSING_OUTPUT_DIR / "summaries" / "preprocessing_summary.txt"
PREPROCESSING_METADATA_PATH = PREPROCESSING_OUTPUT_DIR / "summaries" / "preprocessing_metadata.json"
HOLDOUT_SPLIT_SUMMARY_PATH = SPLIT_OUTPUT_DIR / "summaries" / "holdout_split_summary.txt"
HOLDOUT_SPLIT_METADATA_PATH = SPLIT_OUTPUT_DIR / "summaries" / "holdout_split_metadata.json"
TIME_SERIES_CV_SUMMARY_PATH = SPLIT_OUTPUT_DIR / "summaries" / "time_series_cv_summary.txt"
TIME_SERIES_CV_METADATA_PATH = SPLIT_OUTPUT_DIR / "summaries" / "time_series_cv_metadata.json"
TIME_SERIES_CV_FOLDS_PATH = SPLIT_OUTPUT_DIR / "summaries" / "time_series_cv_folds.csv"
FEATURE_ENGINEERING_SUMMARY_PATH = FEATURE_OUTPUT_DIR / "summaries" / "feature_engineering_summary.txt"
FEATURE_ENGINEERING_METADATA_PATH = FEATURE_OUTPUT_DIR / "summaries" / "feature_engineering_metadata.json"
FEATURE_COLUMNS_PATH = FEATURE_OUTPUT_DIR / "summaries" / "feature_columns.json"

# ---------------------------------------------------------------------------
# 3. KOLOM DAN TIMEZONE
# ---------------------------------------------------------------------------
TIMESTAMP_COL = "timestamp"      # Nama kolom waktu standar setelah data loading
TIMESTAMP_COL_ALIASES = ["pickup_hour"]  # Nama kolom waktu yang mungkin ada pada data mentah
TARGET_COL = "trip_count"        # Variabel target forecasting

LOCAL_TZ = "America/New_York"    # Timezone lokal NYC untuk calendar features
MODELING_TZ = "UTC"              # Timezone untuk index modeling jika tersedia

# ---------------------------------------------------------------------------
# 4. PARAMETER FORECASTING DAN VALIDASI
# ---------------------------------------------------------------------------
FORECAST_HORIZON = 24            # Horizon prediksi: 24 jam ke depan

# Time Series Cross Validation
CV_N_FOLDS = 5
CV_VAL_HORIZON_HOURS = 168       # 7 hari validasi per fold
CV_GAP_HOURS = 0                 # Gap antar train dan validation (default 0)

# Hold-out split
FINAL_TEST_DAYS = 30             # 30 hari terakhir sebagai final test

# Reproducibility
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 5. SEARCH SPACE HYPERPARAMETER
# ---------------------------------------------------------------------------
# Prophet search space 
PROPHET_SEARCH_SPACE = {
    "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.5],
    "seasonality_prior_scale": [1.0, 5.0, 10.0],
    "seasonality_mode": ["additive", "multiplicative"],
}

# XGBoost search space 
XGB_SEARCH_SPACE = {
    "n_estimators": [300, 600, 1000],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 5, 10],
}

# ---------------------------------------------------------------------------
# 6. FEATURE SETS 
# ---------------------------------------------------------------------------
# XGBoost-Basic: lag dan calendar minimal
XGB_BASIC_LAGS = [1, 24, 168]
XGB_BASIC_CALENDAR = ["hour", "day_of_week"]

# XGBoost-Advanced: lag lebih banyak, rolling stats, calendar richer
XGB_ADVANCED_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
XGB_ADVANCED_ROLLING_WINDOWS = [3, 24, 168]
XGB_ADVANCED_ROLLING_STD_WINDOWS = [24]
XGB_ADVANCED_CALENDAR = ["hour", "day_of_week", "is_weekend", "month"]

# ---------------------------------------------------------------------------
# 7. LOGGING DAN RUNTIME
# ---------------------------------------------------------------------------
RUNTIME_LOG_CSV = LOGS_DIR / "runtime_logs.csv"
EXPERIMENT_RUNS_JSONL = LOGS_DIR / "experiment_runs.jsonl"

# Schema kolom runtime log (untuk referensi konsistensi antar script)
RUNTIME_LOG_COLUMNS = [
    "timestamp_run",
    "experiment_name",
    "model_name",
    "feature_set",
    "fold",
    "parameter_set_id",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "n_train_rows",
    "n_prediction_rows",
    "train_time_seconds",
    "prediction_time_seconds",
    "total_runtime_seconds",
    "status",
    "error_message",
]

# ---------------------------------------------------------------------------
# 8. METRIC CONFIGURATION
# ---------------------------------------------------------------------------
METRICS = ["mae", "rmse", "mape", "smape"]
PRIMARY_METRIC = "mae"   # Metric utama untuk memilih best configuration
SECONDARY_METRIC = "rmse"

# ---------------------------------------------------------------------------
# 9. HELPER UNTUK MEMBUAT DIREKTORI
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    """
    Memastikan seluruh direktori output yang diperlukan sudah ada.
    Fungsi ini idempoten: aman dipanggil berkali-kali.
    Gunakan di awal script eksperimen untuk mencegah FileNotFoundError.
    """
    dirs = [
        PROCESSED_DIR,
        FEATURES_DIR,
        EDA_DIR / "figures",
        EDA_DIR / "summaries",
        PREPROCESSING_OUTPUT_DIR / "summaries",
        SPLIT_OUTPUT_DIR / "summaries",
        FEATURE_OUTPUT_DIR / "summaries",
        PROPHET_OUTPUT_DIR,
        PROPHET_REGRESSOR_BASIC_OUTPUT_DIR,
        XGB_BASIC_OUTPUT_DIR,
        XGB_ADVANCED_OUTPUT_DIR,
        LOGS_DIR,
        FINAL_TEST_DIR / "metrics",
        FINAL_TEST_DIR / "predictions",
        FINAL_TEST_DIR / "figures",
        REPORTS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 10. VALIDASI KONFIGURASI SAAT IMPORT
# ---------------------------------------------------------------------------
def _validate_config() -> None:
    """
    Validasi cepat saat modul di-import untuk mendeteksi masalah konfigurasi.
    - Pastikan dataset mentah ada.
    - Pastikan folder src berada di lokasi yang sesuai.
    - Pastikan parameter logis (horizon positif, fold >= 2, dst).
    """
    if not RAW_DATA_PATH.exists():
        # Bukan eror fatal karena dataset mungkin belum dipindahkan,
        # tapi perlu diperhatikan sebelum preprocessing.
        import warnings
        warnings.warn(
            f"Dataset mentah tidak ditemukan di: {RAW_DATA_PATH}\n"
            "Pastikan file data berada di direktori yang benar sebelum menjalankan pipeline.",
            UserWarning,
        )

    if not CONFIG_FILE.parent.name == "src":
        raise RuntimeError(
            f"config.py diharapkan berada di dalam folder 'src/', "
            f"tapi ditemukan di: {CONFIG_FILE.parent}"
        )

    if FORECAST_HORIZON <= 0:
        raise ValueError("FORECAST_HORIZON harus > 0")

    if CV_N_FOLDS < 2:
        raise ValueError("CV_N_FOLDS minimal 2 agar cross validation bermakna")

    if CV_VAL_HORIZON_HOURS < FORECAST_HORIZON:
        raise ValueError(
            "CV_VAL_HORIZON_HOURS harus >= FORECAST_HORIZON "
            f"({CV_VAL_HORIZON_HOURS} < {FORECAST_HORIZON})"
        )

    if FINAL_TEST_DAYS <= 0:
        raise ValueError("FINAL_TEST_DAYS harus > 0")


_validate_config()

# Panduan Reproduksi Proyek

Dokumen ini menjelaskan cara menjalankan ulang proyek forecasting NYC Taxi
hourly `trip_count` dari kondisi fresh clone. README berfungsi sebagai halaman
depan ringkas, sedangkan dokumen ini dipakai sebagai panduan operasional.

Gunakan panduan ini ketika repository di-clone di komputer lain, misalnya
komputer lab, laptop teman, atau environment baru.

## 1. Prasyarat

Pastikan perangkat memiliki:

- Python 3.9 atau lebih baru
- Git
- Akses terminal, disarankan PowerShell untuk Windows
- Ruang penyimpanan yang cukup untuk dataset, output eksperimen, dan plots

Library utama yang digunakan:

- pandas
- numpy
- matplotlib
- seaborn
- prophet
- xgboost
- scikit-learn
- pytz

Semua dependency utama tercatat di [requirements.txt](requirements.txt) dan
[pyproject.toml](pyproject.toml).

## 2. Clone Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

Setelah masuk ke folder proyek, pastikan struktur utama tersedia:

```text
README.md
GUIDE.md
requirements.txt
src/
data/ atau instruksi penempatan dataset
```

Semua perintah Python pada panduan ini sebaiknya dijalankan dari root
repository, bukan dari dalam folder `src/`.

## 3. Setup Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jika PowerShell menolak aktivasi environment karena execution policy, jalankan:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS atau Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Opsional, jika ingin menjalankan package secara editable:

```bash
pip install -e .
```

## 4. Persiapan Dataset

Dataset utama harus berada pada path:

```text
data/yellow_taxi_hourly_2025_2026_clean.csv
```

Dataset minimal harus memiliki:

| Kolom | Keterangan |
|---|---|
| `pickup_hour` atau `timestamp` | timestamp hourly |
| `trip_count` | jumlah perjalanan taksi per jam |

Konfigurasi path dataset, target, timezone, horizon, dan CV tersedia di:

```text
src/config.py
```

Catatan penting:

- Jangan mengubah final test set setelah dibuat.
- Jangan melakukan random split.
- Calendar features dibuat berdasarkan waktu lokal NYC.
- Index modeling menggunakan UTC.

## 5. Urutan Pipeline Utama

Urutan eksekusi yang direkomendasikan:

```bash
python -m src.preprocessing
python -m src.splits
python -m src.features
python -m src.experiments.tune_prophet
python -m src.experiments.tune_xgb_basic
python -m src.experiments.tune_xgb_advanced
python -m src.experiments.experiment_a
python -m src.experiments.experiment_b
```

Tahap tersebut mencakup:

| Tahap | Script | Output utama |
|---|---|---|
| Preprocessing | `python -m src.preprocessing` | `data/processed/full_preprocessed.csv` |
| Hold-out split | `python -m src.splits` | `data/processed/train_val.csv`, `data/processed/final_test.csv` |
| Feature engineering | `python -m src.features` | `data/processed/features/` |
| Tuning Prophet | `python -m src.experiments.tune_prophet` | `outputs/experiments/prophet/` |
| Tuning XGBoost-Basic | `python -m src.experiments.tune_xgb_basic` | `outputs/experiments/xgb_basic/` |
| Tuning XGBoost-Advanced | `python -m src.experiments.tune_xgb_advanced` | `outputs/experiments/xgb_advanced/` |
| Experiment A | `python -m src.experiments.experiment_a` | `outputs/reports/experiment_a_prophet_vs_xgb_basic.md` |
| Experiment B | `python -m src.experiments.experiment_b` | `outputs/reports/experiment_b_xgb_basic_vs_xgb_advanced.md` |

Catatan scope:

- Panduan ini mengikuti script CLI yang tersedia pada repository saat ini.
- Tahap retraining dan final testing tetap tercatat sebagai bagian metodologi
  penelitian di [RESEARCH_PIPELINE.md](RESEARCH_PIPELINE.md) dan
  [PLAN.md](PLAN.md).
- Jika script retraining atau final testing ditambahkan kemudian, jalankan
  tahap tersebut setelah tuning best configurations dan sebelum penyusunan
  kesimpulan final.

## 6. Smoke Test dengan Runtime Lebih Ringan

Full tuning dapat memakan waktu lama. Untuk memastikan environment berjalan,
gunakan parameter terbatas:

```bash
python -m src.preprocessing
python -m src.splits
python -m src.features
python -m src.experiments.tune_prophet --max-parameter-sets 2 --skip-plots
python -m src.experiments.tune_xgb_basic --max-parameter-sets 3 --skip-plots
python -m src.experiments.tune_xgb_advanced --max-parameter-sets 3 --skip-plots
python -m src.experiments.experiment_a --skip-plots
python -m src.experiments.experiment_b --skip-plots
```

Catatan:

- Smoke test hanya untuk validasi instalasi dan alur program.
- Hasil smoke test tidak dianggap sebagai hasil penelitian final.
- Jika ingin menghasilkan benchmark penuh, jalankan tuning tanpa
  `--max-parameter-sets`.

## 7. Output dan Lokasi Artefak

Artefak penting tersimpan di folder berikut:

```text
data/processed/
outputs/eda/
outputs/preprocessing/
outputs/splits/
outputs/features/
outputs/experiments/
outputs/logs/
outputs/reports/
```

File yang umum dicek setelah pipeline selesai:

| File | Fungsi |
|---|---|
| `outputs/logs/runtime_logs.csv` | log time cost computing terpusat |
| `outputs/logs/experiment_runs.jsonl` | log metadata run eksperimen |
| `outputs/experiments/prophet/best_params.json` | best params Prophet |
| `outputs/experiments/xgb_basic/best_params.json` | best params XGBoost-Basic |
| `outputs/experiments/xgb_advanced/best_params.json` | best params XGBoost-Advanced |
| `outputs/experiments/*/cv_metrics.csv` | metrics per fold dan parameter |
| `outputs/experiments/*/cv_predictions.csv` | prediction output CV |
| `outputs/reports/experiment_a_prophet_vs_xgb_basic.md` | report Experiment A |
| `outputs/reports/experiment_b_xgb_basic_vs_xgb_advanced.md` | report Experiment B |

Setiap eksperimen menyimpan:

- parameter
- metrics
- prediction outputs
- plots, kecuali menjalankan `--skip-plots`
- metadata
- training time
- prediction time
- total runtime

## 8. Exploratory Data Analysis

Notebook EDA tersedia di:

```text
notebooks/01_EDA.ipynb
```

Output EDA yang sudah atau akan dihasilkan:

```text
outputs/eda/figures/
outputs/eda/summaries/
```

Jika environment belum memiliki Jupyter, install terlebih dahulu:

```bash
pip install jupyter
```

Lalu buka notebook:

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

EDA digunakan untuk memahami trend, seasonality, pola weekday/weekend,
autocorrelation, missing timestamp, dan karakteristik target.

## 9. Aturan Leakage Prevention

Sebelum menggunakan hasil eksperimen, pastikan prinsip berikut tetap terpenuhi:

- Tidak ada shuffle data.
- Final test set tidak dipakai saat tuning.
- Split dilakukan berdasarkan waktu.
- Rolling features menggunakan pola `shift` lalu `rolling`.
- Feature matrix XGBoost hanya digunakan untuk training rows.
- Validation XGBoost tidak diprediksi langsung dari feature rows validation.
- Test XGBoost tidak boleh diprediksi langsung dari feature rows test.
- Recursive forecasting digunakan untuk horizon 24 jam.
- Actual validation/test hanya digunakan sebagai label evaluasi.
- Runtime training dan prediction dicatat melalui `src/tracking.py`.

## 10. Membaca Hasil Eksperimen

Untuk melihat hasil tuning model:

```text
outputs/experiments/prophet/
outputs/experiments/xgb_basic/
outputs/experiments/xgb_advanced/
```

Untuk membaca ringkasan komparatif:

```text
outputs/reports/experiment_a_prophet_vs_xgb_basic.md
outputs/reports/experiment_b_xgb_basic_vs_xgb_advanced.md
```

Experiment A membandingkan:

```text
Prophet vs XGBoost-Basic
```

Experiment B membandingkan:

```text
XGBoost-Basic vs XGBoost-Advanced
```

Metric utama untuk pemilihan best configuration adalah MAE. RMSE, MAPE, dan
sMAPE digunakan sebagai metric pendukung.

## 11. Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Pastikan perintah dijalankan dari root repository:

```bash
cd <repository-folder>
python -m src.preprocessing
```

Jika masih bermasalah, install package secara editable:

```bash
pip install -e .
```

### Dataset tidak ditemukan

Pastikan file berada di:

```text
data/yellow_taxi_hourly_2025_2026_clean.csv
```

Nama file dikonfigurasi di `src/config.py` melalui `RAW_DATA_FILENAME`.

### Aktivasi `.venv` gagal di PowerShell

Gunakan:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Perintah tersebut hanya berlaku untuk sesi terminal saat ini.

### Instalasi Prophet gagal

Pastikan `pip` sudah diperbarui:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install prophet
```

Jika menggunakan komputer lab, pastikan environment memiliki akses internet dan
izin untuk menginstall package Python.

### Runtime tuning terlalu lama

Gunakan smoke test:

```bash
python -m src.experiments.tune_xgb_basic --max-parameter-sets 3 --skip-plots
```

Untuk hasil penelitian penuh, jalankan ulang tanpa `--max-parameter-sets`.

### Report Experiment A atau B gagal dibuat

Pastikan tuning tiga model sudah selesai dan menghasilkan:

```text
outputs/experiments/prophet/best_params.json
outputs/experiments/xgb_basic/best_params.json
outputs/experiments/xgb_advanced/best_params.json
```

Experiment A membutuhkan artefak Prophet dan XGBoost-Basic. Experiment B
membutuhkan artefak XGBoost-Basic dan XGBoost-Advanced.

## 12. Catatan untuk Push ke GitHub

Jika repository akan dibagikan melalui GitHub, periksa ukuran file sebelum
push. Dataset dan folder `outputs/` dapat berukuran besar. Jika tidak ingin
menyimpan data atau output di GitHub, tetap pastikan struktur folder dan
instruksi path di dokumen ini jelas.

Untuk reproduksi oleh orang lain, minimal repository harus menyediakan:

- kode di `src/`
- `requirements.txt`
- `pyproject.toml`
- `README.md`
- `GUIDE.md`
- `RESEARCH_PIPELINE.md`
- `FEATURE_SET.md`
- instruksi memperoleh atau meletakkan dataset

## 13. Dokumen Pendukung

- [README.md](README.md): ringkasan proyek dan quick start.
- [RESEARCH_PIPELINE.md](RESEARCH_PIPELINE.md): alur metodologi penelitian.
- [FEATURE_SET.md](FEATURE_SET.md): daftar fitur basic dan advanced.
- [PLAN.md](PLAN.md): rencana implementasi dan tracking progress.
- [AGENTS.md](AGENTS.md): aturan global proyek dan checklist leakage prevention.

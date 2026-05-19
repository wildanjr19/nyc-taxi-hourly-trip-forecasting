"""
Script tahap 10: Hyperparameter tuning XGBoost-Basic.

Validation XGBoost diprediksi dengan recursive forecasting melalui
forecast_validation_window(), bukan model.predict() langsung pada feature
rows validation.

Contoh:
    python -m src.experiments.tune_xgb_basic
    python -m src.experiments.tune_xgb_basic --max-parameter-sets 3 --skip-plots
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from src.config import CV_N_FOLDS, CV_VAL_HORIZON_HOURS, FORECAST_HORIZON
from src.features import BASIC_FEATURE_SET
from src.experiments.tuning_utils import run_xgb_tuning


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune XGBoost-Basic with time series CV.")
    parser.add_argument("--max-parameter-sets", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=CV_N_FOLDS)
    parser.add_argument("--val-horizon", type=int, default=CV_VAL_HORIZON_HOURS)
    parser.add_argument("--forecast-horizon", type=int, default=FORECAST_HORIZON)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    metadata = run_xgb_tuning(
        feature_set=BASIC_FEATURE_SET,
        n_folds=args.n_folds,
        val_horizon=args.val_horizon,
        forecast_horizon=args.forecast_horizon,
        max_parameter_sets=args.max_parameter_sets,
        skip_plots=args.skip_plots,
    )
    best = metadata["best"]
    print("Tuning XGBoost-Basic selesai.")
    print(f"Best parameter set: {best['parameter_set_id']}")
    print(f"Best params: {best['params']}")
    print(f"Output dir: {metadata['outputs']['base']}")


if __name__ == "__main__":
    main()

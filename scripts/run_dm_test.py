"""
Wrapper untuk menjalankan Diebold-Mariano test final test revisi 1.

Contoh:
    python scripts/run_dm_test.py
    python scripts/run_dm_test.py --skip-plots
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.dm_test import main


if __name__ == "__main__":
    main()

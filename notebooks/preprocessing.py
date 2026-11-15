"""Lets the notebooks keep `from preprocessing import ...`; the code lives in src/loan_default."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loan_default.preprocessing import *  # noqa: E402, F403

"""Shared test setup.

XGBoost and PyTorch each ship their own OpenMP runtime on macOS, and the two do not mix in one process: loading PyTorch
first crashes XGBoost, and multi-threaded PyTorch ops crash after XGBoost. Loading XGBoost first and keeping PyTorch to
one thread is safe, and the test models are tiny. The training scripts never load both libraries in one process.

Tests that read METR-LA are skipped, with a message, until `make data` has been run.
"""
# ruff: noqa: I001
from pathlib import Path

import pytest
import xgboost  # noqa: F401  (must be imported before torch; see above)
import torch

torch.set_num_threads(1)
RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "metr-la.h5"
NEEDS_DATA = {"data", "speed", "small", "built", "full_plan"}


def pytest_collection_modifyitems(items):
    if RAW.exists():
        return
    skip = pytest.mark.skip(reason="METR-LA is not downloaded: run `make data`")
    for item in items:
        if NEEDS_DATA & set(item.fixturenames):
            item.add_marker(skip)

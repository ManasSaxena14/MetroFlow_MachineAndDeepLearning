"""Shared test setup.

XGBoost and PyTorch each ship their own OpenMP runtime on macOS, and the two do not mix in one process: loading PyTorch
first crashes XGBoost, and multi-threaded PyTorch ops crash after XGBoost. Loading XGBoost first and keeping PyTorch to
one thread is safe, and the test models are tiny. The training scripts never load both libraries in one process.
"""
import xgboost  # noqa: F401, I001  (must be imported before torch; see above)
import torch  # noqa: I001

torch.set_num_threads(1)

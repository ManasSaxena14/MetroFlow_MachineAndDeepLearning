"""Hybrid checks: chronological fitting split, blend weights, and the residual corrector's layout."""
import numpy as np

from metroflow.features.build import FEATURE_NAMES
from metroflow.models import hybrid


def test_chrono_split_is_chronological_and_by_whole_windows():
    fit, stop = hybrid.chrono_split(100 * hybrid.N_SENSORS)
    assert fit.sum() == 75 * hybrid.N_SENSORS and stop.sum() == 25 * hybrid.N_SENSORS
    assert not (fit & stop).any()
    assert np.flatnonzero(fit).max() < np.flatnonzero(stop).min()      # fitting rows all come before stopping rows


def test_blend_recovers_the_better_model_and_stays_in_range():
    rng = np.random.default_rng(0)
    Y = rng.uniform(20, 70, (5000, 12)).astype(np.float32)
    good, bad = Y + rng.normal(0, 1, Y.shape), Y + rng.normal(0, 8, Y.shape)
    w = hybrid.fit_blend(good, bad, Y, np.ones_like(Y, bool), np.ones(len(Y), bool))
    assert ((w >= 0) & (w <= 1)).all() and (w > 0.8).all()


def test_corrector_learns_a_systematic_bias_and_keeps_layout():
    rng = np.random.default_rng(1)
    n = 40 * hybrid.N_SENSORS
    F = rng.normal(size=(n, len(FEATURE_NAMES))).astype(np.float32)
    Y = rng.uniform(30, 70, (n, 12)).astype(np.float32)
    M = np.ones_like(Y, bool)
    M[::7] = False                                                      # masked targets are ignored
    dl = Y - 3.0                                                        # the DL model is always 3 mph too slow
    ml = Y + rng.normal(0, 5, Y.shape).astype(np.float32)
    fit, stop = hybrid.chrono_split(n)
    models, _ = hybrid.fit_corrector(F, dl, ml, Y, M, fit, stop, horizons=[1, 12], rounds=300)
    out = hybrid.apply_corrector(models, F, dl, ml)
    assert out.shape == dl.shape
    assert np.allclose(out[:, 1:11], dl[:, 1:11])                       # untouched horizons stay as the DL forecast
    assert np.abs(out[:, 0] - Y[:, 0]).mean() < 1.0 < np.abs(dl[:, 0] - Y[:, 0]).mean()

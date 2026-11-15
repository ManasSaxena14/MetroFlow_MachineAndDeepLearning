"""Phase 6 checks: day-level error aggregation and the significance tests behave as they should."""
import numpy as np

from src.eval import compare


def test_daily_errors_sum_by_target_day_and_ignore_masked():
    Y = np.array([[50.0], [60.0], [np.nan], [40.0]]).repeat(12, 1)
    P = Y + 2.0
    P[2] = 999.0                                                   # garbage where the sensor was down
    M = ~np.isnan(Y)
    days = np.array([[1], [1], [2], [2]]).repeat(12, 1)
    err, n = compare.daily_errors(P, Y, M, days, h=12)
    assert np.allclose(err, [4.0, 2.0]) and np.allclose(n, [2, 1])


def test_dm_test_detects_a_consistent_gain_and_not_noise():
    rng = np.random.default_rng(0)
    stat, p = compare.dm_test(0.3 + rng.normal(0, 0.1, 24))       # better on every one of 24 days
    assert stat > 0 and p < 0.001
    _, p_noise = compare.dm_test(rng.normal(0, 0.1, 24))
    assert p_noise > 0.01
    assert compare.dm_test(np.zeros(24)) == (0.0, 1.0)             # identical forecasts

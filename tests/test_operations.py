"""Alerts, forecast ranges and failure testing: definitions, causality, calibration and layout."""
import numpy as np
import pytest

from metroflow.data import preprocess as pp
from metroflow.data.load import load_speed
from metroflow.eval import alerts, robustness
from metroflow.eval import uncertainty as unc
from metroflow.train import hybrid_cv

FF = np.array([60.0, 60.0])


# ----------------------------------------------------------------------------- alerts
def series(*levels):
    """One (T, 2) speed array: sensor 0 follows `levels`, sensor 1 always flows."""
    v = np.full((len(levels), 2), 60.0)
    v[:, 0] = levels
    return v


def test_jam_start_needs_persistence_and_a_clear_half_hour():
    one_dip = series(*[60] * 10, 20, *[60] * 10)
    assert not alerts.jam_starts(one_dip, FF).any()                       # a single slow reading is not a jam
    jam = series(*[60] * 10, *[20] * 6, *[60] * 3, *[20] * 6)
    s = alerts.jam_starts(jam, FF)
    assert list(np.flatnonzero(s[:, 0])) == [10]                          # the relapse after 15 clear minutes is the same jam
    assert not s[:, 1].any()


def test_jam_start_ignores_outages():
    v = series(*[60] * 10, *[20] * 6)
    v[11, 0] = np.nan
    assert list(np.flatnonzero(alerts.jam_starts(v, FF)[:, 0])) == []     # no three consecutive REAL slow readings at the start
    v = series(*[60] * 10, *[20] * 6)
    v[4:9, 0] = np.nan
    assert not alerts.jam_starts(v, FF)[:, 0].any()                       # too little real history to know it was clear


def test_alerts_respect_eligibility_and_quiet_period():
    score = np.ones((30, 2))
    ok = np.ones_like(score, bool)
    ok[:, 1] = False
    a = alerts.raise_alerts(score, ok, 0.5, quiet=12)
    assert list(np.flatnonzero(a[:, 0])) == [0, 12, 24] and not a[:, 1].any()


def test_match_lead_time_hits_and_false_alarms():
    S, H = 60, 12
    a, st, real = np.zeros((S, 2), bool), np.zeros((S + H, 2), bool), np.ones((S + H, 2), bool)
    st[40, 0] = True
    a[30, 0] = a[36, 0] = True                                            # 50 and 20 minutes before the jam
    a[5, 0] = True                                                        # 175 minutes before: a false alarm
    a[40, 1] = True                                                       # sensor 1 never jams: a false alarm
    m = alerts.match(a, st, real)
    assert m["hit"].sum() == 2 and m["false"].sum() == 2 and list(m["lead"]) == [50]
    r = alerts.metrics(m, days=2)
    assert r["false_alarms_per_day"] == 1 and r["recall_30min_pct"] == 100 and r["precision_pct"] == 50


def test_alert_at_the_jam_itself_is_not_a_warning():
    S, H = 60, 12
    a, st, real = np.zeros((S, 1), bool), np.zeros((S + H, 1), bool), np.ones((S + H, 1), bool)
    st[40], a[40] = True, True
    m = alerts.match(a, st, real)
    assert list(m["lead"]) == [0] and m["hit"].sum() == 0


def test_alerts_into_an_outage_are_not_scored():
    S, H = 60, 12
    a, st, real = np.zeros((S, 1), bool), np.zeros((S + H, 1), bool), np.ones((S + H, 1), bool)
    a[20] = True
    real[21:33] = False
    assert alerts.match(a, st, real)["scored"].sum() == 0


def test_threshold_keeps_the_false_alarm_budget():
    rng = np.random.default_rng(0)
    S, H, n = 600, 12, 20
    score, st = rng.random((S, n)), rng.random((S + H, n)) < 0.002
    ok, real = np.ones((S, n), bool), np.ones((S + H, n), bool)
    thr = alerts.threshold_for_budget(score, ok, st, real, days=2, budget=30)
    n_false = alerts.match(alerts.raise_alerts(score, ok, thr), st, real)["false"].sum() / 2
    assert 20 <= n_false <= 30


# ----------------------------------------------------------------------------- uncertainty
@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    n = 40_000
    ff = np.full(n, 65.0)
    f = rng.uniform(15, 65, (n, 12))
    sd = np.where(f < 32.5, 8.0, 2.0)                                     # congested forecasts are four times as uncertain
    y = np.maximum(f + rng.normal(0, 1, f.shape) * sd, 0.5)
    return f, rng.random((n, 12)), y, rng.random((n, 12)) > 0.1, ff


def test_ranges_reach_their_target_and_adapt(toy):
    f, g, y, m, ff = toy
    half = len(f) // 2
    c = unc.GroupedConformal().fit(f[:half], g[:half], y[:half], m[:half], ff[:half])
    lo, hi = c.interval(f[half:, 5], g[half:, 5], ff[half:], 5, 0.8)
    inside = (y[half:, 5] >= lo) & (y[half:, 5] <= hi)
    jam = f[half:, 5] < 32.5
    assert abs(inside.mean() - 0.8) < 0.015 and abs(inside[jam].mean() - 0.8) < 0.03
    assert (hi - lo)[jam].mean() > 2.5 * (hi - lo)[~jam].mean()


def test_masked_targets_never_enter_calibration(toy):
    f, g, y, m, ff = toy
    y2 = np.where(m, y, np.nan)
    a, b = unc.GroupedConformal().fit(f, g, y, m, ff), unc.GroupedConformal().fit(f, g, y2, m, ff)
    assert np.array_equal(a.table, b.table)


def test_probability_matches_frequency_and_stays_in_range(toy):
    f, g, y, m, ff = toy
    half = len(f) // 2
    c = unc.GroupedConformal().fit(f[:half], g[:half], y[:half], m[:half], ff[:half])
    p = c.prob_below(f[half:, 2], g[half:, 2], ff[half:], 2)
    hit = y[half:, 2] < 0.5 * ff[half:]
    assert 0 <= p.min() and p.max() <= 1
    mid = (p > 0.3) & (p < 0.7)
    assert abs(p[mid].mean() - hit[mid].mean()) < 0.08


def test_interval_score_punishes_misses():
    y, lo, hi = np.array([50.0, 50.0]), np.array([45.0, 52.0]), np.array([55.0, 62.0])
    s = unc.interval_score(y, lo, hi, 0.8)
    assert s[0] == 10 and s[1] == pytest.approx(10 + 10 * 2)


def test_calibration_folds_are_separated():
    t0 = np.datetime64("2012-05-23T00:00")
    tt = (t0 + np.arange(10 * 288) * np.timedelta64(5, "m"))[:, None].repeat(12, 1)
    folds = hybrid_cv.day_folds(tt)
    assert set(folds) == {0, 1, 2, 3} and (folds[:288] == 0).all() and (folds[288:576] == 1).all()
    fit = hybrid_cv.fit_mask(folds, 1)
    assert not fit[folds == 1].any()
    assert not fit[288 - hybrid_cv.GAP:576 + hybrid_cv.GAP].any() and fit[288 - hybrid_cv.GAP - 1] and fit[576 + hybrid_cv.GAP]


# ----------------------------------------------------------------------------- failure testing
@pytest.fixture(scope="module")
def small():
    cfg = pp.load_config()
    speed = load_speed().iloc[:3000]
    ds = pp.build_dataset(speed, cfg, "strict")
    v = speed.values.astype(np.float32)
    raw = np.where(v > 0, v, np.nan)
    slots = (speed.index.hour * 12 + speed.index.minute // 5).to_numpy()
    tab = pp.slot_mean_table(raw, speed.index, ds["train_end"])
    return ds, raw, (pp.time_features(speed.index), tab, slots, ds["scaler"], cfg)


def test_rebuilt_inputs_equal_the_pipeline_when_nothing_is_broken(small):
    ds, raw, rest = small
    s = ds["splits"]["test"]
    assert np.array_equal(robustness.rebuild_inputs(s["origin_idx"], raw, *rest), s["X"])


def test_delay_hides_exactly_the_late_readings(small):
    ds, raw, rest = small
    o = ds["splits"]["test"]["origin_idx"]
    X = robustness.rebuild_inputs(o, raw, *rest, delay=3)
    assert not X[:, -3:, :, 1].any()
    assert np.array_equal(X[:, :-3], ds["splits"]["test"]["X"][:, :-3])
    changed = raw.copy()
    changed[int(o[0]) - 2:] = 1.0                                         # readings that have not arrived cannot matter
    assert np.array_equal(robustness.rebuild_inputs(o[:1], changed, *rest, delay=3), X[:1])


def test_corruption_touches_only_what_it_should(small):
    _, raw, _ = small
    ff = np.nanpercentile(raw, 85, axis=0)
    for kind in ("dead", "stuck"):
        s = {"kind": kind, "level": 20, "draw": 1}
        out, aff = robustness.corrupt(raw, s, ff)[0], robustness.affected_sensors(s, 207)
        assert aff.sum() == 41 and np.array_equal(out[:, ~aff], raw[:, ~aff], equal_nan=True)
        if kind == "dead":
            assert np.isnan(out[:, aff]).all()
        else:
            assert np.array_equal(np.isnan(out), np.isnan(raw))          # silent: still looks like a real reading
    lo, hi = (robustness.affected_sensors({"kind": "dead", "level": k, "draw": 0}, 207) for k in (10, 30))
    assert (hi | ~lo).all()                                               # 30% contains the 10%
    spikes = robustness.corrupt(raw, {"kind": "spikes", "level": 10, "draw": 0}, ff)[0]
    changed = (spikes != raw) & ~np.isnan(raw)
    assert 0.09 < changed.sum() / (~np.isnan(raw)).sum() < 0.11 and np.array_equal(np.isnan(spikes), np.isnan(raw))

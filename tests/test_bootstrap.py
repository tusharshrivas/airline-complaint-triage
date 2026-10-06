import numpy as np

from triage.stats import bootstrap_macro_f1_diff, macro_f1_idx

LABELS = ["a", "b", "c"]


def data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    y = rng.choice(LABELS, n)
    good = np.where(rng.random(n) < 0.8, y, rng.choice(LABELS, n))
    bad = np.where(rng.random(n) < 0.5, y, rng.choice(LABELS, n))
    return y, good, bad


def test_deterministic_with_seed():
    y, a, b = data()
    r1 = bootstrap_macro_f1_diff(y, a, b, LABELS, 200, 42)
    r2 = bootstrap_macro_f1_diff(y, a, b, LABELS, 200, 42)
    assert r1 == r2
    assert r1 != bootstrap_macro_f1_diff(y, a, b, LABELS, 200, 43)


def test_detects_clear_difference_and_includes_zero_for_identical():
    y, a, b = data()
    clear = bootstrap_macro_f1_diff(y, a, b, LABELS, 300, 42)
    assert clear["point_estimate"] > 0 and not clear["includes_zero"]
    same = bootstrap_macro_f1_diff(y, a, a, LABELS, 300, 42)
    assert same["includes_zero"] and same["point_estimate"] == 0


def test_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score

    y, a, _ = data()
    idx = {lab: i for i, lab in enumerate(LABELS)}
    yt, yp = np.array([idx[v] for v in y]), np.array([idx[v] for v in a])
    assert abs(macro_f1_idx(yt, yp, 3) - f1_score(yt, yp, average="macro")) < 1e-9

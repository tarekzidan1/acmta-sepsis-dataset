import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from acmta_sepsis import evaluate as E


def test_weighted_metrics_match_sklearn_with_ties():
    rng = np.random.RandomState(0)
    y = (rng.rand(3000) < 0.1).astype(int); p = np.round(rng.rand(3000) * 0.6 + 0.3 * y, 1)
    o, g = E._tie_groups(p); w = np.ones(len(y))
    assert abs(E._weighted_auc(y[o].astype(float), w, g) - roc_auc_score(y, p)) < 1e-12
    assert abs(E._weighted_ap(y[o].astype(float), w, g) - average_precision_score(y, p)) < 1e-12


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.RandomState(1)
    y = (rng.rand(4000) < 0.1).astype(int); p = rng.rand(4000) * 0.5 + 0.4 * y; pat = np.arange(4000) // 8
    ci = E.bootstrap_ci(y, p, pat, n_boot=200)
    assert ci['auroc'][0] <= roc_auc_score(y, p) <= ci['auroc'][1]


def test_delong_identical_and_holm():
    rng = np.random.RandomState(2)
    y = (rng.rand(500) < 0.3).astype(int); p = rng.rand(500) + y
    r = E.delong_test(y, p, p)
    assert r['p'] == 1.0 or r['z'] == 0.0
    adj = E.holm([0.01, 0.02, 0.04])
    assert adj == sorted(adj) and abs(adj[0] - 0.03) < 1e-12


def test_cucts_and_lead_time_toy():
    y = np.array([0, 0, 0, 0, 1, 1]); p = np.array([0.1, 0.2, 0.3, 0.35, 0.8, 0.4])
    stay = np.zeros(6, int); hours = np.array([24])
    curve = E.operating_curve(y, p, stay, hours)
    sel = E.cucts(curve, sens_floor=1.0, budget=10)
    assert sel['feasible'] and abs(sel['threshold'] - 0.4) < 1e-12
    t = np.array([0, 1, 2, 3, 4, 5]); t0 = np.array([12])
    lt = E.lead_times(stay, t, y, p, t0, 0.4)
    assert lt['first_true_positive_alarm']['median'] == 8.0     # first TP alarm at t=4 -> lead 12-4


def test_ece_perfect_calibration_near_zero():
    rng = np.random.RandomState(3)
    p = rng.rand(200000); y = (rng.rand(200000) < p).astype(int)
    e, _ = E.ece(y, p)
    assert e < 0.01


def test_rule_scores(syn_prep):
    from acmta_sepsis.scores import qsofa, sofa_proxy, news_proxy
    pp, co, sp = syn_prep
    rows = np.arange(200)
    for fn, mx in ((qsofa, 2), (sofa_proxy, 13), (news_proxy, 15)):
        s, p = fn(pp, rows)
        assert s.min() >= 0 and s.max() <= mx and ((p >= 0) & (p <= 1)).all()


def test_engineered_features(syn_prep):
    from acmta_sepsis.features import window_features, FEATS
    pp, co, sp = syn_prep
    s = np.array([0]); t = np.array([20])
    F, names = window_features(pp, s, t)
    j = pp.dyn.index('HR'); a = pp.offsets[0] + 20
    w = pp.raw[a - 11:a + 1, j]
    k = names.index('HR_mean')
    assert np.isclose(F[0, k], np.nanmean(w), equal_nan=True)
    assert np.isclose(F[0, names.index('HR_missfrac')], 1 - np.isfinite(w).mean())

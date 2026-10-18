"""Evaluation metrics and statistical procedures (thesis §3.8-§3.9, §4.4-§4.8)."""
from __future__ import annotations
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score


# ---------------------------------------------------------------- discrimination & calibration
def auroc(y, p): return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float('nan')
def auprc(y, p): return float(average_precision_score(y, p)) if y.sum() > 0 else float('nan')
def brier(y, p): return float(np.mean((p - y) ** 2))


def ece(y, p, bins=10):
    """Expected calibration error with equal-width probability bins (Pakdaman Naeini et al., 2015; Guo et al., 2017)."""
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0; rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            conf, acc = p[m].mean(), y[m].mean(); e += m.mean() * abs(acc - conf)
            rows.append({'bin': b, 'n': int(m.sum()), 'mean_pred': float(conf), 'obs_freq': float(acc)})
    return float(e), rows


# ---------------------------------------------------------------- operating points
def operating_curve(y, p, patient, hours_per_patient):
    """Window-level sensitivity, alarms and false alarms per patient-day for every distinct threshold."""
    order = np.argsort(-p, kind='mergesort'); ps = p[order]; ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); n_alarm = np.arange(1, len(ps) + 1)
    last = np.r_[ps[1:] != ps[:-1], True]               # keep the last index of each tie group
    thr = ps[last]; tp = tp[last]; fp = fp[last]; n_alarm = n_alarm[last]
    patient_days = hours_per_patient.sum() / 24.0
    return {'threshold': thr, 'sensitivity': tp / max(ys.sum(), 1), 'alarms_per_patient_day': n_alarm / patient_days,
            'false_alarms_per_patient_day': fp / patient_days, 'ppv': tp / n_alarm, 'specificity': 1 - fp / max((1 - ys).sum(), 1)}


def cucts(curve, sens_floor=0.85, budget=3.0, budget_metric='alarms_per_patient_day'):
    """Clinical-Utility-Calibrated Threshold Selection (§3.9; parameters from §4.6: sensitivity floor 0.85, at most
    three alarms per patient per ICU-day). Minimises the alarm rate subject to both constraints; if they cannot be
    met simultaneously the two constraint-limited operating points are reported (as in §5.1)."""
    s = curve['sensitivity']; a = curve[budget_metric]; thr = curve['threshold']
    ok = (s >= sens_floor) & (a <= budget)
    res = {'sens_floor': sens_floor, 'budget': budget, 'budget_metric': budget_metric, 'feasible': bool(ok.any())}
    if ok.any():
        k = np.flatnonzero(ok)[np.argmin(a[ok])]
        res.update(threshold=float(thr[k]), sensitivity=float(s[k]), rate=float(a[k]))
    i_s = np.flatnonzero(s >= sens_floor)
    if len(i_s):
        k = i_s[0]; res['at_sens_floor'] = {'threshold': float(thr[k]), 'sensitivity': float(s[k]), 'rate': float(a[k])}
    i_b = np.flatnonzero(a <= budget)
    if len(i_b):
        k = i_b[-1]; res['at_budget'] = {'threshold': float(thr[k]), 'sensitivity': float(s[k]), 'rate': float(a[k])}
    return res


def youden(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p); k = np.argmax(tpr - fpr)
    return {'threshold': float(thr[k]), 'sensitivity': float(tpr[k]), 'specificity': float(1 - fpr[k])}


def at_threshold(y, p, thr, hours_total):
    a = p >= thr; tp = int((a & (y == 1)).sum()); fp = int((a & (y == 0)).sum()); fn = int((~a & (y == 1)).sum()); tn = int((~a & (y == 0)).sum())
    days = hours_total / 24.0
    return {'threshold': float(thr), 'sensitivity': tp / max(tp + fn, 1), 'specificity': tn / max(tn + fp, 1),
            'ppv': tp / max(tp + fp, 1), 'npv': tn / max(tn + fn, 1), 'alarms_per_patient_day': (tp + fp) / days,
            'false_alarms_per_patient_day': fp / days}


def decision_curve(y, p, thresholds=np.linspace(0.01, 0.5, 50)):
    """Decision curve analysis (Vickers & Elkin, 2006): net benefit vs threshold, with treat-all / treat-none."""
    n = len(y); prev = y.mean(); out = []
    for pt in thresholds:
        a = p >= pt; tp = (a & (y == 1)).sum() / n; fp = (a & (y == 0)).sum() / n
        w = pt / (1 - pt)
        out.append({'threshold': float(pt), 'net_benefit': float(tp - fp * w), 'treat_all': float(prev - (1 - prev) * w), 'treat_none': 0.0})
    return out


def lead_times(stay, t, y, p, t0, thr):
    """Lead time = T0 - time of the first true-positive alarm (§4.7), per septic stay; also first alarm at any time."""
    res_tp, res_any = [], []
    for s in np.unique(stay[y == 1]):
        m = stay == s; ts = t[m]; ps = p[m]; ys = y[m]
        a = ps >= thr
        if (a & (ys == 1)).any(): res_tp.append(int(t0[s] - ts[a & (ys == 1)].min()))
        if a.any(): res_any.append(int(t0[s] - ts[a].min()))
    def summ(v, n_septic):
        if not v: return {'n_alerted': 0, 'n_septic': n_septic}
        v = np.array(v)
        return {'n_alerted': int(len(v)), 'n_septic': n_septic, 'median': float(np.median(v)), 'iqr': [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
                'share_ge_1h': float((v >= 1).sum() / n_septic), 'share_ge_6h': float((v >= 6).sum() / n_septic), 'share_ge_12h': float((v >= 12).sum() / n_septic)}
    n_sep = len(np.unique(stay[y == 1]))
    return {'first_true_positive_alarm': summ(res_tp, n_sep), 'first_alarm_any_time': summ(res_any, n_sep)}


# ---------------------------------------------------------------- patient-level bootstrap (weighted, presorted, tie-aware)
def _tie_groups(p):
    """Descending order of scores and a group id per element for tied scores."""
    order = np.argsort(-p, kind='mergesort'); ps = p[order]
    gid = np.r_[0, np.cumsum(ps[1:] != ps[:-1])]
    return order, gid


def _weighted_auc(ys, w, gid):
    wp = np.bincount(gid, weights=w * ys); wn = np.bincount(gid, weights=w * (1 - ys))
    below = np.cumsum(wn[::-1])[::-1] - wn                 # negative weight strictly below each tie group
    P, N = wp.sum(), wn.sum()
    return float((wp * (below + 0.5 * wn)).sum() / (P * N)) if P > 0 and N > 0 else float('nan')


def _weighted_ap(ys, w, gid):
    wp = np.bincount(gid, weights=w * ys); wa = np.bincount(gid, weights=w)
    tp = np.cumsum(wp); al = np.cumsum(wa); P = tp[-1]
    if P <= 0: return float('nan')
    return float(((tp / np.maximum(al, 1e-12)) * wp).sum() / P)


def bootstrap_ci(y, p, patient, n_boot=2000, seed=42, alpha=0.05):
    """95% CIs by resampling patients with replacement (\u00a73.8: 2,000 patient-level resamples)."""
    rng = np.random.RandomState(seed)
    order, gid = _tie_groups(p); ys = y[order].astype(float); ps = p[order]; pat = patient[order]
    upat, inv = np.unique(pat, return_inverse=True)
    vals = {'auroc': [], 'auprc': [], 'brier': []}
    sq = (ps - ys) ** 2
    for _ in range(n_boot):
        cnt = np.bincount(rng.randint(0, len(upat), len(upat)), minlength=len(upat)).astype(float)
        w = cnt[inv]
        vals['auroc'].append(_weighted_auc(ys, w, gid)); vals['auprc'].append(_weighted_ap(ys, w, gid))
        vals['brier'].append(float((sq * w).sum() / w.sum()))
    out = {}
    for k, v in vals.items():
        v = np.array(v); v = v[np.isfinite(v)]
        out[k] = [float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))]
    return out


def paired_bootstrap_auprc(y, p1, p2, patient, n_boot=2000, seed=42):
    """Two-sided paired bootstrap test of the AUPRC difference over the same patient resamples (\u00a73.8)."""
    rng = np.random.RandomState(seed)
    o1, g1 = _tie_groups(p1); o2, g2 = _tie_groups(p2)
    upat, inv = np.unique(patient, return_inverse=True)
    y1, y2 = y[o1].astype(float), y[o2].astype(float); i1, i2 = inv[o1], inv[o2]
    d = []
    for _ in range(n_boot):
        cnt = np.bincount(rng.randint(0, len(upat), len(upat)), minlength=len(upat)).astype(float)
        d.append(_weighted_ap(y1, cnt[i1], g1) - _weighted_ap(y2, cnt[i2], g2))
    d = np.array(d); obs = average_precision_score(y, p1) - average_precision_score(y, p2)
    pval = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {'diff': float(obs), 'ci': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], 'p': float(min(1.0, pval))}


# ---------------------------------------------------------------- DeLong (Sun & Xu, 2014 fast algorithm)
def _midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]: k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1; i = k
    out = np.empty(n); out[j] = t; return out


def delong_test(y, p1, p2):
    """Paired DeLong test for two correlated ROC curves (DeLong et al., 1988)."""
    pos = y == 1; m, n = pos.sum(), (~pos).sum()
    aucs, v10, v01 = [], [], []
    for p in (p1, p2):
        x, yy = p[pos], p[~pos]
        tx, ty, tz = _midrank(x), _midrank(yy), _midrank(np.r_[x, yy])
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tz[:m] - tx) / n); v01.append(1 - (tz[m:] - ty) / m)
    v10 = np.array(v10); v01 = np.array(v01)
    s = np.cov(v10) / m + np.cov(v01) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return {'auc1': float(aucs[0]), 'auc2': float(aucs[1]), 'z': float(z), 'p': float(2 * stats.norm.sf(abs(z)))}


def holm(pvals):
    """Holm-Bonferroni step-down adjustment (Holm, 1979)."""
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, k in enumerate(order):
        run = max(run, (m - r) * p[k]); adj[k] = min(1.0, run)
    return adj.tolist()


def subgroup_auroc(y, p, groups: dict):
    out = {}
    for name, mask in groups.items():
        if mask.sum() > 0 and 0 < y[mask].sum() < mask.sum():
            out[name] = {'n_windows': int(mask.sum()), 'n_pos': int(y[mask].sum()), 'auroc': auroc(y[mask], p[mask])}
        else:
            out[name] = {'n_windows': int(mask.sum()), 'n_pos': int(y[mask].sum()), 'auroc': None}
    return out


def full_report(y, p, stay, t, t0, hours_per_stay, val=None, n_boot=2000, seed=42):
    """All §3.8-3.9 quantities for one model. `val` = (y_val, p_val, stay_val, hours_val) for threshold selection."""
    rep = {'n_windows': int(len(y)), 'n_pos': int(y.sum()), 'prevalence': float(y.mean()),
           'auroc': auroc(y, p), 'auprc': auprc(y, p), 'brier': brier(y, p), 'brier_constant_prevalence': float(y.mean() * (1 - y.mean()))}
    rep['ece'], rep['reliability'] = ece(y, p)
    if n_boot:
        rep['ci95'] = bootstrap_ci(y, p, stay, n_boot=n_boot, seed=seed)
    hrs = hours_per_stay[np.unique(stay)]
    curve = operating_curve(y, p, stay, hrs)
    rep['cucts_test_curve'] = cucts(curve)
    rep['cucts_test_curve_false_alarm_budget'] = cucts(curve, budget_metric='false_alarms_per_patient_day')
    rep['youden_test'] = youden(y, p)
    if val is not None:
        yv, pv, sv, hv = val
        cv = operating_curve(yv, pv, sv, hv[np.unique(sv)])
        sel = cucts(cv)
        rep['cucts_validation_selected'] = sel
        thr = sel.get('threshold', sel.get('at_sens_floor', {}).get('threshold'))
        if thr is not None:
            rep['at_cucts_threshold_on_test'] = at_threshold(y, p, thr, hrs.sum())
            rep['lead_time_at_cucts_threshold'] = lead_times(stay, t, y, p, t0, thr)
        yd = youden(yv, pv)
        rep['at_youden_threshold_on_test'] = at_threshold(y, p, yd['threshold'], hrs.sum())
    rep['decision_curve'] = decision_curve(y, p)
    return rep

#!/usr/bin/env python
"""Statistical comparison of all real-data models on the held-out test set (thesis §3.8): DeLong tests for AUROC,
paired patient-level bootstrap for AUPRC, Holm-Bonferroni adjustment. Also the post-hoc recalibration study
(an EXPERIMENTAL CHANGE recommended in thesis §4.5/§5.1 but not part of the described pipeline)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from acmta_sepsis import evaluate as E
from acmta_sepsis.experiments import load, RESULTS
from acmta_sepsis.utils import dump

NAME = 'real_ch3'
P = load(NAME); t0s = P['cohort']['t0']; hours = P['hours']
base = np.load(os.path.join(RESULTS, NAME, 'baselines', 'predictions.npz'), allow_pickle=True)
y = base['y_test']; s = base['s_test']; t = base['t_test']
models = {k: base[f'{k}_test_p'] for k in ('qsofa', 'sofa_proxy', 'news_proxy', 'lr', 'gbm')}
val = {}
runs = ['bilstm_acmta', 'transformer_acmta', 'bilstm_early', 'bilstm_late', 'bilstm_vitals_only', 'bilstm_labs_only']
for r in runs:
    z = np.load(os.path.join(RESULTS, NAME, 'runs', r, 'predictions.npz'))
    assert (z['s_test'] == s).all() and (z['t_test'] == t).all() and (z['y_test'] == y).all()   # identical test windows
    models[r] = z['p_test']; val[r] = (z['y_val'], z['p_val'], z['s_val'])
ref = 'bilstm_acmta'
comparisons = [(ref, m) for m in models if m != ref] + [('transformer_acmta', 'bilstm_acmta'), ('bilstm_early', 'bilstm_acmta'), ('bilstm_late', 'bilstm_acmta')]
seen = set(); rows = []
for a, b in comparisons:
    if (a, b) in seen or (b, a) in seen: continue
    seen.add((a, b))
    dl = E.delong_test(y, models[a], models[b])
    pb = E.paired_bootstrap_auprc(y, models[a], models[b], s, n_boot=1000)
    rows.append({'model_a': a, 'model_b': b, 'auroc_a': dl['auc1'], 'auroc_b': dl['auc2'], 'delong_z': dl['z'], 'delong_p': dl['p'],
                 'auprc_diff': pb['diff'], 'auprc_diff_ci': pb['ci'], 'auprc_boot_p': pb['p']})
adj_auc = E.holm([r['delong_p'] for r in rows]); adj_pr = E.holm([r['auprc_boot_p'] for r in rows])
for r, a1, a2 in zip(rows, adj_auc, adj_pr): r['delong_p_holm'] = a1; r['auprc_p_holm'] = a2
# ---------------- post-hoc recalibration (fitted on validation predictions only)
calib = {}
for r in runs:
    yv, pv, sv = val[r]; pt = models[r]
    pl = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(pv / (1 - pv + 1e-12) + 1e-12).reshape(-1, 1), yv)
    pt_platt = pl.predict_proba(np.log(pt / (1 - pt + 1e-12) + 1e-12).reshape(-1, 1))[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(pv, yv); pt_iso = iso.predict(pt)
    out = {}
    for nm, pp_ in (('uncalibrated', pt), ('platt', pt_platt), ('isotonic', pt_iso)):
        e, _ = E.ece(y, pp_)
        out[nm] = {'auroc': E.auroc(y, pp_), 'auprc': E.auprc(y, pp_), 'brier': E.brier(y, pp_), 'ece': e}
    calib[r] = out
dump({'test_windows': int(len(y)), 'test_positive': int(y.sum()), 'comparisons': rows, 'recalibration': calib,
      'brier_constant_prevalence': float(y.mean() * (1 - y.mean()))}, os.path.join(RESULTS, NAME, 'model_comparison.json'))
for r in rows: print('%-20s vs %-20s AUROC %.4f/%.4f p_holm=%.3g | dAUPRC %.4f p_holm=%.3g' % (r['model_a'], r['model_b'], r['auroc_a'], r['auroc_b'], r['delong_p_holm'], r['auprc_diff'], r['auprc_p_holm']))
for r in runs: print(r, {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in calib[r].items()})

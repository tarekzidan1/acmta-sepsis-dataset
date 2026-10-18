#!/usr/bin/env python
"""Re-execute the Chapter 4 "pipeline-validation" protocol (thesis §4.2-4.6) on a SYNTHETIC cohort.

The thesis generator (scripts/synthesize_pnet2019_format.py) and its seed were not supplied, so this uses the
independent generator in acmta_sepsis/synthetic.py. Values are therefore NOT expected to equal the thesis values;
the run tests whether the documented protocol produces the same kind of evidence (cohort/window statistics,
near-chance baselines, CUCTS infeasibility on a weak-signal cohort).

Protocol (as documented in Chapter 4): 2,000 generated stays, stays < 8 h excluded (§4.3), all 40 PhysioNet
variables (§4.10, Fig. 4.10), 12-h windows with a 2-h stride (§4.10.1), positive when T0 falls in the 12 h after
the 4-h buffer (§4.3), five-fold patient-grouped cross-validation over the whole cohort (Table 4.3), gradient
boosting and logistic regression on engineered features (§4.4), qSOFA and a SOFA proxy (§4.4), CUCTS with a
sensitivity floor of 0.85 and three alarms per patient per ICU-day (§4.6).
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from acmta_sepsis import synthetic, data as D, evaluate as E
from acmta_sepsis.preprocessing import Preprocessed, VARIABLE_SETS
from acmta_sepsis.windows import make_windows
from acmta_sepsis.features import window_features
from acmta_sepsis.scores import qsofa, sofa_proxy
from acmta_sepsis.baselines import cv_oof
from acmta_sepsis.utils import dump

def run(seed=7, signal=0.35, stride=2, label_mode='ch4', varset='ch4_full', min_icu_hours=8, min_history=1, fold_seed=42):
    ds = synthetic.as_dataset(synthetic.generate(n_patients=2000, signal=signal, seed=seed))
    co = D.cohort(ds, VARIABLE_SETS[varset]['vitals'], min_icu_hours=min_icu_hours)
    pp = Preprocessed(ds, co['index'], varset)
    n = len(co['index'])
    rng = np.random.RandomState(fold_seed); fold = np.full(n, -1)
    for cls in (True, False):
        m = np.flatnonzero(co['septic'] == cls); rng.shuffle(m); fold[m] = np.arange(len(m)) % 5
    pp.fit_scaler(np.arange(n)).transform()
    s, t, y = make_windows(pp.offsets, co['t0'], np.arange(n), stride=stride, label_mode=label_mode, min_history=min_history)
    hours = np.diff(pp.offsets)
    out = {'settings': dict(seed=seed, signal=signal, stride=stride, label_mode=label_mode, varset=varset, min_icu_hours=min_icu_hours, min_history=min_history),
           'cohort': {'generated': int(ds.n), 'kept': n, 'septic': int(co['septic'].sum()), 'prevalence_pct': 100 * float(co['septic'].mean()),
                      'los_median': float(np.median(hours)), 'los_iqr': [float(np.percentile(hours, 25)), float(np.percentile(hours, 75))],
                      'age_mean': float(np.nanmean(pp.static_raw[:, 0])), 'age_sd': float(np.nanstd(pp.static_raw[:, 0])),
                      'male_pct': 100 * float(np.nanmean(pp.static_raw[:, 1])),
                      'windows': int(len(y)), 'positive_windows': int(y.sum()), 'window_prevalence_pct': 100 * float(y.mean())}}
    X, names = window_features(pp, s, t)
    wf = fold[s]
    preds = {'gbm': cv_oof('gbm', X, y, wf), 'lr': cv_oof('lr', X, y, wf)}
    rows = pp.offsets[s] + t
    preds['qsofa'] = qsofa(pp, rows)[1]; preds['sofa_proxy'] = sofa_proxy(pp, rows)[1]
    res = {}
    for k, p in preds.items():
        curve = E.operating_curve(y, p, s, hours)
        res[k] = {'auroc': E.auroc(y, p), 'auprc': E.auprc(y, p), 'brier': E.brier(y, p), 'ci95': E.bootstrap_ci(y, p, s, n_boot=500),
                  'cucts_alarms': E.cucts(curve), 'cucts_false_alarms': E.cucts(curve, budget_metric='false_alarms_per_patient_day')}
    out['baselines'] = res
    out['brier_constant_prevalence'] = float(y.mean() * (1 - y.mean()))
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='results/synthetic_ch4_protocol')
    ap.add_argument('--variant', default='ch4'); a = ap.parse_args()
    if a.variant == 'ch4':
        r = run()
    elif a.variant == 'ch3_windows':
        r = run(stride=1, label_mode='ch3')
    elif a.variant == 'ch4_strong_signal':
        r = run(signal=1.5)
    dump(r, os.path.join(a.out, f'{a.variant}.json'))
    print(json.dumps({'cohort': r['cohort'], **{k: {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r['baselines'][k].items() if kk in ('auroc', 'auprc', 'brier')} for k in r['baselines']}}, indent=1))

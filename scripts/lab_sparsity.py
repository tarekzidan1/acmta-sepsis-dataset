"""Laboratory-sparsity stress test (specified before it was run; reported as an additional, exploratory analysis).

Question: does fusing laboratory results at their own measurement times (ACMTA) degrade less than early fusion
(forward-filled hourly grid) when laboratory tests become sparser, e.g. in units that order tests less often?

Protocol (fixed before any result was seen):
  * Test-time perturbation only; models are NOT retrained and the training/validation data are untouched.
  * In every held-out test stay, each laboratory observation is removed independently with probability q,
    q in {0.25, 0.5, 0.75, 1.0}; one uniform draw per observation (seed 2026) makes the removed sets nested and
    identical for all models. Vital signs are unchanged. Masks, forward-filled values and elapsed times are
    recomputed exactly as in preprocessing.py; engineered features of the classical models are recomputed too.
  * Primary comparison: ACMTA (BiLSTM) vs early fusion, AUROC at each q (three training seeds; paired DeLong for
    seed 42 with Holm correction over the four levels). Secondary: ablations, late fusion, gradient boosting,
    logistic regression; the vital-sign-only model (unaffected by construction) is the floor.
Usage: python scripts/lab_sparsity.py [budget_seconds]   (resumable; repeat until it prints "ALL DONE")
"""
import os, sys, json, copy, time, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acmta_sepsis import experiments as X
from acmta_sepsis import evaluate as E
from acmta_sepsis.preprocessing import last_obs_index, DELTA_CAP
from acmta_sepsis.train import load_best, predict
from acmta_sepsis.features import window_features

T0 = time.time(); BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 140.0
LEVELS = [0.25, 0.5, 0.75, 1.0]
RES = os.path.join(os.environ.get('ACMTA_RESULTS', 'results'), 'real_ch3'); OUT = os.path.join(RES, 'lab_sparsity'); os.makedirs(OUT, exist_ok=True)
DEEP = ['bilstm_acmta', 'bilstm_early'] + [f'{r}_s{s}' for s in (43, 44) for r in ('bilstm_acmta', 'bilstm_early')] + \
       ['bilstm_acmta_notimebias', 'bilstm_acmta_nomamg', 'bilstm_late'] + \
       [f'{r}_s{s}' for s in (43, 44) for r in ('bilstm_acmta_notimebias', 'bilstm_acmta_nomamg', 'bilstm_late')]
CLASSICAL = ['gbm', 'lr']
JOBS = [(q, m) for q in LEVELS for m in DEEP[:4] + CLASSICAL + DEEP[4:]]      # primary models first at every level

def done(q, m): return os.path.exists(os.path.join(OUT, f'{m}_q{q}.json'))

def thinned(pp, test_stays, q, seed=2026):
    """Copy of pp in which each laboratory observation of the test stays is removed with probability q."""
    rows = np.concatenate([np.arange(pp.offsets[s], pp.offsets[s + 1]) for s in test_stays])
    lc = pp.lab_cols; raw = pp.raw.copy()
    u = np.random.RandomState(seed).random_sample((len(rows), len(lc)))     # one draw per (row, lab) -> nested removal sets
    sub = raw[np.ix_(rows, lc)]; obs = np.isfinite(sub); drop = obs & (u < q)
    sub[drop] = np.nan; raw[np.ix_(rows, lc)] = sub
    p2 = copy.copy(pp); p2.raw = raw; p2.mask = np.isfinite(raw)
    lo = last_obs_index(p2.mask, pp.offsets); r = np.arange(len(raw))[:, None]
    delta = np.where(lo >= 0, r - lo, pp.t_of_row[:, None] + 1).astype(np.float32)
    p2.delta = np.minimum(delta, DELTA_CAP) / DELTA_CAP
    p2.ffill_raw = np.where(lo >= 0, raw[np.maximum(lo, 0), np.arange(raw.shape[1])[None, :]], np.nan).astype(np.float32)
    p2.transform()
    return p2, {'lab_obs_test': int(obs.sum()), 'removed': int(drop.sum())}

pending = [j for j in JOBS if not done(*j)]
if not pending: print('ALL DONE'); sys.exit(0)
P = X.load('real_ch3'); pp = P['pp']; s_te, t_te, y_te = P['windows']['test']
test_stays = np.unique(s_te); cache = {}
print(f'loaded in {time.time() - T0:.0f}s; pending {len(pending)}', flush=True)
for q, m in pending:
    if time.time() - T0 > BUDGET: break
    if q not in cache: cache.clear(); cache[q] = thinned(pp, test_stays, q)
    p2, info = cache[q]; tt = time.time()
    if m in CLASSICAL:
        with open(os.path.join(RES, 'baselines', f'{m}_model.pkl'), 'rb') as f: mdl = pickle.load(f)
        Xt, _ = window_features(p2, s_te, t_te); p = mdl.predict_proba(Xt)[:, 1]
    else:
        mdl, cfg, _ = load_best(os.path.join(RES, 'runs', m)); p = predict(mdl, p2, s_te, t_te, cfg.fusion)
    rec = {'model': m, 'q': q, 'auroc': E.auroc(y_te, p), 'auprc': E.auprc(y_te, p), 'seconds': time.time() - tt, **info}
    if m in ('bilstm_acmta', 'bilstm_early', 'gbm', 'bilstm_acmta_notimebias', 'bilstm_acmta_nomamg', 'bilstm_late', 'lr'):
        np.savez_compressed(os.path.join(OUT, f'{m}_q{q}.npz'), p=p.astype(np.float32))
    json.dump(rec, open(os.path.join(OUT, f'{m}_q{q}.json'), 'w'))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
left = [j for j in JOBS if not done(*j)]
print('ALL DONE' if not left else f'paused; {len(left)} jobs left')

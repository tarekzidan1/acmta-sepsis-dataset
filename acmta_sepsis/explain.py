"""Explainability (thesis §3.10, §4.9): permutation feature importance (Breiman, 2001) and SHAP (Lundberg & Lee, 2017;
TreeExplainer) for the gradient-boosting baseline; for ACMTA-Sepsis, cross-modal attention profiles over relative hours
and an integrated-gradients attribution per variable and hour (SHAP on the deep model was not run - ISSUE X-01)."""
from __future__ import annotations
import numpy as np
import torch
from . import evaluate as E
from .batching import build_batch


def permutation_importance(model, X, y, names, n_repeats=3, seed=0):
    rng = np.random.RandomState(seed)
    base = E.auroc(y, model.predict_proba(X)[:, 1]); out = []
    for j, n in enumerate(names):
        d = []
        for _ in range(n_repeats):
            Xp = X.copy(); Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            d.append(base - E.auroc(y, model.predict_proba(Xp)[:, 1]))
        out.append((n, float(np.mean(d)), float(np.std(d))))
    out.sort(key=lambda r: -r[1])
    return base, out


def tree_shap(model, X, names, n=2000, seed=0):
    import shap
    rng = np.random.RandomState(seed); idx = rng.choice(len(X), min(n, len(X)), replace=False)
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X[idx])
    imp = np.abs(sv).mean(0)
    order = np.argsort(-imp)
    return [(names[k], float(imp[k])) for k in order]


def attention_profile(model, pp, s, t, fusion='acmta'):
    """Mean attention of vital-sign queries over laboratory keys by key hour offset (and the reverse)."""
    model.eval()
    with torch.no_grad():
        b = build_batch(pp, s, t, None, fusion)
        _, aux = model(b, return_attention=True)
    a = aux['att_v2l'].mean(1)            # (B, Tq, Tk) averaged over heads
    a = a.mean(1).numpy()                 # average over queries -> (B, Tk)
    tl = b['tl'].numpy(); pl = b['pl'].numpy()
    prof = np.zeros(12); cnt = np.zeros(12)
    for i in range(len(s)):
        for k in range(a.shape[1]):
            if not pl[i, k]:
                h = int(round(tl[i, k])) + 11; prof[h] += a[i, k]; cnt[h] += 1
    return {'hour_offset': list(range(-11, 1)), 'mean_attention_vitals_to_labs': (prof / np.maximum(cnt, 1)).tolist(), 'n_keys': cnt.tolist()}


def integrated_gradients(model, pp, s, t, fusion='acmta', steps=16):
    """Integrated gradients (zero baseline = population mean after standardisation) for vital values (per hour)."""
    model.eval()
    b = build_batch(pp, s, t, None, fusion)
    xv = b['xv'].clone(); total = torch.zeros_like(xv)
    for k in range(1, steps + 1):
        bb = dict(b); x = (xv * k / steps).requires_grad_(True); bb['xv'] = x
        out = torch.sigmoid(model(bb)).sum(); g, = torch.autograd.grad(out, x); total += g
    attr = (xv * total / steps).detach().numpy()          # (B, T, n_vitals) with left-aligned valid steps
    tv = b['tv'].numpy(); pv = b['pv'].numpy()
    heat = np.zeros((len(pp.vitals), 12)); cnt = np.zeros(12)
    for i in range(len(s)):
        for k in range(attr.shape[1]):
            if not pv[i, k]:
                h = int(round(tv[i, k])) + 11; heat[:, h] += np.abs(attr[i, k]); cnt[h] += 1
    return {'variables': pp.vitals, 'hour_offset': list(range(-11, 1)), 'mean_abs_attribution': (heat / np.maximum(cnt, 1)).tolist()}

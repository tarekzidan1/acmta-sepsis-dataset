"""Engineered window features for the classical baselines (thesis §3.7.2; §4.4: minimum, maximum, average,
standard deviation, slope, last recorded value and proportion of missing values over the 12-hour window)."""
from __future__ import annotations
import numpy as np
from .windows import window_rows

FEATS = ['min', 'max', 'mean', 'std', 'slope', 'last', 'missfrac']


def window_features(pp, stay, t, chunk: int = 100_000, include_static: bool = True) -> tuple[np.ndarray, list[str]]:
    n = len(stay)
    nv = len(pp.dyn)
    names = [f'{v}_{f}' for v in pp.dyn for f in FEATS]
    if include_static:
        names += list(pp.static_names) + list(pp.dynamic_static or [])
    out = np.empty((n, len(names)), dtype=np.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        rows, valid, rel = window_rows(pp.offsets, stay[a:b], t[a:b])
        X = pp.raw[rows]                                   # (m, 12, nv) raw (with NaN for missing)
        X[~valid] = np.nan
        obs = np.isfinite(X)
        cnt = obs.sum(axis=1)
        with np.errstate(all='ignore'):
            mn = np.nanmin(X, axis=1); mx = np.nanmax(X, axis=1); mean = np.nanmean(X, axis=1)
            sd = np.nanstd(X, axis=1)
            tt = np.where(obs, rel[:, :, None], np.nan)
            tm = np.nanmean(tt, axis=1)
            cov = np.nansum((tt - tm[:, None, :]) * (X - mean[:, None, :]), axis=1)
            var = np.nansum((tt - tm[:, None, :]) ** 2, axis=1)
            slope = np.where(cnt >= 2, cov / np.where(var > 0, var, np.nan), np.nan)
        last = pp.ffill_raw[rows[:, -1]]                   # last recorded value up to t (may precede the window)
        nvalid = valid.sum(axis=1, keepdims=True)
        miss = 1.0 - cnt / np.maximum(nvalid, 1)
        F = np.stack([mn, mx, mean, sd, slope, last, miss], axis=2).reshape(b - a, nv * len(FEATS))
        if include_static:
            st = pp.static_raw[stay[a:b]]
            parts = [F, st]
            if pp.dyn_static_raw is not None:
                parts.append(pp.dyn_static_raw[rows[:, -1]])
            F = np.concatenate(parts, axis=1)
        out[a:b] = F
    return out, names

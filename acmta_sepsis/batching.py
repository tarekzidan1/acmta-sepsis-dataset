"""Assemble model inputs for a set of windows: a dense hourly vital-sign stream and an asynchronous laboratory
stream that keeps only the hours at which laboratory values were actually observed (§3.6.3: modalities are fused
"without requiring the two streams to be resampled onto a common time grid")."""
from __future__ import annotations
import numpy as np
import torch
from .windows import window_rows


def build_batch(pp, stay, t, y=None, fusion='acmta', as_torch=True):
    rows, valid, rel = window_rows(pp.offsets, stay, t)
    B, W = rows.shape
    vc, lc = pp.vital_cols, pp.lab_cols
    out = {}
    # ---- vital-sign stream (hourly grid; valid steps left-aligned for packing)
    order = np.argsort(~valid, axis=1, kind='stable')
    rv = np.take_along_axis(rows, order, 1); okv = np.take_along_axis(valid, order, 1); tv = np.take_along_axis(rel, order, 1)
    okf = okv[..., None].astype(np.float32)
    if fusion in ('acmta', 'late', 'vitals_only', 'early'):
        out['xv'] = pp.x[rv][:, :, vc] * okf
        out['mv'] = pp.mask[rv][:, :, vc].astype(np.float32) * okf
        out['dv'] = pp.delta[rv][:, :, vc] * okf
        out['tv'] = np.where(okv, tv, 0.0).astype(np.float32)
        out['pv'] = ~okv
    if fusion == 'early':   # early fusion: both modalities concatenated on the hourly grid (labs forward-filled)
        out['xe'] = pp.x[rv] * okf
        out['me'] = pp.mask[rv].astype(np.float32) * okf
        out['de'] = pp.delta[rv] * okf
    # ---- laboratory stream: observed laboratory hours inside the window (asynchronous events)
    if fusion in ('acmta', 'late', 'labs_only'):
        ev = valid & pp.mask[rows][:, :, lc].any(-1)
        none = ~ev.any(1)
        # PROVISIONAL (ISSUE M-06): a window without any laboratory observation is represented by one "state" token
        # at the prediction time carrying the last observed (forward-filled) values, mask 0 and elapsed time.
        ev[none, -1] = True
        order_l = np.argsort(~ev, axis=1, kind='stable')
        rl = np.take_along_axis(rows, order_l, 1); okl = np.take_along_axis(ev, order_l, 1); tl = np.take_along_axis(rel, order_l, 1)
        okf_l = okl[..., None].astype(np.float32)
        out['xl'] = pp.x[rl][:, :, lc] * okf_l
        out['ml'] = pp.mask[rl][:, :, lc].astype(np.float32) * okf_l
        out['dl'] = pp.delta[rl][:, :, lc] * okf_l
        out['tl'] = np.where(okl, tl, 0.0).astype(np.float32)
        out['pl'] = ~okl
        # trim padding columns shared by the whole batch
        k = int(okl.sum(1).max())
        for key in ('xl', 'ml', 'dl', 'tl', 'pl'): out[key] = out[key][:, :k]
    st = pp.static[stay]
    if pp.dyn_static is not None:
        st = np.concatenate([st, pp.dyn_static[rows[:, -1]]], axis=1)
    out['static'] = st.astype(np.float32)
    if y is not None: out['y'] = y.astype(np.float32)
    if as_torch:
        out = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in out.items()}
    return out

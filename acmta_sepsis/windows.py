"""Prediction windows and labels (thesis §1.6, §3.3, §3.5; Figure 2.1).

At every hour t of a stay the preceding twelve hours [t-11, t] form the observation window; the window is
labelled positive when onset T0 falls within the following twelve hours. Predictions stop four hours before T0
(pre-onset exclusion buffer). Hours after the buffer start are not used for septic stays.
"""
from __future__ import annotations
import numpy as np

WINDOW = 12
HORIZON = 12
BUFFER = 4


def make_windows(offsets: np.ndarray, t0: np.ndarray, stays: np.ndarray, stride: int = 1, label_mode: str = 'ch3',
                 min_history: int = 1, horizon: int = HORIZON, buffer: int = BUFFER):
    """Return arrays (stay, t, y) for the given stays.

    label_mode 'ch3': y = 1 iff buffer <= T0 - t <= horizon (thesis §3.3: onset within the next 12 h, predictions
                      stop 4 h before T0).
    label_mode 'ch4': y = 1 iff buffer < T0 - t <= buffer + horizon (thesis §4.3: "T0 occurred in the twelve hours
                      after the elimination buffer") - ISSUE D-08.
    stride: 1 h (thesis §1.6, §3.5) or 2 h (thesis §4.10.1) - ISSUE D-04.
    min_history: minimum number of observed hours before the first window (PROVISIONAL, default 1 = from the first hour).
    """
    S, T, Y = [], [], []
    for s in stays:
        L = int(offsets[s + 1] - offsets[s])
        ts = np.arange(min_history - 1, L, stride)
        o = int(t0[s])
        if o >= 0:
            lead = o - ts
            ts = ts[lead >= buffer] if label_mode == 'ch3' else ts[lead > buffer]
            lead = o - ts
            y = (lead <= horizon) if label_mode == 'ch3' else (lead <= buffer + horizon)
        else:
            y = np.zeros(len(ts), dtype=bool)
        S.append(np.full(len(ts), s, dtype=np.int32)); T.append(ts.astype(np.int32)); Y.append(y.astype(np.int8))
    if not S:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int8)
    return np.concatenate(S), np.concatenate(T), np.concatenate(Y)


def window_rows(offsets: np.ndarray, stay: np.ndarray, t: np.ndarray, window: int = WINDOW):
    """Global row indices (n, window) of each observation window and a validity mask (False = before admission)."""
    rel = np.arange(-(window - 1), 1)[None, :]
    tt = t[:, None] + rel
    valid = tt >= 0
    rows = offsets[stay][:, None] + np.maximum(tt, 0)
    return rows, valid, rel.repeat(len(t), axis=0).astype(np.float32)

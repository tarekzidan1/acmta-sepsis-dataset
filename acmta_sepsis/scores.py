"""Rule-based clinical scores at prediction time (thesis §3.7.2: qSOFA, SOFA, NEWS; §4.4: qSOFA and a SOFA proxy).

PhysioNet 2019 contains no Glasgow Coma Scale, PaO2, vasopressor, urine-output, supplemental-oxygen or
consciousness information, so every score is a PROXY restricted to the available components (ISSUE M-09).
Nominal probabilities for the Brier score are obtained by dividing by the maximum score of the full instrument
(thesis §4.5: rule-based scores converted "to nominal probabilities using normalisation"; exact rule not stated).
"""
from __future__ import annotations
import numpy as np


def _v(pp, rows, name):
    j = pp.dyn.index(name)
    return pp.ffill_raw[rows, j]


def qsofa(pp, rows):
    rr = _v(pp, rows, 'Resp'); sbp = _v(pp, rows, 'SBP')
    s = (np.nan_to_num(rr, nan=0) >= 22).astype(np.float32) + (np.nan_to_num(sbp, nan=1e9) <= 100).astype(np.float32)
    return s, s / 3.0


def _bands(x, cuts, pts, missing=0.0):
    out = np.full(x.shape, missing, dtype=np.float32)
    ok = np.isfinite(x)
    for c, p in zip(cuts, pts):
        out = np.where(ok & c(x), p, out)
    return out


def sofa_proxy(pp, rows):
    plt = _v(pp, rows, 'Platelets'); bil = _v(pp, rows, 'Bilirubin_total'); mp = _v(pp, rows, 'MAP'); cr = _v(pp, rows, 'Creatinine')
    coag = _bands(plt, [lambda x: x < 150, lambda x: x < 100, lambda x: x < 50, lambda x: x < 20], [1, 2, 3, 4])
    liver = _bands(bil, [lambda x: x >= 1.2, lambda x: x >= 2.0, lambda x: x >= 6.0, lambda x: x >= 12.0], [1, 2, 3, 4])
    cardio = _bands(mp, [lambda x: x < 70], [1])
    renal = _bands(cr, [lambda x: x >= 1.2, lambda x: x >= 2.0, lambda x: x >= 3.5, lambda x: x >= 5.0], [1, 2, 3, 4])
    s = coag + liver + cardio + renal
    if 'FiO2' in pp.dyn and 'SaO2' in pp.dyn:   # respiratory proxy (SaO2/FiO2) only when available (ch4_full set)
        fio2 = _v(pp, rows, 'FiO2'); sao2 = _v(pp, rows, 'SaO2')
        sf = sao2 / np.where(fio2 > 0, fio2, np.nan)
        s = s + _bands(sf, [lambda x: x < 400, lambda x: x < 315, lambda x: x < 235, lambda x: x < 150], [1, 2, 3, 4])
    return s, s / 24.0


def news_proxy(pp, rows):
    rr = _v(pp, rows, 'Resp'); spo2 = _v(pp, rows, 'O2Sat'); sbp = _v(pp, rows, 'SBP'); hr = _v(pp, rows, 'HR'); tmp = _v(pp, rows, 'Temp')
    s = (_bands(rr, [lambda x: x <= 8, lambda x: (x >= 9) & (x <= 11), lambda x: (x >= 21) & (x <= 24), lambda x: x >= 25], [3, 1, 2, 3])
         + _bands(spo2, [lambda x: x <= 91, lambda x: (x > 91) & (x <= 93), lambda x: (x > 93) & (x <= 95)], [3, 2, 1])
         + _bands(sbp, [lambda x: x <= 90, lambda x: (x > 90) & (x <= 100), lambda x: (x > 100) & (x <= 110), lambda x: x >= 220], [3, 2, 1, 3])
         + _bands(hr, [lambda x: x <= 40, lambda x: (x > 40) & (x <= 50), lambda x: (x > 90) & (x <= 110), lambda x: (x > 110) & (x <= 130), lambda x: x > 130], [3, 1, 1, 2, 3])
         + _bands(tmp, [lambda x: x <= 35.0, lambda x: (x > 35.0) & (x <= 36.0), lambda x: (x > 38.0) & (x <= 39.0), lambda x: x > 39.0], [3, 1, 1, 2]))
    return s, s / 20.0

"""Schema-matched SYNTHETIC PhysioNet-2019-format generator.

!! This is NOT the generator used in the thesis (scripts/synthesize_pnet2019_format.py was not supplied). !!
It implements the description in thesis §4.2 so that (a) the pipeline can be unit/smoke tested without real data
and (b) the Chapter 4 "pipeline-validation" protocol can be re-executed. Numbers obtained with it are NOT
comparable to the thesis values except qualitatively. Every quantity below is a PROVISIONAL modelling choice.

§4.2: forty clinical columns + SepsisLabel, one pipe-separated file per ICU stay, one row per hour, PhysioNet-like
missingness, sepsis prevalence 7.96 %, physiological drift along six pathways (leukocytosis, fever, tachycardia,
hypotension, tachypnoea, acute organ dysfunction), deliberately weak signal; 2,000 generated stays of which
stays shorter than eight hours were excluded (§4.3).
"""
from __future__ import annotations
import os
import numpy as np
from .data import COLUMNS, COL

BASE = {  # (mean, sd) of a stable adult ICU patient (PROVISIONAL)
    'HR': (84, 15), 'O2Sat': (97, 2.5), 'Temp': (36.9, 0.6), 'SBP': (122, 20), 'MAP': (82, 13), 'DBP': (63, 11),
    'Resp': (18.5, 4.5), 'EtCO2': (33, 6), 'BaseExcess': (-0.5, 3.5), 'HCO3': (24, 3.5), 'FiO2': (0.5, 0.15),
    'pH': (7.39, 0.06), 'PaCO2': (41, 8), 'SaO2': (93, 6), 'AST': (60, 50), 'BUN': (23, 15), 'Alkalinephos': (95, 50),
    'Calcium': (8.1, 0.9), 'Chloride': (105, 5), 'Creatinine': (1.4, 1.0), 'Bilirubin_direct': (0.6, 0.8),
    'Glucose': (135, 45), 'Lactate': (2.3, 1.4), 'Magnesium': (2.0, 0.35), 'Phosphate': (3.5, 1.2),
    'Potassium': (4.1, 0.5), 'Bilirubin_total': (1.4, 1.6), 'TroponinI': (2.0, 6.0), 'Hct': (31, 5), 'Hgb': (10.4, 1.8),
    'PTT': (40, 15), 'WBC': (11, 5), 'Fibrinogen': (290, 120), 'Platelets': (200, 90)}
DRIFT = {  # direction (in SD units at full drift) along the six pathways named in §4.2
    'WBC': +1.0, 'Temp': +1.0, 'HR': +1.0, 'SBP': -1.0, 'MAP': -1.0, 'DBP': -0.7, 'Resp': +1.0,
    'Creatinine': +0.8, 'Bilirubin_total': +0.6, 'Platelets': -0.8, 'Lactate': +1.0}
VITAL_P = {'HR': 0.92, 'O2Sat': 0.88, 'Temp': 0.33, 'SBP': 0.85, 'MAP': 0.88, 'DBP': 0.5, 'Resp': 0.85, 'EtCO2': 0.04}


def generate(n_patients=2000, prevalence=0.08, signal=0.35, seed=7, short_frac=0.28):
    """Return list of (pid, array[L, 41]). `signal` = drift amplitude in SD units at onset (weak by design)."""
    rng = np.random.RandomState(seed)
    out = []
    for i in range(n_patients):
        if rng.rand() < short_frac:
            L = rng.randint(1, 8)
        else:
            L = int(np.clip(np.exp(rng.normal(np.log(55), 0.55)), 8, 336))
        septic = (rng.rand() < prevalence) and L >= 14
        t0 = int(rng.randint(10, L + 6)) if septic else -1
        a = np.full((L, 41), np.nan, dtype=np.float32)
        age = float(np.clip(rng.normal(64.2, 15.6), 18, 100)); male = float(rng.rand() < 0.495)
        unit = rng.rand(); hosp = -abs(rng.normal(30, 60))
        a[:, COL['Age']] = round(age, 2); a[:, COL['Gender']] = male
        if unit < 0.4: a[:, COL['Unit1']], a[:, COL['Unit2']] = 1, 0
        elif unit < 0.8: a[:, COL['Unit1']], a[:, COL['Unit2']] = 0, 1
        a[:, COL['HospAdmTime']] = round(hosp, 2); a[:, COL['ICULOS']] = np.arange(1, L + 1)
        t = np.arange(L)
        ramp = np.zeros(L) if not septic else np.clip((t - (t0 - 12)) / 12.0, 0, 1.5)    # drift starts 12 h before onset
        for name, (mu, sd) in BASE.items():
            j = COL[name]
            person = rng.normal(0, 0.6 * sd)
            series = mu + person + np.cumsum(rng.normal(0, 0.08 * sd, L)) + rng.normal(0, 0.4 * sd, L)
            if name in DRIFT:
                series = series + DRIFT[name] * signal * sd * ramp
            if name in VITAL_P:
                obs = rng.rand(L) < VITAL_P[name]
            else:
                period = rng.choice([6, 12, 24]); obs = np.zeros(L, bool)
                if rng.rand() < 0.7:
                    obs[rng.randint(0, period)::period] = True
                obs &= rng.rand(L) < 0.85
            a[obs, j] = np.round(series[obs], 2)
        if septic:
            a[np.maximum(t0 - 6, 0):, COL['SepsisLabel']] = 1      # PhysioNet convention: label = 1 for t >= t_sepsis - 6
            a[:np.maximum(t0 - 6, 0), COL['SepsisLabel']] = 0
        else:
            a[:, COL['SepsisLabel']] = 0
        out.append(('s%06d' % i, a))
    return out


def write_psv(stays, root):
    os.makedirs(os.path.join(root, 'training_setA'), exist_ok=True)
    for pid, a in stays:
        with open(os.path.join(root, 'training_setA', pid + '.psv'), 'w') as f:
            f.write('|'.join(COLUMNS) + '\n')
            for row in a:
                f.write('|'.join('NaN' if not np.isfinite(v) else ('%g' % v) for v in row) + '\n')


def as_dataset(stays):
    from .data import PhysioNet2019
    values = np.concatenate([a for _, a in stays]); offsets = np.r_[0, np.cumsum([len(a) for _, a in stays])]
    return PhysioNet2019.from_arrays(values, offsets, [p for p, _ in stays])

"""Variable selection, plausibility filtering, missingness channels, imputation and standardisation (thesis §3.4)."""
from __future__ import annotations
import numpy as np
from .data import COL, VITALS_ALL, LABS_ALL

# Thesis §3.4: vital-sign modality = heart rate, systolic BP, MAP, respiratory rate, SpO2, temperature;
# laboratory modality = WBC, lactate, creatinine, bilirubin, platelet count; static = age, sex.
VARIABLE_SETS = {
    'ch3': {'vitals': ['HR', 'SBP', 'MAP', 'Resp', 'O2Sat', 'Temp'],
            'labs': ['WBC', 'Lactate', 'Creatinine', 'Bilirubin_total', 'Platelets'],
            'static': ['Age', 'Gender'], 'dynamic_static': []},
    # Thesis §4.10 (implementation description): 8 vitals, 26 labs, 6 static/administrative features (ISSUE D-05)
    'ch4_full': {'vitals': list(VITALS_ALL), 'labs': list(LABS_ALL),
                 'static': ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime'], 'dynamic_static': ['ICULOS']},
}

# PROVISIONAL (ISSUE M-04): the thesis removes "physiologically implausible values ... using fixed clinical ranges"
# without listing them. Wide ranges are used so that only clear recording errors are removed.
PLAUSIBLE = {
    'HR': (20, 300), 'O2Sat': (40, 100), 'Temp': (25, 45), 'SBP': (40, 300), 'MAP': (20, 250), 'DBP': (10, 200),
    'Resp': (2, 80), 'EtCO2': (5, 100), 'BaseExcess': (-40, 40), 'HCO3': (2, 60), 'FiO2': (0.15, 1.0), 'pH': (6.5, 8.0),
    'PaCO2': (5, 150), 'SaO2': (20, 100), 'AST': (1, 20000), 'BUN': (1, 300), 'Alkalinephos': (5, 5000),
    'Calcium': (2, 20), 'Chloride': (50, 150), 'Creatinine': (0.1, 30), 'Bilirubin_direct': (0.01, 50),
    'Glucose': (10, 2000), 'Lactate': (0.1, 40), 'Magnesium': (0.2, 10), 'Phosphate': (0.2, 20),
    'Potassium': (1, 12), 'Bilirubin_total': (0.05, 80), 'TroponinI': (0.0, 500), 'Hct': (5, 75), 'Hgb': (1, 25),
    'PTT': (10, 250), 'WBC': (0.05, 300), 'Fibrinogen': (20, 2000), 'Platelets': (1, 3000),
}
DELTA_CAP = 48.0   # PROVISIONAL: elapsed-time channel capped at 48 h and scaled to [0, 1]


def plausibility_filter(values: np.ndarray, names: list[str]) -> tuple[np.ndarray, int]:
    v = values.copy(); removed = 0
    for j, n in enumerate(names):
        lo, hi = PLAUSIBLE.get(n, (-np.inf, np.inf))
        bad = np.isfinite(v[:, j]) & ((v[:, j] < lo) | (v[:, j] > hi))
        removed += int(bad.sum()); v[bad, j] = np.nan
    return v, removed


def last_obs_index(mask: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """For each row/channel: global row index of the most recent observation within the same stay (-1 if none)."""
    n, c = mask.shape
    rows = np.arange(n, dtype=np.int32)[:, None].repeat(c, axis=1)
    idx = np.where(mask, rows, np.int32(-1)).astype(np.int32)
    out = np.empty_like(idx)
    for s in range(len(offsets) - 1):
        a, b = offsets[s], offsets[s + 1]
        out[a:b] = np.maximum.accumulate(idx[a:b], axis=0)
    return out


class Preprocessed:
    """Hourly per-stay arrays for one variable set (rows of the selected cohort only)."""
    def __init__(self, ds, cohort_index: np.ndarray, varset: str = 'ch3'):
        spec = VARIABLE_SETS[varset]
        self.varset = varset
        self.vitals, self.labs = spec['vitals'], spec['labs']
        self.dyn = self.vitals + self.labs
        self.static_names = spec['static']; self.dynamic_static = spec['dynamic_static']
        offs = [0]; chunks = []
        for i in cohort_index:
            s = ds.stay(int(i)); chunks.append(s); offs.append(offs[-1] + len(s))
        raw = np.concatenate(chunks, axis=0)
        self.offsets = np.array(offs, dtype=np.int64)
        self.stay_of_row = np.repeat(np.arange(len(cohort_index)), np.diff(self.offsets))
        self.t_of_row = np.arange(len(raw)) - self.offsets[self.stay_of_row]
        v = raw[:, [COL[n] for n in self.dyn]]
        self.raw, self.n_implausible = plausibility_filter(v, self.dyn)
        self.mask = np.isfinite(self.raw)
        lo = last_obs_index(self.mask, self.offsets)
        rows = np.arange(len(raw))[:, None]
        delta = np.where(lo >= 0, rows - lo, self.t_of_row[:, None] + 1).astype(np.float32)   # PROVISIONAL: never observed -> hours since ICU admission
        self.delta = np.minimum(delta, DELTA_CAP) / DELTA_CAP
        ff = np.where(lo >= 0, self.raw[np.maximum(lo, 0), np.arange(self.raw.shape[1])[None, :]], np.nan)
        self.ffill_raw = ff.astype(np.float32)    # forward-filled, unscaled (used by rule-based scores)
        st = raw[self.offsets[:-1]][:, [COL[n] for n in self.static_names]].astype(np.float32)
        self.static_raw = st
        self.dyn_static_raw = raw[:, [COL[n] for n in self.dynamic_static]].astype(np.float32) if self.dynamic_static else None
        self.label_raw = raw[:, COL['SepsisLabel']]
        self.lab_cols = np.arange(len(self.vitals), len(self.dyn))
        self.vital_cols = np.arange(len(self.vitals))

    def fit_scaler(self, train_stays: np.ndarray):
        rows = np.concatenate([np.arange(self.offsets[s], self.offsets[s + 1]) for s in train_stays])
        r = self.raw[rows]
        self.mean = np.nanmean(r, axis=0).astype(np.float32)
        self.std = (np.nanstd(r, axis=0) + 1e-6).astype(np.float32)
        st = self.static_raw[train_stays]
        self.s_mean = np.nanmean(st, axis=0).astype(np.float32); self.s_std = (np.nanstd(st, axis=0) + 1e-6).astype(np.float32)
        for k, n in enumerate(self.static_names):
            if n in ('Gender', 'Unit1', 'Unit2'):
                self.s_mean[k], self.s_std[k] = 0.0, 1.0
        if self.dyn_static_raw is not None:
            ds_ = self.dyn_static_raw[rows]
            self.ds_mean = np.nanmean(ds_, axis=0); self.ds_std = np.nanstd(ds_, axis=0) + 1e-6
        return self

    def transform(self):
        """Thesis §3.4: last observation carried forward; values before the first observation filled with the
        training-partition population mean; z-score standardisation with training statistics."""
        x = np.where(np.isfinite(self.ffill_raw), self.ffill_raw, self.mean[None, :])
        self.x = ((x - self.mean[None, :]) / self.std[None, :]).astype(np.float32)
        st = np.where(np.isfinite(self.static_raw), self.static_raw, self.s_mean[None, :])
        self.static = ((st - self.s_mean) / self.s_std).astype(np.float32)
        if self.dyn_static_raw is not None:
            d = np.where(np.isfinite(self.dyn_static_raw), self.dyn_static_raw, self.ds_mean)
            self.dyn_static = ((d - self.ds_mean) / self.ds_std).astype(np.float32)
        else:
            self.dyn_static = None
        return self

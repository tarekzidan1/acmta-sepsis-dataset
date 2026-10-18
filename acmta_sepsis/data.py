"""Dataset loading, cohort construction, onset (T0) derivation and patient-level splitting.

Dataset: PhysioNet/Computing in Cardiology Challenge 2019, v1.0.0 (DOI 10.13026/v64v-d857, CC BY 4.0),
training_setA (20,336 PSV files) + training_setB (20,000 PSV files); one file per ICU stay, one row per hour,
40 variables + SepsisLabel (thesis §3.2, Table 3.1).
"""
from __future__ import annotations
import os, glob, json, hashlib
import numpy as np

COLUMNS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2',
           'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose',
           'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
           'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel']
COL = {c: i for i, c in enumerate(COLUMNS)}
VITALS_ALL = COLUMNS[0:8]
LABS_ALL = COLUMNS[8:34]
DEMOG_ALL = COLUMNS[34:40]


def _parse_psv(path: str) -> np.ndarray:
    with open(path, 'r') as f:
        header = f.readline().strip().split('|')
        if header != COLUMNS:
            raise ValueError(f'unexpected header in {path}')
        rows = [line.rstrip('\n').split('|') for line in f if line.strip()]
    a = np.array(rows, dtype=object)
    a[a == 'NaN'] = 'nan'
    return a.astype(np.float32)


def build_cache(raw_dir: str, cache_path: str) -> dict:
    """Read all PSV files once into a single NPZ: values (N_rows x 41, float32), offsets, patient ids, set label."""
    files = sorted(glob.glob(os.path.join(raw_dir, 'training_setA', '*.psv'))) + \
            sorted(glob.glob(os.path.join(raw_dir, 'training_setB', '*.psv')))
    if not files:
        raise FileNotFoundError(f'no PSV files under {raw_dir}/training_set[AB]; see scripts/download_physionet2019.sh')
    arrays, pids, sets, offsets = [], [], [], [0]
    for k, f in enumerate(files):
        a = _parse_psv(f)
        arrays.append(a)
        pids.append(os.path.splitext(os.path.basename(f))[0])
        sets.append('A' if 'training_setA' in f else 'B')
        offsets.append(offsets[-1] + len(a))
    values = np.concatenate(arrays, axis=0)
    np.savez_compressed(cache_path, values=values, offsets=np.array(offsets, dtype=np.int64),
                        pids=np.array(pids), sets=np.array(sets))
    return {'files': len(files), 'rows': int(values.shape[0])}


class PhysioNet2019:
    """In-memory view of the cached dataset (all stays)."""
    def __init__(self, cache_path: str):
        z = np.load(cache_path, allow_pickle=False)
        self.values = z['values']            # (rows, 41)
        self.offsets = z['offsets']          # (n+1,)
        self.pids = z['pids']
        self.sets = z['sets']
        self.n = len(self.pids)
        self.lengths = np.diff(self.offsets)

    @classmethod
    def from_arrays(cls, values, offsets, pids, sets=None):
        obj = cls.__new__(cls)
        obj.values = values.astype(np.float32); obj.offsets = np.asarray(offsets, dtype=np.int64)
        obj.pids = np.asarray(pids); obj.sets = np.asarray(sets if sets is not None else ['S'] * len(pids))
        obj.n = len(obj.pids); obj.lengths = np.diff(obj.offsets)
        return obj

    def stay(self, i: int) -> np.ndarray:
        return self.values[self.offsets[i]:self.offsets[i + 1]]


def onset_index(labels: np.ndarray, onset_shift_hours: int = 6):
    """Row index of Sepsis-3 onset T0 within a stay, or None for non-septic stays.

    Thesis §3.3: "The time of onset (T0) was taken from the labels supplied with the 2019 Challenge dataset".
    The challenge documentation states SepsisLabel = 1 for t >= t_sepsis - 6, i.e. the labels are shifted six
    hours ahead of onset. PROVISIONAL (ISSUE D-07): T0 = first labelled row + onset_shift_hours (default 6).
    For stays whose first row is already labelled, T0 is censored (onset may be earlier); flag returned.
    """
    pos = np.flatnonzero(labels > 0.5)
    if len(pos) == 0:
        return None, False
    first = int(pos[0])
    return first + onset_shift_hours, first == 0


def cohort(ds: PhysioNet2019, vitals: list[str], min_age: float = 18.0, min_icu_hours: int = 0,
           require_vital: bool = True, onset_shift_hours: int = 6) -> dict:
    """Thesis §3.3: adults (>= 18 y) admitted to an adult ICU with at least one recorded vital sign.
    "No further exclusions were applied" (§3.3). Chapter 4 (§4.3) additionally excluded ICU stays shorter than
    eight hours in the synthetic run -> optional `min_icu_hours` (ISSUE D-03)."""
    keep, t0, censored, septic, reasons = [], [], [], [], {'age': 0, 'no_vital': 0, 'short_stay': 0}
    vidx = [COL[v] for v in vitals]
    for i in range(ds.n):
        s = ds.stay(i)
        age = np.nanmax(s[:, COL['Age']]) if np.isfinite(s[:, COL['Age']]).any() else np.nan
        if not (age >= min_age):
            reasons['age'] += 1; continue
        if require_vital and not np.isfinite(s[:, vidx]).any():
            reasons['no_vital'] += 1; continue
        if len(s) < min_icu_hours:
            reasons['short_stay'] += 1; continue
        o, cens = onset_index(s[:, COL['SepsisLabel']], onset_shift_hours)
        keep.append(i); t0.append(-1 if o is None else o); censored.append(cens); septic.append(o is not None)
    return {'index': np.array(keep), 't0': np.array(t0), 'censored': np.array(censored), 'septic': np.array(septic),
            'excluded': reasons}


def split_patients(septic: np.ndarray, test_frac: float = 0.2, n_folds: int = 5, seed: int = 42) -> dict:
    """Thesis §3.7.1: patient-level 80:20 development/test split; patient-grouped, stratified five-fold CV on the
    development set (k = 5). Stratification by patient-level sepsis status (PROVISIONAL for the 80:20 split,
    which the thesis does not say is stratified)."""
    rng = np.random.RandomState(seed)
    n = len(septic)
    idx = np.arange(n)
    test = []
    for cls in (True, False):
        members = idx[septic == cls]
        rng.shuffle(members)
        test.extend(members[:int(round(test_frac * len(members)))])
    test = np.array(sorted(test))
    dev = np.setdiff1d(idx, test)
    folds = np.full(n, -1)
    for cls in (True, False):
        members = dev[septic[dev] == cls].copy()
        rng.shuffle(members)
        for k, m in enumerate(members):
            folds[m] = k % n_folds
    return {'test': test, 'dev': dev, 'fold': folds}


def file_manifest(raw_dir: str) -> dict:
    """SHA-256 of every PSV file (to compare a fresh official download with the copy used here)."""
    out = {}
    for sub in ('training_setA', 'training_setB'):
        for f in sorted(glob.glob(os.path.join(raw_dir, sub, '*.psv'))):
            out[f'{sub}/{os.path.basename(f)}'] = hashlib.sha256(open(f, 'rb').read()).hexdigest()
    return out

import numpy as np
from acmta_sepsis import data as D
from acmta_sepsis.windows import make_windows, window_rows


def test_onset_shift_six_hours():
    lab = np.array([0] * 10 + [1] * 5, dtype=float)
    o, cens = D.onset_index(lab, 6)
    assert o == 16 and not cens            # SepsisLabel starts 6 h before t_sepsis
    assert D.onset_index(np.zeros(5), 6) == (None, False)
    o, cens = D.onset_index(np.ones(5), 6)
    assert o == 6 and cens                 # onset censored when the first row is already labelled


def test_cohort_exclusions():
    vals = np.full((30, 41), np.nan, np.float32)
    vals[:, D.COL['Age']] = 17; vals[:10, D.COL['HR']] = 80          # stay 0: minor -> excluded
    vals[10:20, D.COL['Age']] = 50                                    # stay 1: no vital signs -> excluded
    vals[20:30, D.COL['Age']] = 60; vals[20:30, D.COL['HR']] = 90     # stay 2: kept
    vals[:, D.COL['SepsisLabel']] = 0
    ds = D.PhysioNet2019.from_arrays(vals, [0, 10, 20, 30], ['a', 'b', 'c'])
    co = D.cohort(ds, ['HR'])
    assert list(co['index']) == [2] and co['excluded']['age'] == 1 and co['excluded']['no_vital'] == 1


def test_split_patient_level_disjoint_and_stratified():
    rng = np.random.RandomState(0); septic = rng.rand(5000) < 0.073
    sp = D.split_patients(septic, seed=42)
    assert len(np.intersect1d(sp['test'], sp['dev'])) == 0
    assert len(sp['test']) + len(sp['dev']) == 5000
    assert abs(len(sp['test']) / 5000 - 0.2) < 0.01
    assert (sp['fold'][sp['test']] == -1).all() and set(np.unique(sp['fold'][sp['dev']])) == {0, 1, 2, 3, 4}
    for k in range(5):
        m = sp['fold'] == k
        assert abs(septic[m].mean() - septic[sp['dev']].mean()) < 0.01


def test_window_labels_ch3_and_buffer():
    offsets = np.array([0, 40, 80]); t0 = np.array([30, -1])
    s, t, y = make_windows(offsets, t0, np.array([0, 1]))
    ts0, y0 = t[s == 0], y[s == 0]
    assert ts0.max() == 26                         # predictions stop 4 h before T0
    assert set(ts0[y0 == 1]) == set(range(18, 27))  # onset within the next 12 h, at least 4 h ahead
    assert (y[s == 1] == 0).all() and len(t[s == 1]) == 40
    s2, t2, y2 = make_windows(offsets, t0, np.array([0]), label_mode='ch4')
    assert set(t2[y2 == 1]) == set(range(14, 26))   # Chapter 4 wording: T0 within the 12 h after the buffer
    s3, t3, _ = make_windows(offsets, t0, np.array([1]), stride=2)
    assert (np.diff(t3) == 2).all()


def test_window_rows_never_cross_stays_or_look_ahead():
    offsets = np.array([0, 5, 25])
    rows, valid, rel = window_rows(offsets, np.array([1, 1]), np.array([0, 15]))
    assert (rows[0][valid[0]] == 5).all() and valid[0].sum() == 1     # first hour of stay 1: only itself
    assert rows[1][-1] == 20 and rows[1][0] == 9 and valid[1].all()   # [t-11, t], no future rows
    assert rel[0, -1] == 0 and rel[0, 0] == -11

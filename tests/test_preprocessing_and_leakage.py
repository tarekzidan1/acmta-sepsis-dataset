import numpy as np
from acmta_sepsis.preprocessing import last_obs_index, plausibility_filter


def test_last_obs_index_within_stay_only():
    mask = np.array([[1], [0], [0], [0], [1], [0]], bool)
    lo = last_obs_index(mask, np.array([0, 3, 6]))
    assert list(lo[:, 0]) == [0, 0, 0, -1, 4, 4]     # stay 2 cannot use stay 1's observation


def test_plausibility_filter():
    v = np.array([[300.5, 98.0], [80.0, 101.0]], np.float32)
    out, n = plausibility_filter(v, ['HR', 'O2Sat'])
    assert n == 2 and np.isnan(out[0, 0]) and np.isnan(out[1, 1])


def test_forward_fill_uses_past_only_and_train_mean(syn_prep):
    pp, co, sp = syn_prep
    j = pp.dyn.index('WBC')
    s = 0; a, b = pp.offsets[s], pp.offsets[s + 1]
    raw = pp.raw[a:b, j]; ff = pp.ffill_raw[a:b, j]
    last = np.nan
    for k in range(b - a):
        if np.isfinite(raw[k]): last = raw[k]
        assert (np.isnan(last) and np.isnan(ff[k])) or ff[k] == last
    never = ~np.isfinite(ff)
    if never.any():
        assert np.allclose(pp.x[a:b, j][never], 0.0, atol=1e-5)     # train mean -> 0 after standardisation


def test_scaler_uses_training_patients_only(syn_ds):
    from acmta_sepsis import data as D
    from acmta_sepsis.preprocessing import Preprocessed
    co = D.cohort(syn_ds, ['HR', 'SBP', 'MAP', 'Resp', 'O2Sat', 'Temp'], min_icu_hours=8)
    pp1 = Preprocessed(syn_ds, co['index'], 'ch3').fit_scaler(np.arange(50))
    pp2 = Preprocessed(syn_ds, co['index'], 'ch3')
    pp2.raw[pp2.offsets[100]:pp2.offsets[150]] += 1000.0              # perturb non-training stays
    pp2.fit_scaler(np.arange(50))
    assert np.allclose(pp1.mean, pp2.mean) and np.allclose(pp1.std, pp2.std)

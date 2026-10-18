import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from acmta_sepsis import synthetic, data as D
from acmta_sepsis.preprocessing import Preprocessed, VARIABLE_SETS


@pytest.fixture(scope='session')
def syn_ds():
    """Small SYNTHETIC dataset (not real patient data)."""
    return synthetic.as_dataset(synthetic.generate(n_patients=300, prevalence=0.2, signal=1.0, seed=3))


@pytest.fixture(scope='session')
def syn_prep(syn_ds):
    co = D.cohort(syn_ds, VARIABLE_SETS['ch3']['vitals'], min_icu_hours=8)
    sp = D.split_patients(co['septic'], seed=1)
    pp = Preprocessed(syn_ds, co['index'], 'ch3')
    train = sp['dev'][sp['fold'][sp['dev']] != 0]
    pp.fit_scaler(train).transform()
    return pp, co, sp

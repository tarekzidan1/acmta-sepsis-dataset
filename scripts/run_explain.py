#!/usr/bin/env python
import os, sys, pickle, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from acmta_sepsis.experiments import load, RESULTS
from acmta_sepsis.features import window_features
from acmta_sepsis.explain import permutation_importance, tree_shap, attention_profile, integrated_gradients
from acmta_sepsis.train import load_best
from acmta_sepsis.utils import dump
NAME = 'real_ch3'; P = load(NAME); pp = P['pp']; s, t, y = P['windows']['test']
rng = np.random.RandomState(0)
pos = np.flatnonzero(y == 1); neg = rng.choice(np.flatnonzero(y == 0), 30000, replace=False)
idx = np.sort(np.r_[pos, neg])
X, names = window_features(pp, s[idx], t[idx])
gbm = pickle.load(open(os.path.join(RESULTS, NAME, 'baselines', 'gbm_model.pkl'), 'rb'))
base, perm = permutation_importance(gbm, X, y[idx], names, n_repeats=2)
shp = tree_shap(gbm, X, names)
model, cfg, ck = load_best(os.path.join(RESULTS, NAME, 'runs', 'bilstm_acmta'))
tp = pos[:600]
att = attention_profile(model, pp, s[tp], t[tp]); ig = integrated_gradients(model, pp, s[tp], t[tp])
dump({'note': 'test-set subsample: all positive windows + 30,000 random negative windows (AUROC on this subsample differs from the full test AUROC)',
      'gbm_auroc_subsample': base, 'gbm_permutation_importance_top30': perm[:30], 'gbm_mean_abs_shap_top30': shp[:30],
      'acmta_attention_profile_true_positive_windows': att, 'acmta_integrated_gradients_vitals': ig}, os.path.join(RESULTS, NAME, 'explainability.json'))
print('perm top10', [(n, round(v, 4)) for n, v, _ in perm[:10]]); print('shap top10', [(n, round(v, 4)) for n, v in shp[:10]])
print('attention', [round(x, 3) for x in att['mean_attention_vitals_to_labs']])

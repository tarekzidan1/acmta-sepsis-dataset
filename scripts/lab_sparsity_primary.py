# pre-specified primary comparison of the stress test: ACMTA (BiLSTM) vs early fusion (seed 42), paired DeLong, Holm over the 4 levels
import numpy as np, json, os, sys
sys.path.insert(0, '.')
from acmta_sepsis.evaluate import delong_test, holm, auroc
R = 'results/real_ch3'
y = np.load(f'{R}/runs/bilstm_acmta/predictions.npz')['y_test']
assert (y == np.load(f'{R}/runs/bilstm_early/predictions.npz')['y_test']).all()
res = {}
for q in (0.25, 0.5, 0.75, 1.0):
    pa = np.load(f'{R}/lab_sparsity/bilstm_acmta_q{q}.npz')['p']; pe = np.load(f'{R}/lab_sparsity/bilstm_early_q{q}.npz')['p']
    res[q] = {'auroc_acmta': auroc(y, pa), 'auroc_early': auroc(y, pe), 'delong': delong_test(y, pa, pe)}
ph = holm([r['delong']['p'] if isinstance(r['delong'], dict) else r['delong'][1] for r in res.values()])
for (q, r), p in zip(res.items(), ph):
    r['p_holm'] = float(p)
json.dump({str(k): v for k, v in res.items()}, open(f'{R}/lab_sparsity_primary_acmta_vs_early_seed42.json', 'w'), indent=1, default=float)
print(json.dumps({str(k): v for k, v in res.items()}, indent=1, default=float))

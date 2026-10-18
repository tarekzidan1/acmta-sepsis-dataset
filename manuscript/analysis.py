"""Analyses for the journal manuscript (real PhysioNet/CinC 2019 data, Chapter Three protocol).
Reads the saved predictions of the reproduction package; writes paper/analysis.json and paper/curves.npz.
Seed 42 = pre-specified primary run (bootstrap CIs, operating points, subgroups); seeds 43/44 = training-seed spread."""
import os, sys, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve
PKG = os.environ.get('ACMTA_PKG', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get('ACMTA_PAPER', 'manuscript_out')
sys.path.insert(0, PKG)
os.environ.setdefault('ACMTA_WORK', 'work')
os.environ.setdefault('ACMTA_RESULTS', PKG + '/results')
from acmta_sepsis import experiments as X, evaluate as E
from scipy.stats import rankdata
E._midrank = lambda x: rankdata(x)   # vectorised midranks (identical values to the package's loop implementation)

R = PKG + '/results/real_ch3'
P = X.load('real_ch3'); pp = P['pp']; t0 = P['cohort']['t0']; hours = P['hours']
s_dev, t_dev, y_dev = P['windows']['dev']; fold = P['dev_fold']; vf = P['meta']['val_fold']
s_te, t_te, y_te = P['windows']['test']
src = np.load(os.environ.get('ACMTA_DATA', 'data/physionet2019.npz'), allow_pickle=True)
hosp = np.array(src['sets'])[np.asarray(P['cohort']['index'])]          # 'A'/'B' per cohort stay
names = list(pp.static_names)
age = pp.static_raw[:, names.index('Age')]; male = pp.static_raw[:, names.index('Gender')] == 1

DEEP = {'ACMTA (BiLSTM)': 'bilstm_acmta', 'ACMTA (Transformer)': 'transformer_acmta', 'Early fusion': 'bilstm_early',
        'Late fusion': 'bilstm_late', 'Vital signs only': 'bilstm_vitals_only', 'Laboratory only': 'bilstm_labs_only',
        'ACMTA without MAMG': 'bilstm_acmta_nomamg', 'ACMTA without time-gap bias': 'bilstm_acmta_notimebias'}
SEEDS = {42: '', 43: '_s43', 44: '_s44'}
BASE = {'qSOFA (proxy)': 'qsofa', 'SOFA (proxy)': 'sofa_proxy', 'NEWS2 (proxy)': 'news_proxy', 'Logistic regression': 'lr', 'Gradient boosting': 'gbm'}

def platt(pv, yv, pt):  # recalibration fitted on validation predictions only
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lg(pv)[:, None], yv)
    return m.predict_proba(lg(pt)[:, None])[:, 1]

pred = {}
for name, run in DEEP.items():
    for seed, suf in SEEDS.items():
        f = f'{R}/runs/{run}{suf}/predictions.npz'
        if os.path.exists(f):
            z = np.load(f)
            assert np.array_equal(z['s_test'], s_te) and np.array_equal(z['t_test'], t_te)
            pred[(name, seed)] = dict(p_val=z['p_val'], y_val=z['y_val'], s_val=z['s_val'], p_test=z['p_test'])
zb = np.load(f'{R}/baselines/predictions.npz')
assert np.array_equal(zb['s_test'], s_te) and np.array_equal(zb['s_dev'], s_dev)
vmask = fold == vf
for name, k in BASE.items():
    pred[(name, 42)] = dict(p_val=zb[f'{k}_dev_p'][vmask], y_val=y_dev[vmask], s_val=s_dev[vmask], p_test=zb[f'{k}_test_p'])


MODE = sys.argv[1] if len(sys.argv) > 1 else 'metrics'
if MODE.startswith('sub:'):
    name = MODE[4:]
    st = s_te
    groups = {'Age 18-44': age[st] < 45, 'Age 45-64': (age[st] >= 45) & (age[st] < 65), 'Age 65-79': (age[st] >= 65) & (age[st] < 80), 'Age >=80': age[st] >= 80,
              'Female': ~male[st], 'Male': male[st], 'Hospital system A': hosp[st] == 'A', 'Hospital system B': hosp[st] == 'B'}
    p = pred[(name, 42)]['p_test']; out = {}
    for g, msk in groups.items():
        ci = E.bootstrap_ci(y_te[msk], p[msk], s_te[msk], n_boot=300, seed=42)
        out[g] = {'n_windows': int(msk.sum()), 'n_pos': int(y_te[msk].sum()), 'n_stays': int(len(np.unique(s_te[msk]))),
                  'auroc': E.auroc(y_te[msk], p[msk]), 'ci95': ci['auroc']}
    f = f'{OUT}/subgroups.json'; d = json.load(open(f)) if os.path.exists(f) else {}
    d[name] = out; json.dump(d, open(f, 'w'), indent=1); print('subgroups', name, {g: round(v['auroc'], 3) for g, v in out.items()}); sys.exit(0)
AJ = f'{OUT}/analysis.json'
res = json.load(open(AJ)) if (MODE != 'metrics' and os.path.exists(AJ)) else {'metrics': {}, 'seeds': {}, 'tests': {}, 'operating': {}, 'subgroups': {}, 'cohort': {}}
def save(): json.dump(res, open(AJ, 'w'), indent=1, default=float)
brep = json.load(open(f'{R}/baselines/baseline_report.json'))
if MODE == 'metrics':
  def cohort(st_, y_w):
      sep = np.asarray(P['cohort']['septic'])[st_]; los = hours[st_]
      return {'stays': int(len(st_)), 'septic': int(sep.sum()), 'septic_pct': float(100 * sep.mean()),
              'age_mean': float(age[st_].mean()), 'age_sd': float(age[st_].std(ddof=1)), 'male_pct': float(100 * male[st_].mean()),
              'hospital_A_pct': float(100 * (hosp[st_] == 'A').mean()), 'los_median': float(np.median(los)),
              'los_iqr': [float(np.percentile(los, 25)), float(np.percentile(los, 75))],
              'windows': int(len(y_w)), 'pos_windows': int(y_w.sum()), 'pos_pct': float(100 * y_w.mean()),
              'censored_septic': int(np.asarray(P['cohort']['censored'])[st_].sum())}
  res['cohort'] = {'development': cohort(np.unique(s_dev), y_dev), 'test': cohort(np.unique(s_te), y_te), 'meta': P['meta']}


  for (name, seed), d in pred.items():
      p = d['p_test']; pc = platt(d['p_val'], d['y_val'], p)
      m = {'auroc': E.auroc(y_te, p), 'auprc': E.auprc(y_te, p), 'brier': E.brier(y_te, p), 'ece': E.ece(y_te, p)[0],
           'brier_platt': E.brier(y_te, pc), 'ece_platt': E.ece(y_te, pc)[0]}
      if seed == 42:
          rep = json.load(open(f'{R}/runs/{DEEP[name]}/test_report.json')) if name in DEEP else brep[BASE[name]]['test']
          m['ci95'] = rep['ci95']
          if name in ('ACMTA without MAMG', 'ACMTA without time-gap bias'):   # queued with 200 resamples -> recompute with 2,000
              m['ci95'] = E.bootstrap_ci(y_te, p, s_te, n_boot=2000, seed=42)
      res['metrics'][f'{name}|{seed}'] = m
  for name in DEEP:
      vals = [res['metrics'][f'{name}|{s}'] for s in SEEDS if f'{name}|{s}' in res['metrics']]
      res['seeds'][name] = {k: {'mean': float(np.mean([v[k] for v in vals])), 'sd': float(np.std([v[k] for v in vals], ddof=1)) if len(vals) > 1 else None,
                                'n': len(vals), 'values': [v[k] for v in vals]} for k in ('auroc', 'auprc', 'brier', 'ece', 'brier_platt', 'ece_platt')}

  ref = 'ACMTA (BiLSTM)'
  comps = [n for n in list(DEEP) + list(BASE) if n != ref]
  for seed in SEEDS:
      if (ref, seed) not in pred: continue
      pa = pred[(ref, seed)]['p_test']; rows = []
      for c in comps:
          key = (c, seed) if (c, seed) in pred else ((c, 42) if c in BASE else None)
          if key is None or key not in pred: continue
          pb = pred[key]['p_test']; dl = E.delong_test(y_te, pa, pb)
          rows.append({'comparator': c, 'auroc_ref': E.auroc(y_te, pa), 'auroc_cmp': E.auroc(y_te, pb), 'delong': dl})
      ps = E.holm([r['delong']['p'] for r in rows])
      for r, ph in zip(rows, ps): r['p_holm'] = float(ph)
      res['tests'][str(seed)] = rows

ref = 'ACMTA (BiLSTM)'
hrs_te = hours[np.unique(s_te)].sum()
def opp(name, seed):
    d = pred[(name, seed)]
    cv = E.operating_curve(d['y_val'], d['p_val'], d['s_val'], hours[np.unique(d['s_val'])])
    sel = E.cucts(cv); out = {'feasible_on_validation': sel['feasible']}
    for lab, k in (('budget', 'at_budget'), ('sens_floor', 'at_sens_floor')):
        thr = sel[k]['threshold']
        o = E.at_threshold(y_te, d['p_test'], thr, hrs_te)
        o['lead_time'] = E.lead_times(s_te, t_te, y_te, d['p_test'], t0, thr); out[lab] = o
    yd = E.youden(d['y_val'], d['p_val']); out['youden'] = E.at_threshold(y_te, d['p_test'], yd['threshold'], hrs_te)
    return out
if MODE == 'metrics':
    save(); print('metrics saved')
if MODE == 'ops':
    for (name, seed) in pred:
        if (seed == 42 or name == ref) and f'{name}|{seed}' not in res['operating']:
            res['operating'][f'{name}|{seed}'] = opp(name, seed)
    save(); print('operating points saved', len(res['operating'])); sys.exit(0)
if MODE != 'curves':
    for n, v in res['seeds'].items():
        sd = v['auroc']['sd']; print(f"{n:28s} AUROC {v['auroc']['mean']:.4f} +/- {sd if sd is not None else float('nan'):.4f} (n={v['auroc']['n']})  AUPRC {v['auprc']['mean']:.4f}")
    for r in res['tests']['42']: print(f"vs {r['comparator']:28s} dAUROC {r['auroc_ref']-r['auroc_cmp']:+.4f}  p_holm {r['p_holm']:.3g}")
    sys.exit(0)

curves = {}
FIGM = ['ACMTA (BiLSTM)', 'ACMTA (Transformer)', 'Early fusion', 'Late fusion', 'Vital signs only', 'Laboratory only', 'Gradient boosting', 'Logistic regression', 'NEWS2 (proxy)', 'qSOFA (proxy)']
def thin(a, b, n=400):
    idx = np.unique(np.r_[np.linspace(0, len(a) - 1, n).astype(int), len(a) - 1]); return a[idx], b[idx]
th = np.linspace(0.002, 0.10, 50)
for i, name in enumerate(FIGM):
    d = pred[(name, 42)]; p = d['p_test']
    fpr, tpr, _ = roc_curve(y_te, p); fpr, tpr = thin(fpr, tpr)
    pr, rc, _ = precision_recall_curve(y_te, p); rc, pr = thin(rc[::-1], pr[::-1])
    oc = E.operating_curve(y_te, p, s_te, hours[np.unique(s_te)])
    a, sns = thin(oc['alarms_per_patient_day'], oc['sensitivity'], 800)
    curves[f'roc_fpr_{i}'] = fpr; curves[f'roc_tpr_{i}'] = tpr; curves[f'pr_rec_{i}'] = rc; curves[f'pr_prec_{i}'] = pr
    curves[f'alarm_rate_{i}'] = a; curves[f'alarm_sens_{i}'] = sns
    pc = platt(d['p_val'], d['y_val'], p)
    for tag, pv in (('platt', pc), ('raw', p)):
        q = np.quantile(pv, np.linspace(0, 1, 11)); idx = np.clip(np.searchsorted(q[1:-1], pv, side='right'), 0, 9)
        curves[f'rel_{tag}_pred_{i}'] = np.array([pv[idx == b].mean() for b in range(10)])
        curves[f'rel_{tag}_obs_{i}'] = np.array([y_te[idx == b].mean() for b in range(10)])
    dca = E.decision_curve(y_te, pc, th)
    curves[f'dca_nb_{i}'] = np.array([x['net_benefit'] for x in dca])
curves['dca_thresholds'] = th; curves['dca_treat_all'] = np.array([x['treat_all'] for x in dca])
res['figure_models'] = FIGM
np.savez_compressed(f'{OUT}/curves.npz', **curves)
save(); print('curves saved')

#!/usr/bin/env python
"""Build the thesis-versus-code comparison table and the real-data results table from the saved result files."""
import os, sys, json, csv
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
syn = json.load(open(os.path.join(R, 'synthetic_ch4_protocol', 'ch4.json')))
lat = json.load(open(os.path.join(R, 'real_ch3', 'engineering_latency.json')))
TOL = {'count': 0.0, 'pct': 0.1, 'auroc': 0.02, 'auprc': 0.005, 'brier': 0.005, 'latency_rel': 0.2}
STATUS = ('NOT REPRODUCIBLE under the same conditions: the thesis generator (scripts/synthesize_pnet2019_format.py) and its seed were '
          'not supplied; measured value comes from an independent generator implementing the §4.2 description (protocol comparison only)')
c = syn['cohort']; b = syn['baselines']
rows = []
def add(exp, metric, thesis, measured, tol_key, note='', status=STATUS, cond='Synthetic Ch4 protocol (2,000 stays, >=8 h, 40 variables, 2-h stride, Ch4 labels, 5-fold patient CV), generator seed 7, fold seed 42'):
    if measured is None or thesis is None:
        absd = rel = None
    else:
        absd = measured - thesis; rel = (absd / thesis) if thesis else None
    tol = TOL.get(tol_key)
    within = None if absd is None else (abs(absd) <= (tol if tol_key != 'latency_rel' else abs(thesis) * tol))
    rows.append({'experiment': exp, 'metric': metric, 'thesis_value': thesis, 'measured_value': measured,
                 'abs_diff': None if absd is None else round(absd, 4), 'rel_diff': None if rel is None else round(rel, 4),
                 'pre_specified_tolerance': tol, 'numerically_within_tolerance': within, 'dataset_condition': cond, 'status': status, 'explanation': note})
add('Cohort (Table 4.1/4.2)', 'stays generated', 2000, c['generated'], 'count')
add('Cohort (Table 4.1/4.2)', 'stays kept (>= 8 h)', 1445, c['kept'], 'count', 'Generator calibrated to the published summary statistics; agreement is by construction, not reproduction.')
add('Cohort (Table 4.1/4.2)', 'septic stays', 115, c['septic'], 'count')
add('Cohort (Table 4.1/4.2)', 'sepsis prevalence (%)', 7.96, round(c['prevalence_pct'], 2), 'pct')
add('Cohort (Table 4.1/4.2)', 'median ICU LOS (h)', 55, c['los_median'], 'count')
add('Cohort (Table 4.1/4.2)', 'mean age (y)', 64.2, round(c['age_mean'], 1), 'pct')
add('Cohort (Table 4.1/4.2)', 'male (%)', 49.5, round(c['male_pct'], 1), 'pct')
add('Cohort (Table 4.1/4.2)', 'prediction windows', 21636, c['windows'], 'count', 'Window count depends on stride, minimum history and LOS distribution; no combination tried (stride 1-4 h, min history 1/12 h) gives 21,636 with this generator.')
add('Cohort (Table 4.1/4.2)', 'positive windows', 383, c['positive_windows'], 'count')
add('Cohort (Table 4.1/4.2)', 'window-level prevalence (%)', 1.77, round(c['window_prevalence_pct'], 2), 'pct')
for k, nm, au, ap, br in (('gbm', 'Gradient boosting', 0.440, 0.021, 0.0991), ('lr', 'Logistic regression', 0.512, 0.022, 0.1719), ('qsofa', 'qSOFA', 0.501, 0.016, 0.5814), ('sofa_proxy', 'SOFA proxy', 0.542, 0.019, 0.0965)):
    note = ('Thesis flags its own below-chance GBM AUROC as a probable label/feature or fold-assignment error (§4.4); this implementation passes label/leakage tests and gives AUROC >= 0.5 on a weak-signal cohort.' if k == 'gbm' else
            'Nominal probability mapping of rule scores is not specified in the thesis (here score / maximum score); the Brier value depends on it.' if k in ('qsofa', 'sofa_proxy') else '')
    add(f'Table 4.3 - {nm}', 'AUROC', au, round(b[k]['auroc'], 4), 'auroc', note)
    add(f'Table 4.3 - {nm}', 'AUPRC', ap, round(b[k]['auprc'], 4), 'auprc')
    add(f'Table 4.3 - {nm}', 'Brier', br, round(b[k]['brier'], 4), 'brier')
add('§4.4/§5.1', 'Brier of constant prediction at window prevalence', 0.017, round(syn['brier_constant_prevalence'], 4), 'brier', 'Analytic: p(1-p).')
g = b['gbm']['cucts_false_alarms']
add('§5.1 CUCTS (GBM)', 'sensitivity at 3 false alarms per patient-day', 0.25, round(g['at_budget']['sensitivity'], 3), 'auprc', 'Thesis gives a range "about 0.2 to 0.3"; midpoint used. Qualitatively reproduced: constraints cannot be met simultaneously (infeasible).')
add('§5.1 CUCTS (GBM)', 'false alarms per patient-day at sensitivity 0.85', 11.0, round(g['at_sens_floor']['rate'], 2), 'latency_rel', 'Thesis: "approximately eleven".')
bl = lat['bilstm_acmta']
add('Figure 4.13', 'real-time inference latency per hourly prediction, CPU (ms)', 84.0, round(bl['realtime_ms_per_hour_threads1']['median'], 3), 'latency_rel',
    'Different hardware (Intel Core Ultra 7 265H, 2 threads) and model size (hidden 64, 110,731 parameters; thesis size not reported). Measured on the real-data BiLSTM-ACMTA model.',
    status='NOT COMPARABLE (hardware and model size differ; thesis configuration not reported)', cond='Real PhysioNet 2019 test patient, batch size 1')
add('§3.11/§4.10.1', 'peak GPU memory (GB)', 4.0, None, 'latency_rel', 'No GPU available in the execution environment.', status='NOT RUN (no GPU)', cond='-')
with open(os.path.join(R, 'comparison_table.csv'), 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
# real-data results table
real = []
br = json.load(open(os.path.join(R, 'real_ch3', 'baselines', 'baseline_report.json')))
names = {'qsofa': 'qSOFA (proxy: RR, SBP)', 'sofa_proxy': 'SOFA proxy (platelets, bilirubin, MAP, creatinine)', 'news_proxy': 'NEWS2 proxy (5 vital signs)',
         'lr': 'Logistic regression, engineered features', 'gbm': 'Gradient boosting (XGBoost), engineered features'}
for k, nm in names.items():
    t = br[k]['test']
    real.append({'model': nm, 'AUROC': t['auroc'], 'AUROC_95CI': t['ci95']['auroc'], 'AUPRC': t['auprc'], 'AUPRC_95CI': t['ci95']['auprc'], 'Brier': t['brier'], 'ECE': t['ece']})
for r, nm in (('bilstm_acmta', 'ACMTA-Sepsis, BiLSTM encoder'), ('transformer_acmta', 'ACMTA-Sepsis, Transformer encoder'), ('bilstm_early', 'Early fusion (BiLSTM)'),
              ('bilstm_late', 'Late fusion (BiLSTM)'), ('bilstm_vitals_only', 'Vital signs only (BiLSTM)'), ('bilstm_labs_only', 'Laboratory only (BiLSTM)')):
    t = json.load(open(os.path.join(R, 'real_ch3', 'runs', r, 'test_report.json')))
    st = json.load(open(os.path.join(R, 'real_ch3', 'runs', r, 'train_state.json')))
    real.append({'model': nm, 'AUROC': t['auroc'], 'AUROC_95CI': t['ci95']['auroc'], 'AUPRC': t['auprc'], 'AUPRC_95CI': t['ci95']['auprc'], 'Brier': t['brier'], 'ECE': t['ece'],
                 'best_epoch': t['checkpoint']['epoch'], 'epochs_run': st['epoch'], 'train_minutes': round(st['train_seconds'] / 60, 1), 'params': t['params']})
with open(os.path.join(R, 'real_data_results.csv'), 'w', newline='') as f:
    keys = sorted({k for r in real for k in r}, key=lambda k: ['model', 'AUROC', 'AUROC_95CI', 'AUPRC', 'AUPRC_95CI', 'Brier', 'ECE', 'best_epoch', 'epochs_run', 'train_minutes', 'params'].index(k))
    wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(real)
print('rows', len(rows), 'real', len(real))

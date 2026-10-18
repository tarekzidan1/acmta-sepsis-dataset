"""Formatted numbers for the manuscript text, tables and slides (single source of truth: analysis.json)."""
import json, os, numpy as np
D = os.environ.get('ACMTA_PAPER', 'manuscript_out')
A = json.load(open(D + '/analysis.json')); S = json.load(open(D + '/subgroups.json'))
EX = json.load(open(os.path.join(os.environ.get('ACMTA_RESULTS', 'results'), 'real_ch3', 'explainability.json')))
OPT = json.load(open(os.path.join(os.environ.get('ACMTA_RESULTS', 'results'), 'real_ch3', 'optuna_bilstm', 'study.json')))
f2 = lambda x: f'{x:.2f}'; f3 = lambda x: f'{x:.3f}'; f4 = lambda x: f'{x:.4f}'; c0 = lambda x: f'{int(round(x)):,}'
def pfmt(p):
    if p < 0.001: return 'p < 0.001'
    return f'p = {p:.3f}' if p < 0.01 else f'p = {p:.2f}'
F = {}
co = A['cohort']; meta = co['meta']; dv = co['development']; te = co['test']
F.update(n_source=c0(meta['n_stays_source']), n_cohort=c0(meta['cohort']), excl_age=str(meta['excluded']['age']), excl_novital=str(meta['excluded']['no_vital']),
         n_septic=c0(meta['septic']), prev_stay=f"{100 * meta['septic_prevalence']:.1f}", n_windows=c0(meta['windows_total']), n_pos=c0(meta['pos_total']),
         win_prev=f"{100 * meta['window_prevalence_total']:.2f}", censored=str(meta['onset_censored']), implausible=c0(meta['implausible_values_removed']),
         dev_stays=c0(dv['stays']), test_stays=c0(te['stays']), test_windows=c0(te['windows']), test_pos=c0(te['pos_windows']), test_prev=f"{te['pos_pct']:.2f}",
         dev_windows=c0(dv['windows']), dev_pos=c0(dv['pos_windows']))
M = A['metrics']; SD = A['seeds']
def ci(name, k='auroc'): lo, hi = M[f'{name}|42']['ci95'][k]; return f'{lo:.3f}–{hi:.3f}'
models = list(SD.keys()) + ['Gradient boosting', 'Logistic regression', 'NEWS2 (proxy)', 'qSOFA (proxy)', 'SOFA (proxy)']
key = lambda n: n.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower()
for n in models:
    k = key(n); m = M[f'{n}|42']
    F[f'{k}_auroc'] = f3(m['auroc']); F[f'{k}_auroc_ci'] = ci(n); F[f'{k}_auprc'] = f3(m['auprc']); F[f'{k}_auprc_ci'] = ci(n, 'auprc')
    F[f'{k}_brier'] = f3(m['brier']); F[f'{k}_ece'] = f3(m['ece']); F[f'{k}_brier_platt'] = f4(m['brier_platt']); F[f'{k}_ece_platt'] = f4(m['ece_platt'])
    if n in SD:
        s = SD[n]; F[f'{k}_auroc_mean'] = f3(s['auroc']['mean']); F[f'{k}_auroc_sd'] = f3(s['auroc']['sd']); F[f'{k}_auprc_mean'] = f3(s['auprc']['mean']); F[f'{k}_auprc_sd'] = f3(s['auprc']['sd'])
        F[f'{k}_auroc_ms'] = f"{s['auroc']['mean']:.3f} ± {s['auroc']['sd']:.3f}"; F[f'{k}_auprc_ms'] = f"{s['auprc']['mean']:.3f} ± {s['auprc']['sd']:.3f}"
# paired tests vs ACMTA (BiLSTM)
T = {str(s): {r['comparator']: r for r in rows} for s, rows in A['tests'].items()}
for n in [x for x in models if x != 'ACMTA (BiLSTM)']:
    k = key(n); r = T['42'][n]; F[f'{k}_d'] = f"{r['auroc_ref'] - r['auroc_cmp']:+.3f}"; F[f'{k}_p'] = pfmt(r['p_holm'])
    sig = [T[s][n]['p_holm'] < 0.05 for s in T if n in T[s]]; F[f'{k}_sig_seeds'] = f'{sum(sig)}/{len(sig)}'
    F[f'{k}_pmax'] = pfmt(max(T[s][n]['p_holm'] for s in T if n in T[s]))
# operating points (validation-selected thresholds applied to the test set)
O = A['operating']
for n in ['ACMTA (BiLSTM)', 'ACMTA (Transformer)', 'Early fusion', 'Late fusion', 'Gradient boosting', 'Logistic regression', 'NEWS2 (proxy)', 'Vital signs only', 'Laboratory only', 'qSOFA (proxy)', 'SOFA (proxy)']:
    k = key(n); o = O[f'{n}|42']
    for lab in ('budget', 'sens_floor'):
        x = o[lab]; lt = x['lead_time']['first_true_positive_alarm']
        F[f'{k}_{lab}_sens'] = f2(x['sensitivity']); F[f'{k}_{lab}_spec'] = f2(x['specificity']); F[f'{k}_{lab}_ppv'] = f"{100 * x['ppv']:.1f}"
        F[f'{k}_{lab}_alarms'] = f"{x['alarms_per_patient_day']:.1f}"; F[f'{k}_{lab}_falarms'] = f"{x['false_alarms_per_patient_day']:.1f}"
        F[f'{k}_{lab}_alerted'] = f"{lt.get('n_alerted', 0)}/{lt['n_septic']}"; F[f'{k}_{lab}_alerted_pct'] = f"{100 * lt.get('n_alerted', 0) / lt['n_septic']:.0f}"
        F[f'{k}_{lab}_lead'] = (f"{lt['median']:.0f} ({lt['iqr'][0]:.0f}–{lt['iqr'][1]:.0f})" if lt.get('n_alerted') else '–')
    F[f'{k}_feasible'] = 'yes' if o['feasible_on_validation'] else 'no'
acm = [O[f'ACMTA (BiLSTM)|{s}']['budget']['sensitivity'] for s in (42, 43, 44) if f'ACMTA (BiLSTM)|{s}' in O]
F['acmta_budget_sens_ms'] = f"{np.mean(acm):.2f} ± {np.std(acm, ddof=1):.2f}" if len(acm) > 1 else f2(acm[0])
best_budget = max(((n, O[f'{n}|42']['budget']['sensitivity']) for n in ['ACMTA (BiLSTM)', 'ACMTA (Transformer)', 'Early fusion', 'Late fusion', 'Gradient boosting', 'Logistic regression']), key=lambda x: x[1])
F['best_budget_model'] = best_budget[0]; F['best_budget_sens'] = f2(best_budget[1])
floor_rates = [O[f'{n}|42']['sens_floor']['alarms_per_patient_day'] for n in ['ACMTA (BiLSTM)', 'ACMTA (Transformer)', 'Early fusion', 'Late fusion', 'Gradient boosting', 'Logistic regression']]
F['floor_alarms_min'] = f'{min(floor_rates):.1f}'; F['floor_alarms_max'] = f'{max(floor_rates):.1f}'
budget_sens = [O[f'{n}|42']['budget']['sensitivity'] for n in ['ACMTA (BiLSTM)', 'ACMTA (Transformer)', 'Early fusion', 'Late fusion', 'Gradient boosting', 'Logistic regression']]
F['budget_sens_min'] = f2(min(budget_sens)); F['budget_sens_max'] = f2(max(budget_sens))
# subgroups
for n in ('ACMTA (BiLSTM)', 'Gradient boosting'):
    k = key(n); v = S[n]; a = [v[g]['auroc'] for g in v]
    F[f'{k}_sub_min'] = f3(min(a)); F[f'{k}_sub_max'] = f3(max(a))
    for g in v: F[f"{k}_sub_{g.replace(' ', '_').replace('>=', 'ge').replace('-', '_').lower()}"] = f3(v[g]['auroc'])
# explainability, optimisation, environment
F['perm_top'] = ', '.join(t[0] for t in EX['gbm_permutation_importance_top30'][:6])
att = EX['acmta_attention_profile_true_positive_windows']['mean_attention_vitals_to_labs']
F['att_last'] = f2(att[-1]); F['att_earlier_min'] = f2(min(att[:-1])); F['att_earlier_max'] = f2(max(att[:-1]))
tr = OPT['trials']; comp = [t for t in tr if t['state'] == 'COMPLETE']
F['opt_trials'] = str(len(tr)); F['opt_complete'] = str(len(comp)); F['opt_default'] = f3(OPT['best']['value'])
F['opt_best_sampled'] = f3(max(t['value'] for t in comp if t['number'] != OPT['best']['number']))
# calibration reference, decision curve, ablation deltas, floor PPV
prev = A['cohort']['test']['pos_pct'] / 100; F['brier_constant'] = f4(prev * (1 - prev))
CV = np.load(D + '/curves.npz'); th = CV['dca_thresholds']; nb = CV['dca_nb_0']; ta = CV['dca_treat_all']
ok = (nb > np.maximum(ta, 0)) ; F['dca_lo'] = f"{100 * th[ok].min():.1f}" if ok.any() else '–'; F['dca_hi'] = f"{100 * th[ok].max():.1f}" if ok.any() else '–'
F['dca_contiguous'] = 'yes' if ok.any() and ok[np.argmax(ok):len(ok) - np.argmax(ok[::-1])].all() else 'no'
fl = [(n, O[f'{n}|42']['sens_floor']['ppv']) for n in ['ACMTA (BiLSTM)', 'ACMTA (Transformer)', 'Early fusion', 'Late fusion', 'Gradient boosting', 'Logistic regression']]
F['max_floor_ppv'] = f"{100 * max(v for _, v in fl):.1f}"; F['min_floor_ppv'] = f"{100 * min(v for _, v in fl):.1f}"
for n in ('ACMTA without MAMG', 'ACMTA without time-gap bias', 'Early fusion', 'ACMTA (Transformer)', 'Late fusion'):
    k = key(n); d = [SD['ACMTA (BiLSTM)']['auroc']['values'][i] - SD[n]['auroc']['values'][i] for i in range(3)]
    mm = lambda x: f"{x:+.3f}".replace('-', '\u2212')
    F[f'{k}_dmean'] = mm(np.mean(d)); F[f'{k}_drange'] = f"{mm(min(d))} to {mm(max(d))}"
    F[f'{k}_dmean3'] = f"{np.mean(d):+.3f}"
F['early_fusion_sub_min'] = f3(min(v['auroc'] for v in S['Early fusion'].values())) if 'Early fusion' in S else '–'
F['early_fusion_sub_max'] = f3(max(v['auroc'] for v in S['Early fusion'].values())) if 'Early fusion' in S else '–'
for n in ('ACMTA (BiLSTM)', 'Gradient boosting'):
    k = key(n); v = S[n]
    for g in v: F[f"{k}_sub_{g.replace(' ', '_').replace('>=', 'ge').replace('-', '_').lower()}_ci"] = f"{v[g]['ci95'][0]:.3f}–{v[g]['ci95'][1]:.3f}"
F['test_septic'] = c0(A['cohort']['test']['septic']); F['dev_septic'] = c0(A['cohort']['development']['septic'])
json.dump(F, open(D + '/facts.json', 'w'), indent=1, ensure_ascii=False)
print(len(F), 'facts')
for k in ['brier_constant', 'dca_lo', 'dca_hi', 'dca_contiguous', 'max_floor_ppv', 'acmta_without_mamg_dmean', 'acmta_without_time_gap_bias_drange', 'acmta_without_mamg_sig_seeds', 'gradient_boosting_sig_seeds', 'early_fusion_sig_seeds', 'acmta_transformer_sig_seeds', 'acmta_bilstm_auroc', 'acmta_bilstm_auroc_ci', 'acmta_bilstm_auroc_ms', 'early_fusion_auroc', 'gradient_boosting_auroc', 'gradient_boosting_p', 'early_fusion_p', 'budget_sens_max', 'floor_alarms_min']:
    print(k, F.get(k))

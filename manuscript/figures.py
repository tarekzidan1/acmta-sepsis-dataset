"""Publication figures (Springer artwork rules: Arial-metric sans font 7-9 pt, no titles inside figures, panel letters a/b/c,
174 mm full width, EPS vector + 1200-dpi TIFF; colour-blind-safe palette plus distinct line styles)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
D = os.environ.get('ACMTA_PAPER', 'manuscript_out'); OUTD = D + '/figs'
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 8,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.title_fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.linewidth': 0.6, 'lines.linewidth': 1.1, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
MM = 1 / 25.4
A = json.load(open(D + '/analysis.json')); C = np.load(D + '/curves.npz'); FIGM = A['figure_models']
EX = json.load(open(os.path.join(os.environ.get('ACMTA_RESULTS', 'results'), 'real_ch3', 'explainability.json')))
STY = {'ACMTA (BiLSTM)': ('#0072B2', '-', 1.6), 'ACMTA (Transformer)': ('#56B4E9', '--', 1.1), 'Early fusion': ('#009E73', '-.', 1.1),
       'Late fusion': ('#7F7F7F', (0, (1, 1)), 1.0), 'Gradient boosting': ('#D55E00', '-', 1.0), 'Logistic regression': ('#E69F00', ':', 1.2),
       'NEWS2 (proxy)': ('#666666', (0, (5, 2, 1, 2)), 1.0), 'Vital signs only': ('#CC79A7', (0, (3, 1, 1, 1, 1, 1)), 1.0),
       'Laboratory only': ('#8C6D31', (0, (4, 2)), 1.0), 'qSOFA (proxy)': ('#AAAAAA', (0, (2, 2)), 1.0)}
def m(name, key): return A['metrics'][f'{name}|42'][key]
def save(fig, name):
    fig.savefig(f'{OUTD}/{name}.eps'); fig.savefig(f'{OUTD}/{name}.png', dpi=300)
    fig.savefig(f'{OUTD}/{name}.tif', dpi=1200, pil_kwargs={'compression': 'tiff_lzw'}); plt.close(fig)
def letter(ax, s, x=-0.16, y=1.04): ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='left')

# ---------------------------------------------------------------- Fig 2: ROC and PR
SEL = ['ACMTA (BiLSTM)', 'ACMTA (Transformer)', 'Early fusion', 'Gradient boosting', 'Logistic regression', 'Vital signs only', 'NEWS2 (proxy)']
fig, axs = plt.subplots(1, 2, figsize=(174 * MM, 78 * MM))
for nm in SEL:
    i = FIGM.index(nm); c, ls, lw = STY[nm]
    axs[0].plot(C[f'roc_fpr_{i}'], C[f'roc_tpr_{i}'], color=c, ls=ls, lw=lw, label=f"{nm} ({m(nm, 'auroc'):.3f})")
    axs[1].plot(C[f'pr_rec_{i}'], C[f'pr_prec_{i}'], color=c, ls=ls, lw=lw, label=f"{nm} ({m(nm, 'auprc'):.3f})")
axs[0].plot([0, 1], [0, 1], color='#BBBBBB', lw=0.6, ls='-')
prev = A['cohort']['test']['pos_pct'] / 100
axs[1].axhline(prev, color='#BBBBBB', lw=0.6); axs[1].text(0.03, prev - 0.003, f'no skill (prevalence {100 * prev:.1f}%)', ha='left', va='top', fontsize=8, color='#555555')
axs[0].set(xlabel='1 - specificity', ylabel='Sensitivity', xlim=(0, 1), ylim=(0, 1)); axs[1].set(xlabel='Recall (sensitivity)', ylabel='Precision', xlim=(0, 1), ylim=(0, 0.30))
axs[0].legend(title='AUROC', loc='lower right', frameon=False, title_fontsize=8, handlelength=2.6)
axs[1].legend(title='AUPRC', loc='upper right', frameon=False, title_fontsize=8, handlelength=2.6)
letter(axs[0], 'a'); letter(axs[1], 'b'); fig.tight_layout(w_pad=2.5); save(fig, 'Fig2')

# ---------------------------------------------------------------- Fig 3: alarm budget trade-off and decision curves
fig, axs = plt.subplots(1, 2, figsize=(174 * MM, 78 * MM))
ax = axs[0]
ax.add_patch(Rectangle((0, 0.85), 3, 0.15, facecolor='none', edgecolor='#333333', hatch='////', lw=0.6))
ax.text(1.5, 0.925, 'target\nregion', ha='center', va='center', fontsize=8, color='#333333', bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
for nm in ['ACMTA (BiLSTM)', 'Early fusion', 'Gradient boosting', 'Logistic regression', 'NEWS2 (proxy)']:
    i = FIGM.index(nm); c, ls, lw = STY[nm]
    ax.plot(C[f'alarm_rate_{i}'], C[f'alarm_sens_{i}'], color=c, ls=ls, lw=lw, label=nm)
ax.axvline(3, color='#333333', lw=0.6, ls='--'); ax.axhline(0.85, color='#333333', lw=0.6, ls='--')
ax.text(3.4, 0.03, 'budget: 3 alarms\nper patient-day', fontsize=8, va='bottom', ha='left'); ax.text(3.4, 0.865, 'floor 0.85', fontsize=8, ha='left')
ax.set(xlabel='Alarms per patient-day (window level)', ylabel='Sensitivity', xlim=(0, 24), ylim=(0, 1))
ax.legend(loc='lower right', frameon=False, handlelength=2.6)
ax = axs[1]; th = C['dca_thresholds'] * 100
ax.plot(th, C['dca_treat_all'], color='#999999', lw=0.8, ls='-', label='Treat all'); ax.axhline(0, color='#333333', lw=0.6, label='Treat none')
for nm in ['ACMTA (BiLSTM)', 'Early fusion', 'Gradient boosting', 'Logistic regression']:
    i = FIGM.index(nm); c, ls, lw = STY[nm]
    ax.plot(th, C[f'dca_nb_{i}'], color=c, ls=ls, lw=lw, label=nm + ' (recalibrated)')
ax.set(xlabel='Threshold probability (%)', ylabel='Net benefit', xlim=(0.2, 10), ylim=(-0.002, 0.012))
ax.legend(loc='upper right', frameon=False, handlelength=2.6)
letter(axs[0], 'a'); letter(axs[1], 'b'); fig.tight_layout(w_pad=2.5); save(fig, 'Fig3')

# ---------------------------------------------------------------- Fig 4: calibration and subgroups
fig, axs = plt.subplots(1, 2, figsize=(174 * MM, 80 * MM), gridspec_kw={'width_ratios': [1, 1.25]})
ax = axs[0]; i = FIGM.index('ACMTA (BiLSTM)'); j = FIGM.index('Gradient boosting')
ax.plot([1e-4, 1], [1e-4, 1], color='#BBBBBB', lw=0.6)
ax.plot(C[f'rel_raw_pred_{i}'], C[f'rel_raw_obs_{i}'], color=STY['ACMTA (BiLSTM)'][0], ls='--', marker='o', ms=3, mfc='white', lw=1.0, label='ACMTA, as trained')
ax.plot(C[f'rel_platt_pred_{i}'], C[f'rel_platt_obs_{i}'], color=STY['ACMTA (BiLSTM)'][0], ls='-', marker='o', ms=3, lw=1.2, label='ACMTA, recalibrated')
ax.plot(C[f'rel_platt_pred_{j}'], C[f'rel_platt_obs_{j}'], color=STY['Gradient boosting'][0], ls='-', marker='s', ms=3, lw=1.0, label='Boosting, recalibrated')
ax.set(xscale='log', yscale='log', xlim=(5e-4, 1), ylim=(5e-4, 0.6), xlabel='Mean predicted risk (decile)', ylabel='Observed proportion positive')
ax.legend(loc='upper left', frameon=True, framealpha=1, edgecolor='white', handlelength=2.2, borderaxespad=0.2)
ax = axs[1]; sub = json.load(open(D + '/subgroups.json'))
G = list(sub['ACMTA (BiLSTM)'].keys()); y0 = np.arange(len(G))[::-1]
for k, (nm, off, mk) in enumerate([('ACMTA (BiLSTM)', 0.16, 'o'), ('Gradient boosting', -0.16, 's')]):
    v = sub[nm]; c = STY[nm][0]
    au = np.array([v[g]['auroc'] for g in G]); lo = np.array([v[g]['ci95'][0] for g in G]); hi = np.array([v[g]['ci95'][1] for g in G])
    ax.errorbar(au, y0 + off, xerr=[au - lo, hi - au], fmt=mk, color=c, ms=3.5, lw=0.9, capsize=1.5, label=nm, mfc='white' if k else c)
ax.axvline(m('ACMTA (BiLSTM)', 'auroc'), color=STY['ACMTA (BiLSTM)'][0], lw=0.5, ls=':')
lab = [f"{g.replace('>=', '≥')}  (n={sub['ACMTA (BiLSTM)'][g]['n_stays']:,})" for g in G]
ax.set_yticks(y0); ax.set_yticklabels(lab); ax.set(xlabel='AUROC (95% CI)', xlim=(0.6, 0.85))
ax.legend(loc='upper left', frameon=False, borderaxespad=0.2)
letter(axs[0], 'a', x=-0.2); letter(axs[1], 'b', x=-0.45); fig.tight_layout(w_pad=2.0); save(fig, 'Fig4')

# ---------------------------------------------------------------- Fig 5: explainability
fig = plt.figure(figsize=(174 * MM, 74 * MM)); gs = fig.add_gridspec(1, 6, width_ratios=[1.05, 0.3, 1.1, 0.05, 0.85, 0.9], wspace=0.22)
ax = fig.add_subplot(gs[0]); top = EX['gbm_permutation_importance_top30'][:10][::-1]
pretty = lambda s: s.replace('_total', '').replace('missfrac', 'missing fraction').replace('_', ' ')
ax.barh(range(len(top)), [t[1] for t in top], xerr=[t[2] for t in top], color='#D55E00', height=0.65, error_kw={'lw': 0.6, 'capsize': 1.2})
ax.set_yticks(range(len(top))); ax.set_yticklabels([pretty(t[0]) for t in top]); ax.set_xlabel('ΔAUROC when permuted')
letter(ax, 'a', x=-0.75)
ax = fig.add_subplot(gs[2]); ig = EX['acmta_integrated_gradients_vitals']; M = np.array(ig['mean_abs_attribution'])
im = ax.imshow(M / M.max(), aspect='auto', cmap='Blues', vmin=0, vmax=1)
ax.set_yticks(range(len(ig['variables']))); ax.set_yticklabels(ig['variables']); ax.set_xticks(range(1, 12, 2)); ax.set_xticklabels([str(h) for h in ig['hour_offset'][1::2]])
ax.set_xlabel('Hour relative to prediction')
cax = fig.add_subplot(gs[3]); cb = fig.colorbar(im, cax=cax); cb.set_label('Relative attribution', fontsize=8); cb.ax.tick_params(labelsize=8)
letter(ax, 'b', x=-0.3)
ax = fig.add_subplot(gs[5]); at = EX['acmta_attention_profile_true_positive_windows']
ax.plot(at['hour_offset'], at['mean_attention_vitals_to_labs'], color='#0072B2', marker='o', ms=3, lw=1.1)
ax.set(xlabel='Lab key time (h)', ylabel='Mean attention per key', ylim=(0.3, 0.95), xticks=[-10, -5, 0])
letter(ax, 'c', x=-0.42)
save(fig, 'Fig5')

# ---------------------------------------------------------------- Fig 1: prediction set-up and architecture (schematic, 8-pt lettering)
W_, H_ = 172, 122
fig = plt.figure(figsize=(W_ * MM, H_ * MM))
plt.rcParams['mathtext.fontset'] = 'dejavusans'
ax = fig.add_axes([0, 0.63, 1, 0.37]); ax.set_xlim(-30.5, 4.6); ax.set_ylim(0, 3.5); ax.axis('off')
ax.annotate('', xy=(4.4, 0.6), xytext=(-30.3, 0.6), arrowprops=dict(arrowstyle='-|>', lw=0.8, color='black'))
for h in range(-28, 1, 4): ax.plot([h, h], [0.5, 0.7], color='black', lw=0.6); ax.text(h, 0.08, f'{h}' if h else '0', ha='center', fontsize=8)
ax.text(4.2, 0.08, 'h', fontsize=8)
ax.add_patch(Rectangle((-30, 0.75), 18, 0.6, facecolor='#F2F2F2', edgecolor='#999999', lw=0.6)); ax.text(-21, 1.05, 'negative windows (onset > 12 h ahead, or none)', ha='center', va='center', fontsize=8)
ax.add_patch(Rectangle((-12, 0.75), 8, 0.6, facecolor='#CFE3F3', edgecolor='#0072B2', lw=0.8)); ax.text(-8, 1.05, 'positive windows', ha='center', va='center', fontsize=8)
ax.add_patch(Rectangle((-4, 0.75), 4, 0.6, facecolor='white', edgecolor='#333333', hatch='////', lw=0.6))
ax.text(-2, 1.5, 'buffer\n(excluded)', ha='center', va='bottom', fontsize=8)
ax.plot([0, 0], [0.6, 3.1], color='#D55E00', lw=1.2)
ax.text(0.3, 2.8, 'onset T$_0$\n(label + 6 h)', color='#D55E00', fontsize=8, va='center', ha='left')
ax.plot([-9, -9], [0.6, 2.45], color='#0072B2', lw=0.6, ls=':')
ax.annotate('', xy=(-9, 2.25), xytext=(-21, 2.25), arrowprops=dict(arrowstyle='|-|', lw=0.8, color='#0072B2', shrinkA=0, shrinkB=0, mutation_scale=3))
ax.text(-15, 2.45, '12-h observation window', ha='center', va='bottom', fontsize=8, color='#0072B2')
ax.text(-8.7, 2.55, 'prediction time t', ha='left', va='bottom', fontsize=8, color='#0072B2')
ax.text(-30.5, 3.45, 'a', fontsize=10, fontweight='bold', va='top')
ax = fig.add_axes([0, 0, 1, 0.6]); ax.set_xlim(0, W_); ax.set_ylim(0, 0.6 * H_); ax.axis('off')
def box(x, y, w, h, txt, fc='#FFFFFF', ec='#333333', fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.5', facecolor=fc, edgecolor=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs, linespacing=1.3)
def arr(x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=7, lw=0.8, color='#333333', shrinkA=0, shrinkB=0))
Y1, Y2, BH = 46, 14, 22
box(1, Y1, 27, BH, 'Vital signs\n6 variables,\nhourly grid\n+ masks and\nelapsed time', fc='#E8F1FA')
box(1, Y2, 27, BH, 'Laboratory\n5 variables at\nmeasurement\ntimes + masks\nand elapsed time', fc='#FDF0E6')
for y in (Y1, Y2):
    box(33, y, 27, BH, 'Encoder\nBiLSTM or\nTransformer\n+ time encoding')
    box(65, y, 31, BH, 'MAMG gate\n$g=\\sigma(W[h;\\mu;\\delta]+b)$\n$\\tilde{h}=g\\odot h$', fc='#EFEFEF')
    arr(28, y + BH / 2, 33, y + BH / 2); arr(60, y + BH / 2, 65, y + BH / 2)
box(101, 25, 34, 36, 'ACMTA\nbidirectional\ncross-modal\nattention with\nlearned time-gap\nbias $B(\\Delta t)$', fc='#DCEAF7', ec='#0072B2')
arr(96, Y1 + BH / 2, 101, 54); arr(96, Y2 + BH / 2, 101, 32)
box(140, 47, 31, 15, 'Attention pooling\n+ age, sex')
box(140, 24, 31, 17, 'MLP head\nP(onset in\n4–12 h)', fc='#E5F4EC', ec='#009E73')
arr(135, 51, 140, 54); arr(155.5, 47, 155.5, 41)
ax.text(86, 5, 'Score between a query at $t_i$ and a key at $t_j$:  $e_{ij} = q_i k_j^{\\top}/\\sqrt{d} + B(t_i - t_j)$', ha='center', va='center', fontsize=8, color='#0072B2')
ax.text(0, 0.6 * H_ - 1, 'b', fontsize=10, fontweight='bold', va='top')
save(fig, 'Fig1')
print('figures written')

# ---------------------------------------------------------------- Fig 6: robustness to sparser laboratory testing (test-time thinning)
import glob, os as _os
LS = {}
for f in glob.glob(os.path.join(os.environ.get('ACMTA_RESULTS', 'results'), 'real_ch3', 'lab_sparsity', '*.json')):
    r = json.load(open(f)); LS[(r['model'], r['q'])] = r['auroc']
QS = [0.0, 0.25, 0.5, 0.75, 1.0]
def series(run, name):
    seeds = [run, run + '_s43', run + '_s44'] if run.startswith('bilstm') else [run]
    base = {run: m(name, 'auroc')}
    if run.startswith('bilstm'):
        base[run + '_s43'] = A['metrics'][f'{name}|43']['auroc']; base[run + '_s44'] = A['metrics'][f'{name}|44']['auroc']
    vals = np.array([[base[s] if q == 0 else LS[(s, q)] for q in QS] for s in seeds])
    return vals.mean(0), (vals.std(0, ddof=1) if len(seeds) > 1 else np.zeros(len(QS))), vals - vals[:, :1]
SER = [('bilstm_acmta', 'ACMTA (BiLSTM)'), ('bilstm_early', 'Early fusion'), ('bilstm_late', 'Late fusion'), ('gbm', 'Gradient boosting'), ('lr', 'Logistic regression')]
fig, axs = plt.subplots(1, 2, figsize=(174 * MM, 74 * MM))
x = np.array(QS) * 100
vit = np.mean([A['metrics'][f'Vital signs only|{s}']['auroc'] for s in (42, 43, 44)])
for run_, nm in SER:
    mu, sd, d = series(run_, nm); c, ls, lw = STY[nm]
    axs[0].plot(x, mu, color=c, ls=ls, lw=lw, marker='o', ms=3, label=nm)
    if sd.any(): axs[0].fill_between(x, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
    axs[1].plot(x, d.mean(0), color=c, ls=ls, lw=lw, marker='o', ms=3, label=nm)
axs[0].axhline(vit, color=STY['Vital signs only'][0], lw=0.8, ls=(0, (3, 1, 1, 1, 1, 1)), label='Vital signs only (reference)')
axs[0].set(xlabel='Laboratory observations removed (%)', ylabel='AUROC', xticks=x, ylim=(0.64, 0.78))
axs[1].axhline(0, color='#999999', lw=0.6)
axs[1].set(xlabel='Laboratory observations removed (%)', ylabel='Change in AUROC', xticks=x, ylim=(-0.11, 0.005))
h, l = axs[0].get_legend_handles_labels(); axs[1].legend(h, l, loc='lower left', frameon=False, handlelength=2.4, fontsize=8)
letter(axs[0], 'a'); letter(axs[1], 'b'); fig.tight_layout(w_pad=2.5); save(fig, 'Fig6')
print('Fig6 written')

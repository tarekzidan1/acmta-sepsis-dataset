"""Command-line entry point for every experiment of the reproduction package.

python -m acmta_sepsis.experiments prepare   --data real --name real_ch3
python -m acmta_sepsis.experiments baselines --name real_ch3
python -m acmta_sepsis.experiments train     --name real_ch3 --run bilstm_acmta --encoder bilstm --fusion acmta --max-minutes 9
python -m acmta_sepsis.experiments evaluate  --name real_ch3 --run bilstm_acmta
"""
from __future__ import annotations
import argparse, os, json, time, pickle
import numpy as np
from . import data as D
from .preprocessing import Preprocessed, VARIABLE_SETS
from .windows import make_windows, window_rows
from .utils import set_seed, dump, environment

ROOT = os.environ.get('ACMTA_WORK', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work'))
RESULTS = os.environ.get('ACMTA_RESULTS', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))


def cache_dir(name): return os.path.join(ROOT, name)


def prepare(a):
    t0 = time.time()
    set_seed(a.seed)
    if a.data == 'real':
        ds = D.PhysioNet2019(a.cache)
    else:
        from . import synthetic
        ds = synthetic.as_dataset(synthetic.generate(n_patients=a.n_synthetic, signal=a.signal, seed=a.synthetic_seed))
    spec = VARIABLE_SETS[a.varset]
    co = D.cohort(ds, spec['vitals'], min_icu_hours=a.min_icu_hours, onset_shift_hours=a.onset_shift)
    sp = D.split_patients(co['septic'], test_frac=0.2, n_folds=5, seed=a.seed)
    pp = Preprocessed(ds, co['index'], a.varset)
    train_stays = sp['dev'][sp['fold'][sp['dev']] != a.val_fold]
    pp.fit_scaler(train_stays).transform()
    t0s = co['t0']
    hours = np.diff(pp.offsets)
    W = {}
    for part, stays in (('dev', sp['dev']), ('test', sp['test'])):
        s, t, y = make_windows(pp.offsets, t0s, stays, stride=a.stride, label_mode=a.label_mode, min_history=a.min_history)
        W[part] = (s, t, y)
    s, t, y = W['dev']
    fold = sp['fold'][s]
    meta = {'name': a.name, 'data': a.data, 'varset': a.varset, 'stride': a.stride, 'label_mode': a.label_mode,
            'min_icu_hours': a.min_icu_hours, 'min_history': a.min_history, 'onset_shift_hours': a.onset_shift, 'seed': a.seed,
            'val_fold': a.val_fold, 'n_stays_source': int(ds.n), 'cohort': int(len(co['index'])), 'excluded': co['excluded'],
            'septic': int(co['septic'].sum()), 'septic_prevalence': float(co['septic'].mean()),
            'onset_censored': int(co['censored'].sum()), 'implausible_values_removed': int(pp.n_implausible),
            'los_median': float(np.median(hours)), 'los_iqr': [float(np.percentile(hours, 25)), float(np.percentile(hours, 75))],
            'age_mean': float(np.nanmean(pp.static_raw[:, 0])), 'age_sd': float(np.nanstd(pp.static_raw[:, 0])),
            'male_pct': float(100 * np.nanmean(pp.static_raw[:, 1])),
            'windows': {k: {'n': int(len(v[0])), 'pos': int(v[2].sum()), 'prev': float(v[2].mean()) if len(v[2]) else None,
                            'stays': int(len(np.unique(v[0])))} for k, v in W.items()},
            'windows_total': int(sum(len(v[0]) for v in W.values())), 'pos_total': int(sum(v[2].sum() for v in W.values())),
            'seconds': time.time() - t0}
    meta['window_prevalence_total'] = meta['pos_total'] / max(meta['windows_total'], 1)
    d = cache_dir(a.name); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'prep.pkl'), 'wb') as f:
        pickle.dump({'pp': pp, 'cohort': co, 'split': sp, 'windows': W, 'dev_fold': fold, 'hours': hours, 'meta': meta}, f, protocol=4)
    dump(meta, os.path.join(RESULTS, a.name, 'cohort_meta.json'))
    print(json.dumps(meta, indent=1))


def load(name):
    with open(os.path.join(cache_dir(name), 'prep.pkl'), 'rb') as f:
        return pickle.load(f)


def baselines(a):
    from .features import window_features
    from .scores import qsofa, sofa_proxy, news_proxy
    from .baselines import cv_oof, fit_predict
    from . import evaluate as E
    P = load(a.name); pp = P['pp']; W = P['windows']; t0s = P['cohort']['t0']; hours = P['hours']
    out_dir = os.path.join(RESULTS, a.name, 'baselines'); os.makedirs(out_dir, exist_ok=True)
    s_d, t_d, y_d = W['dev']; s_t, t_t, y_t = W['test']; fold = P['dev_fold']
    timings = {}
    tt = time.time(); Xd, names = window_features(pp, s_d, t_d); Xt, _ = window_features(pp, s_t, t_t); timings['features_s'] = time.time() - tt
    preds = {}
    for nm, fn in (('qsofa', qsofa), ('sofa_proxy', sofa_proxy), ('news_proxy', news_proxy)):
        rows_d = pp.offsets[s_d] + t_d; rows_t = pp.offsets[s_t] + t_t
        sd_, pd_ = fn(pp, rows_d); st_, pt_ = fn(pp, rows_t)
        preds[nm] = {'dev_score': sd_, 'dev_p': pd_, 'test_score': st_, 'test_p': pt_}
    for kind in ('lr', 'gbm'):
        if a.skip_gbm and kind == 'gbm': continue
        if a.models and kind not in a.models.split(','): continue
        tt = time.time(); oof = cv_oof(kind, Xd, y_d, fold, seed=a.seed); timings[f'{kind}_cv_s'] = time.time() - tt
        tt = time.time(); model, pt_ = fit_predict(kind, Xd, y_d, Xt, seed=a.seed); timings[f'{kind}_fit_s'] = time.time() - tt
        preds[kind] = {'dev_p': oof, 'test_p': pt_}
        with open(os.path.join(out_dir, f'{kind}_model.pkl'), 'wb') as f: pickle.dump(model, f)
    pred_path = os.path.join(out_dir, 'predictions.npz')
    old_preds = dict(np.load(pred_path, allow_pickle=True)) if os.path.exists(pred_path) else {}
    old_preds.update(dict(names=np.array(names), s_dev=s_d, t_dev=t_d, y_dev=y_d, s_test=s_t, t_test=t_t, y_test=y_t,
                          **{f'{k}_{kk}': v for k, d_ in preds.items() for kk, v in d_.items()}))
    np.savez_compressed(pred_path, **old_preds)
    dump(timings, os.path.join(out_dir, 'timings_%s.json' % (a.models or 'all').replace(',', '_')))
    if a.n_boot < 0:
        print('predictions saved', list(preds)); return
    report = {'timings': timings, 'n_features': len(names)}
    for k, d_ in preds.items():
        cv = {'auroc': E.auroc(y_d, d_['dev_p']), 'auprc': E.auprc(y_d, d_['dev_p']), 'brier': E.brier(y_d, d_['dev_p'])}
        if 'dev_score' in d_: cv['auroc_raw_score'] = E.auroc(y_d, d_['dev_score'])
        rep = E.full_report(y_t, d_['test_p'], s_t, t_t, t0s, hours, val=(y_d, d_['dev_p'], s_d, hours), n_boot=a.n_boot, seed=a.seed)
        report[k] = {'dev_cv_or_rule': cv, 'test': rep}
        print(k, 'dev', {kk: round(v, 4) for kk, v in cv.items()}, 'test AUROC %.4f AUPRC %.4f Brier %.4f' % (rep['auroc'], rep['auprc'], rep['brier']))
    dump(report, os.path.join(out_dir, 'baseline_report.json'))


def report_baselines(a):
    from . import evaluate as E
    P = load(a.name); t0s = P['cohort']['t0']; hours = P['hours']
    out_dir = os.path.join(RESULTS, a.name, 'baselines')
    z = dict(np.load(os.path.join(out_dir, 'predictions.npz'), allow_pickle=True))
    s_d, y_d, s_t, t_t, y_t = z['s_dev'], z['y_dev'], z['s_test'], z['t_test'], z['y_test']
    path = os.path.join(out_dir, 'baseline_report.json')
    report = json.load(open(path)) if os.path.exists(path) else {}
    for k in [m for m in a.models.split(',') if m]:
        pd_, pt_ = z[f'{k}_dev_p'], z[f'{k}_test_p']
        cv = {'auroc': E.auroc(y_d, pd_), 'auprc': E.auprc(y_d, pd_), 'brier': E.brier(y_d, pd_)}
        if f'{k}_dev_score' in z: cv['auroc_raw_score'] = E.auroc(y_d, z[f'{k}_dev_score'])
        rep = E.full_report(y_t, pt_, s_t, t_t, t0s, hours, val=(y_d, pd_, s_d, hours), n_boot=a.n_boot, seed=a.seed)
        report[k] = {'dev_cv_or_rule': cv, 'test': rep}
        print(k, 'dev', {kk: round(v, 4) for kk, v in cv.items()}, 'test AUROC %.4f AUPRC %.4f Brier %.4f' % (rep['auroc'], rep['auprc'], rep['brier']), 'CI', rep.get('ci95'))
        dump(report, path)


def train_cmd(a):
    from .model import ACMTAConfig
    from .train import train
    P = load(a.name); pp = P['pp']; W = P['windows']; fold = P['dev_fold']
    s, t, y = W['dev']
    tr = fold != P['meta']['val_fold']; va = ~tr
    if a.train_subsample < 1.0:   # used only for optimisation runs, never for the reproduction runs
        rng = np.random.RandomState(a.seed); keep = rng.rand(len(s)) < a.train_subsample; tr = tr & keep
    cfg = ACMTAConfig(n_vitals=len(pp.vitals), n_labs=len(pp.labs), n_static=pp.static.shape[1] + (pp.dyn_static.shape[1] if pp.dyn_static is not None else 0),
                      hidden=a.hidden, heads=a.heads, dropout=a.dropout, encoder=a.encoder, fusion=a.fusion, mamg=not a.no_mamg,
                      time_bias=not a.no_time_bias)
    out_dir = os.path.join(RESULTS, a.name, 'runs', a.run)
    res = train(cfg, pp, (s[tr], t[tr], y[tr]), (s[va], t[va], y[va]), out_dir, lr=a.lr, weight_decay=a.weight_decay,
                batch_size=a.batch_size, max_epochs=a.max_epochs, patience=a.patience, seed=a.seed, max_minutes=a.max_minutes,
                threads=a.threads)
    dump({'args': vars(a), 'result': res}, os.path.join(out_dir, 'train_call_last.json'))
    print(json.dumps(res))


def evaluate_cmd(a):
    from .train import load_best, predict
    from . import evaluate as E
    import torch
    P = load(a.name); pp = P['pp']; W = P['windows']; fold = P['dev_fold']; t0s = P['cohort']['t0']; hours = P['hours']
    out_dir = os.path.join(RESULTS, a.name, 'runs', a.run)
    model, cfg, ck = load_best(out_dir)
    if a.threads: torch.set_num_threads(a.threads)
    s, t, y = W['dev']; va = fold == P['meta']['val_fold']
    tt = time.time(); p_val = predict(model, pp, s[va], t[va], cfg.fusion); t_val = time.time() - tt
    s_t, t_t, y_t = W['test']
    tt = time.time(); p_test = predict(model, pp, s_t, t_t, cfg.fusion); t_test = time.time() - tt
    np.savez_compressed(os.path.join(out_dir, 'predictions.npz'), s_val=s[va], t_val=t[va], y_val=y[va], p_val=p_val, s_test=s_t, t_test=t_t, y_test=y_t, p_test=p_test)
    rep = E.full_report(y_t, p_test, s_t, t_t, t0s, hours, val=(y[va], p_val, s[va], hours), n_boot=a.n_boot, seed=a.seed)
    rep['val'] = {'auroc': E.auroc(y[va], p_val), 'auprc': E.auprc(y[va], p_val), 'brier': E.brier(y[va], p_val)}
    rep['inference_seconds'] = {'val': t_val, 'test': t_test, 'per_window_ms': 1000 * t_test / max(len(y_t), 1)}
    rep['checkpoint'] = {'epoch': ck['epoch'], 'val_loss': ck['val_loss'], 'config': cfg.to_dict()}
    rep['params'] = int(sum(p.numel() for p in model.parameters()))
    dump(rep, os.path.join(out_dir, 'test_report.json'))
    print(a.run, 'test AUROC %.4f AUPRC %.4f Brier %.4f | val AUROC %.4f' % (rep['auroc'], rep['auprc'], rep['brier'], rep['val']['auroc']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('--name', required=True)
    ap.add_argument('--data', default='real'); ap.add_argument('--cache', default='data/physionet2019.npz')  # cache built by data.build_cache (docs/DATASET.md)
    ap.add_argument('--varset', default='ch3'); ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--label-mode', default='ch3'); ap.add_argument('--min-icu-hours', type=int, default=0)
    ap.add_argument('--min-history', type=int, default=1); ap.add_argument('--onset-shift', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--val-fold', type=int, default=0)
    ap.add_argument('--n-synthetic', type=int, default=2000); ap.add_argument('--signal', type=float, default=0.35)
    ap.add_argument('--synthetic-seed', type=int, default=7)
    ap.add_argument('--n-boot', type=int, default=2000); ap.add_argument('--skip-gbm', action='store_true'); ap.add_argument('--models', default='')
    ap.add_argument('--run', default='bilstm_acmta'); ap.add_argument('--encoder', default='bilstm'); ap.add_argument('--fusion', default='acmta')
    ap.add_argument('--hidden', type=int, default=128); ap.add_argument('--heads', type=int, default=4); ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--batch-size', type=int, default=256); ap.add_argument('--max-epochs', type=int, default=20); ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--max-minutes', type=float, default=None); ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--no-mamg', action='store_true'); ap.add_argument('--no-time-bias', action='store_true')
    ap.add_argument('--train-subsample', type=float, default=1.0)
    a = ap.parse_args()
    {'prepare': prepare, 'baselines': baselines, 'report-baselines': report_baselines, 'train': train_cmd, 'evaluate': evaluate_cmd}[a.cmd](a)


if __name__ == '__main__':
    main()

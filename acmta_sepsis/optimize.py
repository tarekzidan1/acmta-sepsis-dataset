"""Controlled hyper-parameter optimisation with Optuna (Akiba et al., 2019) over the thesis search space (Table 3.3),
using ONLY the training and validation partitions of the development set. The held-out test set is never used here.

Budget-limited protocol (PROVISIONAL, documented in docs/OPTIMISATION.md): each trial runs a fixed budget of 700 AdamW
steps on batches sampled from a fixed random 50 % of the training-partition patients and is scored by the AUROC on a
fixed random 50 % of the validation-fold patients (selection criterion). Trials persist in an SQLite study so the search can be resumed inside time-limited jobs.
"""
from __future__ import annotations
import os, json, time, argparse
import numpy as np
import optuna
import torch
from .experiments import load, RESULTS, ROOT
from .model import ACMTAConfig
from .train import train, load_best, predict
from . import evaluate as E
from .utils import dump

SPACE = {'hidden': (64, 256), 'heads': [2, 4, 8], 'dropout': (0.1, 0.5), 'lr': (1e-4, 1e-2), 'weight_decay': (1e-6, 1e-3), 'batch_size': [64, 128, 256]}


def objective_factory(P, a, out_root):
    from .model import ACMTASepsis, weighted_bce
    from .batching import build_batch
    from .utils import set_seed
    pp = P['pp']; s, t, y = P['windows']['dev']; fold = P['dev_fold']; vf = P['meta']['val_fold']
    rng = np.random.RandomState(123)
    tr_pat = np.unique(s[fold != vf]); va_pat = np.unique(s[fold == vf])
    tr_keep = set(rng.choice(tr_pat, int(a.train_frac * len(tr_pat)), replace=False))
    va_keep = set(rng.choice(va_pat, int(a.val_frac * len(va_pat)), replace=False))
    trm = np.flatnonzero(np.array([x in tr_keep for x in s])); vam = np.array([x in va_keep for x in s])
    pos_weight = (len(trm) - y[trm].sum()) / max(y[trm].sum(), 1)
    def objective(trial):
        hidden = trial.suggest_int('hidden', 64, 256, step=32)
        heads = trial.suggest_categorical('heads', SPACE['heads'])
        if hidden % heads: raise optuna.TrialPruned()
        cfg = ACMTAConfig(n_vitals=len(pp.vitals), n_labs=len(pp.labs), n_static=pp.static.shape[1], hidden=hidden, heads=heads,
                          dropout=trial.suggest_float('dropout', 0.1, 0.5), encoder=a.encoder, fusion='acmta')
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True); wd = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        bs = trial.suggest_categorical('batch_size', SPACE['batch_size'])
        t0 = time.time(); set_seed(42)
        m = ACMTASepsis(cfg); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
        r = np.random.RandomState(trial.number)
        m.train()
        for step in range(a.steps):      # fixed optimiser-step budget per trial (comparable cost across configurations)
            idx = trm[r.randint(0, len(trm), bs)]
            b = build_batch(pp, s[idx], t[idx], y[idx], 'acmta')
            loss = weighted_bce(m(b), b['y'], pos_weight); opt.zero_grad(); loss.backward(); opt.step()
        p = predict(m, pp, s[vam], t[vam], 'acmta')
        auc = E.auroc(y[vam], p)
        trial.set_user_attr('val_auprc', E.auprc(y[vam], p)); trial.set_user_attr('seconds', time.time() - t0)
        return auc
    return objective


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--name', default='real_ch3'); ap.add_argument('--encoder', default='bilstm')
    ap.add_argument('--trials', type=int, default=12); ap.add_argument('--budget-min', type=float, default=2.2)
    ap.add_argument('--train-frac', type=float, default=0.5); ap.add_argument('--val-frac', type=float, default=0.5); ap.add_argument('--steps', type=int, default=700); ap.add_argument('--enqueue-default', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(2)
    out_root = os.path.join(RESULTS, a.name, 'optuna_%s' % a.encoder); os.makedirs(out_root, exist_ok=True)
    storage = 'sqlite:///' + os.path.join(ROOT, a.name, 'optuna_%s.db' % a.encoder)
    study = optuna.create_study(study_name='acmta_%s' % a.encoder, storage=storage, direction='maximize', load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5))
    done = [t for t in study.trials if t.state.name in ('COMPLETE', 'PRUNED')]
    for t in study.trials:   # trials interrupted by a time limit are marked failed and re-run
        if t.state.name == 'RUNNING': study.tell(t.number, state=optuna.trial.TrialState.FAIL)
    if a.enqueue_default and not any(t.user_attrs.get('reference_default') for t in study.trials):
        # reference point: the PROVISIONAL defaults of the reproduction runs, scored under the same budgeted protocol
        study.enqueue_trial({'hidden': 64, 'heads': 4, 'dropout': 0.2, 'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 256}, user_attrs={'reference_default': True})
    P = load(a.name)
    obj = objective_factory(P, a, out_root)
    t0 = time.time()
    while len([t for t in study.trials if t.state.name in ('COMPLETE', 'PRUNED')]) < a.trials and (time.time() - t0) < a.budget_min * 60 * 0.5:
        study.optimize(obj, n_trials=1)
    trials = [{'number': t.number, 'state': t.state.name, 'value': t.value, 'params': t.params, **t.user_attrs} for t in study.trials]
    best = study.best_trial if any(t.state.name == 'COMPLETE' for t in study.trials) else None
    dump({'space': {k: (list(v) if isinstance(v, tuple) else v) for k, v in SPACE.items()}, 'protocol': vars(a), 'trials': trials,
          'best': {'number': best.number, 'value': best.value, 'params': best.params} if best else None}, os.path.join(out_root, 'study.json'))
    n_done = len([t for t in study.trials if t.state.name in ('COMPLETE', 'PRUNED')])
    print('trials done', n_done, '/', a.trials, 'best', best.value if best else None, best.params if best else None)
    if n_done >= a.trials: print('SEARCH COMPLETE')


if __name__ == '__main__':
    main()

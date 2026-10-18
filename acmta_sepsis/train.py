"""Training with class-weighted BCE (Eq. 3.10), AdamW (§3.6.4), early stopping on validation loss, resumable
checkpoints (to run inside time-limited jobs) and deterministic seeding."""
from __future__ import annotations
import os, json, time, math
import numpy as np
import torch
from .model import ACMTASepsis, ACMTAConfig, weighted_bce
from .batching import build_batch
from .utils import set_seed, dump


def predict(model, pp, stay, t, fusion, batch_size=2048, return_attention=False):
    model.eval(); out = []; att = []
    with torch.no_grad():
        for a in range(0, len(stay), batch_size):
            b = build_batch(pp, stay[a:a + batch_size], t[a:a + batch_size], None, fusion)
            if return_attention:
                logit, aux = model(b, return_attention=True); att.append(aux)
            else:
                logit = model(b)
            out.append(torch.sigmoid(logit).numpy())
    p = np.concatenate(out) if out else np.zeros(0)
    return (p, att) if return_attention else p


def eval_loss(model, pp, stay, t, y, fusion, pos_weight, batch_size=2048):
    model.eval(); tot = 0.0; n = 0
    with torch.no_grad():
        for a in range(0, len(stay), batch_size):
            b = build_batch(pp, stay[a:a + batch_size], t[a:a + batch_size], y[a:a + batch_size], fusion)
            l = weighted_bce(model(b), b['y'], pos_weight).item()
            tot += l * len(b['y']); n += len(b['y'])
    return tot / max(n, 1)


def train(cfg: ACMTAConfig, pp, train_w, val_w, out_dir: str, lr=1e-3, weight_decay=1e-4, batch_size=256,
          max_epochs=20, patience=3, seed=42, max_minutes=None, log_every=200, val_batch=4096, threads=None):
    """train_w/val_w: tuples (stay, t, y). Resumes automatically from out_dir/ckpt_last.pt.
    Returns a dict with status 'done' or 'paused' (time budget reached)."""
    os.makedirs(out_dir, exist_ok=True)
    if threads: torch.set_num_threads(threads)
    t_start = time.time()
    s_tr, t_tr, y_tr = train_w; s_va, t_va, y_va = val_w
    n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
    pos_weight = n_neg / max(n_pos, 1)            # §3.4: weight inversely proportional to class frequency
    set_seed(seed)
    model = ACMTASepsis(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    state = {'epoch': 0, 'step_in_epoch': 0, 'best_val': math.inf, 'bad_epochs': 0, 'history': [], 'train_seconds': 0.0,
             'pos_weight': pos_weight, 'config': cfg.to_dict(), 'lr': lr, 'weight_decay': weight_decay,
             'batch_size': batch_size, 'seed': seed, 'max_epochs': max_epochs, 'patience': patience}
    last = os.path.join(out_dir, 'ckpt_last.pt'); best = os.path.join(out_dir, 'ckpt_best.pt')
    if os.path.exists(last):
        ck = torch.load(last, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); state = ck['state']
        torch.set_rng_state(ck['torch_rng']); np.random.set_state(ck['np_rng'])
    log = open(os.path.join(out_dir, 'train_log.txt'), 'a')
    def save_last():
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'state': state,
                    'torch_rng': torch.get_rng_state(), 'np_rng': np.random.get_state()}, last)
    while state['epoch'] < max_epochs and state['bad_epochs'] < patience:
        ep = state['epoch']
        perm = np.random.RandomState(seed + ep).permutation(len(s_tr))
        nb = int(math.ceil(len(perm) / batch_size))
        model.train(); t_ep = time.time(); run_loss = 0.0; k = 0
        for bi in range(state['step_in_epoch'], nb):
            idx = perm[bi * batch_size:(bi + 1) * batch_size]
            b = build_batch(pp, s_tr[idx], t_tr[idx], y_tr[idx], cfg.fusion)
            loss = weighted_bce(model(b), b['y'], pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item(); k += 1
            state['step_in_epoch'] = bi + 1
            if (bi + 1) % log_every == 0:
                log.write(f'epoch {ep} step {bi+1}/{nb} train_loss {run_loss/k:.4f} elapsed_epoch {time.time()-t_ep:.0f}s\n'); log.flush()
            if max_minutes and (time.time() - t_start) > max_minutes * 60:
                state['train_seconds'] += time.time() - t_ep
                save_last(); log.write(f'PAUSED at epoch {ep} step {bi+1}/{nb}\n'); log.close()
                return {'status': 'paused', 'epoch': ep, 'step': bi + 1, 'of': nb}
        state['train_seconds'] += time.time() - t_ep
        t_v = time.time()
        vl = eval_loss(model, pp, s_va, t_va, y_va, cfg.fusion, pos_weight, batch_size=val_batch)
        state['history'].append({'epoch': ep, 'train_loss': run_loss / max(k, 1), 'val_loss': vl, 'val_seconds': time.time() - t_v})
        if vl < state['best_val'] - 1e-5:
            state['best_val'] = vl; state['bad_epochs'] = 0; state['best_epoch'] = ep
            torch.save({'model': model.state_dict(), 'config': cfg.to_dict(), 'epoch': ep, 'val_loss': vl}, best)
        else:
            state['bad_epochs'] += 1
        log.write(f'EPOCH {ep} done: train_loss {run_loss/max(k,1):.4f} val_loss {vl:.4f} best {state["best_val"]:.4f} bad {state["bad_epochs"]}\n'); log.flush()
        state['epoch'] = ep + 1; state['step_in_epoch'] = 0
        save_last()
        if max_minutes and (time.time() - t_start) > max_minutes * 60 and state['epoch'] < max_epochs and state['bad_epochs'] < patience:
            log.close(); return {'status': 'paused', 'epoch': state['epoch'], 'step': 0}
    log.close()
    dump(state, os.path.join(out_dir, 'train_state.json'))
    return {'status': 'done', 'best_epoch': state.get('best_epoch'), 'best_val': state['best_val'], 'epochs': state['epoch']}


def load_best(out_dir: str):
    ck = torch.load(os.path.join(out_dir, 'ckpt_best.pt'), weights_only=False)
    cfg = ACMTAConfig(**ck['config']); m = ACMTASepsis(cfg); m.load_state_dict(ck['model']); m.eval()
    return m, cfg, ck

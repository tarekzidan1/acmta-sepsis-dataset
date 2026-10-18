#!/usr/bin/env python
"""ENGINEERING measurements (method-preserving): real-time single-patient inference latency (thesis §4.10.1,
Figure 4.13: 84 ms per hour on CPU), bulk inference throughput, effect of thread count, and dynamic int8
quantisation of linear layers (an engineering change whose numerical effect on AUROC is measured)."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from acmta_sepsis.experiments import load, RESULTS
from acmta_sepsis.train import load_best, predict
from acmta_sepsis.batching import build_batch
from acmta_sepsis import evaluate as E
from acmta_sepsis.utils import dump, environment

NAME = 'real_ch3'; P = load(NAME); pp = P['pp']; W = P['windows']
s_t, t_t, y_t = W['test']
out = {'environment': environment()}
for run in ('bilstm_acmta', 'transformer_acmta'):
    model, cfg, ck = load_best(os.path.join(RESULTS, NAME, 'runs', run))
    res = {'params': int(sum(p.numel() for p in model.parameters()))}
    # real-time replay: one septic test patient, one forward pass per hour (batch of 1), as in Figure 4.13
    stay = int(np.unique(s_t[y_t == 1])[0]); hrs = np.flatnonzero(s_t == stay)
    for threads in (1, 2):
        torch.set_num_threads(threads)
        lat = []
        with torch.inference_mode():
            for k in hrs:
                t0 = time.perf_counter(); b = build_batch(pp, s_t[k:k + 1], t_t[k:k + 1], None, cfg.fusion); model(b); lat.append(1000 * (time.perf_counter() - t0))
        res[f'realtime_ms_per_hour_threads{threads}'] = {'median': float(np.median(lat[1:])), 'p95': float(np.percentile(lat[1:], 95)), 'n_hours': len(lat)}
    torch.set_num_threads(2)
    # bulk throughput on 20,000 test windows for several batch sizes
    idx = np.arange(20000)
    for bs in (256, 1024, 4096):
        t0 = time.perf_counter(); p = predict(model, pp, s_t[idx], t_t[idx], cfg.fusion, batch_size=bs); dt = time.perf_counter() - t0
        res[f'bulk_ms_per_window_bs{bs}'] = 1000 * dt / len(idx)
    # dynamic int8 quantisation of nn.Linear / nn.LSTM (CPU)
    try:
        q = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8)
        idx = np.arange(60000)
        p32 = predict(model, pp, s_t[idx], t_t[idx], cfg.fusion, batch_size=4096)
        t0 = time.perf_counter(); p8 = predict(q, pp, s_t[idx], t_t[idx], cfg.fusion, batch_size=4096); dt8 = time.perf_counter() - t0
        t0 = time.perf_counter(); predict(model, pp, s_t[idx], t_t[idx], cfg.fusion, batch_size=4096); dt32 = time.perf_counter() - t0
        res['int8_dynamic_quantisation'] = {'ms_per_window_fp32': 1000 * dt32 / len(idx), 'ms_per_window_int8': 1000 * dt8 / len(idx),
                                            'auroc_fp32_subset': E.auroc(y_t[idx], p32), 'auroc_int8_subset': E.auroc(y_t[idx], p8),
                                            'max_abs_prob_diff': float(np.abs(p32 - p8).max())}
    except Exception as ex:
        res['int8_dynamic_quantisation'] = {'status': 'not supported for this architecture in PyTorch 2.3 (%s)' % type(ex).__name__}
    out[run] = res
    print(run, json.dumps(res, indent=0)[:900])
dump(out, os.path.join(RESULTS, NAME, 'engineering_latency.json'))

#!/usr/bin/env python
"""Advance a queue of training/evaluation jobs within a wall-clock budget (for time-limited job slots).
Re-run until it prints QUEUE COMPLETE. Usage: python scripts/run_queue.py <queue.json> [budget_minutes]"""
import json, os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acmta_sepsis import experiments as X

def args_for(job, name):
    a = argparse.Namespace(name=name, run=job['run'], encoder=job.get('encoder', 'bilstm'), fusion=job.get('fusion', 'acmta'),
                           hidden=job.get('hidden', 64), heads=job.get('heads', 4), dropout=job.get('dropout', 0.2),
                           lr=job.get('lr', 1e-3), weight_decay=job.get('weight_decay', 1e-4), batch_size=job.get('batch_size', 256),
                           max_epochs=job.get('max_epochs', 10), patience=job.get('patience', 3), seed=job.get('seed', 42),
                           threads=job.get('threads', 2), no_mamg=job.get('no_mamg', False), no_time_bias=job.get('no_time_bias', False),
                           train_subsample=job.get('train_subsample', 1.0), n_boot=job.get('n_boot', 2000), max_minutes=None)
    return a

def main():
    q = json.load(open(sys.argv[1])); budget = float(sys.argv[2]) if len(sys.argv) > 2 else 2.2
    t0 = time.time(); name = q['name']
    for job in q['jobs']:
        d = os.path.join(X.RESULTS, name, 'runs', job['run'])
        if not os.path.exists(os.path.join(d, 'train_state.json')):
            a = args_for(job, name); a.max_minutes = max(0.2, budget - (time.time() - t0) / 60)
            X.train_cmd(a); return
        if not os.path.exists(os.path.join(d, 'test_report.json')):
            X.evaluate_cmd(args_for(job, name)); return
    print('QUEUE COMPLETE')

if __name__ == '__main__':
    main()

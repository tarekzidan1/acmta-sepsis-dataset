from __future__ import annotations
import os, json, random, time, platform, subprocess, sys
import numpy as np


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def environment() -> dict:
    env = {'python': sys.version.split()[0], 'platform': platform.platform(), 'processor': platform.processor(),
           'cpu_count': os.cpu_count()}
    for mod in ('numpy', 'pandas', 'torch', 'sklearn', 'xgboost', 'scipy', 'optuna', 'shap', 'matplotlib'):
        try:
            m = __import__(mod); env[mod] = getattr(m, '__version__', '?')
        except Exception:
            env[mod] = None
    try:
        import torch
        env['torch_threads'] = torch.get_num_threads(); env['cuda'] = torch.cuda.is_available()
    except Exception:
        pass
    try:
        env['cpu_model'] = [l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')][0]
        env['mem_total_gb'] = round(int([l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemTotal')][0]) / 1024 ** 2, 2)
    except Exception:
        pass
    return env


class Timer:
    def __init__(self): self.t0 = time.time()
    def elapsed(self): return time.time() - self.t0


def dump(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

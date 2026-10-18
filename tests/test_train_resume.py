import os, numpy as np, torch
from acmta_sepsis.model import ACMTAConfig
from acmta_sepsis.train import train, load_best, predict
from acmta_sepsis.windows import make_windows


def test_train_checkpoint_resume_and_predict(syn_prep, tmp_path):
    pp, co, sp = syn_prep
    s, t, y = make_windows(pp.offsets, co['t0'], sp['dev'])
    tr = np.arange(len(s)) % 5 != 0; va = ~tr
    cfg = ACMTAConfig(n_vitals=6, n_labs=5, n_static=2, hidden=16, heads=2)
    d1 = str(tmp_path / 'a'); d2 = str(tmp_path / 'b')
    r1 = train(cfg, pp, (s[tr], t[tr], y[tr]), (s[va], t[va], y[va]), d1, max_epochs=2, patience=5, batch_size=128, seed=0)
    assert r1['status'] == 'done'
    # interrupted run: pause after a tiny time budget, then resume -> same final weights as uninterrupted
    r2 = train(cfg, pp, (s[tr], t[tr], y[tr]), (s[va], t[va], y[va]), d2, max_epochs=2, patience=5, batch_size=128, seed=0, max_minutes=1e-9)
    assert r2['status'] == 'paused'
    while r2['status'] == 'paused':
        r2 = train(cfg, pp, (s[tr], t[tr], y[tr]), (s[va], t[va], y[va]), d2, max_epochs=2, patience=5, batch_size=128, seed=0, max_minutes=0.02)
    a = torch.load(os.path.join(d1, 'ckpt_last.pt'), weights_only=False)['model']; b = torch.load(os.path.join(d2, 'ckpt_last.pt'), weights_only=False)['model']
    assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)
    m, c, ck = load_best(d1)
    p = predict(m, pp, s[va], t[va], c.fusion)
    assert p.shape == (va.sum(),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()

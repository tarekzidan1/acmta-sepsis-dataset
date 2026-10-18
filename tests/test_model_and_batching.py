import numpy as np, torch, pytest
from acmta_sepsis.model import ACMTAConfig, ACMTASepsis, TimeGapBias, MAMGGate, CrossModalAttention
from acmta_sepsis.batching import build_batch
from acmta_sepsis.windows import make_windows

FUSIONS = ['acmta', 'early', 'late', 'vitals_only', 'labs_only']


def _windows(pp, co, sp, n=64):
    s, t, y = make_windows(pp.offsets, co['t0'], sp['dev'])
    return s[:n], t[:n], y[:n]


def test_lab_stream_contains_only_observed_hours(syn_prep):
    pp, co, sp = syn_prep
    s, t, y = _windows(pp, co, sp, 200)
    b = build_batch(pp, s, t, y, 'acmta', as_torch=False)
    for i in range(len(s)):
        valid = ~b['pl'][i]
        assert valid.sum() >= 1
        rows = pp.offsets[s[i]] + t[i] + b['tl'][i][valid].astype(int)
        obs = pp.mask[rows][:, pp.lab_cols].any(1)
        if b['ml'][i][valid].sum() > 0:
            assert obs.all()               # every non-state lab token is an hour with a laboratory observation
        assert (b['tl'][i][valid] <= 0).all() and (b['tl'][i][valid] >= -11).all()


@pytest.mark.parametrize('fusion', FUSIONS)
@pytest.mark.parametrize('encoder', ['bilstm', 'transformer'])
def test_forward_shapes_no_nan(syn_prep, fusion, encoder):
    pp, co, sp = syn_prep
    s, t, y = _windows(pp, co, sp)
    torch.manual_seed(0)
    m = ACMTASepsis(ACMTAConfig(n_vitals=6, n_labs=5, n_static=2, hidden=32, heads=4, encoder=encoder, fusion=fusion))
    b = build_batch(pp, s, t, y, fusion)
    out = m(b)
    assert out.shape == (len(s),) and torch.isfinite(out).all()


def test_attention_masks_and_normalisation(syn_prep):
    pp, co, sp = syn_prep
    s, t, y = _windows(pp, co, sp)
    m = ACMTASepsis(ACMTAConfig(n_vitals=6, n_labs=5, n_static=2, hidden=32, heads=4)).eval()
    b = build_batch(pp, s, t, y, 'acmta')
    _, aux = m(b, return_attention=True)
    a = aux['att_v2l']                                    # (B, H, Tq, Tk)
    assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)
    pad = b['pl'][:, None, None, :].expand_as(a)
    assert (a[pad] == 0).all()                            # padded laboratory keys receive no attention


def test_time_gap_bias_depends_on_interval_and_mamg_range():
    torch.manual_seed(0)
    tb = TimeGapBias(heads=2)
    dt = torch.tensor([[[0.0, -3.0, 6.0]]])
    out = tb(dt)
    assert out.shape == (1, 2, 1, 3) and not torch.allclose(out[..., 0], out[..., 1])
    g = MAMGGate(8, 3)
    h, gate = g(torch.randn(2, 5, 8), torch.ones(2, 5, 3), torch.zeros(2, 5, 3))
    assert (gate > 0).all() and (gate < 1).all()


def test_no_look_ahead(syn_prep):
    """Changing measurements recorded after the prediction hour must not change the prediction."""
    pp, co, sp = syn_prep
    s, t, y = make_windows(pp.offsets, co['t0'], sp['dev'])
    keep = t + 8 < np.diff(pp.offsets)[s]
    s, t = s[keep], t[keep]
    _, first = np.unique(s, return_index=True)          # one window per stay so perturbations cannot leak across windows
    sel = first[:40]
    s, t = s[sel], t[sel] + 5
    m = ACMTASepsis(ACMTAConfig(n_vitals=6, n_labs=5, n_static=2, hidden=32, heads=4)).eval()
    with torch.no_grad():
        p1 = m(build_batch(pp, s, t, None, 'acmta'))
        backup = pp.x.copy()
        for si, ti in zip(s, t):
            a = pp.offsets[si] + ti + 1; b = pp.offsets[si + 1]
            pp.x[a:b] += 5.0
        p2 = m(build_batch(pp, s, t, None, 'acmta'))
        pp.x[:] = backup
    assert torch.allclose(p1, p2, atol=1e-6)

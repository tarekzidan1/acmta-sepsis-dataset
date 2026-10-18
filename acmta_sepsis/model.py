"""ACMTA-Sepsis architecture (thesis §3.6, Eqs. 3.1-3.10; Figure 3.1; implementation outline §4.10-4.10.1).

Class names follow thesis §4.10.1 (ACMTAConfig, MAMGGate, VariableEmbedding, BiLSTMEncoder, TransformerEncoder,
TimeGapBias, ACMTABlock, AttentionPool, ACMTASepsis). Unspecified details are marked PROVISIONAL.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ACMTAConfig:
    n_vitals: int = 6
    n_labs: int = 5
    n_static: int = 2
    hidden: int = 128           # Table 3.3 search range 64-256 (PROVISIONAL default: 128)
    heads: int = 4              # Table 3.3 search range 2-8 (PROVISIONAL default: 4)
    dropout: float = 0.2        # Table 3.3 search range 0.1-0.5 (PROVISIONAL default: 0.2)
    encoder: str = 'bilstm'     # 'bilstm' | 'transformer'  (RQ2, §3.6.1)
    fusion: str = 'acmta'       # 'acmta' | 'early' | 'late' | 'vitals_only' | 'labs_only'  (RQ3, Table 3.4)
    mamg: bool = True           # §3.6.2 (ablation: False)
    time_bias: bool = True      # §3.6.3 learned time-gap bias B(dt) (ablation: False)
    n_freq: int = 16            # PROVISIONAL: size of the sinusoidal expansion phi(dt)
    transformer_layers: int = 1 # PROVISIONAL
    def to_dict(self): return asdict(self)


class TimeEncoding(nn.Module):
    """Continuous-time sinusoidal encoding of measurement times in hours (PROVISIONAL realisation of the
    "positional encoding adapted to represent non-uniform measurement intervals", §3.6.1)."""
    def __init__(self, d: int, max_period: float = 96.0):
        super().__init__()
        k = torch.arange(d // 2, dtype=torch.float32)
        self.register_buffer('freq', 2 * math.pi / (max_period ** (k / max(1, d // 2 - 1))))
        self.d = d
    def forward(self, t):                          # t: (B, T) hours
        a = t.unsqueeze(-1) * self.freq
        e = torch.cat([torch.sin(a), torch.cos(a)], dim=-1)
        if e.shape[-1] < self.d: e = F.pad(e, (0, self.d - e.shape[-1]))
        return e


class VariableEmbedding(nn.Module):
    """Embeds value, missingness mask and elapsed-time channels of each step (§3.4, §4.10)."""
    def __init__(self, n_vars: int, hidden: int):
        super().__init__()
        self.proj = nn.Linear(3 * n_vars, hidden)
        self.time = TimeEncoding(hidden)
    def forward(self, x, m, d, t):
        return self.proj(torch.cat([x, m, d], dim=-1)) + self.time(t)


class BiLSTMEncoder(nn.Module):
    """Eq. 3.2: h_t = [forward h_t ; backward h_t]."""
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
    def forward(self, e, pad):                     # pad: (B, T) True = padding (valid steps are left-aligned)
        lengths = (~pad).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=e.shape[1])
        return self.drop(out)


class TransformerEncoder(nn.Module):
    """Eq. 3.3: scaled dot-product self-attention over all time steps."""
    def __init__(self, hidden: int, heads: int, dropout: float, layers: int = 1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=2 * hidden, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, e, pad):
        return self.drop(self.enc(e, src_key_padding_mask=pad))


class MAMGGate(nn.Module):
    """Eqs. 3.4-3.5: g_t = sigma(W_g [h_t ; mu_t ; delta_t] + b_g);  h~_t = g_t (Hadamard) h_t."""
    def __init__(self, hidden: int, n_vars: int):
        super().__init__()
        self.lin = nn.Linear(hidden + 2 * n_vars, hidden)
    def forward(self, h, mu, delta):
        g = torch.sigmoid(self.lin(torch.cat([h, mu, delta], dim=-1)))
        return g * h, g


class TimeGapBias(nn.Module):
    """Eq. 3.7: B(dt) = W2 ReLU(W1 phi(dt) + b1) + b2, phi = sinusoidal expansion of the signed interval (one bias per head)."""
    def __init__(self, heads: int, n_freq: int = 16, hidden: int = 32):
        super().__init__()
        k = torch.arange(n_freq, dtype=torch.float32)
        self.register_buffer('freq', 2 * math.pi / (48.0 ** (k / max(1, n_freq - 1))))
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq, hidden), nn.ReLU(), nn.Linear(hidden, heads))
    def forward(self, dt):                        # dt: (B, Tq, Tk) hours (t_i - t_j)
        a = dt.unsqueeze(-1) * self.freq
        phi = torch.cat([torch.sin(a), torch.cos(a)], dim=-1)
        return self.mlp(phi).permute(0, 3, 1, 2)  # (B, H, Tq, Tk)


class CrossModalAttention(nn.Module):
    """Eqs. 3.6 and 3.8 for one direction (queries from one modality, keys/values from the other)."""
    def __init__(self, hidden: int, heads: int, dropout: float, time_bias: bool, n_freq: int):
        super().__init__()
        assert hidden % heads == 0
        self.h, self.d = heads, hidden // heads
        self.wq = nn.Linear(hidden, hidden); self.wk = nn.Linear(hidden, hidden); self.wv = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, hidden)
        self.bias = TimeGapBias(heads, n_freq) if time_bias else None
        self.drop = nn.Dropout(dropout); self.norm = nn.LayerNorm(hidden)
    def forward(self, q_seq, q_t, k_seq, k_t, k_pad):
        B, Tq, _ = q_seq.shape; Tk = k_seq.shape[1]
        Q = self.wq(q_seq).view(B, Tq, self.h, self.d).transpose(1, 2)
        K = self.wk(k_seq).view(B, Tk, self.h, self.d).transpose(1, 2)
        V = self.wv(k_seq).view(B, Tk, self.h, self.d).transpose(1, 2)
        e = Q @ K.transpose(-1, -2) / math.sqrt(self.d)               # (B, H, Tq, Tk)
        if self.bias is not None:
            e = e + self.bias(q_t.unsqueeze(2) - k_t.unsqueeze(1))       # dt_ij = t_i - t_j
        e = e.masked_fill(k_pad[:, None, None, :], float('-inf'))
        a = torch.softmax(e, dim=-1)
        z = (self.drop(a) @ V).transpose(1, 2).reshape(B, Tq, self.h * self.d)
        return self.norm(q_seq + self.drop(self.out(z))), a           # PROVISIONAL: residual + LayerNorm


class ACMTABlock(nn.Module):
    """Asynchronous Cross-Modal Temporal Attention, computed in both directions (§3.6.3)."""
    def __init__(self, cfg: ACMTAConfig):
        super().__init__()
        self.v2l = CrossModalAttention(cfg.hidden, cfg.heads, cfg.dropout, cfg.time_bias, cfg.n_freq)
        self.l2v = CrossModalAttention(cfg.hidden, cfg.heads, cfg.dropout, cfg.time_bias, cfg.n_freq)
    def forward(self, hv, tv, pv, hl, tl, pl):
        zv, av = self.v2l(hv, tv, hl, tl, pl)     # vital-sign queries attend to laboratory keys
        zl, al = self.l2v(hl, tl, hv, tv, pv)     # laboratory queries attend to vital-sign keys
        return zv, zl, av, al


class AttentionPool(nn.Module):
    """Attention pooling over the observation window (§3.6.4 "pooled over the observation window"; §4.10)."""
    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
    def forward(self, h, pad):
        s = self.score(h).squeeze(-1).masked_fill(pad, float('-inf'))
        a = torch.softmax(s, dim=-1)
        return (a.unsqueeze(-1) * h).sum(1), a


def make_encoder(cfg):
    return BiLSTMEncoder(cfg.hidden, cfg.dropout) if cfg.encoder == 'bilstm' else \
        TransformerEncoder(cfg.hidden, cfg.heads, cfg.dropout, cfg.transformer_layers)


class Head(nn.Module):
    """Prediction head: multilayer perceptron with a logistic output (Eq. 3.9; sigmoid applied in the loss)."""
    def __init__(self, d_in: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, z): return self.net(z).squeeze(-1)


class ACMTASepsis(nn.Module):
    def __init__(self, cfg: ACMTAConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        if cfg.fusion in ('acmta', 'late', 'vitals_only'):
            self.emb_v = VariableEmbedding(cfg.n_vitals, h); self.enc_v = make_encoder(cfg)
            self.gate_v = MAMGGate(h, cfg.n_vitals) if cfg.mamg else None
            self.pool_v = AttentionPool(h)
        if cfg.fusion in ('acmta', 'late', 'labs_only'):
            self.emb_l = VariableEmbedding(cfg.n_labs, h); self.enc_l = make_encoder(cfg)
            self.gate_l = MAMGGate(h, cfg.n_labs) if cfg.mamg else None
            self.pool_l = AttentionPool(h)
        if cfg.fusion == 'early':
            nv = cfg.n_vitals + cfg.n_labs
            self.emb_e = VariableEmbedding(nv, h); self.enc_e = make_encoder(cfg)
            self.gate_e = MAMGGate(h, nv) if cfg.mamg else None
            self.pool_e = AttentionPool(h)
        if cfg.fusion == 'acmta':
            self.acmta = ACMTABlock(cfg)
            self.head = Head(2 * h + cfg.n_static, h, cfg.dropout)
        elif cfg.fusion == 'late':
            self.head_v = Head(h + cfg.n_static, h, cfg.dropout); self.head_l = Head(h + cfg.n_static, h, cfg.dropout)
        else:
            self.head = Head(h + cfg.n_static, h, cfg.dropout)

    def _stream(self, emb, enc, gate, x, m, d, t, pad):
        hseq = enc(emb(x, m, d, t), pad)
        g = None
        if gate is not None:
            hseq, g = gate(hseq, m, d)
        return hseq, g

    def forward(self, b, return_attention: bool = False):
        cfg = self.cfg; aux = {}
        if cfg.fusion in ('acmta', 'late', 'vitals_only'):
            hv, gv = self._stream(self.emb_v, self.enc_v, self.gate_v, b['xv'], b['mv'], b['dv'], b['tv'], b['pv'])
        if cfg.fusion in ('acmta', 'late', 'labs_only'):
            hl, gl = self._stream(self.emb_l, self.enc_l, self.gate_l, b['xl'], b['ml'], b['dl'], b['tl'], b['pl'])
        if cfg.fusion == 'acmta':
            zv, zl, av, al = self.acmta(hv, b['tv'], b['pv'], hl, b['tl'], b['pl'])
            pv_, wv = self.pool_v(zv, b['pv']); pl_, wl = self.pool_l(zl, b['pl'])
            logit = self.head(torch.cat([pv_, pl_, b['static']], dim=-1))
            if return_attention: aux = {'att_v2l': av, 'att_l2v': al, 'pool_v': wv, 'pool_l': wl}
            return (logit, aux) if return_attention else logit
        if cfg.fusion == 'late':
            pv_, _ = self.pool_v(hv, b['pv']); pl_, _ = self.pool_l(hl, b['pl'])
            p = 0.5 * (torch.sigmoid(self.head_v(torch.cat([pv_, b['static']], -1))) + torch.sigmoid(self.head_l(torch.cat([pl_, b['static']], -1))))
            logit = torch.logit(p.clamp(1e-6, 1 - 1e-6))   # output-level averaging of probabilities (Table 3.4)
            return (logit, aux) if return_attention else logit
        if cfg.fusion == 'early':
            he, _ = self._stream(self.emb_e, self.enc_e, self.gate_e, b['xe'], b['me'], b['de'], b['tv'], b['pv'])
            pe, _ = self.pool_e(he, b['pv'])
            logit = self.head(torch.cat([pe, b['static']], -1))
            return (logit, aux) if return_attention else logit
        if cfg.fusion == 'vitals_only':
            p_, _ = self.pool_v(hv, b['pv'])
        else:
            p_, _ = self.pool_l(hl, b['pl'])
        logit = self.head(torch.cat([p_, b['static']], -1))
        return (logit, aux) if return_attention else logit


def weighted_bce(logit, y, pos_weight: float):
    """Eq. 3.10: L = -(1/N) sum [w+ y log p + (1 - y) log(1 - p)]."""
    return F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(pos_weight, dtype=logit.dtype))

# Implementation issue log

Every place where the thesis is silent, ambiguous or self-contradictory, and the choice made in this package.
"PROVISIONAL" in the code refers to these entries. None of these choices should be described as a documented
thesis method.

## Data and labels

| ID | Thesis evidence | Problem | Choice in this package |
|---|---|---|---|
| D-01 | §3.2, Table 3.1: PhysioNet 2019 is "openly available at no cost"; §3.12: access "obtained under the PhysioNet Credentialed Health Data Use Agreement"; §5.4.1: synthetic data used because of "the difficulty of obtaining credentialed real sepsis data" | PhysioNet/CinC 2019 v1.0.0 is Open Access (CC BY 4.0); no credentialing is needed | The real data are used (see docs/DATASET.md) |
| D-02 | §3.3 vs §4.3 | §3.3 applies no exclusion beyond age and missing vital signs; §4.3 excludes stays < 8 h | `min_icu_hours` option (0 = §3.3; 8 = §4.3). Every real PhysioNet stay has >= 8 rows, so both give the same real-data cohort |
| D-03 | §3.3 "T0 was taken from the labels supplied" | The challenge labels are shifted 6 h ahead of onset (SepsisLabel = 1 for t >= t_sepsis - 6) | T0 = first labelled hour + 6 (option `--onset-shift`); 425 septic stays are labelled from their first hour, so their onset is censored (onset <= 6 h after ICU admission) |
| D-04 | §1.6, §3.3, §3.5 (hourly) vs §4.10.1 ("two-hour stride") | Window stride | `stride` option; 1 h for the reproduction runs |
| D-05 | §3.4 + Table 3.2 vs §4.10 + Figure 4.10 | Variable set: 6 vitals / 5 labs / age, sex versus 8 vitals / 26 labs / 6 static | `ch3` (default) and `ch4_full` variable sets |
| D-06 | §1.7 ("72-hour window"), Figure 2.1 ("±72 h") | Sepsis-3 window definition differs between chapters and from the PhysioNet definition | Not needed for PhysioNet labels; documented only |
| D-07 | §3.3 vs §4.3 | Positive label: onset within the next 12 h with predictions stopping 4 h before T0 (lead 4-12 h) versus onset in the 12 h after the 4-h buffer (lead 4-16 h) | `label_mode` = `ch3` (default) or `ch4` |
| D-08 | not specified | Windows in the first 11 hours of a stay | Included and left-padded (`min_history` = 1) |
| D-09 | not specified | Septic stays after the buffer start | Not used (no predictions after T0 - 4 h) |

## Pre-processing and model

| ID | Thesis evidence | Problem | Choice |
|---|---|---|---|
| M-01 | §3.4 "fixed clinical ranges" | Ranges not listed | Wide physiological ranges in `preprocessing.PLAUSIBLE` (1,053 real values removed) |
| M-02 | §3.4 elapsed-time channel | Scaling and value before the first observation not given | Hours since last observation, or since ICU admission if never observed, capped at 48 h and divided by 48 |
| M-03 | §3.4 vs §3.6.3 | Measurements are "resampled at hourly intervals" onto "a common temporal index", yet ACMTA operates "without requiring the two streams to be resampled onto a common time grid" | Vital signs on the hourly grid; laboratory stream = only the hours at which a laboratory value was observed, with their actual time stamps |
| M-04 | not specified | Window without any laboratory observation | One state token at the prediction time carrying the last observed values (mask 0) |
| M-05 | §3.6.2 vs §4.10/§4.10.1/Fig. 4.12 | MAMG after the encoder on both modalities versus per-variable gating of the laboratory stream before the encoder | §3.6.2 (Eqs. 3.4-3.5) implemented; can be disabled (`--no-mamg`) |
| M-06 | §3.6.1 | Form of the "positional encoding adapted to non-uniform intervals" | Continuous-time sinusoidal encoding of the hour offset, added to the embeddings of both encoders |
| M-07 | §3.6.3 | Residual connections, normalisation, number of layers, size of phi | Residual + LayerNorm after each attention direction; 16 frequencies; one Transformer layer |
| M-08 | §3.6.4 / §4.10 | Pooling and how static features enter | Attention pooling of each fused stream; static features concatenated before the MLP head (§4.10) |
| M-09 | §3.7.2, §4.4 | qSOFA/SOFA/NEWS need variables absent from PhysioNet (GCS, PaO2, vasopressors, oxygen therapy, consciousness) | Proxy scores from available components; nominal probability = score / maximum of the full instrument |
| M-10 | §3.7.2 | Hyper-parameters of logistic regression / gradient boosting | L2 logistic regression (C = 1, balanced weights); XGBoost 300 trees, depth 4, learning rate 0.05, scale_pos_weight = N_neg/N_pos |
| M-11 | §3.6.4, §3.7.1 | Epochs, patience, seeds, number of Optuna trials, chosen hyper-parameters | Max 10 epochs, patience 3, seed 42; reproduction runs use mid-range values (hidden 64, 4 heads, dropout 0.2, lr 1e-3, weight decay 1e-4, batch 256) |
| M-12 | §3.7.1 | Deep models: full 5-fold CV + test | Compute budget: one CV fold (fold 0) used for early stopping/threshold selection, then a single evaluation on the held-out 20 % test set. Classical baselines use the full 5-fold CV on the development set |
| M-13 | §3.8 | Alarm-budget unit (alarms vs false alarms per patient-day) | Both reported; CUCTS selection uses alarms per patient-day (Figure 4.7) |
| M-14 | §3.8 DeLong | Windows of the same patient are correlated | DeLong p-values computed as specified but are optimistic; patient-level bootstrap CIs are also reported |

## Explainability and infrastructure

| ID | Problem | Choice |
|---|---|---|
| X-01 | SHAP on the deep model (§3.10) is not specified (explainer, background set) and is expensive | SHAP (TreeExplainer) and permutation importance for gradient boosting; attention profiles and integrated gradients for ACMTA-Sepsis |
| X-02 | Clinical expert rubric (§3.10) | Not implementable in code |
| X-03 | §3.11 PyTorch 2.2, NVIDIA A100 | Executed with PyTorch 2.3.1 (2.2 is not available from conda-forge for linux-64/py3.10) on 2 CPU threads; no GPU |
| X-04 | Time-limited execution slots (about 3 minutes per job) | Resumable training (`train.py` checkpoints, `scripts/run_queue.py`); interrupted and uninterrupted training give identical weights (tests/test_train_resume.py) |

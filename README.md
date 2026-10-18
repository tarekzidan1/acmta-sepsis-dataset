# ACMTA-Sepsis — reference implementation and reproduction package

Re-implementation of the methodology of the M.Sc. thesis *"Early Prediction of Sepsis in Intensive Care Units Using
Multimodal Temporal Deep Learning Models"* (Tarek Ghassan Zidan, Arab American University): data pipeline (§3.2-3.5),
ACMTA-Sepsis model with Modality-Aware Missingness Gating and Asynchronous Cross-Modal Temporal Attention (§3.6,
Eqs. 3.1-3.10), baselines and comparison conditions (§3.7, Table 3.4), evaluation metrics, CUCTS threshold selection
and statistics (§3.8-3.9), explainability (§3.10) and the Chapter 4 pipeline-validation protocol.

**Important:** the thesis's own code, synthetic data generator and configurations were not supplied. This package is an
independent implementation from the text. Where the text is silent or contradictory the choice is marked
`PROVISIONAL` in the code and listed in [docs/ISSUES.md](docs/ISSUES.md).

## Layout

| Path | Content |
|---|---|
| `acmta_sepsis/data.py` | PSV loading/cache, cohort (§3.3), onset T0, patient-level 80:20 split + stratified 5-fold CV (§3.7.1) |
| `acmta_sepsis/preprocessing.py` | variable sets, plausibility filter, masks, elapsed time, LOCF, train-mean fill, standardisation (§3.4) |
| `acmta_sepsis/windows.py` | 12-h windows, 12-h horizon, 4-h buffer, labels (§3.3, §3.5) |
| `acmta_sepsis/batching.py` | dense vital-sign stream + asynchronous laboratory-event stream |
| `acmta_sepsis/model.py` | ACMTA-Sepsis, BiLSTM/Transformer encoders, MAMG, time-gap bias, early/late/single-modality variants |
| `acmta_sepsis/train.py` | class-weighted BCE (Eq. 3.10), AdamW, early stopping, resumable checkpoints |
| `acmta_sepsis/evaluate.py` | AUROC, AUPRC, Brier, ECE, reliability, DCA, alarm rates, CUCTS, lead time, bootstrap CIs, DeLong, Holm |
| `acmta_sepsis/features.py`, `scores.py`, `baselines.py` | engineered features, qSOFA/SOFA/NEWS proxies, logistic regression, XGBoost |
| `acmta_sepsis/explain.py` | permutation importance, SHAP (trees), attention profiles, integrated gradients |
| `acmta_sepsis/synthetic.py` | schema-matched synthetic generator (tests and Chapter 4 protocol; NOT the thesis generator) |
| `acmta_sepsis/optimize.py` | Optuna search on training/validation data only |
| `acmta_sepsis/experiments.py` | CLI: `prepare`, `baselines`, `report-baselines`, `train`, `evaluate` |
| `scripts/` | dataset download, resumable job queue, Chapter 4 protocol, model comparison, latency, explainability, summary |
| `configs/` | `default.yaml` (all settings), `queue_real_ch3.json` (the six deep-learning runs) |
| `tests/` | 31 tests: labels/buffer, no look-ahead, split integrity, scaler leakage, shapes, attention masking, metrics vs scikit-learn, checkpoint resume |
| `results/` | logs, checkpoints, predictions, metrics JSON/CSV, comparison table |

## Setup

```bash
conda env create -f environment.yml && conda activate acmta-sepsis     # exact tested versions (conda-forge, CPU)
# or: pip install -r requirements.txt   (see the NumPy note in the file)
```

## Data

See [docs/DATASET.md](docs/DATASET.md) (open access, CC BY 4.0; no credentialing):
```bash
scripts/download_physionet2019.sh data/physionet2019
python -c "from acmta_sepsis.data import build_cache; build_cache('data/physionet2019', 'data/physionet2019.npz')"
```

## Tests

```bash
pytest -q tests          # synthetic data only; ~5 s
```

## Reproduce the reported runs

```bash
export ACMTA_WORK=work ACMTA_RESULTS=results
python -m acmta_sepsis.experiments prepare --name real_ch3 --data real --cache data/physionet2019.npz --varset ch3   # cohort, split, windows
python -m acmta_sepsis.experiments baselines --name real_ch3 --models lr --n-boot -1
python -m acmta_sepsis.experiments baselines --name real_ch3 --models gbm --n-boot -1
python -m acmta_sepsis.experiments report-baselines --name real_ch3 --models qsofa,sofa_proxy,news_proxy,lr,gbm
python scripts/run_queue.py configs/queue_real_ch3.json 600   # repeat until "QUEUE COMPLETE" (resumable)
python scripts/compare_models.py                              # DeLong/bootstrap tests + recalibration study
python scripts/ch4_protocol.py --variant ch4                  # Chapter 4 protocol on the synthetic generator
python scripts/latency_and_engineering.py; python scripts/run_explain.py
python -m acmta_sepsis.optimize --name real_ch3 --trials 15 --enqueue-default --budget-min 600
python scripts/summarize.py                                   # results/comparison_table.csv, results/real_data_results.csv
python scripts/lab_sparsity.py 600                            # robustness to sparser laboratory testing (resumable)
python scripts/lab_sparsity_primary.py                        # pre-specified primary comparison: ACMTA vs early fusion
```
Seeds: split/fold 42, training 42, synthetic generator 7, Optuna 42. Hardware used: Intel Core Ultra 7 265H, 2 CPU
threads, 3.8 GB RAM, no GPU (see `results/real_ch3/engineering_latency.json`). Total deep-learning training time for the
six runs was about 35 minutes (incl. validation passes).

## Results in brief (details: results/*.csv, results/*/ *.json)

* **Thesis numbers (Chapter 4) are not reproducible as such**: they come from a synthetic generator that was not supplied.
  Re-running the documented protocol on an independent generator gives the same qualitative picture (near-chance
  baselines; CUCTS infeasible: sensitivity 0.30 at 3 false alarms per patient-day, 8.9 false alarms per patient-day at
  sensitivity 0.85) - see `results/comparison_table.csv`.
* **Real PhysioNet 2019 data, Chapter 3 protocol (reported in the updated thesis, Section 4.11, and in the manuscript; three training seeds):** test AUROC
  0.758 (95 % CI 0.740–0.777; 0.758 ± 0.002 over seeds) for BiLSTM-ACMTA,
  0.754 ± 0.003 Transformer-ACMTA, 0.760 ± 0.002 early fusion, 0.752 ± 0.005 late fusion,
  0.683 ± 0.003 vital signs only, 0.725 ± 0.005 laboratory only; 0.749 gradient boosting,
  0.690 logistic regression, 0.580/0.546/0.534 NEWS2/qSOFA/SOFA proxies.
  ACMTA did not outperform early fusion, and the MAMG and time-gap-bias ablations had no consistent effect
  (−0.001 to +0.002 and −0.003 to +0.006 AUROC across seeds). Under CUCTS no model met both
  constraints (sensitivity at most 0.46 within 3 alarms per patient-day). With 25/50/75/100 % of the test-set laboratory
  observations deleted, ACMTA stayed above gradient boosting at every level but, contrary to the pre-specified
  expectation, degraded more than early fusion (Holm-adjusted DeLong p = 0.030, 0.025 and < 0.001 at 50/75/100 %). Analyses and figures: `manuscript/`
  (see `manuscript/README.md`); runs: `configs/queue_real_ch3.json` and `configs/queue_real_ch3_seeds.json`.

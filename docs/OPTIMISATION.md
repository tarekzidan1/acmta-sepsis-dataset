# Optimisation study (kept separate from the reproduction)

All optimisation used the **training and validation partitions only** (development set, fold 0 as validation). The
held-out test set was never used to choose a configuration; test numbers below are reported once, after selection.

## 1. Hyper-parameter search (method-preserving: the thesis prescribes an Optuna search over Table 3.3)

* **Search space (Table 3.3):** hidden 64-256 (step 32); heads {2, 4, 8} (hidden must be divisible); dropout 0.1-0.5;
  learning rate 1e-4 to 1e-2 (log); weight decay 1e-6 to 1e-3 (log); batch size {64, 128, 256}. Encoder: BiLSTM; fusion: ACMTA.
* **Sampler / budget:** Optuna 5.0 TPE (seed 42, 5 random start-up trials), run until 15 trials had completed: 17 trials
  were recorded - 1 enqueued reference trial (the reproduction defaults) and 16 sampled trials, of which 2 were interrupted
  by the job time limit (state FAIL) and 2 repeated an earlier configuration after a resume (the seeded sampler re-proposed
  them); 12 distinct sampled configurations completed. Each trial: fixed budget of 700 AdamW steps on batches drawn from a
  fixed random 50 % of the training-partition patients; scored on a fixed random 50 % of the validation-fold patients.
* **Selection criterion:** validation AUROC.
* **Outcome (results/real_ch3/optuna_bilstm/study.json):** best sampled trial #16, validation AUROC 0.7309 (hidden 224,
  4 heads, dropout 0.22, lr 1.9e-3, weight decay 1.4e-4, batch 256); the enqueued reproduction defaults (trial #15: hidden 64,
  4 heads, dropout 0.2, lr 1e-3, weight decay 1e-4, batch 256) scored **0.7409** and remained the best configuration. Two
  trials with lr > 3e-3 collapsed (AUROC 0.5). Because the budget is a fixed number of steps, larger batches see more data,
  which favours batch 256 (the only two completed batch-256 trials are the two best); the search is therefore under-powered and does not
  justify changing the reproduction configuration. **No re-training was needed: the selected configuration is the
  reproduction configuration** (test results in results/real_data_results.csv).

## 2. Engineering measurements (method-preserving)

From results/real_ch3/engineering_latency.json (Intel Core Ultra 7 265H, 2 threads, PyTorch 2.3.1 CPU):

* Real-time replay of a septic test patient, one forward pass per hour, batch 1: median 0.86 ms (BiLSTM-ACMTA, 1 thread) and
  0.63 ms (Transformer-ACMTA). The thesis reports 84 ms per hour on CPU (Figure 4.13) for an unreported model size and CPU.
* Bulk inference: 0.03-0.07 ms per window (batch 256-4096).
* Dynamic int8 quantisation (BiLSTM-ACMTA): no speed-up (0.069 vs 0.066 ms/window), AUROC change -0.0004 on 60,000 test
  windows, but individual probabilities changed by up to 0.29 - **not recommended**. Not supported for the Transformer
  encoder in PyTorch 2.3 (fast-path incompatibility).
* Training throughput: 1.9-2.0 min per epoch (980,000 windows, hidden 64, batch 256) for the ACMTA models; resumable
  checkpoints make training robust to time-limited jobs without changing the result (bitwise-equivalent weights in tests).

## 3. Experimental change (alters the method; new research result)

**Post-hoc recalibration** (recommended by thesis §4.5 and §5.1 but not part of the described pipeline): Platt scaling and
isotonic regression fitted on the validation predictions and applied to the test predictions
(results/real_ch3/model_comparison.json). For BiLSTM-ACMTA: Brier 0.189 -> 0.0145, ECE 0.351 -> 0.0003, AUROC unchanged
(0.758), AUPRC unchanged (Platt). Class-weighted training therefore over-predicts risk exactly as §4.5 anticipates, and a
recalibration step is necessary before probabilities are interpreted as risks.

# Manuscript analyses

Run after all queued experiments have finished (from the package root, with `ACMTA_RESULTS` pointing at the results folder):

```bash
export ACMTA_PAPER=manuscript_out && mkdir -p $ACMTA_PAPER/figs
python manuscript/analysis.py metrics      # discrimination, calibration, seeds, DeLong/Holm, cohort table
python manuscript/analysis.py ops          # CUCTS / Youden operating points and lead times
python manuscript/analysis.py curves       # ROC, PR, alarm, reliability and decision curves
python manuscript/analysis.py "sub:ACMTA (BiLSTM)"; python manuscript/analysis.py "sub:Gradient boosting"; python manuscript/analysis.py "sub:Early fusion"
python manuscript/facts.py                 # formatted numbers used in the text and tables
python manuscript/figures.py               # Fig1-Fig5 (EPS, 1200-dpi TIFF, PNG)
```

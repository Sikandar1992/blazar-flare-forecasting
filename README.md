# Advance warning of γ-ray blazar flares from Fermi-LAT light curves: a strictly causal machine-learning backtest

Code accompanying:

**Shah, Z. & Akbar, S. (2026)**, "Advance warning of γ-ray blazar flares
from *Fermi*-LAT light curves: a strictly causal machine-learning
backtest."

- Zahir Shah  — Manipal Centre for Natural
  Sciences, Centre of Excellence, Manipal Academy of Higher Education,
  Manipal 576104, India (zahir.shah@manipal.edu)
- Sikandar Akbar — Department of Physics, University of Kashmir,
  Srinagar 190006, India (darprince46@gmail.com) — corresponding author

## What this code does

This repository contains the full analysis pipeline described in the
paper: Bayesian-Blocks flare identification, rolling-window feature
extraction (42 variability features per window), WATCH/TRIGGER label
assignment, model training (Logistic Regression, Polynomial Logistic
Regression, Random Forest), TRAIN-only calibration and threshold
selection, held-out evaluation (ROC AUC, AP, BSS, block-permutation
and block-bootstrap tests), and generation of the diagnostic and
timeline figures used in the manuscript.

## Requirements

See `requirements.txt` for the exact package versions used to
generate the results reported in the paper (Table 3). Install with:

```
pip install -r requirements.txt
```

**Note on reproducibility:** LR and PLR results (including PLR, the
paper's best-performing model) were exactly reproducible across the
environments tested. RF results showed small variation across
package versions/machines, consistent with known sensitivity of parallelized
ensemble methods to environment differences even under a fixed
random seed. Use the pinned versions in `requirements.txt` to most
closely reproduce the reported numbers.

## Input data

The script expects 14 Fermi-LAT daily light-curve CSV files (as
downloaded from the Fermi-LAT Light Curve Repository; Abdollahi et
al. 2023), placed in the same working directory as the script, with
the following exact filenames:

```
4FGL_J1048.4+7143_daily_3_23_2026.csv      (target source)
4FGL_J1512.8-0906_daily_3_24_2026.csv
4FGL_J1224.9+2122_daily_3_24_2026.csv
4FGL_J0904.9-5734_daily_3_23_2026.csv
4FGL_J1256.1-0547_daily_3_23_2026.csv
4FGL_J2253.9+1609_daily_3_23_2026.csv
4FGL_J0538.8-4405_daily_3_24_2026.csv
4FGL_J0739.2+0137_daily_3_24_2026.csv
4FGL_J1159.5+2914_daily_3_24_2026.csv
4FGL_J0730.3-1141_daily_3_24_2026.csv
4FGL_J1310.5+3221_daily_3_24_2026.csv
4FGL_J1443.9+2501_daily_3_24_2026.csv
4FGL_J1522.1+3144_daily_3_24_2026.csv
4FGL_J0403.9-3605_daily_3_24_2026.csv
```

If your downloaded filenames differ, either rename them to match the
above, or edit the `LC_FILE` and `EXTRA_TRAIN_FILES` settings near the
top of the script to point to your own paths.

The already-processed feature/label tables derived from these light
curves (i.e., the actual model training data) are archived separately
as a Dataset deposit — see "Related data" below — so re-running this
pipeline from raw light curves is not required to inspect or reuse
the training data itself.

## Running

```
python3 code_final_upload.py
```

Outputs (figures, timelines, lead-time tables, the model comparison
summary, and the exported training dataset CSV) are written to
`pdf_flare_outputs_v3/` by default.

## A note on scope

This script also includes an XGBoost classifier and a dual-threshold
alert-state system, retained here for exploratory reference. Neither
is part of the analysis reported in the manuscript — only the
Logistic Regression, Polynomial Logistic Regression, and Random
Forest results (Table 3) are discussed in the paper.

## Related data

The derived training dataset (42-feature vectors and WATCH/TRIGGER
labels per rolling window, for all 14 sources) is archived as a
separate Zenodo Dataset deposit: (https://zenodo.org/records/22643599).

Raw Fermi-LAT light curves are not redistributed here; they are
publicly available from the Fermi-LAT Light Curve Repository
(Abdollahi et al. 2023).

## License

MIT License — see `LICENSE`.

## Citation

If you use this code, please cite:

Shah, Z. & Akbar, S. 2026, [journal / DOI once assigned]

and the archived software release:

[Software Zenodo (https://zenodo.org/records/22643599)

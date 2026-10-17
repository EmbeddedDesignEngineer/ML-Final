# MLE-AI-201 Machine Learning — Final Project

Build **two models from scratch** (a decision tree and an SVM) on the
**bina.az** real-estate dataset, then benchmark honestly against
scikit-learn. Read the full brief in the project statement PDF handed out
with this pack.

- **Released:** Sun 04 Oct 2026
- **Due:** **Sun 18 Oct 2026, 23:59 (Baku)** — two weeks, no late submissions.
- **Demo:** defended in the session after the deadline.

## What you submit (on Moodle)
Exactly **three** things; **one member submits** for the whole team:
1. **GitHub link** to your repo (tagged `v1.0-final`).
2. **Report PDF** (IEEE two-column — use `report/report.tex`).
3. **Slides PDF**.

Keep `contribution_report.md` in the repo root (checked at the demo).

## The two tasks
- **Task A — Regression:** predict `price` (consider `log(price)`).
- **Task B — Classification:** predict a derived **price tier**
  (premium vs standard, median split on the training set).

## Golden rule
Your **decision tree** and your **SVM** must be **your own NumPy code**.
scikit-learn is for the comparison baselines (and the optional bonuses)
only. For the SVM you do **not** need a QP/SMO solver — use the
**Pegasos** sub-gradient method (see the statement and `src/svm.py`).

## Repo layout
```
.
├── README.md               # this file
├── requirements.txt        # pin your versions
├── .gitignore
├── contribution_report.md  # who did what (keep in repo root)
├── data/                   # dataset lives here (not committed — see data/README.md)
├── src/                    # YOUR from-scratch code
│   ├── data_prep.py        #   loading, cleaning, split, tier label
│   ├── decision_tree.py    #   DecisionTree (Gini/entropy/MSE)
│   ├── svm.py              #   PegasosSVM (hinge-loss sub-gradient)
│   ├── evaluate.py         #   metrics + comparison helpers
│   └── run_all.py          #   ONE command reproduces every number/figure
├── report/                 # IEEE report (self-contained, no IEEEtran.cls)
│   ├── report.tex
│   └── figures/
└── presentation/           # your slides PDF
```

## Reproducibility (graded)
- Pin dependencies in `requirements.txt`; fix all random seeds.
- `python -m src.run_all` must reproduce **every** headline number and
  figure from a clean checkout.
- Do **not** commit the raw data archive (see `.gitignore`).

## Quick start
```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m src.run_all
```

These commands use Windows paths. On Linux/macOS use `.venv/bin/python`.
Place the real raw dataset at `data/house_sale.csv` before running. The default
run writes to `results/final/` and includes cleaning, EDA, all 33 SVM validation
configurations, tree validation sweeps, Ridge selection, five-fold evaluation,
and final model comparisons. The SVC fits can take substantial time.

For a new reproduction choose an unused directory:

```bash
python -m src.run_all --output results/reproduction
```

The runner preserves earlier output instead of deleting it. After a later-stage
failure, a completed SVM search can be reused with
`python -m src.run_all --output results/reproduction --resume`.
Data and SVM source hashes are checked before reuse. An incomplete SVM search
cannot be resumed; choose a new output directory in that case.

`--skip-cv` and `--skip-svm-search` are diagnostic shortcuts. They do not produce
the complete experiment evidence. Install `requirements-dev.txt` and run
`python -m pytest -q` for all checks.

## Decision tree module

The NumPy tree supports Gini/entropy classification and MSE regression.
Run its checks with `python -m pytest tests/test_decision_tree.py tests/test_tree_sklearn.py tests/test_tree_experiments.py -q` (pytest required).
See [tree methodology and integration notes](experiments/TREE_REPORT_NOTES.md)
for the model API, validation sweeps, sklearn comparison caveats and remaining
real-data evaluation steps. The tree helper does not replace the shared runner.

## Evaluation module

The starter metrics are implemented in NumPy in `src/evaluate.py`.
Run `python -m pytest -q` after installing `pytest==9.1.1`.

With the real `data/house_sale.csv` available, reproduce the evaluation with:

```bash
python -m src.data_prep
python -m experiments.evaluation_experiments --output results/evaluation_reproduction
```

The evaluation command performs validation-only tree selection, reuses the
four previously selected SVM configurations, runs five-fold uncertainty checks
without final test rows, and saves final metrics, subgroup analysis and figures.
Use a fresh output directory to preserve earlier test results.
Committed results are in `evaluation/artifacts/`; methodology and interpretation
are in `evaluation/REPORT_NOTES.md`. This evaluation is integrated into Bailar's shared `src/run_all.py` entry point.

## Ridge and shared experiments

`src/linear.py` implements Ridge with a NumPy SVD solve and an unpenalized
intercept. Ridge is the optional regression bonus; custom and sklearn versions
use training-fitted scaling and the same alpha grid. `src/run_all.py` is the
shared entry point and `experiments/pipeline.py` connects the team's modules.
See `experiments/PIPELINE_REPORT_NOTES.md` for results, reproduction details and
timing/uncertainty limits. Committed evidence is under
`experiments/artifacts/final/`; local prepared CSV/NPZ files remain ignored.

## Regression and SVM quality study

Run `python -m src.run_all --quality-study --output results/quality-new` with
the raw CSV at `data/house_sale.csv` (or pass `--raw`). Use a new output directory.
This compares the old settings with deeper trees, our bootstrap tree ensemble,
and longer averaged SVM training. RFF candidates also vary regularization and gamma.
The study fits preprocessing on train and selects configurations on validation.
It never scores the historic test partition. `selected.json` records the best
searched configuration in each family; `validation.json` contains every candidate.
The manifest records source/data hashes, sizes and completion status. Partial
results from an interrupted run are not a complete comparison.

The original full-pipeline command preserves the historical experiment settings.
See `experiments/MODEL_QUALITY_REPORT.md` for measured improvements and limits.

For deeper regression comparisons, run
`python -m src.run_all --price-study --clean data/house_sale_clean.csv --output results/price-new`.
This uses the cleaned CSV from preprocessing, compares raw and log-price-per-area
targets, and reports median/90th-percentile error, relative accuracy, and price bands.
Custom residual boosting is included as a candidate, not assumed to outperform bagging.
The selected model can differ for RMSE and MAE; see `experiments/PRICE_ERROR_REPORT.md`.

This completes the experiment code path. The team must still assemble and
review its final report PDF, presentation PDF and named contribution report
before tagging the submission `v1.0-final`.

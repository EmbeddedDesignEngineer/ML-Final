# Evaluation usage

The original seven function names and signatures remain unchanged. Metric
calculations use NumPy; sklearn is used in tests and reference models.

## Reproduce real evaluation

From the repository root, after installing requirements.txt and obtaining
the real house_sale.csv described in data/README.md:

```powershell
python -m src.data_prep
python -m experiments.evaluation_experiments --output results/evaluation_reproduction
python -m pip install pytest==9.1.1
python -m pytest -q
```

The output directory must be empty. The full command includes five-fold
evaluation. --skip-cv omits uncertainty and is only a shorter diagnostic run.
The labelled development demo remains available as
`python -m experiments.evaluation_demo`; it is not final evidence.

## Metric policies

- Inputs must be non-empty, finite, real one-dimensional arrays of equal length.
- MAE and RMSE use the supplied target units. This run uses AZN directly.
- Constant-target R2 is 1 for perfect predictions and 0 otherwise. One row
  cannot define R2 and raises ValueError.
- Class 1 is premium. Confusion matrices are [[TN, FP], [FN, TP]].
- Undefined precision, recall and F1 ratios return zero.
- ROC-AUC requires both true classes, uses decision scores or class-1
  probabilities, and assigns average ranks to equal scores. Higher means premium.
- A single-class subgroup has blank AUC rather than a misleading zero.

## Shared model interfaces

Davud's split dictionary uses X_te for features, y_te for AZN prices and t_te
for binary labels. Farid's prepared NPZ uses y_test for binary labels.
Keep these meanings distinct. The evaluator uses exact shared split row IDs
and verifies price alignment before attaching original locations.

SVM features are standardized from training statistics. Tree features are
unscaled. Tree probabilities are matched to classes_ to find class 1.
Feature importance is available as tree.feature_importances_. Linear SVM
weights are model.w; original feature interpretation is only valid when
model.rff_ is None. RFF weights describe transformed features.

## Results and integration

Committed real-data outputs are in evaluation/artifacts/. Methodology,
tables, interpretation and limits are in evaluation/REPORT_NOTES.md.
report/evaluation_section.tex is a supporting section for the shared report.

The program exposes run_evaluation(clean_path, output, include_cv=True) for
Bailar's shared runner. It does not change src/run_all.py or implement Ridge.
Its configuration source, split rules, data hashes and package versions are
recorded in the manifest. Wall-clock times vary between runs. CV is descriptive
for fixed selected settings, not nested model-selection uncertainty.

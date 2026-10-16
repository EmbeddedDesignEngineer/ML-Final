# Evaluation module

## Scope and status

The seven starter functions in `src/evaluate.py` retain their original names
and signatures. Their calculations use NumPy, with sklearn used in tests as
an independent reference. Plotting and grouped regression error summaries
are in `src/evaluation_plots.py`.

The code and development tests are implemented. Real Bina.az evaluation,
the final comparison across all models, and integration into `src/run_all.py`
remain unfinished. The demo uses artificial inputs and is not report evidence.

## Running checks

From the repository root, in an environment with `requirements.txt` installed:

```powershell
python -m pip install pytest==9.1.1
python -m pytest tests/test_evaluate.py tests/test_evaluation_plots.py -q
python -m experiments.evaluation_demo
python -m pytest -q
```

The demo writes labelled development figures to `results/evaluation_demo/`.
The complete test suite passed with Python 3.12, NumPy 2.5.4, pandas 3.0.6,
matplotlib 3.11.2, scikit-learn 1.9.1 and pytest 9.1.1 during development.

## Metric definitions and edge cases

- MAE averages absolute prediction errors.
- RMSE takes the square root of the average squared error, penalizing large
  errors more strongly. Prices and predictions must use the same units.
- R-squared is `1 - SS_res / SS_tot`. Constant targets score 1 for exact
  predictions and 0 otherwise, matching sklearn's default finite-score policy.
  A single observation raises ValueError. Negative R-squared is allowed.
- The binary confusion matrix is `[[TN, FP], [FN, TP]]`. Rows are true labels
  and columns are predicted labels. Class 1 means premium.
- Precision, recall and F1 describe class 1. Undefined zero-denominator
  ratios return 0, matching sklearn with `zero_division=0`.
- ROC-AUC uses ranks of continuous scores. Equal scores receive their average
  rank. Higher scores must support class 1. Both true classes are required;
  a single-class input raises ValueError. Do not supply hard predictions.
- Inputs must be non-empty, finite, real, one-dimensional arrays with equal
  lengths. Classification labels must be 0 or 1.
- `compare` prints a union of metric columns. Missing entries appear as `-`.
  Separate regression and classification metrics rather than treating SVM
  class predictions as price predictions.

## Integration with existing model APIs

Use a fitted model whose settings were selected on validation data. The code
below assumes `model`, `X_test` and `y_test` have already been prepared.

```python
from src.evaluate import precision_recall_f1, roc_auc, compare
from src.evaluation_plots import plot_confusion_matrix

predictions = model.predict(X_test)
scores = model.decision_function(X_test)  # SVM
precision, recall, f1 = precision_recall_f1(y_test, predictions)
compare({"SVM": {"Precision": precision, "Recall": recall,
                 "F1": f1, "ROC-AUC": roc_auc(y_test, scores)}})
plot_confusion_matrix(y_test, predictions, "report/figures/svm_confusion.png")
```

For a classification tree, use the probability column corresponding to class
1 instead of `decision_function`. Keep row order and feature order identical.

Davud's `get_splits()` returns `X_te` for test features, `y_te` for AZN prices,
and `t_te` for binary tier labels. Farid's prepared NPZ uses `X_test` and
`y_test`, where `y_test` means binary labels. SVM features must be standardized
using training statistics, as in the existing preparation adapter.

For regression, reverse the exact training transformation before calculating
AZN errors: `exp` reverses `log`, while `expm1` reverses `log1p`. Label all
figures and tables with the actual target units.

## Error analysis and interpretation

```python
import numpy as np
from src.evaluation_plots import summarize_price_errors, summarize_errors_by_group

threshold = np.quantile(train_prices, 0.90)
price_errors = summarize_price_errors(test_prices, predicted_prices, threshold)
location_errors = summarize_errors_by_group(test_prices, predicted_prices, test_locations)
```

Freeze the expensive-property threshold using training data before test
evaluation. Identify underrepresented locations using training counts too.
Location metadata must match test observations in exactly the same order;
`get_splits()` currently does not return location names or row IDs, so arrange
aligned metadata with the data-preparation owner. Replace missing names with
`Unknown`. Each summary reports count, MAE, RMSE and bias (predicted - actual).
Negative bias indicates average underprediction. Small groups have uncertain
estimates; a plausible explanation is not proof of causation.

`plot_feature_values` displays tree impurity importances or linear SVM weights
ranked by absolute magnitude. Farid's linear weights are `model.w`. Only label
them with original feature names when `model.rff_ is None`; RFF weights refer
to transformed features. Weight interpretation depends on feature scaling.

Final reporting still needs real model predictions, sklearn baselines,
cross-validation mean and standard deviation, timing, and tree importances.
The existing SVM experiment runner still uses its own sklearn evaluation;
it has not automatically been rewired to this module.

## AI assistance disclosure

OpenAI Codex assisted with implementation, test generation, documentation and
local test execution for these evaluation files. The student has not yet
confirmed an independent code review or completed the real-data analysis.
Update the final report with the student's actual verification and analysis,
and record contributions accurately rather than claiming unperformed work.

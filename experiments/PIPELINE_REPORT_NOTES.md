# Ridge bonus and experiment execution — bailar-dev

## Implementation

The Ridge implementation in src/linear.py uses NumPy. For alpha > 0 it solves
`(X_centered.T @ X_centered + alpha * I) @ w = X_centered.T @ y_centered`
with `np.linalg.solve`. The intercept is `mean(y) - mean(X) @ w` and is not
penalized. Alpha 0 uses least squares to handle singular designs. fit returns
the model; predict returns one number per row. Invalid shapes and non-finite
inputs fail explicitly. No sklearn regression algorithm is used inside it.

Both implementations use training-fitted feature standardization. The custom
model uses a small scaling wrapper; the sklearn reference uses a Pipeline
with StandardScaler and Ridge. Each CV fold refits its encoder and scaler.

## Validation design

The alpha grid is 0.1, 1, 10, 100, 1000. Each model is fitted on training rows
only and selected by minimum validation RMSE in AZN. Ties keep the first grid
entry. The final custom alpha is 1000; the sklearn alpha is
1000. This is the best tested value, not a claim
of a global optimum; the winner is at the grid boundary. No test labels select
alpha or any other hyperparameter. Price is modeled directly, without a log
transformation or clipping negative predictions.

The shared runner also executes all 33 of Farid's SVM configurations and
Senan's tree depth/leaf/criterion sweeps. SVM winners are frozen using validation
F1, then validation ROC-AUC. Tree regression uses validation RMSE and tree
classification uses validation F1. Custom and reference winners are selected
independently, so final settings can differ. The complete parameter records
and validation tables are saved before final evaluation.

## Real-data regression results

| Model | Test RMSE (AZN) | Test MAE (AZN) | Test R2 |
| --- | --- | --- | --- |
| custom_tree_regression | 109163.81 | 45555.13 | 0.7595 |
| sklearn_tree_regression | 119375.81 | 52230.16 | 0.7124 |
| custom_ridge | 128440.98 | 57859.18 | 0.6671 |
| sklearn_ridge | 128440.98 | 57859.18 | 0.6671 |

The raw data hash is b6a4c67e0d3c712de31b9db53db53688792bc03099a213c1994c43114758aec2. Davud's unchanged cleaning
produced 57,363 residential Baku rows. Seed 42 yields 40,155 training, 8,604
validation and 8,604 test rows. The premium threshold is 209,000 AZN, computed
from training only. No raw listings or prepared feature arrays are committed.

Ridge is an optional linear baseline. Its score should be compared with the
tree as evidence about these fitted models and this cleaned population; a
lower score is not an implementation failure. The custom and sklearn Ridge
metrics agree to numerical precision in the test set and all five folds.

## Uncertainty

| Model | Metric | Five-fold mean | Sample std |
| --- | --- | --- | --- |
| custom_ridge | rmse | 136830.6652 | 5549.7019 |
| custom_ridge | mae | 60471.7661 | 1118.0533 |
| custom_ridge | r2 | 0.6525 | 0.0190 |
| sklearn_ridge | rmse | 136830.6652 | 5549.7019 |
| sklearn_ridge | mae | 60471.7661 | 1118.0533 |
| sklearn_ridge | r2 | 0.6525 | 0.0190 |

The shared five-fold study includes all ten models, giving 50 fold fits and
36 mean/std rows. It uses original training + validation rows; final test rows
are excluded. Fold-specific encoding, scaling and tier thresholds are fitted
from that fold's training rows. Settings remain frozen. Std uses ddof=1.
Because selection already used the original validation pool, this is
descriptive uncertainty for fixed settings, not nested model-selection CV.

## Reproduction

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m src.run_all
.\.venv\Scripts\python.exe -m pytest -q
```

Place the raw CSV at data/house_sale.csv first. On Linux/macOS use
.venv/bin/python. The default result folder is results/final. Choose a new
--output for another complete reproduction. Diagnostic flags --skip-cv and
--skip-svm-search shorten checks but omit full experiment evidence.

The complete run saves cleaning/EDA, SVM studies and objective histories,
tree validation curves, Ridge validation, test metrics, fold summaries,
subgroup errors, feature effects and measured fit/predict/score times. JSON
manifests record seeds, data hashes, source hashes and package versions.
The raw and prepared CSV/NPZ data stays local under the ignored results folder.

## Recovery and verification

The verified real run reused a completed, hash-checked SVM search after a Ridge
integration correction, then completed the remaining stages through
`python -m src.run_all --output results/bailar_full_run --resume`.
The full SVM study was executed, not replaced with synthetic data or old
headline values. Resume checks data hashes, SVM source hashes and completed
search status. It preserves prior evaluation attempts in separate directories.
Only the completed final attempt is copied into the aggregate evidence.

309 tests passed with warnings treated as errors. Tests cover sklearn Ridge
agreement, singular designs, an unpenalized intercept, train-only scaling,
test-independent selection, pipeline execution, missing-data failure and safe
resume. Additional result audits verify ten test rows, 50 fold rows, all
subgroup totals and recomputed mean/std values. All new Python files retain
simple fit/predict interfaces and contain no comments or TODO stubs.

Wall-clock times depend on hardware and load. On a resumed run the manifest's
elapsed time covers that invocation; reused SVM fitting times remain separately
recorded. Do not treat these as repeated benchmark averages. The combined
report PDF, final presentation and final submission tag still need team review.

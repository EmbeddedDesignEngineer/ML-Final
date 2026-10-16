# Decision tree: implementation, experiments, and report notes

## Status and ownership

Senan's tree module is implemented in `src/decision_tree.py` using NumPy only.
It supports price regression and price-tier classification. The shared data
preparation, metrics, SVM, and `run_all.py` are maintained by the other members.
This change does not claim real Bina.az results: no dataset is present in this
checkout. Synthetic tests establish correctness, not real-market accuracy.

## Model contract

```python
from src.decision_tree import DecisionTree

regressor = DecisionTree(task="regression", criterion="mse", max_depth=8,
                         min_samples_leaf=5).fit(X_train, y_price_train)
prices = regressor.predict(X_test)

classifier = DecisionTree(task="classification", criterion="entropy",
                          max_depth=8, min_samples_leaf=5).fit(X_train, y_tier_train)
tiers = classifier.predict(X_test)
premium_scores = classifier.predict_proba(X_test)[:, 1]  # classes_ == [0, 1]
```

These depths/leaf sizes illustrate the API; they are not selected final settings.
`X` must be finite numeric `(rows, features)` data and `y` a one-dimensional
array. Fit encoding and imputation on training rows before calling the model.
Trees do not require standardization. Classification supports sorted numeric or
string labels, including a single class; use `classes_` to identify probability
columns. Numeric regression labels are not class-encoded. Empty prediction
batches are supported. Missing or infinite values fail explicitly.

## Method text for the report

At each node we sort each feature and consider thresholds between distinct
adjacent values. A sample goes left when its feature value is less than or equal
to the threshold. We choose the largest impurity reduction:

`gain = I(parent) - n_left/n_node * I(left) - n_right/n_node * I(right)`.

Classification impurity is either `1 - sum(p_k**2)` (Gini) or
`-sum(p_k * log2(p_k))` (entropy), omitting zero probabilities. Regression
impurity is the population variance of the target. Prefix class counts or
centered target sums/squares evaluate all valid thresholds after sorting.
Centering reduces cancellation when prices have a large common offset.

Leaves predict the majority class or the mean regression target. Class
probabilities are empirical leaf frequencies. Growth stops at a pure node,
insufficient split samples, the depth limit, no valid partition, or insufficient
global gain. Candidate partitions must each satisfy `min_samples_leaf`.
`min_impurity_decrease` compares `(n_node/n_train) * gain` with the configured
threshold, matching sklearn's sample-weighted definition. Zero-gain splits are
allowed at a zero minimum: a first uninformative split can unlock an XOR pattern.
A positive minimum stops such splits. Exactly tied gains retain the first feature
and lowest candidate threshold; class ties choose the first sorted class.

Construction is recursive for ordinary trees. Beyond depth 128 it uses an
explicit continuation stack with the same split and stopping rules, so an
unlimited-depth tree cannot exhaust Python's call stack. Prediction and tree
inspection use iteration. `random_state` is retained for the shared API but
exhaustive search uses no randomness.

Feature importance sums `(n_node/n_train) * gain` for each split feature and
normalizes the vector to sum to one. A tree with no positive gains returns zeros.
This measures training impurity reduction, not causal influence; continuous and
high-cardinality features may be favored. Correlated features can share or
substitute importance.

## Validation experiments: integration for Bailar

```python
import json
from pathlib import Path
from experiments.tree_experiments import compare_tree_configs, save_validation_curves

rows, best = compare_tree_configs(
    X_train, y_tier_train, X_val, y_tier_val,
    task="classification", depths=(2, 4, 8, 12), leaf_sizes=(1, 5, 20), seed=42,
)
out = Path("results/tree")
out.mkdir(parents=True, exist_ok=True)
(out / "classification_validation.json").write_text(
    json.dumps({"rows": rows, "best": best}, indent=2), encoding="utf-8")
save_validation_curves(rows, "report/figures/tree_classification_validation.png")
```

Repeat with `task="regression"` and price targets for regression curves.
The helper compares custom/sklearn models at matched depths and leaf sizes.
Classification searches both Gini and entropy and selects maximal validation
F1 for premium class 1. Regression selects minimal validation RMSE in the
supplied target units. Exact score ties keep the first grid entry. It records
train/validation scores, fit/validation-predict time, actual depth and leaf count.
It deliberately accepts no test split. Do not repeatedly alter the grid after
looking at final test scores.

After freezing the configurations, the shared runner should:

1. Evaluate the frozen train-fitted models on untouched test rows. Use Sabuhi's
   metrics for final RMSE/MAE/R2 and precision/recall/F1/AUC/confusion matrices.
2. If fitting on log-price, label those metrics clearly and inverse-transform
   predictions for AZN metrics. Do not compare log RMSE directly with AZN RMSE.
3. Perform the agreed five-fold uncertainty study without final test rows.
   Refit encoding/imputation and the price-tier threshold inside each fold;
   report mean and sample standard deviation separately from held-out metrics.
4. Export named importances and residual summaries by price range/location.
5. Wire these outputs into the shared one-command runner and report.

The local validation helper is not a substitute for that final integration.

## How to interpret the curves

- Gini versus entropy: compare validation F1 at matched depth and leaf size;
  report the measured winner, not an assumed theoretical winner.
- Increasing depth usually improves training fit. Falling validation F1 or
  rising validation RMSE while training improves indicates overfitting.
- Larger minimum leaves smooth predictions and reduce tiny-region memorization,
  but may underfit rare expensive homes. Validate this on actual results.
- A large train/validation gap is evidence to investigate, not proof by itself:
  check split representativeness and leakage before attributing it to depth.

## sklearn agreement and limitations

Tiny examples with unique best splits match sklearn thresholds, predictions,
probabilities and impurity. Random regression examples compare predictions and
structure as well. Equal-gain features need not match: sklearn permutes feature
order, whereas ours chooses the first. This can change importances even when
predictions agree. sklearn also internally uses float32 features, while our tree
uses float64; extremely close feature values may therefore produce different
thresholds. `max_depth=0` is a supported custom stump but sklearn expects >=1;
the comparison grid requires positive depths.

Cost per node is dominated by sorting, roughly `O(d*n*log(n))`, plus class-count
scans. Balanced full-tree cost includes this at every level; highly unbalanced
trees are much more expensive. Data is copied into child subsets. Prefix class
counts require `O(n*K)` memory per feature, appropriate for the required binary
task; this is not a large-class-count production implementation. Values near
float64 overflow are not a supported financial magnitude.

## Verification and environment

Run from the repository root:

```bash
python -m pytest tests/test_decision_tree.py tests/test_tree_sklearn.py tests/test_tree_experiments.py -q
python -m pytest -q
```

Tests cover known leaf probabilities/means, Gini/entropy, all stopping rules,
weighted gain, constant and adjacent-float inputs, refitting, invalid arrays,
XOR, an unbalanced 1,100-row tree, sklearn comparisons and plot export.

On 16 October 2026 the complete repository suite passed: **190 tests**,
including **50 tree tests**. In the Windows sandbox pytest's default temporary
directory was unavailable, so verification used an empty directory under
`results/` via `--basetemp`. Plot output was also inspected visually.

Local verification used Python 3.14, NumPy 2.3.4, scikit-learn 1.9.0,
matplotlib 3.10.8 and pytest 8.4.2. These differ from the existing repository
requirements pins. The team should reconcile its final environment before
claiming clean-install reproducibility; shared dependency pins were not changed
as part of the tree module. pytest must also be installed to run the test suite.

## Two-slide outline

1. **How our tree learns:** impurity formulas, one numeric threshold example,
   weighted gain, majority/mean leaves, four stopping controls.
2. **How we checked it:** tiny sklearn parity, validation depth/leaf curves,
   Gini versus entropy, measured importances and limitations. Add actual
   dataset numbers only after the shared experiment run.

## Demo explanation (Azərbaycanca)

Ağac hər əlamət üzrə mümkün hədləri yoxlayır. Ən çox qarışıqlığı azaldan
bölünməni seçir və bunu alt qruplarda təkrarlayır. Klassifikasiyada qarışıqlığı
Gini və ya entropy, qiymət proqnozunda isə variance ilə ölçürük. Yarpaqda ən
çox rast gəlinən sinfi və ya orta qiyməti qaytarırıq. Dərinlik və minimum
nümunə məhdudiyyətləri məlumatı əzbərləməyin qarşısını almağa kömək edir;
son ayarları validation nəticələrinə görə seçirik. Test hissəsi bu seçimə
qarışmır. Sklearn ilə fərq hər zaman səhv demək deyil: bərabər gain zamanı
feature seçimi və floating-point dəqiqliyi ağacın strukturunu dəyişə bilər.

The final report must disclose AI assistance as required by the project brief.
Describe the actual assistance with implementation, tests and review; each team
member should independently understand and verify their submitted contribution.

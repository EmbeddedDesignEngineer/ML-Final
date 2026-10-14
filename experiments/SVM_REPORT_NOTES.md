# SVM methodology and real Bina.az experiments

## Verified status — 14 October 2026

All **33 predeclared configurations** completed on the real local Bina.az sale
data. Selection used validation F1, with validation ROC-AUC as the tie-breaker.
Four family winners were frozen before test evaluation. The manifest at
`results/svm/configurations.json` has `status="complete"`; `metrics.csv` contains
33 validation rows and four test rows. No synthetic observations contributed
to these results.

The additional **five-fold SVM cross-validation study is complete**: 20 fits on
the original training-plus-validation pool, with the final test rows excluded.
Each fold fits its own preprocessing and training-price threshold. Fixed model
configurations were reused without tuning. Verified mean ± sample standard
deviations are reported in the cross-validation section below.

Davud's train-only encoding fix, commit `4f3d34c`, is present in merged checkout
`7ded9863059fe9db324b17b777db74c88f0220ac`. Source inspection, real-array checks
and integration tests confirm that the previous encoding blocker is resolved.
The current audit is `results/svm/preprocessing_audit.json`; it supersedes the
historical failed audit at `results/svm_preprocessing_audit.json`.

The existing SVM, original experiment runner, preprocessing and other team
modules were preserved. The export adapter and preparation tests remain intact.
The CV extension adds `experiments/svm_cross_validation.py` and
`tests/test_svm_cross_validation.py`. This SVM notes file was updated; the shared
report and presentation were not edited. **140 tests passed in 2.62 seconds**
with warnings treated as errors: the previous 124 tests plus 16 focused CV checks.

## Data preparation and leakage audit

The existing cleaned CSV was compared with a fresh execution of Davud's
`clean(load_raw())` and matched within CSV floating-point precision. Neither
input CSV was overwritten. The adapter calls `get_splits` and `standardize`;
it does not implement another cleaning, encoding or splitting pipeline.

| Property | Verified value |
|---|---:|
| Raw observations / columns | 100,775 / 51 |
| Observations after listing deduplication | 64,454 |
| Cleaned observations / columns | 57,363 / 17 |
| Training / validation / test observations | 40,155 / 8,604 / 8,604 |
| Numeric features | 65 |
| Split and model seed | 42 |
| Training-price median | 209,000 AZN |
| Premium proportion: train / validation / test | 49.7099% / 49.7327% / 49.7211% |
| Maximum absolute standardized training-column mean | 7.73e-12 |
| Constant training features | 0 |

`get_splits` splits cleaned rows first, then fits `FeatureEncoder` **once on
training rows**. It learns 45 locations whose training counts reach 200.
Rare or unseen locations map to `loc_other`. The three residential categories
are fixed domain definitions in `CATEGORY_MAP`, not vocabulary discovered from
validation/test. Together with 14 numeric inputs and two row-level engineered
features, this produces 65 columns in a fixed order. Transforming held-out rows
with a new location or changed prices leaves the fitted state and training
features unchanged.

All three feature arrays use the training mean and population standard deviation
(`ddof=0`), including indicator columns. All entries are numeric and finite;
training standard deviations are one. The threshold is exactly the training-price
median. Labels are `1` for `price > 209000`, otherwise `0`, including prices equal
to the threshold. Both classes occur in every split.

The shared splitter uses price deciles to stratify approximately 70/15/15.
Full-data quantiles decide split membership only; they are not model features,
scaling statistics or the tier threshold. Cleaned-row IDs recovered using the
same public splitter are unique, disjoint and exhaustive. Cleaning deduplicates
listing URLs before splitting. Because original IDs are then removed, exported
IDs are cleaned-row positions tied to the cleaned-file hash, not original ad IDs.

Source review confirms that features exclude price, price-derived unit/total
prices, tier labels and the discarded post-listing fields. Numeric engineering
uses coordinates and floor/total-floor ratio. Missing-value fills are fixed
structural rules, not held-out estimates. Price and price-per-area sanity filters
use fixed domain limits to define the cleaned population; conclusions apply to
that filtered Baku residential population, not every raw ad.

The NPZ was loaded back with `allow_pickle=False` and accepted by the unchanged
runner's `validate_prepared_data`. Its contract is:

| Keys | Content |
|---|---|
| `X_train`, `X_val`, `X_test` | Standardized arrays: `(40155,65)`, `(8604,65)`, `(8604,65)` |
| `y_train`, `y_val`, `y_test` | Corresponding one-dimensional `{0,1}` tier labels |
| `train_ids`, `val_ids`, `test_ids` | Disjoint cleaned-row indices, excluded from features |
| `feature_names` | 65 Unicode names in input-column order |
| `metadata_json` | Scalar JSON string with source/data hashes, seed, split policy, training fit scopes, threshold, vocabulary, scaler mean/std and checks |

The raw SHA-256 is
`b6a4c67e0d3c712de31b9db53db53688792bc03099a213c1994c43114758aec2`;
the cleaned SHA-256 is
`e0befc41a917e5346b3c7b69b782692a896d2849d0d3e61710babf9351b80c5e`.
The executed NPZ SHA-256 is
`6f9dba08d6306069bcc874b766439728ed3c8205b54b819dc8d81f8d6e722dcc`.
Metadata declarations alone are insufficient to prove absence of leakage; source
review and the numerical checks above supply the supporting evidence.

## Linear soft-margin SVM and Pegasos

`PegasosSVM` maps external labels `{0,1}` to internal labels `{-1,+1}` and learns
`f(x) = w^T x + b` with the objective

\[
J(w,b)=\frac{\lambda}{2}\|w\|_2^2+
\frac{1}{n}\sum_{i=1}^{n}\max(0,1-y_i(w^T x_i+b)).
\]

Hinge loss penalizes errors and correctly classified points inside the margin.
The quadratic term discourages large weights. Bias is neither regularized nor
projected. Negative scores predict standard (`0`); nonnegative scores predict
premium (`1`). Scores are not probabilities.

Training starts from zero weights and bias. Each iteration draws one training
row uniformly with replacement. The standard schedule is `eta_t=1/(lambda*t)`.
The margin is tested before the update. If `y_i*(w^T x_i+b) < 1`,

\[
w\leftarrow(1-\eta_t\lambda)w+\eta_t y_i x_i,
\qquad b\leftarrow b+\eta_t y_i.
\]

Otherwise only weight shrinkage is applied. At margin exactly one, the zero
hinge subgradient is chosen. Weights are then projected onto
`||w|| <= 1/sqrt(lambda)`; bias is excluded. The constant-rate experiment uses
`eta_t=0.01` with the same objective, subgradients and projection. It is an
alternative schedule in `PegasosSVM`, not the standard diminishing schedule.

`loss_history_` records the full regularized training objective at the steps in
`loss_iterations_`. Every run here has 51 checkpoints, including initialization
and the final update: every 100 steps for 5,000-update runs and every 400 steps
for 20,000-update runs. Stochastic fluctuations are expected. Objectives at
different lambdas represent different functions and are not used for selection.
The unregularized bias means the objective is not jointly strongly convex in
`(w,b)`; an unconditional strongly convex Pegasos rate claim is inappropriate.

The distance between the score planes -1 and +1 is `2/||w||_2`. Training points
with `y_i*f(x_i) <= 1` are margin-active; these are not recovered nonzero dual
coefficients. Zero norm gives an infinite, degenerate margin in the model and
a null diagnostic in exported records. No zero-norm case occurred here.

## Random Fourier Features

The nonlinear extension approximates `k(x,x')=exp(-gamma*||x-x'||^2)` using

\[
z(x)=\sqrt{2/D}\cos(xW+c),\qquad
W_{jk}\sim\mathcal{N}(0,2\gamma),\quad c_k\sim U(0,2\pi).
\]

Frequency standard deviation is `sqrt(2*gamma)`. Frequencies and phases are sampled
during fit with a fixed seed and reused unchanged at inference. The same linear
optimizer operates on `z(X)`; no duplicate optimizer or full kernel matrix is
needed. RFF gives a nonlinear decision boundary with a configurable feature
budget. Its norm and margin belong to RFF space, so their magnitudes are not
directly comparable with margins in the standardized original feature space.

## Experiment design and sklearn comparison

Every configuration used the same complete training and validation arrays.
There was no subsampling, resplitting, test tuning or refitting on validation.
The complete grid ran once:

| Family | Grid | Fits |
|---|---|---:|
| Custom linear | lambda `{1e-4,1e-3,1e-2}` x updates `{5000,20000}` x schedule `{Pegasos,constant(0.01)}` | 12 |
| Custom RFF | same lambdas x gamma `{0.01,0.1}` x components `{128,512}`; 20,000 Pegasos updates | 12 |
| sklearn linear SVC | same three lambdas mapped to C | 3 |
| sklearn RBF SVC | same three lambdas x same two gammas | 6 |

Updates are individual draws, not epochs. These budgets correspond to about
0.125 and 0.498 training-set-size units of draws; replacement sampling does not
guarantee visiting every row. This is a limited optimization budget, especially
for small lambda.

Each family winner maximized validation F1 for premium listings. Ties would be
resolved by validation ROC-AUC, then grid order. Selection was saved before the
first test prediction. Exactly those four models were evaluated once on test
at their unchanged zero score threshold.

Our loss averages the hinge term. SVC minimizes
`0.5*||w||^2 + C*sum(hinge)`, so matched regularization is

\[
C=\frac{1}{n_{train}\lambda}.
\]

For 40,155 training rows, lambdas `1e-4`, `1e-3`, `1e-2` correspond to C values
`0.2490349894`, `0.02490349894`, `0.002490349894`. Linear-kernel SVC avoids the
default squared-hinge/intercept differences of `LinearSVC`. RBF SVC uses the same
gamma grid as RFF, but an exact kernel and a different optimizer. Baselines use
the already train-standardized features, a 256 MiB kernel cache and the recorded
sklearn version's other defaults. No sklearn estimator is used inside the custom
algorithm. Because `src/evaluate.py` remains a stub, the experiment script uses
sklearn metrics independently. ROC-AUC uses decision scores, not hard labels.

## Selected configurations and actual results

All metrics are on a 0–1 scale; the positive class is premium (`1`).

| Family / configuration ID | Selected parameters | Validation F1 | Validation ROC-AUC |
|---|---|---:|---:|
| Custom linear / `custom_linear_003` | lambda=1e-4, T=20,000, constant eta=0.01 | 0.886890 | 0.951508 |
| Custom RFF / `custom_rff_017` | lambda=1e-3, T=20,000, Pegasos, gamma=0.01, D=512 | 0.840682 | 0.934187 |
| Linear SVC / `sklearn_linear_024` | lambda=1e-4, C=0.2490349894 | 0.907798 | 0.963376 |
| RBF SVC / `sklearn_rbf_027` | lambda=1e-4, C=0.2490349894, gamma=0.01 | 0.894755 | 0.966132 |

| Model | Validation accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Custom linear | 0.885983 | 0.875284 | 0.898808 | 0.886890 | 0.951508 |
| Custom RFF | 0.821943 | 0.757354 | 0.944613 | 0.840682 | 0.934187 |
| Linear SVC | 0.908066 | 0.905581 | 0.910026 | 0.907798 | 0.963376 |
| RBF SVC | 0.894816 | 0.890509 | 0.899042 | 0.894755 | 0.966132 |

| Model | Test accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Custom linear | 0.880521 | 0.874251 | 0.887331 | 0.880742 | 0.948866 |
| Custom RFF | 0.828917 | 0.765720 | 0.945068 | 0.845993 | 0.934133 |
| Linear SVC | 0.903301 | 0.904650 | 0.900421 | 0.902530 | 0.963139 |
| RBF SVC | 0.893422 | 0.893836 | 0.891538 | 0.892686 | 0.964495 |

Test confusion matrices have orientation `[[TN,FP],[FN,TP]]`:

| Model | Confusion matrix | Training (s) | Test predict (s) | Test scores (s) |
|---|---|---:|---:|---:|
| Custom linear | `[[3780,546],[482,3796]]` | 0.102832 | 0.000766 | 0.000256 |
| Custom RFF | `[[3089,1237],[235,4043]]` | 0.751286 | 0.072273 | 0.072344 |
| Linear SVC | `[[3920,406],[426,3852]]` | 14.330181 | 1.187338 | 1.189327 |
| RBF SVC | `[[3873,453],[464,3814]]` | 16.225550 | 4.462808 | 4.437761 |

Training times include RFF mapping and objective logging where applicable.
Predict and score extraction are timed separately on all 8,604 test rows;
metric calculation is excluded. Test rows repeat the original fit time. The
33 fits took 213.287 seconds in total, excluding inference, data preparation and
plotting. These are single-run measurements, not benchmark averages. Linear SVC
uses its support-vector prediction path, so its timing does not represent the
fastest possible linear predictor.

Linear SVC had the highest validation F1 and is the overall winner under the
predeclared rule. RBF SVC had higher validation ROC-AUC but lower F1; the
secondary metric does not override the primary metric. Custom linear also
outperformed custom RFF on validation and test F1. This grid supplies no evidence
of an F1 advantage from nonlinearity, but does not establish that nonlinear
models are inherently worse: bandwidth, finite random features and optimization
budget all affect the comparison.

RFF recovered 94.51% of test premium listings but mislabeled 1,237 standard
listings as premium, versus 546 for custom linear. Its lower precision explains
why high recall did not yield the best F1. Linear SVC made 832 errors versus
1,028 for custom linear. No district-specific or luxury-subgroup error claim
is supported by these aggregate confusion matrices.

## Learning rate, iterations and convergence

At fixed lambda `1e-4`, more updates improved validation F1 for both schedules.
The constant schedule performed better within these finite budgets:

| Schedule | Updates | Final training objective | Validation F1 |
|---|---:|---:|---:|
| Pegasos | 5,000 | 10.889866 | 0.790801 |
| Pegasos | 20,000 | 2.687266 | 0.850672 |
| Constant, eta=0.01 | 5,000 | 0.339413 | 0.866490 |
| Constant, eta=0.01 | 20,000 | 0.309367 | 0.886890 |

The selected linear objective decreased from 1 to 0.309367. Its last five
checkpoints ranged from 0.308661 to 0.328767, showing stochastic fluctuations.
The selected RFF objective decreased from 1 to 0.513867; its final five values
ranged from 0.461358 to 0.530478. A lower earlier checkpoint was not substituted
after viewing test results: the predeclared runner uses the final iterate.

**Convergence is not established for every configuration.** In particular,
lambda `1e-4` with Pegasos ends above the zero-initialized objective after 20,000
updates. Its initial rate is 10,000 and final rate is 0.5; large early steps and
the short budget are consistent with slow settling. This is an optimization
limitation, not proof of a subgradient error. Longer training studied exclusively
on development data would be needed to establish satisfactory convergence.
The observed constant-rate result is not a general superiority claim.

## Margin and regularization analysis

| Selected custom model | Final objective | Weight norm | Margin width | Active training points | Active fraction |
|---|---:|---:|---:|---:|---:|
| Linear | 0.309367 | 4.418389 | 0.452654 | 10,532 / 40,155 | 26.2284% |
| RFF | 0.513867 | 12.491877 | 0.160104 | 21,330 / 40,155 | 53.1192% |

Linear widths are in standardized input space; RFF widths are in transformed
space. The selected sklearn linear/RBF models had 10,765/14,170 dual support
vectors, respectively. Dual counts and custom margin-active counts are different
diagnostics; sklearn objective/norm/margin fields are left empty in the CSV.

Holding the linear budget at 20,000 updates and constant eta=0.01:

| Lambda | Weight norm | Margin width | Active fraction | Validation F1 |
|---|---:|---:|---:|---:|
| 1e-4 | 4.418389 | 0.452654 | 0.262284 | 0.886890 |
| 1e-3 | 4.194020 | 0.476869 | 0.270701 | 0.883866 |
| 1e-2 | 2.810818 | 0.711537 | 0.319387 | 0.870623 |

Stronger regularization widened the margin but reduced validation F1 in this
controlled comparison. A wider margin alone does not imply better prediction.
Pegasos runs show a different F1 trend because lambda also changes learning-rate
scale, mixing optimization and regularization effects.

## RFF parameter effects and linear weights

At lambda `1e-3` and 20,000 Pegasos updates:

| Gamma | Components | Validation F1 |
|---|---:|---:|
| 0.01 | 128 | 0.811900 |
| 0.01 | 512 | 0.840682 |
| 0.1 | 128 | 0.728720 |
| 0.1 | 512 | 0.771356 |

Across the tested grid, 512 components improved F1 over 128 at every fixed
lambda/gamma, and gamma=0.01 outperformed gamma=0.1 at every fixed lambda/D.
These observations are specific to this seed and grid. Larger D increases
mapping cost and memory for the materialized `n_train x D` matrix. The gap to
exact RBF SVC can reflect both approximation and incomplete primal optimization.

The selected linear model's largest absolute standardized weights were:

| Feature | Weight |
|---|---:|
| `area_m2` | +3.631352 |
| `cat_house` | -0.992582 |
| `dist_centre_km` | -0.847366 |
| `rooms` | +0.830128 |
| `loc_Masazır q.` | -0.769445 |
| `cat_new_build` | +0.662262 |
| `loc_Mərdəkan q.` | +0.580166 |
| `loc_Şüvəlan q.` | +0.563510 |

The intercept is 0.42. Larger area and more rooms increase the fitted score;
greater distance from the fixed city center decreases it, holding other encoded
inputs fixed. Correlated size/geographic features and redundant indicator coding
limit isolated interpretation. Indicator weights apply to standardized columns,
not directly to a raw 0-to-1 switch. These are conditional model associations,
not causal price effects. All 65 weights are in `linear_weights.csv`.

## Reproducibility, tests and outputs

The environment used Python 3.12.3, NumPy 2.5.4, pandas 3.0.6, sklearn 1.9.1,
matplotlib 3.11.2, SciPy 1.18.1 and pytest 9.1.1. Hardware was an AMD Ryzen 9
8940HX; BLAS/OpenMP thread limits were one. The virtual environment was outside
the repository. Environment details, source hashes and data provenance are
recorded in the result JSON files.

After the grid, the two frozen custom configurations were replayed on **training
only** for reproducibility and weight export. All 51 objective checkpoints,
weight norms, margin widths and active-point diagnostics matched exactly. These
two inexpensive replays did not calculate new validation/test predictions or
change selection. The full baseline grid was not repeated; elapsed times are
not expected to reproduce exactly.

Verification checked the 37-row schema, winners recomputed from validation only,
matching frozen test configurations, finite metrics, confusion counts totaling
each split's size, loss checkpoints, source/NPZ hashes and six readable PNGs.
Every figure was visually inspected. `verification.json` records the checks.
No mathematical or functional SVM bug requiring an implementation change was found.

Actual test command (Python from the temporary project environment):

```bash
MPLCONFIGDIR=/tmp/ml-final-svm-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest -q -p no:cacheprovider -W error
```

Latest result after the CV addition: **140 passed in 2.62s**, no warnings. The
previous 124 tests were preserved. The 16 new CV checks cover final-test
exclusion, fold membership, train-only imputation/encoding/scaling/thresholds,
fixed configurations, the fold-size-dependent C mapping, sample standard
deviations, output preservation and reproducibility. Fit/evaluation spies verify
that each model receives only the correct training/validation rows. Synthetic
fixtures remain test-only; real-data CV was executed once.

For a fresh checkout with the existing cleaned CSV:

```bash
python -m experiments.prepare_svm_data --output data/svm_prepared.npz
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/ml-final-svm-mpl \
  python -m experiments.svm_experiments \
    --data data/svm_prepared.npz --output results/svm
```

Both commands protect existing output. Current artifacts already exist; use a
new NPZ path/output directory for intentional reproduction. If another checkout
has only raw data, Davud's `python -m src.data_prep` creates the cleaned CSV.
Do not refit preprocessing on held-out data or retune using the reported test set.

The runner reproduces canonical metrics, selections, loss histories and figures.
Supplementary comparison/margin/convergence CSVs summarize those canonical files;
weights come from the documented train-only replay. Outputs in `results/svm/`:

```text
metrics.csv                         # 33 validation + 4 test rows
configurations.json                 # grid, provenance, status=complete
selected_configurations.json        # decisions saved before test
loss_histories.json                 # 24 custom models, 51 checkpoints each
comparison.csv                      # winners: validation and test
margin_analysis.csv                 # all 24 custom configurations
convergence_summary.csv             # initial/final/minimum and tail values
linear_weights.csv
selected_linear_state.json
preprocessing_audit.json
execution_environment.json
verification.json
run.log
cv_fold_metrics.csv                 # 5 folds x 4 fixed models
cv_summary.csv
cv_summary.md
cv_assignments.csv
cv_preprocessing.json
cv_manifest.json
cv_verification.json
cv_run.log
figures/
    objective_vs_iterations.png
    rff_objective_vs_iterations.png
    lambda_vs_validation_f1.png
    lambda_vs_margin.png
    linear_vs_nonlinear.png
    rff_parameter_effects.png
```

Convergence axes use a symmetric log scale, linear near zero, retaining large
early losses. Lambda curves separate schedules/budgets; RFF panels fix lambda.
Performance figures use validation F1 only. In the family comparison plot,
“Pegasos linear” names the custom class; its winning schedule is **constant**,
as recorded in the tables and configuration JSON.

## Five-fold cross-validation uncertainty — completed

`experiments/svm_cross_validation.py` adds a separate development-only study.
It uses the **48,759 original training + validation rows** and excludes all
**8,604 final-test rows** using the saved NPZ row IDs. The NPZ supplies only
membership and provenance; its previously fitted feature arrays, labels,
encoder statistics and tier threshold are not reused. Final-test records are
skipped when reading the cleaned CSV, and no final-test prediction or metric is
recomputed. The original 33-configuration results remain unchanged.

Five shuffled folds are stratified on development-price deciles with seed 42.
Deciles determine membership only; no global median is used to define the
classification labels. Every development row is validated once and belongs to
the training complement of the other four folds. Training folds contain
39,007 rows (folds 1–4) or 39,008 (fold 5); validation folds contain 9,752 or 9,751
rows, respectively. `cv_assignments.csv` records cleaned-row IDs and validation
fold numbers so all memberships can be reconstructed.

The following operations are performed independently inside every fold:

1. Fit numeric imputation medians on training rows only and apply them unchanged
   to validation. This is a safeguard after Davud's fixed structural fills and
   complete-case cleaning. In this real run **zero cells needed imputation** in
   either part of every fold. Targets are never imputed; entirely missing
   training numeric columns cause an explicit error instead of a validation
   fallback.
2. Fit a new Davud `FeatureEncoder` on the imputed training rows. Its fixed
   residential categories and train-frequency/unknown-location policy are
   reused for validation. No vocabulary is learned from the validation fold.
3. Call Davud's `make_tier_label` on training prices to obtain the threshold,
   then apply that same value to validation prices. All five independently
   computed medians happened to equal **209,000 AZN**; this value was not fixed
   globally or copied from the original experiment.
4. Call Davud's `standardize` for custom models using only training moments.
   Each sklearn baseline instead fits a fresh `Pipeline(StandardScaler, SVC)`
   on the same encoded/imputed training matrix. Its scaler moments are checked
   against the custom path. This satisfies the baseline Pipeline requirement
   while retaining identical fold data and scaling conventions.

Every fold produced 65 features. Numeric medians, encoder vocabulary, feature
order, scaler mean/std, threshold, missing-value counts and class balance are
saved in `cv_preprocessing.json`. Validation never fits these quantities.

The four configurations are exactly the saved family winners, not a new grid.
Custom linear retains constant eta=0.01, lambda=1e-4 and 20,000 updates; RFF
retains Pegasos, lambda=1e-3, gamma=0.01, D=512 and 20,000 updates. Both model
seeds remain 42. Both SVC baselines retain lambda=1e-4; RBF retains gamma=0.01.
Because the common objective uses mean hinge loss, C is deterministically
`1/(n_fold_train*lambda)`: **0.2563642423** for folds 1–4 and **0.2563576702**
for fold 5. This preserves lambda rather than retuning C.

For each metric, the table gives the unweighted mean of five validation-fold
scores and the **sample** standard deviation:

\[
\bar m=\frac{1}{5}\sum_{k=1}^{5}m_k,\qquad
s=\sqrt{\frac{\sum_{k=1}^{5}(m_k-\bar m)^2}{5-1}}.
\]

Scores use premium as the positive class and ROC-AUC uses decision scores.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Custom linear | 0.886380 ± 0.004216 | 0.891169 ± 0.008071 | 0.878959 ± 0.015661 | 0.884905 ± 0.005319 | 0.950618 ± 0.002465 |
| Custom RFF | 0.838799 ± 0.005704 | 0.847076 ± 0.013997 | 0.825362 ± 0.028778 | 0.835650 ± 0.009455 | 0.931868 ± 0.001048 |
| Linear SVC | 0.906315 ± 0.001945 | 0.906379 ± 0.001349 | 0.905034 ± 0.003327 | 0.905704 ± 0.002065 | 0.962505 ± 0.001697 |
| RBF SVC | 0.893599 ± 0.001200 | 0.892487 ± 0.005337 | 0.893689 ± 0.006249 | 0.893059 ± 0.001329 | 0.964539 ± 0.001066 |

Individual F1 values, also available alongside all other metrics and confusion
counts in `cv_fold_metrics.csv`, are:

| Fold | Custom linear | Custom RFF | Linear SVC | RBF SVC |
|---|---:|---:|---:|---:|
| 1 | 0.890459 | 0.839363 | 0.903739 | 0.892595 |
| 2 | 0.879082 | 0.818814 | 0.904944 | 0.891107 |
| 3 | 0.889979 | 0.841355 | 0.909035 | 0.894176 |
| 4 | 0.884865 | 0.839703 | 0.904580 | 0.893020 |
| 5 | 0.880143 | 0.839016 | 0.906221 | 0.894397 |

Linear SVC has the highest mean F1, and RBF SVC the highest mean ROC-AUC.
Custom linear exceeds custom RFF in F1 in every fold. RFF's F1 standard
deviation (0.009455) is larger than the custom linear value (0.005319);
fold 2 has its lowest F1. This confirms sensitivity to fold composition at
the fixed optimization/projection settings, without proving a particular
cause. A larger RFF feature map or longer training was not tested in this CV
study. No model or hyperparameter was changed in response to CV or test results.

Mean elapsed times per fold, in seconds:

| Model | Fit | Predict validation fold | Score validation fold |
|---|---:|---:|---:|
| Custom linear | 0.100212 | 0.000621 | 0.000372 |
| Custom RFF | 0.685437 | 0.083750 | 0.082293 |
| Linear SVC | 10.862185 | 1.268104 | 1.281162 |
| RBF SVC | 14.832070 | 4.476268 | 4.472651 |

The 20 model fits total **132.400 seconds**, excluding shared fold preparation,
validation inference and output writing. Fold preparation time is recorded
separately and repeated on each family's row; it must not be summed four times.
Baseline fit/predict times include their Pipeline scaling; custom fit includes
RFF mapping where applicable, while shared input standardization is outside that
timer. BLAS/OpenMP threads were limited to one. These remain single-run timings.

These standard deviations describe variation across development folds, not a
confidence interval or the standard error of the previously reported final-test
score. Training sets overlap across folds. The configurations were originally
selected using the same development pool, so this is **fixed-configuration CV,
not nested CV**; some selection optimism can remain. Model seed 42 was held
constant rather than estimating variation across independent random seeds.
The original held-out test table remains the separate final evaluation.

Reproduce the study from the repository root using the already saved selections
and split provenance:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.svm_cross_validation \
    --clean data/house_sale_clean.csv \
    --prepared data/svm_prepared.npz \
    --selected results/svm/selected_configurations.json \
    --output results/svm
```

Existing CV files are protected. For a deliberate reproduction, choose a new
output directory; the main metrics and selected configurations are never
overwritten. `cv_manifest.json` records all fixed inputs, source hashes, versions,
seed, policies and completed status. `cv_verification.json` records the independent
membership, mean/std, threshold, C and confusion-count checks plus the actual
140-test result. The real 20-fit CV was run once; complete repeated-run metric
reproducibility was checked using the focused synthetic test fixtures.

## Handoff and remaining requirements

Bailar can consume `metrics.csv` rows with `split="test"`, join their IDs with
`selected_configurations.json`, and use the saved figures in the shared report.
`comparison.csv` combines winners' validation/test rows. Use the same seed-42
split and train-derived 209,000 AZN threshold for comparable team models.
Prepared features are already standardized: do not scale them again.
`src/run_all.py` and `src/evaluate.py` remain team-owned integration stubs.

For uncertainty, Bailar can import `cv_summary.csv` or copy `cv_summary.md` beside
the original test table. Label it development 5-fold mean ± sample standard
deviation. Do not attach these deviations to the old test scores as if they were
test confidence intervals. The fold-level CSV provides the supporting values.

The 33-configuration experiment and SVM-specific five-fold uncertainty analysis
are complete. Remaining limitations and team integration work are:

- The CV estimate is conditional on the previously selected configurations.
  An unbiased assessment of the entire tuning procedure would require nested
  selection within each outer training fold; that is not claimed here.
- Small-lambda Pegasos convergence is insufficient at the declared update budget.
  Future iteration/schedule studies must use development data only; current
  test results must not guide a replacement hyperparameter search.
- A single seed does not establish robustness across projections or splits.
  Larger RFF maps and longer optimization are possible improvements, not
  verified gains. Timing comparisons are single-run measurements.
- Confusion matrices and weights support aggregate error interpretation.
  District, temporal and luxury-outlier error studies were not run. Listing-URL
  deduplication cannot prove that different ads never describe the same physical
  property; group/time splits address a different generalization question.
- Data and `results/` are Git-ignored. Bailar must receive artifacts through the
  team's agreed handoff or reproduce them. No raw data, temporary environment
  or generated results were staged. No commit or push occurred.

Methodological references: Shalev-Shwartz, Singer, Srebro and Cotter,
*Pegasos: Primal Estimated Sub-Gradient Solver for SVM* (2011); Rahimi and Recht,
*Random Features for Large-Scale Kernel Machines* (2007). Official requirements
are in `PROJECT_STATEMENT.pdf`, Problems 3–7. The shared report should retain the
required AI-assistance disclosure and contribution attribution.

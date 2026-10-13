# SVM methodology and experiment handoff

## Verified status — 14 October 2026

The existing NumPy SVM implementation and its 84 tests were preserved. A
standalone SVM experiment runner and focused experiment tests have been added.
**No real Bina.az experiment has been executed in this checkout.** No real-data
validation scores, test scores, best configuration, convergence conclusion, or
performance advantage can currently be reported.

Repository inspection found:

- `data/` contains only `README.md`; no raw CSV, cleaned CSV, NPZ, NPY, or
  Parquet dataset is present. No previous experiment results were found.
- `src/data_prep.py` implements cleaning, including duplicate removal, numeric
  parsing, structural missing-value fills, fixed outlier rules, and removal of
  known target-derived columns. It retains `price` as the target and categorical
  `category`/`location` columns that still require feature construction.
- `make_features`, `make_tier_label`, `train_val_test_split`, and `standardize`
  still raise `NotImplementedError`. Completed preprocessing is not available
  here. Its documented interfaces promise training-only scaling and a
  training-price median threshold, but those promises cannot yet be verified.
- `src/evaluate.py` contains unimplemented metric functions. The experiment
  script therefore uses sklearn metrics, as permitted for this task; sklearn
  remains outside the custom SVM algorithm.
- `src/run_all.py` is an unimplemented integration skeleton.
- The data README's example filename differs from `data_prep.DATA_PATH`; the
  eventual export should record the actual dataset file/version used.

Running `python -m experiments.svm_experiments` without prepared data exits with
an explicit missing-data error and does not create metrics or figures. Synthetic
fixtures are confined to tests and labelled as such, with artifacts written to
pytest temporary directories. They are not evidence about Bina.az performance.

Verification on 14 October 2026: **120 tests passed in 2.18 seconds**, consisting
of the unchanged 84 SVM tests and 36 experiment checks. The final run treated
warnings as errors (`-W error`) and completed without warnings. The checks
included actual NumPy/sklearn fits, CSV/JSON and six PNG outputs, reproducible
validation results, and unchanged selected configurations after inverting only
the synthetic test labels. No existing repository file changed during this task,
as verified against its starting SHA-256 hash. `pip check` found no broken
requirements in the temporary verification environment.

## Linear soft-margin SVM

For input labels in `{0, 1}`, the implementation internally maps `0` to `-1`
and `1` to `+1`. It learns the score `f(x) = w^T x + b` by minimising

\[
J(w,b)=\frac{\lambda}{2}\|w\|_2^2+
\frac{1}{n}\sum_{i=1}^{n}\max(0,1-y_i(w^T x_i+b)).
\]

The hinge term is zero for functional margins at least one. It penalises both
misclassified points and correctly classified points inside the margin. The
quadratic term discourages large weights. Bias is a separate scalar and is
neither regularised nor projected. A score below zero predicts standard (`0`);
a zero or positive score predicts premium (`1`). Scores are not probabilities.

At iteration `t`, one training observation is drawn uniformly with replacement.
The default Pegasos learning rate is `eta_t = 1 / (lambda * t)`. The hinge-active
condition is evaluated before changing the weights. If the margin is below one,

\[
w \leftarrow (1-\eta_t\lambda)w+\eta_t y_i x_i,
\qquad b\leftarrow b+\eta_t y_i.
\]

Otherwise only weight shrinkage is applied. At a margin of exactly one, the
implementation chooses the zero hinge subgradient. After the update, weights
are projected onto the Euclidean ball of radius `1 / sqrt(lambda)`; the bias is
excluded. Initial weights and bias are zero. A local seeded NumPy generator
controls the draws, and every `fit` restarts training.

The constant-rate comparison uses `eta_t = eta0` with the same objective,
subgradients, and projection. It may oscillate near a solution. A finite training
budget does not certify the global optimum. In particular, the unregularised
bias means the objective is not jointly strongly convex in `(w,b)`; the usual
strongly convex Pegasos rate should not be asserted without qualification.

## Objective and margin diagnostics

`loss_history_` stores the full training objective, including regularisation,
at the steps in `loss_iterations_`. The default records at most 51 checkpoints,
including the initial objective of one and the final iteration. These are not
mini-batch loss estimates. Stochastic fluctuations are expected; monotonic
decrease is not required. Objective values at different lambdas correspond to
different functions and should not be treated as directly interchangeable
convergence curves or as the model-selection criterion.

The weight norm is `||w||_2`; the distance between the score planes `-1` and `+1`
is `2 / ||w||_2`. Larger regularisation generally favours smaller weights and a
wider margin, with a possible classification-performance tradeoff. That trend
must be checked empirically rather than assumed for every finite SGD run.

Training points satisfying `y_i * f(x_i) <= 1` are reported as margin-active
points, with their count and fraction. These include boundary points and errors;
they are not recovered nonzero coefficients of a dual SVM. If weights are zero,
the implementation returns an infinite, degenerate margin. The experiment CSV
uses an empty cell (`null` in JSON) for that undefined geometric interpretation.
A weight norm of zero identifies this case.

## Nonlinear extension through random Fourier features

For the Gaussian kernel `k(x,x') = exp(-gamma * ||x-x'||^2)`, the feature map is

\[
z(x)=\sqrt{2/D}\cos(xW+c),\quad
W_{jk}\sim\mathcal{N}(0,2\gamma),\quad c_k\sim U(0,2\pi).
\]

The frequency standard deviation is `sqrt(2 * gamma)`, not `gamma`. Frequencies
and phases are sampled during `fit`, stored, and reused during inference. The
same linear Pegasos optimiser then operates on `z(X)`. No kernel matrix or dual
optimiser is implemented. RFF provides a finite-dimensional approximation to
an RBF kernel; increasing `D` changes approximation quality and cost. The norm
and margin for this model belong to RFF space, not the original feature space.
RFF weights therefore cannot be interpreted as original property-feature weights.

## Prepared-data contract for Davud and Bailar

The runner consumes the output of the team's existing preprocessing. It does
not implement a second cleaning, encoding, split, scaling, or threshold pipeline.
This is the proposed export contract; no completed upstream producer is assumed.

Create `data/svm_prepared.npz` after completing and reviewing preprocessing:

| Key | Required content |
|---|---|
| `X_train`, `X_val`, `X_test` | Finite dense numeric arrays, shape `(n_split, n_features)`, with identical feature order |
| `y_train`, `y_val`, `y_test` | One-dimensional labels in `{0,1}`; both classes in every split for ROC-AUC |
| `train_ids`, `val_ids`, `test_ids` | Unique stable listing IDs, disjoint across splits; separate from model features |
| `feature_names` | Unique nonempty Unicode names matching the columns of `X` |
| `metadata_json` | A scalar Unicode string containing the provenance JSON described below |

All features must already be standardised with the **training** mean and
population standard deviation (`ddof=0`). Zero-variance features may be zero.
The script checks training moments, but does not require held-out moments to be
zero/one and never recentres the held-out arrays. If the team deliberately scales
only some columns, agree on an updated validation contract before running this
script; do not claim an unscaled matrix meets the current contract.

The producer must perform the following:

1. Deduplicate listings before splitting, preserve stable IDs, and document the
   classification stratification strategy required by the brief. Do not invent
   different IDs for repeated copies of the same listing.
2. Derive the premium threshold from training prices only; apply that exact
   threshold to validation and test prices. Do not derive a full-data median
   merely to facilitate stratification.
3. Fit all learned encoding, imputation and scaling statistics on training only.
   Handle unseen held-out categories with the fitted encoder's fixed policy.
4. Exclude `price`, `log_price`, `unit_price`, `total_price`, tier labels, and any
   other target-derived feature. The script catches those known names, but a
   renamed or engineered leakage feature still requires source review.

Example **export only**, using arrays and IDs already produced by Davud's code:

```python
import json
import numpy as np

metadata = {
    "dataset_name": "Bina.az sale listings — REPLACE with actual dataset version",
    "preprocessing_source": "REPLACE with producer module and exact Git commit",
    "split_strategy": "REPLACE with actual splitting/stratification method and seed",
    "scaling_fit_split": "train",
    "encoding_fit_split": "train",
    "tier_threshold_fit_split": "train",
    "tier_threshold": float(threshold),  # already calculated from training prices
}
np.savez_compressed(
    "data/svm_prepared.npz",
    X_train=X_tr_std, y_train=y_tr_tier,
    X_val=X_val_std, y_val=y_val_tier,
    X_test=X_te_std, y_test=y_te_tier,
    train_ids=np.asarray(ids_tr, dtype=str),
    val_ids=np.asarray(ids_val, dtype=str),
    test_ids=np.asarray(ids_te, dtype=str),
    feature_names=np.asarray(feature_names, dtype=str),
    metadata_json=json.dumps(metadata),
)
```

NPZ files are loaded with `allow_pickle=False`. Metadata is a declaration, not
proof: arrays alone cannot establish which rows fitted an upstream encoder or
scaler, whether the threshold is the actual training median, whether IDs were
assigned correctly, or whether unknown target-derived columns remain. Review
the recorded preprocessing source before describing the full pipeline as
leakage-free. No producer statistics are learned from test data by this runner.

## Experiment setup and predeclared selection

The default random seed is `42`. The same prepared training and validation
arrays are used for every model, with no resplitting or subsampling. Models are
trained only on training rows. The grid is deliberately bounded:

| Family | Configurations | Fits |
|---|---|---:|
| Custom linear | lambda `{1e-4, 1e-3, 1e-2}` × updates `{5000, 20000}` × schedule `{Pegasos, constant(0.01)}` | 12 |
| Custom RFF | same lambdas × gamma `{0.01, 0.1}` × components `{128, 512}`, 20000 Pegasos updates | 12 |
| Linear-kernel sklearn SVC | same three lambdas mapped to C | 3 |
| RBF sklearn SVC | same three lambdas × same two gammas | 6 |

Total: **33 fits**. Iteration counts are stochastic draws, not full data passes;
the effective pass budget is `n_iters / n_train`. RFF uses the maximum requested
iteration budget. These settings are an initial declared search, not parameters
already validated on Bina.az. CLI flags allow the grid to be changed before test
evaluation. A very small lambda may require more iterations for a useful result.

Each family's winner maximises validation F1 for the premium class. Ties are
broken by validation ROC-AUC, then by the first configuration in grid order.
Training objective, runtime, and test performance do not participate in selection.
The script saves all selected configurations before making any test prediction.
It then evaluates exactly those four already fitted models, once each, without
refitting on validation or changing thresholds. Keeping separate family winners
supports an honest linear/nonlinear and custom/sklearn comparison; a deployment
choice among them must also use validation evidence, not their test ranking.

## Matched sklearn baselines and metrics

The custom objective uses the **mean** hinge loss. SVC uses
`0.5 * ||w||^2 + C * sum(hinge)`. Multiplying the custom objective by `1/lambda`
gives the same minimiser when

\[
C=\frac{1}{n_{train}\lambda}.
\]

The baseline uses `SVC(kernel="linear")` rather than the default `LinearSVC`,
whose default squared-hinge loss and intercept treatment would complicate an
exact objective comparison. RBF SVC uses the same numerical gamma as RFF, but
operates on the exact RBF kernel; approximation and optimiser differences are
expected. Baseline probability calibration is not requested. ROC-AUC uses raw
decision scores. All settings other than the recorded grid use sklearn defaults
for the recorded version, with a 256 MiB kernel cache.

Accuracy, binary precision, recall, F1, ROC-AUC and confusion counts are saved
for every validation configuration and each selected test configuration.
Confusion-matrix orientation is `[[tn, fp], [fn, tp]]`. Undefined precision/recall
from an empty predicted-positive set is recorded as zero. Both true classes
are required in each split, avoiding an undefined ROC-AUC.

`training_seconds` measures one model's `fit`, including RFF mapping and objective
logging where applicable. `prediction_seconds` measures `predict` only;
`decision_seconds` separately measures score extraction. Metric computation is
excluded. Test rows repeat the corresponding training time; they are not extra
fits. Timings naturally vary between executions and are not reproducibility
assertions. NumPy/sklearn/matplotlib/Python versions, source hashes, NPZ hash
(for CLI runs), input provenance, split sizes, and all configurations are recorded.

SVC's `n_support_vectors` is recorded separately. Explicit weight/objective/margin
diagnostics are left empty for sklearn models to avoid confusing dual support
coefficients or RKHS geometry with the custom model's explicit weights.

## Execution, outputs, and main-runner integration

Install the repository's pinned requirements in the team's environment; pytest
is additionally needed for development tests and is not currently listed in
`requirements.txt`. No shared dependency file was changed. The verification
environment is `/tmp/ml-final-svm-venv`, outside the repository; it is temporary.

Run from the repository root:

```bash
python -m experiments.svm_experiments \
    --data data/svm_prepared.npz --output results/svm
python -m pytest tests/test_svm.py tests/test_svm_experiments.py -q
```

For a headless restricted environment, set `MPLCONFIGDIR` to a writable temporary
directory. The plotting backend is `Agg`; no GUI or browser is required.

Once real data is supplied, the runner writes:

```text
results/svm/
    metrics.csv
    configurations.json
    selected_configurations.json
    loss_histories.json
    figures/
        objective_vs_iterations.png
        rff_objective_vs_iterations.png
        lambda_vs_validation_f1.png
        lambda_vs_margin.png
        linear_vs_nonlinear.png
        rff_parameter_effects.png
```

The convergence plot compares schedules/budgets at a fixed linear lambda; a
separate plot shows the selected RFF objective. Convergence axes use a symmetric
log scale (linear near zero) to keep large initial SGD losses visible. Lambda
plots keep schedules/budgets separate. RFF panels fix lambda and show gamma/D
effects. All performance figures use validation F1, not test scores. Numerical
histories are also saved for alternate report plots.

CSV rows identify `split` as `validation` or `test`. All configurations are saved;
the JSON manifest reaches `status="complete"` only after test evaluation and
figure writing finish. Partial failures retain already computed rows and a
non-complete status; they must not be reported as a completed experiment.
Nonempty output directories are rejected to preserve existing work. Use a new
output directory for a deliberate reproduction, and do not retune after seeing
test results. `results/` and prepared data are already ignored by Git; do not
force-add raw data or temporary test environments.

Bailar can call `main` from the main runner with an explicit argument list, or
call `run_experiments(data, metadata, build_configurations(), output)` using the
same arrays in memory. Alternatively, consume the completed `metrics.csv` test
rows and copy chosen figures into the shared report workflow. Do not fit a second
scaler on these prepared arrays, change the price threshold, or introduce a new
split for baselines. This change does not edit `src/run_all.py` or the shared report.

## Results and conclusions currently supported

| Requested result | Verified status |
|---|---|
| Best real-data linear and RFF configurations | Not selected: prepared data unavailable |
| Real validation/test accuracy, precision, recall, F1, ROC-AUC, confusion matrix | Not computed |
| Real training/prediction timings | Not measured |
| Real objective convergence and margin analysis | Not observed |
| Linear versus RFF or sklearn performance advantage | No empirical claim possible |
| PNG/CSV/JSON output and train/validation/test control flow | Verified on labelled synthetic unit-test fixtures only |

After a real run, replace unavailable result statements using the completed
artifacts. Describe objective fluctuations and final values without claiming
monotonic convergence. Discuss margin changes together with validation F1,
and distinguish wider margins from better generalisation. Compare custom/RBF
differences as a combination of finite RFF approximation and optimisation; do
not attribute all differences to implementation correctness. In a report table,
include the selected parameters, all requested test metrics, and measured times.

## Limitations and remaining official requirements

- The missing prepared dataset and incomplete upstream code prevent final
  real-data analysis and an end-to-end leakage audit. Assertions about numeric
  matrices, 0/1 targets and correct scaling remain conditional on a future export.
- The brief also asks for uncertainty through cross-validation. This runner is
  a single held-out study and does **not** produce cross-validated mean/std.
  Cross-validating a matrix scaled once on all development rows would leak fold
  information. Davud/Bailar must supply fold-specific preprocessing that fits
  encoders, scalers and the documented tier-threshold procedure inside each fold,
  with the outer test set sealed. The baseline `Pipeline` requested in the brief
  belongs in that upstream raw-feature workflow; this fixed-split runner consumes
  identical already-standardized arrays and deliberately does not scale them twice.
- Exact kernel SVC can be expensive on large data. No silent subsampling or
  arbitrary early termination is used. Reduce the declared grid if needed before
  opening the test set, and report that change and the actual training sample size.
- RFF materialises an `n_train × D` array. More components increase memory and
  transform time. Projection randomness and finite SGD budgets also affect results.
- A single seed does not establish robustness across seeds. Timing comparisons
  are single-run measurements and depend on hardware, BLAS threads and objective
  logging. A future uncertainty study must respect the same leakage controls.
- Original-feature weight/error analysis and real estate subgroup conclusions
  require actual feature names and data; no such conclusions are fabricated here.

The mathematical references are Shalev-Shwartz, Singer, Srebro and Cotter,
*Pegasos: Primal Estimated Sub-Gradient Solver for SVM* (2011), and Rahimi and
Recht, *Random Features for Large-Scale Kernel Machines* (2007). The project
brief is `PROJECT_STATEMENT.pdf`, Problems 3–7. Team report authors should retain
the project's required AI-assistance disclosure and accurately attribute work.

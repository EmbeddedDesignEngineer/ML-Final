# Evaluation and error analysis — Sabuhi's report material

## Data and procedure

The real Bina.az raw dataset hash is
`b6a4c67e0d3c712de31b9db53db53688792bc03099a213c1994c43114758aec2`.
Davud's unchanged cleaner produced 57,363 residential Baku listings and 65
features. Seed 42 gives 40,155 training, 8,604 validation and 8,604 final-test
rows. The training-price median is 209,000 AZN; premium means strictly above
that value. Regression here predicts price directly in AZN, without a log
transformation. No raw listings or individual predictions are committed.

Tree settings were selected with Senan's validation-only helper over depths
2, 4, 8 and 12, leaf sizes 1, 5 and 20, and Gini/entropy for classification.
RMSE selects regression settings; F1 selects classification settings. Custom
and sklearn models are selected independently from the same grid. The final
custom regressor uses depth 12,
leaf size 5; sklearn uses
depth 8, leaf size
5. Their final settings
therefore differ: do not attribute the entire final score gap to implementation.
Matched-configuration validation results are in `tree_validation.csv`.

The four SVM settings reuse Farid's recorded validation winners, without
retuning from test results. Custom linear uses lambda=0.0001, 20,000 updates
and constant eta=0.01. Custom RFF uses lambda=0.001, 20,000 Pegasos updates,
gamma=0.01 and 512 features. SVC baselines use lambda=0.0001, mapped to
C=1/(n_train*lambda); RBF gamma is 0.01. All SVMs use the same features scaled
from training statistics. Test predictions use models fitted on training only.
The RFF and exact RBF winners have different selected regularization strengths;
their final score gap does not isolate the effect of the RFF approximation.

## Metrics

MAE is the mean absolute price error. RMSE gives large errors more weight.
R² compares squared prediction error with predicting the mean. Precision,
recall and F1 refer to premium class 1. Confusion matrices have true classes
on rows and predictions on columns: [[TN, FP], [FN, TP]]. ROC-AUC uses SVM
decision scores or tree class-1 probabilities; tied scores receive average
ranks. NumPy metric outputs were tested against hand calculations and sklearn.

## Held-out test results

| Model | RMSE (AZN) | MAE (AZN) | R² |
| --- | --- | --- | --- |
| Custom tree | 109163.81 | 45555.13 | 0.7595 |
| sklearn tree | 119375.81 | 52230.16 | 0.7124 |

| Model | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- |
| Custom tree | 0.9286 | 0.9301 | 0.9293 | 0.9486 |
| sklearn tree | 0.9287 | 0.9287 | 0.9287 | 0.9489 |
| Custom linear SVM | 0.8743 | 0.8873 | 0.8807 | 0.9489 |
| Custom RFF SVM | 0.7657 | 0.9451 | 0.8460 | 0.9341 |
| Linear SVC | 0.9047 | 0.9004 | 0.9025 | 0.9631 |
| RBF SVC | 0.8938 | 0.8915 | 0.8927 | 0.9645 |

Under the fixed settings, Custom tree has the highest observed test
F1 (0.9293). This is a descriptive test comparison, not a new selection
rule. The custom tree makes 605 tier errors; the
custom linear SVM makes 1028. The RFF model
has precision 0.7657 and recall 0.9451; its
1237 false positives show why high recall alone does not imply high F1.

## Five-fold uncertainty

| Model | Metric | Five-fold mean | Sample std |
| --- | --- | --- | --- |
| Custom tree | rmse | 119674.73 | 2953.66 |
| Custom tree | mae | 47249.73 | 1243.76 |
| Custom tree | r2 | 0.7340 | 0.0117 |
| sklearn tree | rmse | 123507.54 | 2965.36 |
| sklearn tree | mae | 52608.71 | 1209.55 |
| sklearn tree | r2 | 0.7167 | 0.0132 |
| Custom tree | precision | 0.9239 | 0.0053 |
| Custom tree | recall | 0.9255 | 0.0062 |
| Custom tree | f1 | 0.9247 | 0.0029 |
| Custom tree | roc_auc | 0.9403 | 0.0027 |
| sklearn tree | precision | 0.9243 | 0.0044 |
| sklearn tree | recall | 0.9258 | 0.0058 |
| sklearn tree | f1 | 0.9250 | 0.0028 |
| sklearn tree | roc_auc | 0.9414 | 0.0019 |
| Custom linear SVM | precision | 0.8912 | 0.0081 |
| Custom linear SVM | recall | 0.8790 | 0.0157 |
| Custom linear SVM | f1 | 0.8849 | 0.0053 |
| Custom linear SVM | roc_auc | 0.9506 | 0.0025 |
| Custom RFF SVM | precision | 0.8471 | 0.0140 |
| Custom RFF SVM | recall | 0.8254 | 0.0288 |
| Custom RFF SVM | f1 | 0.8357 | 0.0095 |
| Custom RFF SVM | roc_auc | 0.9319 | 0.0010 |
| Linear SVC | precision | 0.9064 | 0.0013 |
| Linear SVC | recall | 0.9051 | 0.0033 |
| Linear SVC | f1 | 0.9057 | 0.0020 |
| Linear SVC | roc_auc | 0.9625 | 0.0017 |
| RBF SVC | precision | 0.8925 | 0.0053 |
| RBF SVC | recall | 0.8937 | 0.0062 |
| RBF SVC | f1 | 0.8931 | 0.0013 |
| RBF SVC | roc_auc | 0.9645 | 0.0011 |

Each fold uses the original training-plus-validation pool, excluding all final
test rows. Encoding, scaling and the tier threshold are refitted on each fold's
training rows. Settings stay fixed. Standard deviations use ddof=1. These are
descriptive fold-to-fold variations for settings already selected with the
original validation set; this is not nested cross-validation and must not be
claimed as an unbiased estimate of the whole model-selection process. The
held-out test results remain separate from CV means.

## Expensive properties and location coverage

Expensive means price > 455,000 AZN, the original
training 90th percentile. Sparse means the location has fewer than 200 training
listings, including unseen locations. These rules were fixed before test
evaluation. Location names were recovered with the exact shared split IDs,
with price-array equality checks to prevent metadata misalignment.

| Custom tree subgroup | Test count | MAE (AZN) | RMSE (AZN) | Bias (AZN) |
| --- | --- | --- | --- | --- |
| Expensive | 856 | 177315.24 | 304984.77 | -56805.94 |
| Other | 7748 | 30998.26 | 54377.50 | 6655.22 |
| Frequent | 7847 | 44672.94 | 103724.34 | -743.68 |
| Sparse | 757 | 54699.85 | 154662.37 | 11591.09 |

For the custom regressor, expensive-listing MAE is 177315.24 AZN
versus 30998.26 for other listings. Sparse-location MAE is
54699.85 versus 44672.94 for frequent locations.
Signed bias is predicted minus actual price: negative values mean average
underprediction. Different price ranges and group sizes can explain part of
these differences; the analysis does not establish a causal location effect.
See `error_groups.csv` for every model and individual location, with counts.

| Model | Location coverage | Count | Error rate | F1 |
| --- | --- | --- | --- | --- |
| Custom tree | Frequent | 7847 | 0.0679 | 0.9351 |
| Custom tree | Sparse | 757 | 0.0951 | 0.7943 |
| Custom linear SVM | Frequent | 7847 | 0.1156 | 0.8903 |
| Custom linear SVM | Sparse | 757 | 0.1598 | 0.6553 |
| Custom RFF SVM | Frequent | 7847 | 0.1703 | 0.8535 |
| Custom RFF SVM | Sparse | 757 | 0.1797 | 0.6881 |

Single-class subgroups have no ROC-AUC; those cells are left blank rather than
replaced with zero. Small location groups should not be ranked as reliable
evidence without considering their counts and uncertainty.

## Feature interpretation

### Custom tree (regression)

| Feature | Value |
| --- | --- |
| area_m2 | 0.633433 |
| dist_centre_km | 0.163320 |
| lng | 0.049487 |
| lat | 0.042925 |
| total_floors | 0.039704 |
### Custom tree (classification)

| Feature | Value |
| --- | --- |
| area_m2 | 0.582515 |
| dist_centre_km | 0.149929 |
| lng | 0.064988 |
| lat | 0.035117 |
| total_floors | 0.025213 |
### Custom linear SVM (classification)

| Feature | Value |
| --- | --- |
| area_m2 | 3.631352 |
| cat_house | -0.992582 |
| dist_centre_km | -0.847366 |
| rooms | 0.830128 |
| loc_Masazır q. | -0.769445 |

Tree values are normalized training impurity reductions. Linear SVM values are
weights on standardized inputs, not price changes in AZN. Correlated variables
and indicator columns affect interpretation. Importance and weights are not
causal effects. RFF weights are excluded from original-feature explanations
because they describe random transformed features.

## Timing and limits

| Model | Fit (s) | Test predict (s) | Test scores (s) |
| --- | --- | --- | --- |
| Custom tree (regression) | 3.3127 | 0.0200 | - |
| sklearn tree (regression) | 0.2138 | 0.0019 | - |
| Custom tree (classification) | 2.8848 | 0.0203 | 0.0211 |
| sklearn tree (classification) | 0.3111 | 0.0055 | 0.0022 |
| Custom linear SVM (classification) | 0.2865 | 0.0012 | 0.0010 |
| Custom RFF SVM (classification) | 1.1100 | 0.1011 | 0.0987 |
| Linear SVC (classification) | 43.8000 | 2.8892 | 2.9084 |
| RBF SVC (classification) | 54.0628 | 10.8794 | 10.8997 |

These are single-run wall-clock measurements. Fit time includes each model's
own fitting operations; the common scaling step is outside these times. Score
extraction is timed separately from prediction; metric calculation is excluded.
Values depend on hardware and load and are not benchmark averages.

Results apply to the cleaned residential Baku population. Fixed cleaning rules
exclude some extreme or incomplete listings. Price targets remain heavy-tailed.
This evaluation does not include the optional Ridge bonus. The team's final
report, combined slides and `src/run_all.py` integration remain shared tasks.

## Reproduction and evidence

Run from the repo root after installing requirements and placing the raw data
at `data/house_sale.csv`:

```powershell
python -m src.data_prep
python -m experiments.evaluation_experiments --output results/evaluation_reproduction
python -m pytest -q
```

The output folder must be empty. Seeds and metric results are deterministic;
wall-clock timings are not. `manifest.json` records data/source hashes and
versions. The final report should cite the dataset and algorithm references
already documented by the team, and incorporate this section with the other
members' content.

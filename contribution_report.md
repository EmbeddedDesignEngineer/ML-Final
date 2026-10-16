# Contribution Report

Keep this file in the repo root. It is checked at the demo. Be honest and
specific — "helped with everything" is not acceptable. One row per member,
plus a short per-member paragraph.

## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Khamiyev, Sabuhi | To be confirmed | Evaluation metrics, metric tests, model evaluation, plots and subgroup error analysis |
| Bayramov, Bailar | bailar-dev | Ridge bonus, experiment runner, reproducibility and integration tests |
| ---                    | ---           | ---                   |
| ---                    | ---           | ---                   |
| ---                    | ---           | ---                   |

## Per-member summary
**Khamiyev, Sabuhi —** Evaluation work covers the seven NumPy metrics/comparison
functions in `src/evaluate.py`, plotting and subgroup summaries in
`src/evaluation_plots.py`, and their tests. The evaluation experiment connects
the team's data, tree and SVM APIs, preserves validation-only model selection,
and produces real-data results, five-fold uncertainty summaries and report
material. See `experiments/evaluation_experiments.py` and `evaluation/`.
Personal code review and demo preparation remain the member's responsibility.

**Bayramov, Bailar (bailar-dev) —** The assigned work adds NumPy Ridge in
`src/linear.py`, matched Ridge experiments, the shared `src/run_all.py` command,
and pipeline integration tests. Prepared experiment evidence includes validation
selection, five-fold summaries, measured times and reproduction metadata.
See `experiments/pipeline.py`, `experiments/linear_experiments.py` and
`experiments/PIPELINE_REPORT_NOTES.md`. The member must independently review
the prepared implementation and be able to explain it at the demo.

**Surname, Name —** 2–4 sentences: what you designed, implemented, wrote,
and debugged. Reference concrete files/PRs/commits.

**Surname, Name —** ...

**Surname, Name —** ...

## Rough effort split
Give an approximate percentage per member (should sum to ~100%) and note any
imbalance and why.

| Member | Approx. % |
|--------|-----------|
| ---    | ---       |
| ---    | ---       |
| ---    | ---       |

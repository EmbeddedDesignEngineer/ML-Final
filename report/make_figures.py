"""Build the report/slide figures from a completed `python -m src.run_all` output.

Usage:  python -m report.make_figures --results experiments/artifacts/final
Writes PNG files to report/figures/ (ignored by Git; regenerate before LaTeX).
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parent / "figures"
COPIED = {
    "figures/study/eda/price_distribution.png": "price_distribution.png",
    "figures/study/eda/area_rooms.png": "area_rooms.png",
    "figures/study/eda/location_balance.png": "location_balance.png",
    "figures/svm/objective_vs_iterations.png": "svm_objective.png",
    "figures/svm/lambda_vs_margin.png": "svm_margin.png",
    "figures/svm/lambda_vs_validation_f1.png": "svm_lambda_f1.png",
    "figures/svm/rff_parameter_effects.png": "svm_rff_effects.png",
    "figures/evaluation/custom_tree_regression_features.png": "tree_reg_importance.png",
    "figures/evaluation/custom_tree_regression_price.png": "tree_reg_price.png",
    "figures/evaluation/custom_linear_features.png": "svm_linear_weights.png",
    "figures/evaluation/custom_tree_classification_confusion.png": "tree_cls_confusion.png",
    "figures/evaluation/custom_linear_confusion.png": "svm_linear_confusion.png",
}
NAMES = {
    "custom_tree_regression": "Ours: tree",
    "sklearn_tree_regression": "sklearn tree",
    "custom_ridge": "Ours: Ridge",
    "sklearn_ridge": "sklearn Ridge",
    "custom_tree_classification": "Ours: tree",
    "sklearn_tree_classification": "sklearn tree",
    "custom_linear": "Ours: linear SVM",
    "custom_rff": "Ours: RFF SVM",
    "sklearn_linear": "SVC linear",
    "sklearn_rbf": "SVC RBF",
}
OURS, THEIRS = "#2a6fdb", "#9aa5b1"


def tree_curves(results: Path) -> None:
    rows = pd.read_csv(results / "tree_validation.csv")
    custom = rows[rows["implementation"] == "custom"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3.1))
    reg = custom[custom["task"] == "regression"]
    for leaf, colour in zip(sorted(reg["min_samples_leaf"].unique()), ("#c0392b", "#2a6fdb", "#27ae60")):
        part = reg[reg["min_samples_leaf"] == leaf].sort_values("max_depth")
        left.plot(part["max_depth"], part["train_score"] / 1e3, "--", color=colour, alpha=0.6)
        left.plot(part["max_depth"], part["validation_score"] / 1e3, "o-", color=colour, label=f"leaf={leaf}")
    left.set(title="Task A: regression tree", xlabel="max_depth", ylabel="RMSE (thousand AZN)")
    left.legend(title="solid: val, dashed: train", fontsize=7, title_fontsize=7)
    cls = custom[(custom["task"] == "classification") & (custom["min_samples_leaf"] == 1)]
    for criterion, colour in (("gini", "#2a6fdb"), ("entropy", "#e67e22")):
        part = cls[cls["criterion"] == criterion].sort_values("max_depth")
        right.plot(part["max_depth"], part["train_score"], "--", color=colour, alpha=0.6)
        right.plot(part["max_depth"], part["validation_score"], "o-", color=colour, label=criterion)
    right.set(title="Task B: classification tree (leaf=1)", xlabel="max_depth", ylabel="F1 (premium)")
    right.legend(fontsize=7)
    for axis in (left, right):
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "tree_curves.png", dpi=200)
    plt.close(fig)


def results_bars(results: Path) -> None:
    test = pd.read_csv(results / "test_metrics.csv").set_index("model")
    cv = pd.read_csv(results / "cv_summary.csv")
    cv = cv.pivot_table(index="model", columns="metric", values=["mean", "std"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3.1), gridspec_kw={"width_ratios": [2, 3]})
    reg = ["custom_tree_regression", "sklearn_tree_regression", "custom_ridge", "sklearn_ridge"]
    cls = ["custom_tree_classification", "sklearn_tree_classification", "custom_linear",
           "custom_rff", "sklearn_linear", "sklearn_rbf"]
    for axis, models, metric, scale, label in (
        (left, reg, "rmse", 1e3, "RMSE (thousand AZN)"),
        (right, cls, "f1", 1.0, "F1 (premium)"),
    ):
        colours = [OURS if m.startswith("custom") else THEIRS for m in models]
        axis.bar(range(len(models)), test.loc[models, metric] / scale, color=colours,
                 yerr=cv.loc[models, ("std", metric)] / scale, capsize=3)
        axis.scatter(range(len(models)), cv.loc[models, ("mean", metric)] / scale,
                     marker="D", color="black", s=14, zorder=3, label="5-fold CV mean")
        axis.set_xticks(range(len(models)), [NAMES[m] for m in models], rotation=30, ha="right", fontsize=7)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.3)
    left.set_title("Task A (lower is better)")
    right.set_title("Task B (higher is better)")
    right.set_ylim(0.8, 0.95)
    right.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "results_bars.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("experiments/artifacts/final"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for source, target in COPIED.items():
        shutil.copyfile(args.results / source, OUT / target)
    tree_curves(args.results)
    results_bars(args.results)
    print(json.dumps(sorted(p.name for p in OUT.glob("*.png")), indent=1))


if __name__ == "__main__":
    main()

from pathlib import Path

from src.evaluate import rmse, mae, r2, precision_recall_f1, roc_auc, compare
from src.evaluation_plots import (
    plot_confusion_matrix, plot_actual_vs_predicted, summarize_errors_by_group,
    summarize_price_errors, plot_feature_values,
)


def main():
    output = Path("results/evaluation_demo")
    prices = [100000, 200000, 300000, 500000]
    predictions = [110000, 180000, 330000, 450000]
    labels = [0, 0, 1, 1]
    predicted_labels = [0, 1, 0, 1]
    scores = [-0.8, 0.2, -0.1, 0.9]
    precision, recall, f1 = precision_recall_f1(labels, predicted_labels)
    print("DEVELOPMENT EXAMPLE ONLY - not real Bina.az results")
    compare({
        "Example regression": {"RMSE": rmse(prices, predictions), "MAE": mae(prices, predictions), "R2": r2(prices, predictions)},
        "Example classification": {"Precision": precision, "Recall": recall, "F1": f1, "ROC-AUC": roc_auc(labels, scores)},
    })
    plot_confusion_matrix(labels, predicted_labels, output / "confusion_matrix.png", title="Development example: confusion matrix")
    plot_actual_vs_predicted(prices, predictions, output / "price.png", title="Development example: prices")
    plot_feature_values(["area", "rooms", "distance"], [0.8, 0.3, -0.5], output / "weights.png", title="Development example: invented weights", xlabel="Weight")
    print("Location errors:", summarize_errors_by_group(prices, predictions, ["A", "A", "B", "B"]))

    print("Price errors:", summarize_price_errors(prices, predictions, 300000))
    print(f"Figures saved to {output}")


if __name__ == "__main__":
    main()

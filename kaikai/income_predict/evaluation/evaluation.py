import csv
from pathlib import Path


def load_labels(path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if "income" not in reader.fieldnames:
        raise ValueError(f"Missing 'income' column in {path}")
    labels = [row["income"].strip() for row in rows]
    return labels


def compute_metrics(y_true, y_pred, positive_label=">50K"):
    if len(y_true) != len(y_pred):
        raise ValueError("Prediction and ground-truth arrays have different lengths")

    total = len(y_true)
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))

    tp = sum((yt == positive_label) and (yp == positive_label) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt != positive_label) and (yp == positive_label) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == positive_label) and (yp != positive_label) for yt, yp in zip(y_true, y_pred))

    accuracy = correct / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return accuracy, precision, recall


def main():
    base_dir = Path(__file__).parent
    preds_path = base_dir / "results.csv"
    truth_path = base_dir / "test_withlabel.csv"

    y_pred = load_labels(preds_path)
    y_true = load_labels(truth_path)

    accuracy, precision, recall = compute_metrics(y_true, y_pred)

    print(f"Predictions: {len(y_pred)} records")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")


if __name__ == "__main__":
    main()

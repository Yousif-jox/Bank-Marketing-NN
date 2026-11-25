"""Train and evaluate the Week 10 validation baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .data import (
    build_preprocessor,
    create_stratified_splits,
    load_hotel_bookings,
    separate_features_target,
    split_class_distribution,
)


def evaluate_binary_classifier(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, Any], dict[str, Any]]:
    """Calculate all required Week 10 metrics on one untouched validation set."""

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    positive_index = list(model.classes_).index(1)
    positive_probabilities = probabilities[:, positive_index]
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "precision_canceled": precision_score(y, predictions, pos_label=1, zero_division=0),
        "recall_canceled": recall_score(y, predictions, pos_label=1, zero_division=0),
        "f1_canceled": f1_score(y, predictions, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y, predictions, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, positive_probabilities),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    report = classification_report(y, predictions, labels=[0, 1], output_dict=True, zero_division=0)
    return metrics, report


def make_report_markdown(results: pd.DataFrame, details: dict[str, Any]) -> str:
    """Create a readable validation-only report from computed metrics."""

    def markdown_table(dataframe: pd.DataFrame) -> str:
        """Render a small Markdown table without Pandas' optional tabulate extra."""

        headers = list(dataframe.columns)
        rows = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        for record in dataframe.to_dict(orient="records"):
            values = []
            for header in headers:
                value = record[header]
                values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join(rows)

    metric_columns = [
        "model",
        "accuracy",
        "precision_canceled",
        "recall_canceled",
        "f1_canceled",
        "macro_f1",
        "roc_auc",
    ]
    table = markdown_table(results[metric_columns])
    lines = [
        "# Week 10 Baseline Validation Results",
        "",
        "These metrics were calculated on the validation split only. The test split was not used for model selection.",
        "",
        "## Split distribution",
        "",
        markdown_table(pd.DataFrame(details["split_distribution"])),
        "",
        "## Model comparison",
        "",
        table,
        "",
        "## Confusion matrices",
        "",
    ]
    for model_name, report in details["models"].items():
        confusion = report["confusion_matrix"]
        lines.extend(
            [
                f"### {model_name}",
                "",
                "| | Predicted 0 | Predicted 1 |",
                "|---|---:|---:|",
                f"| Actual 0 | {confusion['true_negatives']} | {confusion['false_positives']} |",
                f"| Actual 1 | {confusion['false_negatives']} | {confusion['true_positives']} |",
                "",
                "Classification report:",
                "",
                "```json",
                json.dumps(report["classification_report"], indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def run_baselines(data_path: str | Path, output_directory: str | Path) -> pd.DataFrame:
    """Fit both Week 10 baselines on train data and evaluate on validation data."""

    dataframe = load_hotel_bookings(data_path)
    X, y = separate_features_target(dataframe)
    splits = create_stratified_splits(X, y)

    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression_balanced": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear",
            random_state=42,
        ),
    }

    result_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "data_path": str(data_path),
        "target": "is_canceled",
        "evaluation_split": "validation",
        "test_set_used_for_model_selection": False,
        "excluded_columns": ["reservation_status", "reservation_status_date", "assigned_room_type", "booking_changes", "days_in_waiting_list", "required_car_parking_spaces", "total_of_special_requests"],
        "split_distribution": split_class_distribution(splits).to_dict(orient="records"),
        "models": {},
    }

    for model_name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(splits.X_train)),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(splits.X_train, splits.y_train)
        metrics, report = evaluate_binary_classifier(pipeline, splits.X_validation, splits.y_validation)
        result_rows.append({"model": model_name, **metrics})
        details["models"][model_name] = {
            "confusion_matrix": {
                key: metrics[key]
                for key in ("true_negatives", "false_positives", "false_negatives", "true_positives")
            },
            "classification_report": report,
        }

    results = pd.DataFrame(result_rows).sort_values("f1_canceled", ascending=False)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path / "baseline_results.csv", index=False)
    (output_path / "baseline_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (output_path / "baseline_report.md").write_text(make_report_markdown(results, details), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week 10 hotel-cancellation baselines.")
    parser.add_argument("--data", default="data/raw/hotel_bookings.csv", help="Path to hotel_bookings.csv")
    parser.add_argument("--output-dir", default="experiments", help="Directory for validation results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pd.set_option("display.max_columns", None)
    print(run_baselines(args.data, args.output_dir).to_string(index=False))

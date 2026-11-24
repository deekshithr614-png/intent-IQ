"""Validate, then train the final Data Sprint 2.0 intent model.

Run from this directory:
    python train.py

Optional paths:
    python train.py --data-dir data --model-dir models --output-dir outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from modeling import build_model
from text_utils import normalize_queries

ROOT = Path(__file__).resolve().parent
EXPECTED_CLASSES = set(range(56))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Hinglish intent classifier")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.data_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Could not find {train_path}. Put train.csv in the data folder.")

    train = pd.read_csv(train_path)
    required = {"example_id", "query_text", "target"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"train.csv is missing required columns: {sorted(missing)}")
    if train["query_text"].isna().any() or train["target"].isna().any():
        raise ValueError("train.csv contains missing query_text or target values")

    y = train["target"].astype(int)
    observed = set(y.unique())
    if observed != EXPECTED_CLASSES:
        raise ValueError(
            "Expected all 56 labels (0..55). "
            f"Missing: {sorted(EXPECTED_CLASSES - observed)}; "
            f"unexpected: {sorted(observed - EXPECTED_CLASSES)}"
        )
    X = normalize_queries(train["query_text"].tolist())

    # Hold out a stratified validation set to estimate generalization.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=y,
    )
    validation_model = build_model(C=args.C)
    print(f"Training validation model on {len(X_train):,} examples...")
    validation_model.fit(X_train, y_train)
    valid_pred = validation_model.predict(X_valid)

    labels = sorted(EXPECTED_CLASSES)
    accuracy = float(accuracy_score(y_valid, valid_pred))
    macro_f1 = float(f1_score(y_valid, valid_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_valid, valid_pred, average="weighted", zero_division=0))
    class55_f1 = float(
        f1_score((y_valid == 55).astype(int), (valid_pred == 55).astype(int), zero_division=0)
    )
    report = classification_report(
        y_valid,
        valid_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "validation_strategy": "stratified train_test_split",
        "validation_size": args.validation_size,
        "random_seed": args.seed,
        "model": "word + char TF-IDF / LinearSVC",
        "C": args.C,
        "train_rows_for_validation": len(X_train),
        "validation_rows": len(X_valid),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "class_55_f1": class55_f1,
    }

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (args.output_dir / "classwise_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    report_rows = []
    for label in labels:
        row = report[str(label)]
        report_rows.append({
            "target": label,
            "precision": row["precision"],
            "recall": row["recall"],
            "f1_score": row["f1-score"],
            "support": int(row["support"]),
        })
    pd.DataFrame(report_rows).to_csv(args.output_dir / "classwise_validation_report.csv", index=False)
    pd.DataFrame(
        confusion_matrix(y_valid, valid_pred, labels=labels),
        index=labels,
        columns=labels,
    ).to_csv(args.output_dir / "confusion_matrix.csv", index_label="actual_target")

    mistakes = pd.DataFrame({
        "query_text": train.iloc[y_valid.index]["query_text"].values,
        "actual_target": y_valid.values,
        "predicted_target": valid_pred,
    })
    mistakes = mistakes[mistakes["actual_target"] != mistakes["predicted_target"]]
    mistakes.to_csv(args.output_dir / "validation_misclassified.csv", index=False)

    print("\nValidation results")
    print(f"Accuracy:       {accuracy:.4%}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Weighted F1:    {weighted_f1:.4f}")
    print(f"Class 55 F1:    {class55_f1:.4f}")
    print(f"Misclassified:  {len(mistakes):,} / {len(X_valid):,}")

    # Refit on every labeled row before producing competition predictions.
    print(f"\nTraining final model on all {len(X):,} labeled examples...")
    final_model = build_model(C=args.C)
    final_model.fit(X, y)
    artifact = {
        "model": final_model,
        "classes": labels,
        "C": args.C,
        "model_name": "word_char_tfidf_linearsvc",
        "normalization": "Unicode NFKC + lowercase + whitespace normalization",
    }
    model_path = args.model_dir / "hinglish_intent_model.joblib"
    joblib.dump(artifact, model_path, compress=3)
    print(f"Saved trained model: {model_path}")
    print(f"Reports saved in: {args.output_dir}")


if __name__ == "__main__":
    main()

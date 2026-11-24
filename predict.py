"""Generate the competition submission using the saved final model.

Run from this directory:
    python predict.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from text_utils import normalize_queries

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Data Sprint 2.0 predictions")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--test-file", type=str, default="test.csv")
    parser.add_argument("--sample-file", type=str, default="sample_submission.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_dir / "hinglish_intent_model.joblib"
    test_path = args.data_dir / args.test_file
    sample_path = args.data_dir / args.sample_file
    for path in (model_path, test_path, sample_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    artifact = joblib.load(model_path)
    model = artifact["model"]
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    if not {"example_id", "query_text"}.issubset(test.columns):
        raise ValueError("test.csv must contain example_id and query_text columns")
    if not {"example_id", "target"}.issubset(sample.columns):
        raise ValueError("sample_submission.csv must contain example_id and target columns")
    if test["example_id"].duplicated().any() or sample["example_id"].duplicated().any():
        raise ValueError("Duplicate example_id values found in test or sample submission")
    if set(test["example_id"]) != set(sample["example_id"]):
        raise ValueError("test.csv IDs and sample_submission.csv IDs do not match")

    queries = normalize_queries(test["query_text"].tolist())
    predictions = model.predict(queries)
    id_to_prediction = dict(zip(test["example_id"], predictions))

    # Preserve the exact order and IDs supplied by the official template.
    submission = sample[["example_id"]].copy()
    submission["target"] = submission["example_id"].map(id_to_prediction).astype(int)
    if len(submission) != len(sample) or submission["target"].isna().any():
        raise RuntimeError("Submission generation failed: missing or invalid predictions")
    if not submission["target"].between(0, 55).all():
        raise RuntimeError("Model generated a target outside the allowed 0..55 range")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Created {output_path}")
    print(f"Rows: {len(submission):,}")
    print("Columns: example_id, target")
    print("Target distribution:")
    print(submission["target"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

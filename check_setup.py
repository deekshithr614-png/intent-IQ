"""Check that dependencies and the bundled BhashaSense model can be loaded."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    try:
        import joblib
        import pandas as pd
        import sklearn
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("Install with: python -m pip install -r requirements-ui.txt")
        return 1

    print(f"scikit-learn: {sklearn.__version__}")
    model_path = ROOT / "models" / "hinglish_intent_model.joblib"
    train_path = ROOT / "data" / "train.csv"
    test_path = ROOT / "data" / "test.csv"
    sample_path = ROOT / "data" / "sample_submission.csv"
    for path in (model_path, train_path, test_path, sample_path):
        if not path.exists():
            print(f"Missing required file: {path.relative_to(ROOT)}")
            return 1

    artifact = joblib.load(model_path)
    model = artifact["model"]
    sample_queries = [
        "kal subah 7 baje alarm laga do",
        "आज मौसम कैसा है",
        "play some music",
    ]
    predictions = model.predict(sample_queries)
    print(f"Training rows: {len(pd.read_csv(train_path)):,}")
    print(f"Test rows: {len(pd.read_csv(test_path)):,}")
    print(f"Classes in saved model: {len(model.classes_)}")
    print("Sample predictions:")
    for query, label in zip(sample_queries, predictions):
        print(f"  {query!r} -> class {int(label)}")
    print("\nSetup check passed. Start the UI with:")
    print("  python -m streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

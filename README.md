# Intent IQ — Data Sprint 2.0 Working NLP Model

BhashaSense is a runnable Hinglish/Hindi intent classifier built from the supplied Data Sprint 2.0 CSV files.

## What it does

- Classifies Romanized Hinglish, Devanagari and noisy text into 56 intent classes (0–55).
- Offers a Streamlit UI for single-query predictions.
- Offers CSV batch prediction and downloadable results.
- Shows validation metrics and class-wise results.
- Generates the competition `example_id,target` submission format.

## Model

The pipeline combines word-level TF-IDF (unigrams and bigrams) with character-level TF-IDF n-grams (2–6), followed by a LinearSVC classifier. Text normalization uses Unicode NFKC normalization, lowercasing and whitespace cleanup while preserving both scripts.

## Windows setup (VS Code)

Open this extracted folder in VS Code, then run these commands in its terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` in a browser.

The trained artifact is included in `models/`, so the UI can run without retraining.

## Retrain and regenerate the submission

```powershell
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe predict.py
```

The model and validation reports are saved under `models/` and `outputs/`. The competition submission is `outputs/submission.csv`.

## Validation

The bundled report records a stratified 80/20 split (seed 42): accuracy 80.67%, macro F1 0.7914, weighted F1 0.8046, and Class 55 F1 0.84. These are local holdout validation metrics, not a competition leaderboard score.

## Notes

- Class 55 is Out-of-Scope. The provided data does not contain human-readable names for Classes 0–54, so the interface displays their numeric IDs.
- LinearSVC decision scores are relative ranking scores, not calibrated probabilities.
- The provided test labels are unavailable. Test-set accuracy cannot be measured from `test.csv` alone.

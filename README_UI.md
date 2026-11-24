# BhashaSense — Data Sprint 2.0 UI

A Streamlit web interface connected to the trained Hinglish intent classifier included in `models/hinglish_intent_model.joblib`.

## Features

- **Overview:** dataset and model summary.
- **Live classifier:** enter Romanized Hinglish, Devanagari Hindi, or ASR-style query text; view the predicted class and top five alternatives.
- **Batch prediction:** upload a CSV with `query_text`; if `example_id` is present, download a two-column `example_id,target` CSV suitable for the competition format.
- **Model insights:** view validation accuracy, macro F1, weighted F1, Class 55 F1, and per-class validation F1.
- **Recent predictions:** session-only history in the live classifier.

## Run on Windows / VS Code

1. Extract the ZIP file.
2. Open the `datasprint2_hinglish_ui` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```

5. Install the app dependencies:

   ```powershell
   py -m pip install -r requirements-ui.txt
   ```

6. Start the interface:

   ```powershell
   py -m streamlit run app.py
   ```

7. Streamlit will print a local address, usually `http://localhost:8501`. Open it in your browser.

You can alternatively double-click `run_app.bat`; it installs dependencies and starts Streamlit.

## Important notes

- The trained model and dataset are bundled in this project; no retraining is required to launch the UI.
- The UI uses the model's existing Unicode normalization and classifier directly.
- `LinearSVC` decision scores are rankings, not calibrated probabilities. The app labels them accordingly.
- The source dataset provides numeric targets, not names for Classes 0–54. The UI therefore displays `Intent #N`; Class 55 is identified as Out-of-Scope based on the dataset description.
- Validation metrics are from the supplied stratified holdout split. They are not public or private leaderboard scores.

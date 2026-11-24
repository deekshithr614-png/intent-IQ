"""Feature extraction and model factory for Data Sprint 2.0."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def build_model(C: float = 0.5) -> Pipeline:
    """Build the tuned word + character TF-IDF Linear SVM classifier.

    Validation on the supplied training file (stratified 80/20, seed 42)
    found this setup stronger than the tested lower-complexity alternatives:
      * word unigrams and bigrams
      * character-within-word n-grams, lengths 2 through 6
      * character features weighted 1.5x
      * LinearSVC with C=0.5
    """
    word_features = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
        min_df=2,
        max_df=0.995,
        sublinear_tf=True,
        strip_accents=None,
        dtype=np.float32,
    )
    char_features = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 6),
        min_df=2,
        max_features=1_200_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    features = FeatureUnion(
        [("word", word_features), ("char", char_features)],
        transformer_weights={"word": 1.0, "char": 1.5},
        n_jobs=1,
    )
    classifier = LinearSVC(
        C=C,
        class_weight=None,
        max_iter=20_000,
        random_state=42,
    )
    return Pipeline([("features", features), ("classifier", classifier)])

"""Text normalization helpers shared by training and inference."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List


def normalize_query(text: object) -> str:
    """Normalize Unicode and whitespace without transliterating either script.

    Keeping both Devanagari and Romanized Hinglish intact is important: the
    character n-gram model learns spelling patterns directly from the data.
    """
    if text is None:
        text = ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    # Replace unusual whitespace/control separators with ordinary spaces.
    text = "".join(" " if unicodedata.category(ch) in {"Zl", "Zp"} else ch for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_queries(values: Iterable[object]) -> List[str]:
    return [normalize_query(value) for value in values]

"""Clean and lightly language-tag scraped Tanglish YouTube comments."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "comments.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "cleaned.csv"

# This starter vocabulary is intentionally small and should grow with annotation.
TAMIL_ROMAN_WORDS = {
    "aama", "adi", "anna", "apdi", "ava", "avanga", "enaku", "enna",
    "epdi", "inga", "iruka", "irukku", "illa", "konjam", "machan",
    "machi", "naan", "nalla", "romba", "semma", "sollu", "thambi",
    "unaku", "ungal", "yaar", "yen", "z super",
}

# Map common spelling variations to a single form before token tagging.
NORMALIZATION_MAP: Dict[str, str] = {
    "simma": "semma",
    "seema": "semma",
    "semmaa": "semma",
    "nallaa": "nalla",
    "rombaa": "romba",
    "illai": "illa",
    "illae": "illa",
    "superrr": "super",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", re.UNICODE)


def normalize_token(token: str) -> str:
    """Normalize one token while preserving punctuation and numbers."""
    lower_token = token.lower()
    return NORMALIZATION_MAP.get(lower_token, lower_token)


def normalize_text(text: str) -> str:
    """Lowercase text and normalize known spelling variants."""
    tokens = TOKEN_PATTERN.findall(str(text))
    normalized_tokens = [normalize_token(token) for token in tokens]
    return " ".join(normalized_tokens).strip()


def tag_token(token: str) -> str:
    """Classify a token using a small Tamil-Roman dictionary and heuristics."""
    normalized_token = normalize_token(token)
    if not normalized_token.isalpha():
        return "Mixed"
    if normalized_token in TAMIL_ROMAN_WORDS:
        return "Tamil-Roman"
    if normalized_token.isascii():
        return "English"
    return "Mixed"


def tag_text(text: str) -> str:
    """Return token tags in the same order as the input text."""
    tokens = TOKEN_PATTERN.findall(str(text))
    return " ".join(f"{token}:{tag_token(token)}" for token in tokens)


def preprocess_comments(input_path: Path, output_path: Path) -> None:
    """Load raw comments, clean them, and create a manual-labeling CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV was not found: {input_path}")

    comments = pd.read_csv(input_path)
    required_columns = {"comment_id", "video_id", "comment_text"}
    missing_columns = required_columns.difference(comments.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    comments["original_text"] = comments["comment_text"].fillna("").astype(str)
    comments["cleaned_text"] = comments["original_text"].map(normalize_text)
    comments["token_tags"] = comments["original_text"].map(tag_text)
    comments["label"] = ""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comments.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Processed {len(comments)} comments.")
    print(f"Saved cleaned data to {output_path}.")


def parse_arguments() -> argparse.Namespace:
    """Read optional input and output paths from the command line."""
    parser = argparse.ArgumentParser(description="Preprocess Tanglish comments.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Run preprocessing using command-line paths or project defaults."""
    args = parse_arguments()
    preprocess_comments(args.input, args.output)


if __name__ == "__main__":
    main()

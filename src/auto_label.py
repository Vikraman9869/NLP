"""Create rough first-pass labels to speed up manual annotation.

These predictions are only a preliminary label source, not final labels. Review
and correct them manually before using the data for model training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from transformers import pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "cleaned.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "cleaned_autolabeled.csv"
MODEL_NAME = "facebook/bart-large-mnli"
CANDIDATE_LABELS = ["positive", "negative", "neutral", "sarcastic"]


def parse_arguments() -> argparse.Namespace:
    """Read optional input and output paths from the command line."""
    parser = argparse.ArgumentParser(
        description="Create preliminary sentiment and sarcasm labels."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_classifier():
    """Create the zero-shot classifier used for preliminary labeling."""
    return pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
    )


def predict_label(classifier, text: str) -> tuple[str, float]:
    """Return the highest-scoring candidate label and its confidence score."""
    result = classifier(
        text,
        candidate_labels=CANDIDATE_LABELS,
        multi_label=False,
    )
    return str(result["labels"][0]), float(result["scores"][0])


def auto_label_comments(input_path: Path, output_path: Path) -> None:
    """Load comments, add preliminary predictions, and save a new CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV was not found: {input_path}")

    comments = pd.read_csv(input_path)
    required_columns = {"cleaned_text", "token_tags", "label"}
    missing_columns = required_columns.difference(comments.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    classifier = build_classifier()
    auto_labels: List[str] = []
    confidence_scores: List[float] = []

    for row_number, text in enumerate(comments["cleaned_text"].fillna(""), start=1):
        predicted_label, confidence_score = predict_label(classifier, str(text))
        auto_labels.append(predicted_label)
        confidence_scores.append(confidence_score)

        if row_number % 100 == 0:
            print(f"Processed {row_number}/{len(comments)} rows.")

    comments["auto_label"] = auto_labels
    comments["confidence_score"] = confidence_scores
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comments.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved auto-labeled data to {output_path}.")
    print("Final label summary:")
    for label in CANDIDATE_LABELS:
        count = int((comments["auto_label"] == label).sum())
        print(f"  {label}: {count}")


def main() -> None:
    """Run auto-labeling using command-line paths or project defaults."""
    args = parse_arguments()
    auto_label_comments(args.input, args.output)


if __name__ == "__main__":
    main()

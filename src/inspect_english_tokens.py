"""Inspect high-frequency tokens tagged as 'English' in preprocessed comments.

This utility audits the Token-level Language Identification (LID) results from
`data/processed/cleaned.csv`. By examining the most frequent tokens classified as
'English', researchers can easily identify Romanized Tamil (Tanglish) words that
might have been missed by the dictionary or suffix heuristics.

Usage:
    python src/inspect_english_tokens.py
    python src/inspect_english_tokens.py --top 50
    python src/inspect_english_tokens.py --input data/processed/cleaned.csv --top 100
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "cleaned.csv"


def extract_english_tokens(csv_path: Path) -> Counter[str]:
    """Parse the 'token_tags' column and count tokens classified as 'English'."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    if "token_tags" not in df.columns:
        raise ValueError("Missing required column 'token_tags' in CSV.")

    english_tokens: List[str] = []
    for tags_row in df["token_tags"].dropna():
        # Each entry is formatted as 'token:Tag token:Tag'
        pairs = str(tags_row).split()
        for pair in pairs:
            if ":" in pair:
                token, tag = pair.rsplit(":", 1)
                if tag == "English":
                    english_tokens.append(token.lower())

    return Counter(english_tokens)


def display_top_tokens(counts: Counter[str], top_n: int = 100) -> None:
    """Print the top N English tokens in a formatted table for easy review."""
    top_tokens: List[Tuple[str, int]] = counts.most_common(top_n)

    print(f"\n{'='*55}")
    print(f"Top {len(top_tokens)} Most Frequent Tokens Tagged as 'English'")
    print(f"{'='*55}")
    print(f"{'Rank':<6}{'Token':<25}{'Frequency':<10}")
    print(f"{'-'*55}")

    for rank, (token, freq) in enumerate(top_tokens, start=1):
        print(f"{rank:<6}{token:<25}{freq:<10}")

    print(f"{'-'*55}")
    print(f"Total unique English tokens: {len(counts)}")
    print(f"Total English token occurrences: {sum(counts.values())}")
    print(f"{'='*55}\n")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the inspection utility."""
    parser = argparse.ArgumentParser(
        description="List the most frequent tokens classified as English."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to cleaned CSV with token_tags (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="Number of top tokens to display (default: 100)",
    )
    return parser.parse_args()


def main() -> None:
    """Execute token inspection."""
    args = parse_arguments()
    counts = extract_english_tokens(args.input)
    display_top_tokens(counts, top_n=args.top)


if __name__ == "__main__":
    main()

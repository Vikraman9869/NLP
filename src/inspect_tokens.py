"""Inspect token language classification results in preprocessed Tanglish comments.

This utility allows auditing:
1. Top Tamil-Roman tokens across the dataset.
2. Top tokens tagged as Tamil-Roman ONLY via morphological suffix heuristics
   (i.e., not present in the hardcoded TAMIL_ROMAN_WORDS lexicon). This is essential
   for catching false positives from loose suffix rules.
3. Top English tokens to detect any missed Tamil words.

Usage:
    python src/inspect_tokens.py --mode all
    python src/inspect_tokens.py --mode suffix-only --top 100
    python src/inspect_tokens.py --mode tamil --top 100
    python src/inspect_tokens.py --mode english --top 100
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.preprocess import TAMIL_ROMAN_WORDS, normalize_token


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "cleaned.csv"


def extract_token_distributions(
    csv_path: Path,
) -> Tuple[Counter[str], Counter[str], Counter[str]]:
    """Parse 'token_tags' and return counters for:

    1. All Tamil-Roman tokens
    2. Suffix-only Tamil-Roman tokens (not in TAMIL_ROMAN_WORDS lexicon)
    3. English tokens
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "token_tags" not in df.columns:
        raise ValueError("Missing 'token_tags' column in CSV.")

    tamil_roman_counts: Counter[str] = Counter()
    suffix_only_counts: Counter[str] = Counter()
    english_counts: Counter[str] = Counter()

    for tags_row in df["token_tags"].dropna():
        pairs = str(tags_row).split()
        for pair in pairs:
            if ":" not in pair:
                continue
            token, tag = pair.rsplit(":", 1)
            norm_token = normalize_token(token)

            if tag == "Tamil-Roman":
                tamil_roman_counts[norm_token] += 1
                if norm_token not in TAMIL_ROMAN_WORDS:
                    suffix_only_counts[norm_token] += 1
            elif tag == "English":
                english_counts[norm_token] += 1

    return tamil_roman_counts, suffix_only_counts, english_counts


def print_table(title: str, counter: Counter[str], top_n: int = 100) -> None:
    """Print formatted token frequency table."""
    items: List[Tuple[str, int]] = counter.most_common(top_n)

    print(f"\n{'='*60}")
    print(f"{title} (Top {len(items)} of {len(counter)} unique)")
    print(f"{'='*60}")
    print(f"{'Rank':<6}{'Token':<28}{'Frequency':<10}")
    print(f"{'-'*60}")

    for rank, (token, freq) in enumerate(items, start=1):
        print(f"{rank:<6}{token:<28}{freq:<10}")

    print(f"{'-'*60}")
    print(f"Total occurrences: {sum(counter.values())}")
    print(f"{'='*60}\n")


def parse_arguments() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(description="Inspect token LID distributions.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to cleaned CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "tamil", "suffix-only", "english"],
        default="all",
        help="Which token category to inspect (default: all)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="Number of tokens to display (default: 100)",
    )
    return parser.parse_args()


def main() -> None:
    """Run token inspection based on CLI arguments."""
    args = parse_arguments()
    tamil_counts, suffix_counts, english_counts = extract_token_distributions(args.input)

    if args.mode in ("all", "tamil"):
        print_table("Tamil-Roman Tokens (Overall)", tamil_counts, top_n=args.top)

    if args.mode in ("all", "suffix-only"):
        print_table(
            "Tamil-Roman Tokens (SUFFIX-ONLY / Not in Lexicon)",
            suffix_counts,
            top_n=args.top,
        )

    if args.mode in ("all", "english"):
        print_table("English Tokens", english_counts, top_n=args.top)


if __name__ == "__main__":
    main()

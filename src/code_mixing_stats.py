"""Compute Code-Mixing Index (CMI) and linguistic distribution of comments.

This module evaluates code-mixing characteristics in Tanglish YouTube comments:
1. Per-comment categorization:
   - Pure English: Contains only English language tokens (and optional 'Other' tokens).
   - Pure Tamil-Roman: Contains only Romanized Tamil tokens (and optional 'Other' tokens).
   - Pure Tamil-Script: Contains only native Tamil script tokens (and optional 'Other' tokens).
   - Code-Mixed: Contains both English and Tamil (Roman or Script) tokens.
   - Non-Linguistic / Other: Contains only non-language tokens (emojis, punctuation, numbers).

2. Code-Mixing Index (CMI) Formulation (Gambäck & Das, 2014):
   CMI measures the degree of multilingual mixing in an utterance:
       CMI = 100 * [1 - (max(w_L1, w_L2, ...) / L)]   if L > 0 else 0
   where L is the count of language-dependent tokens (excluding 'Other') and
   max(w_Li) is the token count of the dominant (matrix) language.
   - For monolingual text: CMI = 0.0%
   - For a perfectly balanced bilingual utterance: CMI = 50.0%

Outputs:
- Generates and saves a dataset-level summary table to `reports/code_mixing_stats.csv`.
- Generates and saves per-comment detailed metrics to `reports/per_comment_code_mixing.csv`.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "cleaned.csv"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "reports" / "code_mixing_stats.csv"
DEFAULT_DETAILS_OUTPUT = PROJECT_ROOT / "reports" / "per_comment_code_mixing.csv"


def calculate_comment_cmi(
    n_english: int, n_tamil: int
) -> float:
    """Calculate Code-Mixing Index (CMI) for a single comment.

    Args:
        n_english: Count of English tokens.
        n_tamil: Count of Tamil tokens (Romanized + Native script).

    Returns:
        float: CMI percentage between 0.0 and 50.0.
    """
    total_lang_tokens = n_english + n_tamil
    if total_lang_tokens == 0:
        return 0.0

    dominant_lang_tokens = max(n_english, n_tamil)
    cmi = 100.0 * (1.0 - (dominant_lang_tokens / total_lang_tokens))
    return round(cmi, 2)


def analyze_code_mixing(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process all comments, classify them, compute CMI, and generate summaries.

    Returns:
        Tuple of (summary_df, per_comment_df)
    """
    per_comment_records: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        comment_id = str(row.get("comment_id", ""))
        video_id = str(row.get("video_id", ""))
        cleaned_text = str(row.get("cleaned_text", ""))
        token_tags = str(row.get("token_tags", ""))

        n_english = 0
        n_tamil_roman = 0
        n_tamil_script = 0
        n_other = 0

        for pair in token_tags.split():
            if ":" not in pair:
                continue
            tag = pair.rsplit(":", 1)[1]
            if tag == "English":
                n_english += 1
            elif tag == "Tamil-Roman":
                n_tamil_roman += 1
            elif tag == "Tamil-Script":
                n_tamil_script += 1
            elif tag == "Other":
                n_other += 1

        n_tamil_total = n_tamil_roman + n_tamil_script
        n_total = n_english + n_tamil_total + n_other
        n_lang = n_english + n_tamil_total

        # Categorization logic
        if n_lang == 0:
            category = "Non-Linguistic / Other"
        elif n_english > 0 and n_tamil_total > 0:
            category = "Code-Mixed"
        elif n_english > 0:
            category = "Pure English"
        elif n_tamil_roman > 0 and n_tamil_script == 0:
            category = "Pure Tamil-Roman"
        else:
            category = "Pure Tamil-Script"

        cmi = calculate_comment_cmi(n_english, n_tamil_total)

        pct_english = round((n_english / n_total * 100), 2) if n_total > 0 else 0.0
        pct_tamil_roman = round((n_tamil_roman / n_total * 100), 2) if n_total > 0 else 0.0
        pct_other = round((n_other / n_total * 100), 2) if n_total > 0 else 0.0

        per_comment_records.append(
            {
                "comment_id": comment_id,
                "video_id": video_id,
                "cleaned_text": cleaned_text,
                "total_tokens": n_total,
                "english_tokens": n_english,
                "tamil_roman_tokens": n_tamil_roman,
                "tamil_script_tokens": n_tamil_script,
                "other_tokens": n_other,
                "pct_english": pct_english,
                "pct_tamil_roman": pct_tamil_roman,
                "pct_other": pct_other,
                "category": category,
                "cmi": cmi,
            }
        )

    per_comment_df = pd.DataFrame(per_comment_records)

    total_comments = len(per_comment_df)
    category_counts = Counter(per_comment_df["category"])

    pure_english_count = category_counts["Pure English"]
    code_mixed_count = category_counts["Code-Mixed"]
    pure_tamil_roman_count = category_counts["Pure Tamil-Roman"]
    pure_tamil_script_count = category_counts["Pure Tamil-Script"]
    other_count = category_counts["Non-Linguistic / Other"]

    mean_cmi_all = round(per_comment_df["cmi"].mean(), 2)
    code_mixed_mask = per_comment_df["category"] == "Code-Mixed"
    mean_cmi_code_mixed = (
        round(per_comment_df.loc[code_mixed_mask, "cmi"].mean(), 2)
        if code_mixed_count > 0
        else 0.0
    )

    summary_rows = [
        {
            "Metric": "Total Comments Analyzed",
            "Count": f"{total_comments}",
            "Percentage": "100.0%",
            "Description": "Total raw comments processed",
        },
        {
            "Metric": "Pure English Comments",
            "Count": f"{pure_english_count}",
            "Percentage": f"{(pure_english_count / total_comments * 100):.2f}%",
            "Description": "Contains only English words + optional emojis/punctuation",
        },
        {
            "Metric": "Code-Mixed Comments (Tanglish)",
            "Count": f"{code_mixed_count}",
            "Percentage": f"{(code_mixed_count / total_comments * 100):.2f}%",
            "Description": "Contains at least 1 English and 1 Tamil token",
        },
        {
            "Metric": "Pure Tamil-Roman Comments",
            "Count": f"{pure_tamil_roman_count}",
            "Percentage": f"{(pure_tamil_roman_count / total_comments * 100):.2f}%",
            "Description": "Contains only Romanized Tamil tokens",
        },
        {
            "Metric": "Pure Tamil-Script Comments",
            "Count": f"{pure_tamil_script_count}",
            "Percentage": f"{(pure_tamil_script_count / total_comments * 100):.2f}%",
            "Description": "Contains only native Tamil script tokens",
        },
        {
            "Metric": "Non-Linguistic / Other Comments",
            "Count": f"{other_count}",
            "Percentage": f"{(other_count / total_comments * 100):.2f}%",
            "Description": "Contains only emojis, numbers, and punctuation",
        },
        {
            "Metric": "Mean CMI (All Comments)",
            "Count": f"{mean_cmi_all:.2f}",
            "Percentage": f"{mean_cmi_all:.2f}%",
            "Description": "Average CMI across entire dataset (0-50% scale)",
        },
        {
            "Metric": "Mean CMI (Code-Mixed Subset Only)",
            "Count": f"{mean_cmi_code_mixed:.2f}",
            "Percentage": f"{mean_cmi_code_mixed:.2f}%",
            "Description": "Average CMI among code-mixed comments",
        },
        {
            "Metric": "Total English Tokens",
            "Count": f"{int(per_comment_df['english_tokens'].sum())}",
            "Percentage": f"{(per_comment_df['english_tokens'].sum() / per_comment_df['total_tokens'].sum() * 100):.2f}%",
            "Description": "Total English token frequency",
        },
        {
            "Metric": "Total Tamil-Roman Tokens",
            "Count": f"{int(per_comment_df['tamil_roman_tokens'].sum())}",
            "Percentage": f"{(per_comment_df['tamil_roman_tokens'].sum() / per_comment_df['total_tokens'].sum() * 100):.2f}%",
            "Description": "Total Romanized Tamil token frequency",
        },
        {
            "Metric": "Total Tamil-Script Tokens",
            "Count": f"{int(per_comment_df['tamil_script_tokens'].sum())}",
            "Percentage": f"{(per_comment_df['tamil_script_tokens'].sum() / per_comment_df['total_tokens'].sum() * 100):.2f}%",
            "Description": "Total Native Tamil script token frequency",
        },
        {
            "Metric": "Total Other Tokens",
            "Count": f"{int(per_comment_df['other_tokens'].sum())}",
            "Percentage": f"{(per_comment_df['other_tokens'].sum() / per_comment_df['total_tokens'].sum() * 100):.2f}%",
            "Description": "Total Emojis, numbers, and symbols",
        },
    ]

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, per_comment_df


def print_summary(summary_df: pd.DataFrame) -> None:
    """Print a clean ASCII summary table to stdout."""
    print(f"\n{'='*75}")
    print("Code-Mixing & Linguistic Distribution Summary")
    print(f"{'='*75}")
    print(f"{'Metric':<36}{'Value / Count':<16}{'Percentage':<12}")
    print(f"{'-'*75}")
    for _, row in summary_df.iterrows():
        metric = str(row["Metric"])
        count_str = str(row["Count"])
        pct_str = str(row["Percentage"])
        print(f"{metric:<36}{count_str:<16}{pct_str:<12}")
    print(f"{'='*75}\n")


def parse_arguments() -> argparse.Namespace:
    """Parse CLI options for code-mixing statistics generator."""
    parser = argparse.ArgumentParser(
        description="Compute Code-Mixing Index (CMI) and distribution stats."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to cleaned CSV with token_tags (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help=f"Path to output summary CSV (default: {DEFAULT_SUMMARY_OUTPUT})",
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        default=DEFAULT_DETAILS_OUTPUT,
        help=f"Path to output per-comment breakdown CSV (default: {DEFAULT_DETAILS_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    """Load cleaned data, calculate CMI and distributions, save and report."""
    args = parse_arguments()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    summary_df, per_comment_df = analyze_code_mixing(df)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_output, index=False, encoding="utf-8")
    print(f"Saved summary table to: {args.summary_output}")

    if args.details_output:
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        per_comment_df.to_csv(args.details_output, index=False, encoding="utf-8")
        print(f"Saved per-comment detailed metrics to: {args.details_output}")

    print_summary(summary_df)


if __name__ == "__main__":
    main()

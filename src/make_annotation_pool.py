"""Build the annotation pool from filtered Tanglish comments.

Filters comments from data/raw/comments_tanglish.csv based on:
1. Minimum linguistic token threshold (excludes < 3 linguistic tokens).
2. Template suppression (excludes date-stamps and 'who is watching' comments).
3. Adds LID confidence flag ('high' for lexicon/script, 'low' for suffix-only).
4. Near-exact text deduplication.

Saves the result to data/annotation/pool.csv.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import TOKEN_PATTERN, TAMIL_ROMAN_WORDS, normalize_token, tag_token
from src.scrape_comments import normalize_for_near_dedupe

DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "comments_tanglish.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "annotation" / "pool.csv"

# Configurable regex patterns for template / spam comments
TEMPLATE_PATTERNS = [
    re.compile(r"\bwho\s+(is|are)\s+(watching|waching|seeing)\b", re.IGNORECASE),
    re.compile(r"\bany\s*one\s+(in|here|watching|waching)\b", re.IGNORECASE),
    re.compile(r"\b(watching|waching)\s+in\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|\d{4})", re.IGNORECASE),
    re.compile(r"\b(attendance|calender|calendar)\b", re.IGNORECASE),
    re.compile(
        r"^\s*(sep|sept|aug|oct|nov|dec|jan|feb|mar|apr|may|jun|july)\s*\d+[\s\W]*(th|st|nd|rd)?[\s\W]*(20\d\d)?[\s\W]*(yarellam|yarallam|yarulam|yaarelam|yaru\s*ellam|yaru|naa)?[\s\W]*(paakureenga|pakuringa|pakkuringa|pakuriga|pakkuriga|pakringa|pakiringa|pakaringa)\b",
        re.IGNORECASE,
    ),
]


def count_linguistic_tokens(text: str) -> int:
    """Count tokens classified as English, Tamil-Roman, or Tamil-Script (ignoring Other)."""
    tokens = TOKEN_PATTERN.findall(str(text))
    return sum(1 for tok in tokens if tag_token(tok) in ("English", "Tamil-Roman", "Tamil-Script"))


def is_template_comment(text: str) -> bool:
    """Return True if text is primarily a template/date-stamp comment with <= 10 linguistic tokens."""
    cleaned = str(text).strip()
    n_ling = count_linguistic_tokens(cleaned)
    for pattern in TEMPLATE_PATTERNS:
        if pattern.search(cleaned):
            if n_ling <= 10:
                return True
    return False


def get_lid_confidence(text: str) -> str:
    """Determine LID confidence level based on source of Tamil classification.

    Returns:
        'high': at least one Tamil token identified from native script or curated lexicon.
        'low' : all Tamil tokens identified strictly via suffix heuristics.
    """
    tokens = TOKEN_PATTERN.findall(str(text))
    has_script = False
    has_lexicon = False
    has_suffix = False

    for tok in tokens:
        tag = tag_token(tok)
        if tag == "Tamil-Script":
            has_script = True
        elif tag == "Tamil-Roman":
            norm = normalize_token(tok)
            if norm in TAMIL_ROMAN_WORDS:
                has_lexicon = True
            else:
                has_suffix = True

    if has_script or has_lexicon:
        return "high"
    if has_suffix:
        return "low"
    return "high"


def build_annotation_pool(
    input_path: Path,
    output_path: Path,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply cleaning rules, compute LID confidence, and save pool.csv."""
    random.seed(seed)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    total_initial = len(df)
    print(f"\nLoaded {total_initial} comments from {input_path}")

    # --- Rule 1: Minimum linguistic tokens ---
    rule1_removed: List[Tuple[str, str]] = []
    rule1_kept: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        text = str(row["comment_text"])
        cid = str(row["comment_id"])
        if count_linguistic_tokens(text) < 3:
            rule1_removed.append((cid, text))
        else:
            rule1_kept.append(dict(row))

    print(f"\n[Rule 1] Comments with < 3 linguistic tokens removed: {len(rule1_removed)}")
    print(f"Sample of 15 comments removed by Rule 1:")
    sample_r1 = random.sample(rule1_removed, min(15, len(rule1_removed))) if rule1_removed else []
    for cid, txt in sample_r1:
        print(f"  {cid}: {ascii(txt)}")

    # --- Rule 2: Template comments ---
    rule2_removed: List[Tuple[str, str]] = []
    rule2_kept: List[Dict[str, object]] = []

    for row in rule1_kept:
        text = str(row["comment_text"])
        cid = str(row["comment_id"])
        if is_template_comment(text):
            rule2_removed.append((cid, text))
        else:
            rule2_kept.append(row)

    print(f"\n[Rule 2] Mostly template / date-stamp comments removed: {len(rule2_removed)}")
    print(f"Sample of 15 comments removed by Rule 2:")
    sample_r2 = random.sample(rule2_removed, min(15, len(rule2_removed))) if rule2_removed else []
    for cid, txt in sample_r2:
        print(f"  {cid}: {ascii(txt)}")

    # --- Rule 4: Near-exact deduplication ---
    rule4_removed: List[Tuple[str, str]] = []
    rule4_kept: List[Dict[str, object]] = []
    seen_texts = set()

    for row in rule2_kept:
        text = str(row["comment_text"])
        cid = str(row["comment_id"])
        norm = normalize_for_near_dedupe(text)
        if norm in seen_texts:
            rule4_removed.append((cid, text))
        else:
            seen_texts.add(norm)
            rule4_kept.append(row)

    print(f"\n[Rule 4] Near-exact duplicate comments removed: {len(rule4_removed)}")
    if rule4_removed:
        print(f"Sample of comments removed by Rule 4:")
        sample_r4 = random.sample(rule4_removed, min(15, len(rule4_removed)))
        for cid, txt in sample_r4:
            print(f"  {cid}: {ascii(txt)}")

    # --- Rule 3: Add LID confidence ---
    for row in rule4_kept:
        row["lid_confidence"] = get_lid_confidence(str(row["comment_text"]))

    pool_df = pd.DataFrame(rule4_kept)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pool_df.to_csv(output_path, index=False, encoding="utf-8")

    low_count = (pool_df["lid_confidence"] == "low").sum()
    high_count = (pool_df["lid_confidence"] == "high").sum()

    print("\n" + "=" * 65)
    print("ANNOTATION POOL SUMMARY")
    print("=" * 65)
    print(f"  Starting Tanglish comments   : {total_initial:>5}")
    print(f"  Removed by Rule 1 (< 3 tokens): {len(rule1_removed):>5}")
    print(f"  Removed by Rule 2 (templates) : {len(rule2_removed):>5}")
    print(f"  Removed by Rule 4 (near-dupes): {len(rule4_removed):>5}")
    print(f"  Final Annotation Pool size   : {len(pool_df):>5}")
    print(f"  LID Confidence breakdown     :")
    print(f"    - High confidence (lexicon/script): {high_count:>5} ({high_count / len(pool_df) * 100:.1f}%)")
    print(f"    - Low confidence (suffix-only)    : {low_count:>5} ({low_count / len(pool_df) * 100:.1f}%)")
    print(f"  Saved pool to                : {output_path}")
    print("=" * 65 + "\n")

    return pool_df


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create clean annotation pool from Tanglish comments.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    build_annotation_pool(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()

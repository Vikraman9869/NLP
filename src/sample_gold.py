"""Sample gold annotations from the pool using two-stage stratification.

Stratifies:
1. By video_id proportionally with an enforced minimum allocation per video.
2. By lid_confidence within each video so low-confidence comments appear at their natural rate.

Outputs blind annotation file to data/annotation/gold_to_label.csv with blank sentiment,
sarcasm, and notes columns (no model predictions).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL_INPUT = PROJECT_ROOT / "data" / "annotation" / "pool.csv"
DEFAULT_GOLD_OUTPUT = PROJECT_ROOT / "data" / "annotation" / "gold_to_label.csv"

GOLD_FIELDNAMES = [
    "comment_id",
    "video_id",
    "comment_text",
    "sentiment",
    "sarcasm",
    "notes",
]


def allocate_stratified_video_samples(
    video_counts: Dict[str, int],
    total_target: int,
    min_per_video: int,
) -> Dict[str, int]:
    """Calculate sample quotas per video proportionally with a minimum constraint."""
    n_videos = len(video_counts)
    if total_target < n_videos * min_per_video:
        raise ValueError(
            f"Target sample size ({total_target}) cannot satisfy min_per_video ({min_per_video}) "
            f"for {n_videos} videos (requires at least {n_videos * min_per_video})."
        )

    # Base minimum allocation
    alloc = {v: min_per_video for v in video_counts}
    remaining = total_target - (n_videos * min_per_video)

    # Proportional distribution of remaining slots using Largest Remainder Method
    pool_total = sum(video_counts.values())
    raw_shares = {v: (c / pool_total) * remaining for v, c in video_counts.items()}
    floor_shares = {v: int(np.floor(w)) for v, w in raw_shares.items()}
    leftover = remaining - sum(floor_shares.values())

    remainders = sorted(
        [(raw_shares[v] - floor_shares[v], v) for v in video_counts],
        reverse=True,
    )
    for i in range(leftover):
        floor_shares[remainders[i][1]] += 1

    for v in video_counts:
        alloc[v] += floor_shares[v]

    return alloc


def draw_stratified_gold_sample(
    pool_path: Path,
    output_path: Path,
    n_sample: int = 500,
    seed: int = 42,
    min_per_video: int = 10,
) -> pd.DataFrame:
    """Draw two-stage stratified gold sample and write blind annotation CSV."""
    if not pool_path.exists():
        raise FileNotFoundError(f"Annotation pool not found at: {pool_path}")

    pool = pd.read_csv(pool_path)
    video_counts = pool["video_id"].value_counts().to_dict()

    # Step 1: Video-level proportional allocation
    allocations = allocate_stratified_video_samples(
        video_counts=video_counts,
        total_target=n_sample,
        min_per_video=min_per_video,
    )

    # Step 2: Second stratification by lid_confidence within each video
    sample_subsets: List[pd.DataFrame] = []

    for v, target_n in allocations.items():
        df_v = pool[pool["video_id"] == v]
        df_low = df_v[df_v["lid_confidence"] == "low"]
        df_high = df_v[df_v["lid_confidence"] == "high"]

        prop_low = len(df_low) / len(df_v) if len(df_v) > 0 else 0.0
        n_low = int(np.round(target_n * prop_low))
        n_low = min(n_low, len(df_low))
        n_high = target_n - n_low

        s_low = (
            df_low.sample(n=n_low, random_state=seed)
            if n_low > 0
            else df_low.iloc[0:0]
        )
        s_high = (
            df_high.sample(n=n_high, random_state=seed)
            if n_high > 0
            else df_high.iloc[0:0]
        )

        sample_subsets.append(pd.concat([s_low, s_high]))

    gold_sample = pd.concat(sample_subsets, ignore_index=True)

    # Ensure blind format (blank annotation targets, no model predictions)
    gold_export = pd.DataFrame()
    gold_export["comment_id"] = gold_sample["comment_id"]
    gold_export["video_id"] = gold_sample["video_id"]
    gold_export["comment_text"] = gold_sample["comment_text"]
    gold_export["sentiment"] = ""
    gold_export["sarcasm"] = ""
    gold_export["notes"] = ""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gold_export.to_csv(output_path, index=False, encoding="utf-8")

    pct_low = (gold_sample["lid_confidence"] == "low").mean() * 100
    print("\n" + "=" * 65)
    print("GOLD SAMPLE SELECTION SUMMARY")
    print("=" * 65)
    print(f"  Target sample size             : {n_sample}")
    print(f"  Actual sampled comments        : {len(gold_export)}")
    print(f"  Unique videos represented      : {gold_export['video_id'].nunique()}")
    print(f"  Minimum allocation per video   : {min_per_video}")
    print(f"  LID Low-confidence in sample   : {(gold_sample['lid_confidence'] == 'low').sum()} ({pct_low:.2f}%)")
    print(f"  Saved blind gold annotations to: {output_path}")
    print("=" * 65 + "\n")

    return gold_export


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample blind gold annotations stratified by video and LID confidence."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_POOL_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_GOLD_OUTPUT)
    parser.add_argument("--n", type=int, default=500, help="Number of comments to sample (default: 500).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument(
        "--min-per-video",
        type=int,
        default=10,
        help="Minimum comments to draw per video (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    draw_stratified_gold_sample(
        pool_path=args.input,
        output_path=args.output,
        n_sample=args.n,
        seed=args.seed,
        min_per_video=args.min_per_video,
    )


if __name__ == "__main__":
    main()

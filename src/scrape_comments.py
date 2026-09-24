"""Collect top-level YouTube comments for the Tanglish NLP project.

The API key is read from YOUTUBE_API_KEY in a .env file or the environment.
Example:
    python src/scrape_comments.py --video-ids VIDEO_ID_1 VIDEO_ID_2
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "comments.csv"


def parse_arguments() -> argparse.Namespace:
    """Read video IDs and scraper settings from the command line."""
    parser = argparse.ArgumentParser(
        description="Fetch top-level comments from YouTube videos."
    )
    parser.add_argument(
        "--video-ids",
        nargs="+",
        required=True,
        help="One or more YouTube video IDs.",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=2000,
        help="Maximum comments to collect across all videos (default: 2000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def create_youtube_client(api_key: str):
    """Create a YouTube Data API v3 client using the supplied API key."""
    return build("youtube", "v3", developerKey=api_key)


def fetch_comments(
    youtube_client,
    video_id: str,
    remaining_comments: int,
) -> List[Dict[str, str]]:
    """Fetch top-level comments for one video, following all page tokens."""
    comments: List[Dict[str, str]] = []
    next_page_token = None

    while remaining_comments > 0:
        request_size = min(100, remaining_comments)
        response = (
            youtube_client.commentThreads()
            .list(
                part="snippet",
                videoId=video_id,
                maxResults=request_size,
                pageToken=next_page_token,
                textFormat="plainText",
            )
            .execute()
        )

        for item in response.get("items", []):
            snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            comments.append(
                {
                    "comment_id": item.get("id", ""),
                    "video_id": video_id,
                    "comment_text": snippet.get("textDisplay", "").strip(),
                }
            )

        remaining_comments = remaining_comments - len(response.get("items", []))
        next_page_token = response.get("nextPageToken")
        if not next_page_token or not response.get("items"):
            break

    return comments


def collect_comments(
    youtube_client,
    video_ids: Iterable[str],
    max_comments: int,
) -> List[Dict[str, str]]:
    """Collect comments from each video until the global limit is reached."""
    all_comments: List[Dict[str, str]] = []

    for video_id in video_ids:
        if len(all_comments) >= max_comments:
            break
        video_limit = max_comments - len(all_comments)
        try:
            video_comments = fetch_comments(youtube_client, video_id, video_limit)
            all_comments.extend(video_comments)
            print(f"Collected {len(video_comments)} comments from video {video_id}.")
        except HttpError as error:
            status = getattr(error.resp, "status", "unknown")
            if status == 403:
                print(
                    f"Skipping {video_id}: API quota/permission error or comments are disabled."
                )
            elif status == 404:
                print(f"Skipping {video_id}: video was not found.")
            else:
                print(f"Skipping {video_id}: YouTube API error (HTTP {status}).")
        except Exception as error:
            print(f"Skipping {video_id}: unexpected error: {error}")

    return all_comments[:max_comments]


def save_comments(comments: List[Dict[str, str]], output_path: Path) -> None:
    """Write collected comments to a CSV file, including an empty-result file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["comment_id", "video_id", "comment_text"]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comments)

    print(f"Saved {len(comments)} comments to {output_path}.")


def main() -> None:
    """Load configuration, collect comments, and save the output CSV."""
    args = parse_arguments()
    if args.max_comments <= 0:
        raise ValueError("--max-comments must be greater than zero.")

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "YOUTUBE_API_KEY was not found. Add it to a .env file in the project root."
        )

    youtube_client = create_youtube_client(api_key)
    comments = collect_comments(youtube_client, args.video_ids, args.max_comments)
    save_comments(comments, args.output)


if __name__ == "__main__":
    main()

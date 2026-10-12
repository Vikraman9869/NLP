"""Collect top-level YouTube comments for the Tanglish NLP project.

Fetches top-level comments across specified YouTube video IDs via the YouTube Data
API v3, normalizes and flags code-mixed Tanglish content, deduplicates by comment_id
(and near-duplicates for Tanglish subsets), and records run statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import TOKEN_PATTERN, tag_token

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "comments.csv"
DEFAULT_TANGLISH_OUTPUT = PROJECT_ROOT / "data" / "raw" / "comments_tanglish.csv"
DEFAULT_STATS_OUTPUT = PROJECT_ROOT / "reports" / "scrape_stats.csv"
DEFAULT_PROGRESS_PATH = PROJECT_ROOT / "reports" / "scrape_progress.json"
DEFAULT_ERROR_LOG = PROJECT_ROOT / "reports" / "scrape_errors.log"

COMMENT_FIELDNAMES = [
    "comment_id",
    "video_id",
    "category",
    "video_title",
    "channel_title",
    "published_at",
    "comment_text",
    "has_tamil_roman",
    "has_tamil_script",
    "is_code_mixed",
]

STATS_FIELDNAMES = [
    "video_id",
    "category",
    "video_title",
    "channel_title",
    "comments_fetched",
    "new_after_dedupe",
    "pct_code_mixed",
]


def extract_video_id(raw_str: str) -> Optional[str]:
    """Extract standard 11-character YouTube video ID from URL or bare string."""
    cleaned = raw_str.strip().strip("<>\"' \t\r\n")
    if not cleaned:
        return None

    # Check if directly an 11-char ID
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", cleaned):
        return cleaned

    # Match common YouTube URL formats (watch?v=, youtu.be/, shorts/, embed/)
    match = re.search(r"(?:v=|\/|youtu\.be\/|shorts\/)([a-zA-Z0-9_-]{11})", cleaned)
    if match:
        return match.group(1)

    return None


def parse_arguments() -> argparse.Namespace:
    """Read CLI arguments for scraping YouTube comments."""
    parser = argparse.ArgumentParser(
        description="Scrape YouTube comments, flag Tanglish tokens, and append to dataset."
    )
    parser.add_argument(
        "--video-file",
        type=Path,
        default=None,
        help="Path to CSV containing video IDs and categories (e.g., data/video_list.csv).",
    )
    parser.add_argument(
        "--video-ids",
        "--video-id",
        dest="video_ids",
        nargs="*",
        default=[],
        help="One or more YouTube video IDs or URLs (space or comma-separated).",
    )
    parser.add_argument(
        "--max-per-video",
        type=int,
        default=500,
        help="Maximum comments to collect per video (default: 500).",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=None,
        help="Optional global limit on total comments to collect across the entire run.",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=None,
        help="Alias for --max-total for backwards compatibility.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path for all comments CSV (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--tanglish-output",
        type=Path,
        default=DEFAULT_TANGLISH_OUTPUT,
        help=f"Path for Tanglish-only comments CSV (default: {DEFAULT_TANGLISH_OUTPUT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print list of target videos and limits without making any API calls.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip videos recorded as completed in reports/scrape_progress.json.",
    )
    return parser.parse_args()


def load_video_targets(
    video_file: Optional[Path],
    cli_video_inputs: List[str],
) -> List[Dict[str, str]]:
    """Load video targets from CSV and/or CLI arguments, deduplicating while preserving order."""
    targets: List[Dict[str, str]] = []
    seen_ids: Set[str] = set()

    # 1. Read from video file if provided
    if video_file is not None:
        if not video_file.exists():
            raise FileNotFoundError(f"Specified --video-file was not found: {video_file}")
        with video_file.open("r", encoding="utf-8", errors="replace") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                raw_id = row.get("video_id") or ""
                vid = extract_video_id(raw_id)
                category = (row.get("category") or "").strip() or "unknown"
                if vid and vid not in seen_ids:
                    seen_ids.add(vid)
                    targets.append({"video_id": vid, "category": category})

    # 2. Read from CLI arguments if provided
    if cli_video_inputs:
        flat_cli: List[str] = []
        for item in cli_video_inputs:
            for part in item.split(","):
                part = part.strip()
                if part:
                    flat_cli.append(part)

        for item in flat_cli:
            vid = extract_video_id(item)
            if vid and vid not in seen_ids:
                seen_ids.add(vid)
                targets.append({"video_id": vid, "category": "unknown"})

    return targets


def compute_tanglish_flags(text: str) -> Tuple[bool, bool, bool]:
    """Compute (has_tamil_roman, has_tamil_script, is_code_mixed) using preprocess LID."""
    tokens = TOKEN_PATTERN.findall(str(text))
    has_tamil_roman = False
    has_tamil_script = False
    has_english = False

    for token in tokens:
        tag = tag_token(token)
        if tag == "Tamil-Roman":
            has_tamil_roman = True
        elif tag == "Tamil-Script":
            has_tamil_script = True
        elif tag == "English":
            has_english = True

    is_code_mixed = (has_tamil_roman or has_tamil_script) and has_english
    return has_tamil_roman, has_tamil_script, is_code_mixed


def normalize_for_near_dedupe(text: str) -> str:
    """Normalize text (lowercase, alphanumeric, collapsed spaces) for near-exact deduplication."""
    cleaned = re.sub(r"[^\w\s]", "", str(text).lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def log_scrape_error(
    log_path: Path,
    video_id: str,
    error_type: str,
    message: str,
) -> None:
    """Append structured error entry to scrape error log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    log_line = f"[{timestamp}] [VIDEO_ID: {video_id}] [{error_type}] {message}\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(log_line)


def load_scrape_progress(progress_path: Path) -> Dict[str, dict]:
    """Load dictionary of video progress records from scrape_progress.json."""
    if not progress_path.exists():
        return {}
    try:
        with progress_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            vids = data.get("completed_videos", {})
            if isinstance(vids, list):
                return {v: {"cap": 0, "exhausted": False} for v in vids}
            if isinstance(vids, dict):
                return vids
            return {}
    except Exception:
        return {}


def is_video_complete_for_cap(progress_data: Dict[str, dict], video_id: str, target_cap: int) -> bool:
    """Check if video is completed for the target cap or ran out of comments."""
    record = progress_data.get(video_id)
    if not record:
        return False
    if record.get("exhausted", False):
        return True
    if record.get("cap", 0) >= target_cap:
        return True
    return False


def save_scrape_progress(progress_path: Path, progress_records: Dict[str, dict]) -> None:
    """Atomically write video progress records to scrape_progress.json."""
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = progress_path.with_suffix(".json.tmp")
    data = {
        "completed_videos": progress_records,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(progress_path)


def save_atomic_csv(
    data: List[Dict[str, object]],
    fieldnames: List[str],
    target_path: Path,
) -> None:
    """Write rows atomically using a temporary file in the same directory."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)
    tmp_path.replace(target_path)


def load_and_upgrade_existing_comments(csv_path: Path) -> List[Dict[str, object]]:
    """Load existing comments and backfill missing schema fields and Tanglish flags."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []

    comments: List[Dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = (row.get("comment_id") or "").strip()
            if not cid:
                continue

            vid = (row.get("video_id") or "").strip() or "YrXAUN3p1IM"
            category = (row.get("category") or "").strip() or "unknown"
            video_title = (row.get("video_title") or "").strip()
            channel_title = (row.get("channel_title") or "").strip()
            published_at = (row.get("published_at") or "").strip()
            comment_text = (row.get("comment_text") or "").strip()

            # Backfill or preserve Tanglish flags
            has_tr_raw = row.get("has_tamil_roman")
            has_ts_raw = row.get("has_tamil_script")
            is_cm_raw = row.get("is_code_mixed")

            if has_tr_raw is None or has_ts_raw is None or is_cm_raw is None:
                has_tr, has_ts, is_cm = compute_tanglish_flags(comment_text)
            else:
                has_tr = str(has_tr_raw).strip().lower() == "true"
                has_ts = str(has_ts_raw).strip().lower() == "true"
                is_cm = str(is_cm_raw).strip().lower() == "true"

            comments.append(
                {
                    "comment_id": cid,
                    "video_id": vid,
                    "category": category,
                    "video_title": video_title,
                    "channel_title": channel_title,
                    "published_at": published_at,
                    "comment_text": comment_text,
                    "has_tamil_roman": has_tr,
                    "has_tamil_script": has_ts,
                    "is_code_mixed": is_cm,
                }
            )

    return comments


def create_youtube_client(api_key: str):
    """Create a YouTube Data API v3 client using the supplied API key."""
    return build("youtube", "v3", developerKey=api_key)


def fetch_video_metadata(
    youtube_client,
    video_id: str,
    error_log: Path,
) -> Tuple[str, str]:
    """Fetch video title and channel title using videos().list (1 quota unit).

    Includes retry with exponential backoff for transient errors.
    """
    backoff_seconds = [2, 5, 10]
    for attempt in range(len(backoff_seconds) + 1):
        try:
            response = (
                youtube_client.videos()
                .list(part="snippet", id=video_id)
                .execute()
            )
            items = response.get("items", [])
            if items:
                snippet = items[0].get("snippet", {})
                return snippet.get("title", "").strip(), snippet.get("channelTitle", "").strip()
            return "", ""
        except HttpError as error:
            status = getattr(error.resp, "status", 0)
            if status == 404:
                log_scrape_error(error_log, video_id, "404", "Video not found during metadata lookup.")
                return "", ""
            if status == 403:
                # Quota or permission error handled during commentThreads
                return "", ""
            if attempt < len(backoff_seconds):
                time.sleep(backoff_seconds[attempt])
            else:
                log_scrape_error(error_log, video_id, f"HTTP_{status}", f"Metadata error: {error}")
                return "", ""
        except Exception as error:
            if attempt < len(backoff_seconds):
                time.sleep(backoff_seconds[attempt])
            else:
                log_scrape_error(error_log, video_id, "NETWORK_ERROR", f"Metadata error: {error}")
                return "", ""

    return "", ""


def parse_http_error_reason(error: HttpError) -> str:
    """Extract granular error reason from YouTube API HttpError response body."""
    try:
        content = error.content.decode("utf-8") if isinstance(error.content, bytes) else str(error.content)
        data = json.loads(content)
        errors = data.get("error", {}).get("errors", [])
        if errors:
            return errors[0].get("reason", "")
    except Exception:
        pass
    return ""


def fetch_comments_for_video(
    youtube_client,
    video_id: str,
    category: str,
    video_title: str,
    channel_title: str,
    max_comments: int,
    error_log: Path,
) -> Tuple[List[Dict[str, object]], bool, bool, str]:
    """Fetch comments for a video with relevance ordering and pagination.

    Returns:
        Tuple of (comments_list, is_exhausted, is_done, termination_status)
        where termination_status is 'OK', 'COMMENTS_DISABLED', 'QUOTA_EXCEEDED', 'NOT_FOUND', or 'ERROR'.
    """
    comments: List[Dict[str, object]] = []
    next_page_token = None
    remaining_comments = max_comments
    backoff_seconds = [2, 5, 10]

    while remaining_comments > 0:
        request_size = min(100, remaining_comments)
        response = None
        success = False

        for attempt in range(len(backoff_seconds) + 1):
            try:
                response = (
                    youtube_client.commentThreads()
                    .list(
                        part="snippet",
                        videoId=video_id,
                        maxResults=request_size,
                        pageToken=next_page_token,
                        order="relevance",
                        textFormat="plainText",
                    )
                    .execute()
                )
                success = True
                break
            except HttpError as error:
                status = getattr(error.resp, "status", 0)
                reason = parse_http_error_reason(error)

                if status == 403:
                    if reason == "commentsDisabled" or "commentsDisabled" in str(error):
                        log_scrape_error(error_log, video_id, "403_COMMENTS_DISABLED", "Comments are disabled for this video.")
                        print(f"  [Warning] Comments are disabled for video {video_id}. Skipping.")
                        return comments, True, True, "COMMENTS_DISABLED"
                    if reason in ("quotaExceeded", "dailyLimitExceeded") or "quotaExceeded" in str(error):
                        log_scrape_error(error_log, video_id, "403_QUOTA_EXCEEDED", "Daily YouTube API quota exceeded.")
                        return comments, False, False, "QUOTA_EXCEEDED"

                    log_scrape_error(error_log, video_id, "403_FORBIDDEN", f"Access forbidden (reason: {reason}).")
                    return comments, False, False, "ERROR"

                if status == 404:
                    log_scrape_error(error_log, video_id, "404_NOT_FOUND", "Video was not found or is private.")
                    print(f"  [Warning] Video {video_id} not found or private. Skipping.")
                    return comments, True, True, "NOT_FOUND"

                # Retry transient 5xx or server errors
                if attempt < len(backoff_seconds):
                    time.sleep(backoff_seconds[attempt])
                else:
                    log_scrape_error(error_log, video_id, f"HTTP_{status}", str(error))
                    return comments, False, False, "ERROR"

            except Exception as error:
                if attempt < len(backoff_seconds):
                    time.sleep(backoff_seconds[attempt])
                else:
                    log_scrape_error(error_log, video_id, "NETWORK_ERROR", str(error))
                    return comments, False, False, "ERROR"

        if not success or response is None:
            break

        items = response.get("items", [])
        for item in items:
            snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            text_display = snippet.get("textDisplay", "").strip()
            comment_id = item.get("id", "").strip()
            published_at = snippet.get("publishedAt", "").strip()

            if comment_id and text_display:
                has_tr, has_ts, is_cm = compute_tanglish_flags(text_display)
                comments.append(
                    {
                        "comment_id": comment_id,
                        "video_id": video_id,
                        "category": category,
                        "video_title": video_title,
                        "channel_title": channel_title,
                        "published_at": published_at,
                        "comment_text": text_display,
                        "has_tamil_roman": has_tr,
                        "has_tamil_script": has_ts,
                        "is_code_mixed": is_cm,
                    }
                )

        remaining_comments -= len(items)
        next_page_token = response.get("nextPageToken")
        if not next_page_token or not items:
            # All pages consumed: exhausted all available comments
            return comments, True, True, "OK"

    # Completed by reaching max_comments cap
    return comments, False, True, "OK"


def update_and_save_tanglish_comments(
    all_comments: List[Dict[str, object]],
    tanglish_output_path: Path,
) -> int:
    """Filter Tamil-bearing comments and write them with near-exact text deduplication."""
    filtered: List[Dict[str, object]] = []
    seen_ids: Set[str] = set()
    seen_texts: Set[str] = set()

    for c in all_comments:
        has_tr = bool(c.get("has_tamil_roman", False))
        has_ts = bool(c.get("has_tamil_script", False))
        if not (has_tr or has_ts):
            continue

        cid = str(c.get("comment_id", ""))
        text = str(c.get("comment_text", ""))
        normalized = normalize_for_near_dedupe(text)

        if not cid or cid in seen_ids:
            continue
        if not normalized or normalized in seen_texts:
            continue

        seen_ids.add(cid)
        seen_texts.add(normalized)
        filtered.append(c)

    save_atomic_csv(filtered, COMMENT_FIELDNAMES, tanglish_output_path)
    return len(filtered)


def main() -> None:
    """Coordinate scraper execution, deduplication, Tanglish tagging, and reporting."""
    args = parse_arguments()

    # Resolve limits
    effective_max_total = args.max_total
    if args.max_comments is not None:
        effective_max_total = args.max_comments

    if args.max_per_video <= 0:
        raise ValueError("--max-per-video must be greater than zero.")
    if effective_max_total is not None and effective_max_total <= 0:
        raise ValueError("--max-total must be greater than zero.")

    # Load targets
    targets = load_video_targets(args.video_file, args.video_ids)
    if not targets:
        raise ValueError("No video IDs provided. Use --video-file or --video-ids.")

    # Handle --dry-run
    if args.dry_run:
        print("\n" + "=" * 65)
        print("DRY RUN MODE: Planned Video Scraping")
        print("=" * 65)
        print(f"Total Videos Configured: {len(targets)}")
        print(f"Per-Video Comment Limit: {args.max_per_video}")
        print(f"Global Run Limit       : {effective_max_total if effective_max_total is not None else 'None (unbounded)'}")
        print("-" * 65)
        print(f"{'#':<4}{'Video ID':<15}{'Category':<15}{'Per-Video Cap':<15}")
        print("-" * 65)

        progress_records = load_scrape_progress(DEFAULT_PROGRESS_PATH) if args.resume else {}
        for idx, t in enumerate(targets, start=1):
            vid = t["video_id"]
            cat = t["category"]
            status_note = " [Resume: Skip]" if is_video_complete_for_cap(progress_records, vid, args.max_per_video) else ""
            print(f"{idx:<4}{vid:<15}{cat:<15}{args.max_per_video}{status_note}")

        print("=" * 65)
        print("Dry run completed. No API calls were made.\n")
        return

    # Load API Key
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError(
            "YOUTUBE_API_KEY was not found. Add it to a .env file in the project root."
        )

    youtube_client = create_youtube_client(api_key)

    # Load existing comments and upgrade schema
    all_comments = load_and_upgrade_existing_comments(args.output)
    existing_count = len(all_comments)
    existing_cids: Set[str] = {str(c["comment_id"]) for c in all_comments}

    # Save upgraded existing comments atomically if output didn't exist or had older schema
    save_atomic_csv(all_comments, COMMENT_FIELDNAMES, args.output)
    tanglish_count = update_and_save_tanglish_comments(all_comments, args.tanglish_output)

    print("\n" + "=" * 65)
    print("Dataset Status Prior to Collection")
    print("=" * 65)
    print(f"  Existing comments in dataset    : {existing_count}")
    print(f"  Existing Tanglish/Tamil comments: {tanglish_count}")
    print("=" * 65 + "\n")

    # Load progress for --resume
    progress_records = load_scrape_progress(DEFAULT_PROGRESS_PATH) if args.resume else {}

    # Load existing stats if available
    stats_records: List[Dict[str, object]] = []
    if DEFAULT_STATS_OUTPUT.exists():
        with DEFAULT_STATS_OUTPUT.open("r", encoding="utf-8", errors="replace") as sf:
            reader = csv.DictReader(sf)
            for row in reader:
                stats_records.append(dict(row))

    total_newly_appended = 0

    for target in targets:
        vid = target["video_id"]
        cat = target["category"]

        if args.resume and is_video_complete_for_cap(progress_records, vid, args.max_per_video):
            print(f"Skipping {vid}: already completed for cap >= {args.max_per_video} (or ran out of comments).")
            continue

        if effective_max_total is not None and total_newly_appended >= effective_max_total:
            print(f"\nReached global collection limit of {effective_max_total}. Stopping run.")
            break

        video_cap = args.max_per_video
        if effective_max_total is not None:
            remaining_global = effective_max_total - total_newly_appended
            video_cap = min(video_cap, remaining_global)

        print(f"\nProcessing Video [{vid}] (Category: {cat}, Target Cap: {video_cap}) ...")

        # 1. Fetch Video Metadata
        video_title, channel_title = fetch_video_metadata(youtube_client, vid, DEFAULT_ERROR_LOG)
        print(f"  Title  : {video_title or '[Not Available]'}")
        print(f"  Channel: {channel_title or '[Not Available]'}")

        # 2. Fetch Comments
        fetched_comments, is_exhausted, is_done, status = fetch_comments_for_video(
            youtube_client=youtube_client,
            video_id=vid,
            category=cat,
            video_title=video_title,
            channel_title=channel_title,
            max_comments=video_cap,
            error_log=DEFAULT_ERROR_LOG,
        )

        if status == "QUOTA_EXCEEDED":
            print("\n" + "!" * 65)
            print("CRITICAL: YouTube API daily quota exceeded!")
            print("Saving progress and stopping entire run immediately.")
            print("!" * 65 + "\n")
            break

        # 3. Deduplicate and Append
        new_for_video: List[Dict[str, object]] = []
        for c in fetched_comments:
            cid = str(c["comment_id"])
            if cid not in existing_cids:
                existing_cids.add(cid)
                new_for_video.append(c)

        all_comments.extend(new_for_video)
        total_newly_appended += len(new_for_video)

        # 4. Compute Video-level stats
        n_fetched = len(fetched_comments)
        n_new = len(new_for_video)
        n_code_mixed = sum(1 for c in fetched_comments if c.get("is_code_mixed", False))
        pct_cm = round((n_code_mixed / n_fetched) * 100, 2) if n_fetched > 0 else 0.0

        print(f"  Fetched: {n_fetched} | New after dedupe: {n_new} | Code-mixed: {n_code_mixed} ({pct_cm}%)")

        # Update stats records
        stat_entry = {
            "video_id": vid,
            "category": cat,
            "video_title": video_title,
            "channel_title": channel_title,
            "comments_fetched": n_fetched,
            "new_after_dedupe": n_new,
            "pct_code_mixed": pct_cm,
        }
        # Replace if video already recorded, else append
        updated = False
        for idx, row in enumerate(stats_records):
            if row.get("video_id") == vid:
                stats_records[idx] = stat_entry
                updated = True
                break
        if not updated:
            stats_records.append(stat_entry)

        # 5. Progressive Atomic Saves
        save_atomic_csv(all_comments, COMMENT_FIELDNAMES, args.output)
        update_and_save_tanglish_comments(all_comments, args.tanglish_output)
        save_atomic_csv(stats_records, STATS_FIELDNAMES, DEFAULT_STATS_OUTPUT)

        # 6. Mark completed in progress tracker if fully completed or hit per-video cap
        if is_done and status in ("OK", "COMMENTS_DISABLED", "NOT_FOUND"):
            exhausted_flag = is_exhausted or (status in ("COMMENTS_DISABLED", "NOT_FOUND"))
            prev_cap = progress_records.get(vid, {}).get("cap", 0)
            new_cap = max(prev_cap, video_cap) if not exhausted_flag else prev_cap
            progress_records[vid] = {
                "cap": new_cap,
                "exhausted": exhausted_flag,
                "comments_fetched": n_fetched,
                "last_scraped": datetime.now(timezone.utc).isoformat(),
            }
            save_scrape_progress(DEFAULT_PROGRESS_PATH, progress_records)

    print("\n" + "=" * 65)
    print("Scraping Session Finished")
    print("=" * 65)
    print(f"  Total comments in dataset      : {len(all_comments)}")
    print(f"  Newly added comments in this run: {total_newly_appended}")
    print(f"  Stats saved to                 : {DEFAULT_STATS_OUTPUT}")
    print(f"  Tanglish comments saved to     : {args.tanglish_output}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

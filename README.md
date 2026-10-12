# NLP_Cat - Tanglish NLP Pipeline

An end-to-end NLP processing pipeline for collecting, preprocessing, language-tagging, and auto-labeling code-mixed **Tanglish** (Tamil written in Latin/Roman script mixed with English) YouTube comments for sentiment analysis and sarcasm detection.

---

## 📌 Project Overview

Code-mixed social media text in Indian languages presents unique linguistic challenges such as non-standard phonetic spellings, code-switching between Tamil and English, and informal colloquialisms. 

This repository provides:
1. **Comment Scraper (`src/scrape_comments.py`)**: Fetches top-level comments across specified YouTube video IDs or an input CSV (`data/video_list.csv`) via the YouTube Data API v3 (`order=relevance`), retrieves video metadata (`videos.list`), flags Tanglish tokens, deduplicates records, and tracks scrape progress.
2. **Preprocessor & Language Tagger (`src/preprocess.py`)**: Cleans raw comments, normalizes elongated character sequences (3+ repeated letters down to 2) and phonetic spelling variants, and applies 4-class token-level language identification (`Tamil-Script`, `Tamil-Roman`, `English`, `Other`).
3. **Token Inspectors (`src/inspect_tokens.py`, `src/inspect_english_tokens.py`)**: Audits token distributions, inspects high-frequency English words to uncover uncaptured Tanglish vocabulary, and tests morphological suffix fallback accuracy.
4. **Code-Mixing Analysis (`src/code_mixing_stats.py`)**: Computes the Code-Mixing Index (CMI) (Gambäck & Das, 2014) and generates dataset-level and per-comment linguistic distribution reports.
5. **Zero-Shot Auto-Labeler (`src/auto_label.py`)**: Leverages transformer zero-shot classification (`facebook/bart-large-mnli`) to generate preliminary sentiment and sarcasm labels with confidence scores for human annotation assistance.

---

## 📁 Repository Structure

```text
NLP_Cat/
├── data/
│   ├── video_list.csv            # Target video IDs, categories, and notes
│   ├── raw/
│   │   ├── comments.csv          # All scraped YouTube comments with metadata & LID flags
│   │   └── comments_tanglish.csv # Filtered Tanglish comments (near-duplicate filtered)
│   └── processed/
│       └── cleaned.csv           # Cleaned & token-tagged comments
├── reports/
│   ├── scrape_stats.csv          # Per-video scrape volume and % code-mixed breakdown
│   ├── scrape_progress.json      # Progress tracker recording per-video caps & completion
│   ├── scrape_errors.log         # Error log for private/disabled comments or API issues
│   ├── code_mixing_stats.csv     # Dataset-level CMI and linguistic distribution
│   └── per_comment_code_mixing.csv # Per-comment breakdown with CMI values
├── src/
│   ├── scrape_comments.py        # YouTube API scraper with Tanglish flagging & resume
│   ├── preprocess.py             # Normalization, repeated-char collapse, 4-class token LID
│   ├── inspect_tokens.py         # Multi-mode inspector (Tamil, suffix-only, English tokens)
│   ├── inspect_english_tokens.py # Utility to audit top English tokens
│   ├── code_mixing_stats.py      # Code-Mixing Index (CMI) calculation & distribution report
│   └── auto_label.py             # Zero-shot classification pipeline
├── .env.example                  # Template for required environment variables
├── .gitignore                    # Git ignore file for secrets and build artifacts
├── requirements.txt              # Python package dependencies
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.9+ (Python 3.11 recommended)
- A YouTube Data API v3 key from the [Google Cloud Console](https://console.cloud.google.com/)

> [!NOTE]
> **API Quota Information**: The default free YouTube Data API quota is **10,000 units per day**. Each `commentThreads.list` request costs **1 quota unit** (up to 100 comments per page), and fetching metadata via `videos.list` costs **1 quota unit** per video. Scraping 17 videos up to 500 comments each requires only **~102 quota units** (~1% of daily quota).

### 2. Clone the Repository

```bash
git clone https://github.com/Vikraman9869/NLP_Cat.git
cd NLP_Cat
```

### 3. Set Up Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API Key

Copy `.env.example` to `.env` and insert your YouTube Data API key:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
YOUTUBE_API_KEY=your_actual_youtube_api_key_here
```

---

## 🛠️ Usage

### Step 1: Scrape YouTube Comments

Configure target video IDs in `data/video_list.csv` (columns: `video_id`, `category`, `notes`). Entries may be bare 11-character IDs or full YouTube URLs (`watch?v=`, `youtu.be/`, `shorts/`).

#### Option A: Scrape using video list file with resume
```bash
python src/scrape_comments.py --video-file data/video_list.csv --max-per-video 500 --resume
```

#### Option B: Dry run (inspect targets and caps without making API calls)
```bash
python src/scrape_comments.py --video-file data/video_list.csv --max-per-video 500 --dry-run
```

#### Option C: Scrape specific video IDs directly
```bash
python src/scrape_comments.py --video-ids VIDEO_ID_1 VIDEO_ID_2 --max-per-video 500
```

- **Outputs**:
  - `data/raw/comments.csv`: All scraped comments with columns `comment_id`, `video_id`, `category`, `video_title`, `channel_title`, `published_at`, `comment_text`, `has_tamil_roman`, `has_tamil_script`, `is_code_mixed`.
  - `data/raw/comments_tanglish.csv`: Curated subset containing only comments with at least one Tamil token (`has_tamil_roman` or `has_tamil_script`), deduplicated by `comment_id` and normalized text.
  - `reports/scrape_stats.csv`: Per-video fetch counts and code-mixed percentages.
  - `reports/scrape_progress.json`: State file tracking scraped caps and completed videos.

### Step 2: Preprocess and Tag Language Tokens

Clean raw comments, collapse repeated elongated characters (3+ down to 2), normalize spelling variants, and apply 4-class token-level language identification (`Tamil-Script`, `Tamil-Roman`, `English`, `Other`):

```bash
python src/preprocess.py --input data/raw/comments.csv --output data/processed/cleaned.csv
```

- Output: Saves `cleaned_text`, `token_tags` (`token:Tag`) to `data/processed/cleaned.csv`.

### Step 3: Audit Token Language Identification

Inspect token distributions to verify accuracy and discover new vocabulary:

```bash
# Audit top tokens tagged as English to discover uncaptured Tanglish words
python src/inspect_tokens.py --mode english --top 100

# Audit tokens tagged as Tamil-Roman only via suffix heuristics
python src/inspect_tokens.py --mode suffix-only --top 100

# Quick inspection of English tokens
python src/inspect_english_tokens.py --top 100
```

### Step 4: Compute Code-Mixing Statistics & CMI

Analyze language distribution and calculate the Code-Mixing Index (Gambäck & Das, 2014):

```bash
python src/code_mixing_stats.py
```

- Outputs:
  - `reports/code_mixing_stats.csv`: Dataset-level summary table (mean CMI, language breakdown).
  - `reports/per_comment_code_mixing.csv`: Granular per-comment metrics and CMI values.

### Step 5: Zero-Shot Auto-Labeling (Optional)

Generate first-pass sentiment and sarcasm classifications:

```bash
python src/auto_label.py --input data/processed/cleaned.csv --output data/processed/cleaned_autolabeled.csv
```

Candidate labels evaluated: `positive`, `negative`, `neutral`, `sarcastic`.

---

## 📄 License

This project is licensed under the MIT License.

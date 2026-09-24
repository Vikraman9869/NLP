# NLP_Cat - Tanglish NLP Pipeline

An end-to-end NLP processing pipeline for collecting, preprocessing, language-tagging, and auto-labeling code-mixed **Tanglish** (Tamil written in Latin/Roman script mixed with English) YouTube comments for sentiment analysis and sarcasm detection.

---

## 📌 Project Overview

Code-mixed social media text in Indian languages presents unique linguistic challenges such as non-standard phonetic spellings, code-switching between Tamil and English, and informal colloquialisms. 

This repository provides:
1. **Comment Scraper (`src/scrape_comments.py`)**: Fetches top-level comments across specified YouTube video IDs via the YouTube Data API v3, managing pagination and quota limitations.
2. **Preprocessor & Language Tagger (`src/preprocess.py`)**: Cleans raw comments, normalizes common phonetic spelling variations, and applies token-level language classification (`Tamil-Roman`, `English`, `Mixed`).
3. **Zero-Shot Auto-Labeler (`src/auto_label.py`)**: Leverages transformer zero-shot classification (`facebook/bart-large-mnli`) to generate preliminary sentiment and sarcasm labels with confidence scores for human annotation assistance.

---

## 📁 Repository Structure

```text
NLP_Cat/
├── data/
│   ├── raw/
│   │   └── comments.csv          # Scraped YouTube comments
│   └── processed/
│       └── cleaned.csv           # Cleaned & token-tagged comments
├── src/
│   ├── scrape_comments.py        # YouTube API scraper
│   ├── preprocess.py             # Normalization and token language tagger
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

Fetch comments from one or more YouTube videos by providing their video IDs:

```bash
python src/scrape_comments.py --video-ids <VIDEO_ID_1> <VIDEO_ID_2> --max-comments 2000
```

- Output: Saved to `data/raw/comments.csv`

### Step 2: Preprocess and Tag Language Tokens

Clean raw comments, normalize informal spelling variations, and tag token languages:

```bash
python src/preprocess.py --input data/raw/comments.csv --output data/processed/cleaned.csv
```

- Output: Adds `original_text`, `cleaned_text`, and `token_tags` to `data/processed/cleaned.csv`

### Step 3: Zero-Shot Auto-Labeling (Optional)

Generate first-pass sentiment and sarcasm classifications:

```bash
python src/auto_label.py --input data/processed/cleaned.csv --output data/processed/cleaned_autolabeled.csv
```

Candidate labels evaluated: `positive`, `negative`, `neutral`, `sarcastic`.

---

## 📄 License

This project is licensed under the MIT License.

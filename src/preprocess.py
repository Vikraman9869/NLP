"""Clean, normalize, and language-tag code-mixed Tanglish YouTube comments.

This module provides preprocessing for Tanglish (Tamil in Latin/Roman script
mixed with English) comments:
1. Normalization: Collapses elongated character sequences (3+ repeated letters
   down to 2) and maps colloquial spelling variants to canonical forms.
2. Token-level Language Identification (LID):
   - 'Tamil-Roman': Romanized Tamil identified via an extensive lexicon and
     tightened morphological suffix heuristics.
   - 'Tamil-Script': Native Tamil script Unicode tokens (U+0B80 to U+0BFF).
   - 'English': Standard English words and ASCII terms not identified as Tamil.
   - 'Other': Punctuation, numerals, emojis, and symbols.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "comments.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "cleaned.csv"

# Regex to detect 3 or more repeated letters (case-insensitive)
REPEATED_CHAR_PATTERN = re.compile(r"([a-zA-Z])\1{2,}", re.IGNORECASE)

# Pattern to detect native Tamil script Unicode characters (U+0B80 to U+0BFF)
TAMIL_SCRIPT_PATTERN = re.compile(r"^[\u0B80-\u0BFF]+$")

# Token extraction pattern: captures native Tamil words, Romanized words
# (with optional contractions), digits, and individual non-whitespace characters/emojis
TOKEN_PATTERN = re.compile(r"[\u0B80-\u0BFF]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", re.UNICODE)


def collapse_repeated_characters(text: str) -> str:
    """Collapse sequences of 3 or more identical letters down to 2.

    Preserves legitimate double letters (e.g., 'good', 'cool', 'mass'),
    while reducing expressive social media elongations:
    - 'superrrrr' -> 'superr' (subsequently mapped to 'super' in normalization)
    - 'massssss'  -> 'mass'
    - 'goooood'   -> 'good'
    - 'ennna'     -> 'enna'
    - 'yarelllam' -> 'yarellam'
    """
    return REPEATED_CHAR_PATTERN.sub(r"\1\1", text)


# Canonical spelling mapping for common colloquial Tanglish & English elongations
NORMALIZATION_MAP: Dict[str, str] = {
    # Emphatic elongations & variations of common descriptors
    "superr": "super",
    "superrr": "super",
    "masss": "mass",
    "semmaa": "semma",
    "seema": "semma",
    "simma": "semma",
    "nallaa": "nalla",
    "rombaa": "romba",
    "rombave": "romba",
    "thalaa": "thala",
    "machaa": "macha",
    "broo": "bro",
    "plss": "pls",
    "plzz": "pls",
    "plz": "pls",
    "pleasee": "please",
    "mee": "me",
    "assamble": "assemble",
    # Negations and confirmations
    "illai": "illa",
    "illae": "illa",
    "illaa": "illa",
    "aamaa": "aama",
    # Question word variants
    "yarelam": "yarellam",
    "yarelllam": "yarellam",
    "yaarelam": "yarellam",
    "yarallam": "yarellam",
    "yaralam": "yarellam",
    "yarulam": "yarellam",
    "yaarellam": "yarellam",
    "yaarulam": "yarellam",
    "yarlam": "yarellam",
    "yarellem": "yarellam",
    "yaarulaam": "yarellam",
    "yarula": "yaru",
    "yaaru": "yaru",
    "epadi": "epdi",
    "apparum": "aprom",
    "eruku": "iruku",
    "erukanga": "irukanga",
    "paathom": "pathom",
    # Verb variants
    "pakuringaa": "pakuringa",
    "pakkuringaa": "pakuringa",
    "pakkuringaaaa": "pakuringa",
    "pakkurigaa": "pakuringa",
    "pakurengaa": "pakuringa",
    "paakuringge": "pakuringa",
    "paakuringe": "pakuringa",
    "pakuriga": "pakuringa",
    "pakkuriga": "pakuringa",
    "pakkuringa": "pakuringa",
    "pakringa": "pakuringa",
    "pakringal": "pakuringa",
    "pakiringa": "pakuringa",
    "pakaringa": "pakuringa",
    "paakureenga": "pakuringa",
    "irukingaa": "irukinga",
    "irukingala": "irukinga",
    # Pronoun & particle variants
    "naa": "naan",
    "thaa": "tha",
    "apo": "appo",
    "yenna": "enna",
    "yellam": "ellam",
    "virundhu": "virunthu",
    "indha": "intha",
    "idhu": "ithu",
    "adhu": "athu",
    "edhu": "ethu",
}

# Curated Romanized Tamil lexicon (Tanglish vocabulary)
TAMIL_ROMAN_WORDS: Set[str] = {
    # Pronouns, Demonstratives & Question words
    "aama", "aamanda", "aamama", "naan", "naa", "na", "nee", "neenga", "neengal",
    "avan", "avanga", "avargal", "ava", "avaru", "athu", "adhu", "ithu", "idhu",
    "edhu", "ethu", "inga", "anga", "enga", "intha", "indha", "andha", "anda",
    "endha", "entha", "ethana", "yaar", "yaaru", "yaru", "yarellam", "yaarelam",
    "yaralam", "yarallam", "yarulam", "yaarukellam", "yarula", "yaravathu", "yar",
    "enna", "ennathu", "ennada", "ennadi", "edhukku", "ethuku", "epdi", "eppadi",
    "apdi", "appadi", "yean", "yen", "yenna", "enaku", "enakku", "unaku", "unakku",
    "ungaluku", "ungalukku", "engaluku", "engalukku", "namakku", "namma", "nammaluku",
    "yarlam", "yellam", "naanum", "avangalum", "namellam", "ivanga", "avara",

    # Particles, Greetings & Colloquial Address Terms
    "da", "di", "machi", "machan", "mams", "thala", "thalapathy", "thalaiva",
    "anna", "thambi", "akka", "thangachi", "nanba", "nanbi", "dosthu", "pa", "ba",
    "ma", "yo", "yow", "nga", "la", "le", "dho", "nu", "thaan", "dhan", "dan",
    "tha", "thaa", "illa", "illada", "illaye", "illamuraya", "kooda", "mattum",
    "mattume", "seri", "sari", "podhum", "pothum", "vaazhthukkal", "vanakkam",
    "nandri", "kondattam", "aatam", "ku", "kku", "lam", "ya", "ah", "haahaa",

    # Sentiment, Evaluative & Quality Descriptors
    "semma", "nalla", "nallava", "nallave", "romba", "rombave", "arputham", "arumai",
    "vera", "gethu", "marana", "attakasam", "mass", "azhagu", "azhaga",
    "supera", "theva", "thevai", "kodumai", "mosam", "kevalam", "bore", "thuppu",
    "kirukku", "paithiyam", "vetti", "bayangara", "bayangarama", "dayangaram",
    "sirappu", "sirappa", "vetri", "sandhosham", "kavala", "azhugai", "siripu",
    "alappara", "verithanam", "theekori", "therikka", "kolla", "tharam", "tharamaana",
    "alavilla", "perumai", "jeyicha", "thothu", "sandhoshama", "kavalapadathe",
    "kolaaru", "gubeer", "uruttu", "vada", "peela", "loosu", "thillalangadi",

    # Verbs, Actions & Verb Inflections
    "paaru", "paarka", "paarkalam", "paapom", "papom", "paathutu", "pathutu", "paakuranga",
    "paakureenga", "paakuringa", "pakuringa", "pakkuringa", "pakuriga", "pakkuriga",
    "pakringa", "pakringal", "pakiringa", "pakaringa", "pakkurigaa", "pakuringaa",
    "pakkuringaa", "paatha", "paathen", "paaka", "pathalum", "paakuringe",
    "paakuringge", "pakkiriga", "sollu", "solra", "solranga", "solreenga", "solren",
    "sonna", "solla", "sola", "pannu", "pannunga", "panuga", "panranga", "panreenga",
    "panran", "panren", "panna", "panniten", "panniyachi", "vaanga", "vaa",
    "vandhutan", "vandhuten", "vanthuten", "vanthu", "vandhu", "vara", "varum",
    "varan", "varuthu", "ponga", "po", "pona", "poiten", "irukku", "iruku", "irukan",
    "iruken", "iruka", "irukanga", "irukeenga", "irukinga", "irukingala", "theriyum",
    "theriyala", "puriya", "puriyala", "puriyudhu", "purinjithu", "kudukka",
    "kudunga", "podu", "podunga", "potanga", "vitu", "vidu", "mudiyum", "mudiyadhu",
    "mudila", "mudiyala", "kekala", "aachu", "aayidichu", "aagum", "aagala",
    "pesu", "pesunga", "pesura", "pesuvom", "oduthu", "odra", "padikira", "paadara",
    "kekkudhu", "venum", "venuma", "vaaram", "kandupidi", "mudinjathu", "poachu",
    "pochu", "pakkala", "erundhom", "tharela",

    # Adverbs, Quantifiers, Entities & Locations
    "konjam", "konjama", "chinna", "periya", "ellam", "ellaam", "ellarum",
    "ellorukum", "ella", "neraiya", "koraivu", "adutha", "munnadi", "pinnaadi",
    "ippo", "eppo", "appo", "apo", "aprom", "inniku", "innaiku", "naalaiku", "nethu",
    "kaalaile", "iravu", "madhiyame", "veliya", "ulla", "virunthu", "virundhu",
    "oru", "rendu", "moonu", "naalu", "anju", "aaru", "ezhu", "ettu", "ombodhu",
    "pathu", "nallaavum", "kodaikanal", "kodai", "madurai", "chennai", "siddhu", "sidhu",
    "yarelam", "pathom", "vaati", "kudi", "yetha", "vinayagar", "anmai", "aanmaiya",
    "carla", "yaarla", "athula", "engala", "ennala", "mela", "fulla", "apdiye",
    "enkaku", "eniku", "irruku", "pakku", "sooriku", "pakurathuku",
}

# Tightened Romanized Tamil grammatical suffixes for unseen token fallback
# Note: Loose patterns such as '-le', '-om', '-ala', and '-ave' have been removed
# to avoid false positives with common English words (e.g., 'while', 'from', 'brave').
TAMIL_ROMAN_SUFFIXES = (
    # Honorific / plural / second-person verb endings:
    "reenga", "reengal", "reengala", "ringal", "ringa", "riga", "unga", "ungal", "nga", "ngal",
    "ngala", "ngalaa", "gala", "galaa",
    # Dative case endings:
    "ukku", "kku", "uku", "thuku", "luku",
    # Locative / ablative endings:
    "kitta", "gitta", "thula",
    # Present / past tense verb person markers:
    "ranga", "rangal", "rangan", "ren", "rom", "ran", "ral",
    # Neuter 3rd person & past/perfective markers:
    "dhu", "thu", "aachu", "achu", "iduchu", "ichu", "pochu", "athu",
    # Negations:
    "yala", "aathu", "aatha",
    # Modal / inclusive / genitive / emphatic:
    "aalam", "alam", "aavathu", "avathu", "oda", "ooda",
    "aave", "laye", "ume",
)

# High-frequency English vocabulary to prevent false positive suffix matches
COMMON_ENGLISH_WORDS: Set[str] = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
    "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my",
    "one", "all", "would", "there", "their", "what", "so", "up", "out", "if",
    "about", "who", "get", "which", "go", "me", "when", "make", "can", "like",
    "time", "no", "just", "him", "know", "take", "people", "into", "year", "your",
    "good", "some", "could", "them", "see", "other", "than", "then", "now",
    "look", "only", "come", "its", "over", "think", "also", "back", "after",
    "use", "two", "how", "our", "work", "first", "well", "way", "even", "new",
    "want", "because", "any", "these", "give", "day", "most", "us", "anyone",
    "anybody", "someone", "watching", "watched", "watch", "watchers", "rewatch",
    "rewatching", "today", "yesterday", "tomorrow", "september", "august", "july",
    "june", "may", "april", "march", "february", "january", "sep", "aug", "jul",
    "jun", "apr", "mar", "feb", "jan", "oct", "nov", "dec", "december", "october",
    "november", "comment", "comments", "section", "series", "video", "videos",
    "fans", "combo", "movie", "assemble", "assembly", "vlog", "vlogs", "room",
    "details", "location", "resort", "stay", "trip", "place", "still", "night",
    "midnight", "calendar", "attendance", "daily", "times", "viewers", "views",
    "celebration", "subscribers", "million", "happy", "best", "super", "nice",
    "love", "dance", "guys", "bro", "name", "am", "pm", "is", "are", "was",
    "were", "has", "had", "been", "wait", "waiting", "full", "part", "same",
    "six", "hit", "squad", "friday", "sunday", "saturday", "monday", "every",
    "birthday", "dubai", "malaysia", "ultimate", "fire", "episode", "season",
    "link", "channel", "please", "thanks", "pls", "album", "forum", "medium",
    "stadium", "children", "film", "show", "story", "level", "while", "middle",
    "title", "file", "male", "whole", "watchable", "brave", "tittle", "uncountable",
    "unforgettable",
}


def normalize_token(token: str) -> str:
    """Normalize a token by collapsing 3+ repeated characters and mapping known variants."""
    lower_token = collapse_repeated_characters(token.lower())
    return NORMALIZATION_MAP.get(lower_token, lower_token)


def normalize_text(text: str) -> str:
    """Extract tokens, normalize elongated spellings, and join into cleaned text."""
    tokens = TOKEN_PATTERN.findall(str(text))
    normalized_tokens = [normalize_token(token) for token in tokens]
    return " ".join(normalized_tokens).strip()


def tag_token(token: str) -> str:
    """Classify token language into 'Tamil-Roman', 'Tamil-Script', 'English', or 'Other'.

    Decision hierarchy:
    1. If token is in native Tamil script (U+0B80 to U+0BFF) -> 'Tamil-Script'.
    2. If token is not purely alphabetic -> 'Other' (numbers, punctuation, emojis).
    3. If normalized token matches known Tanglish lexicon -> 'Tamil-Roman'.
    4. If token is in common English vocabulary -> 'English'.
    5. If token matches tightened Tanglish suffix rules (min length 4) -> 'Tamil-Roman'.
    6. Fallback: If ASCII alphabetic -> 'English'; otherwise -> 'Other'.
    """
    # Check 1: Native Tamil script
    if TAMIL_SCRIPT_PATTERN.match(token):
        return "Tamil-Script"

    normalized_token = normalize_token(token)

    # Check 2: Non-alphabetic tokens (emojis, numbers, punctuation)
    if not normalized_token.isalpha():
        return "Other"

    # Check 3: Curated Tanglish vocabulary
    if normalized_token in TAMIL_ROMAN_WORDS:
        return "Tamil-Roman"

    # Check 4: Known English vocabulary to prevent suffix false positives
    if normalized_token in COMMON_ENGLISH_WORDS:
        return "English"

    # Check 5: Morphological suffix heuristics for unseen Romanized Tamil words
    if len(normalized_token) >= 4 and any(
        normalized_token.endswith(suffix) for suffix in TAMIL_ROMAN_SUFFIXES
    ):
        return "Tamil-Roman"

    # Check 6: Fallback for ASCII English text
    if normalized_token.isascii():
        return "English"

    return "Other"


def tag_text(text: str) -> str:
    """Return token-level language tags in 'token:Tag' format preserving input order."""
    tokens = TOKEN_PATTERN.findall(str(text))
    return " ".join(f"{token}:{tag_token(token)}" for token in tokens)


def preprocess_comments(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Load raw comments, clean them, tag token languages, and save to CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV was not found: {input_path}")

    comments = pd.read_csv(input_path)
    required_columns = {"comment_id", "video_id", "comment_text"}
    missing_columns = required_columns.difference(comments.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    comments["original_text"] = comments["comment_text"].fillna("").astype(str)
    comments["cleaned_text"] = comments["original_text"].map(normalize_text)
    comments["token_tags"] = comments["original_text"].map(tag_text)
    comments["label"] = ""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comments.to_csv(output_path, index=False, encoding="utf-8")

    # Compute and display tag distribution summary for verification
    all_tags: List[str] = []
    for tags_str in comments["token_tags"]:
        for pair in tags_str.split():
            if ":" in pair:
                all_tags.append(pair.rsplit(":", 1)[1])

    tag_series = pd.Series(all_tags)
    print(f"Processed {len(comments)} comments.")
    print(f"Saved cleaned data to {output_path}.")
    print("\nToken Language Tag Distribution:")
    for tag, count in tag_series.value_counts().items():
        percentage = (count / len(tag_series)) * 100
        print(f"  {tag:<14}: {count:>5} tokens ({percentage:.1f}%)")

    return comments


def parse_arguments() -> argparse.Namespace:
    """Read optional input and output paths from the command line."""
    parser = argparse.ArgumentParser(description="Preprocess Tanglish comments.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Run preprocessing using command-line paths or project defaults."""
    args = parse_arguments()
    preprocess_comments(args.input, args.output)


if __name__ == "__main__":
    main()

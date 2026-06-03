"""
Preprocessor — data quality layer for the Lead Scoring Engine.
Cleans, deduplicates, and flags comments before they reach the LLM classifier.
"""
import re
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "comment_id", "lead_id", "username", "comment_text", "platform",
    "post_context", "lead_type", "competitor_mentioned",
    "pain_point_category", "writing_style",
    "comment_timestamp", "comment_age_days", "hour_of_day",
    "day_of_week", "is_recent", "lead_comment_number", "lead_total_comments",
]

ENGLISH_STOPWORDS = {"the", "is", "are", "with", "for", "my", "and", "to", "of", "in", "it", "this", "that", "was"}

SPAM_PATTERNS = [
    r"^[\U00010000-\U0010ffff\s]+$",  # emoji-only
    r"^(#+\w+\s*)+$",                  # pure hashtag string
    r"@(free|follow|win|giveaway)\w*", # promotional handles
    r"\bfollow\s*(me|for\s*follow|back)\b",  # follower-bait
    r"\bfollow\s*4\s*follow\b",
    r"\bf4f\b",
    r"\bl4l\b",
    r"^.{0,20}$",                      # very short (handled also by word count but catch here)
]

_SPAM_RE = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in SPAM_PATTERNS]
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"&\w+;|&#\d+;")
_WS_RE = re.compile(r"\s+")


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    def _clean(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = _URL_RE.sub("", text)
        text = _HTML_RE.sub("", text)
        text = _WS_RE.sub(" ", text)
        return text.strip()

    df = df.copy()
    df["comment_text_clean"] = df["comment_text"].apply(_clean)
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["username", "comment_text"])
    dropped = before - len(df)
    logger.info(f"Deduplication: dropped {dropped} duplicate rows ({before} → {len(df)})")
    return df.reset_index(drop=True)


def detect_language(df: pd.DataFrame) -> pd.DataFrame:
    def _is_english(text: str) -> bool:
        if not isinstance(text, str):
            return False
        words = set(text.lower().split())
        return bool(words & ENGLISH_STOPWORDS)

    df = df.copy()
    df["is_english"] = df["comment_text_clean"].apply(_is_english)
    return df


def flag_spam(df: pd.DataFrame) -> pd.DataFrame:
    def _is_spam(text: str) -> bool:
        if not isinstance(text, str):
            return True
        word_count = len(text.split())
        if word_count < 3:
            return True
        for pattern in _SPAM_RE:
            if pattern.search(text):
                return True
        return False

    df = df.copy()
    df["is_spam"] = df["comment_text_clean"].apply(_is_spam)
    return df


def add_word_count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["word_count"] = df["comment_text_clean"].apply(lambda t: len(str(t).split()) if isinstance(t, str) else 0)
    return df


def mark_classifiable(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_classifiable"] = (
        (df["word_count"] >= 15) &
        (df["is_english"] == True) &
        (df["is_spam"] == False)
    )
    return df


def run_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    df = load_data(input_path)
    total_loaded = len(df)

    df = clean_text(df)
    df = deduplicate(df)
    after_dedup = len(df)

    df = detect_language(df)
    df = flag_spam(df)
    df = add_word_count(df)
    df = mark_classifiable(df)

    non_english = int((~df["is_english"]).sum())
    spam = int(df["is_spam"].sum())
    classifiable = int(df["is_classifiable"].sum())

    df.to_csv(output_path, index=False)

    logger.info("\n--- Preprocessor Summary ---")
    logger.info(f"Total rows loaded:        {total_loaded}")
    logger.info(f"Rows after deduplication: {after_dedup}")
    logger.info(f"Rows flagged non-English: {non_english}")
    logger.info(f"Rows flagged spam:        {spam}")
    logger.info(f"Rows marked classifiable: {classifiable}")
    logger.info(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    run_pipeline(
        input_path="data/processed/mock_leads.csv",
        output_path="data/processed/preprocessed_leads.csv",
    )

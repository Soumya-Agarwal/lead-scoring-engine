"""Tests for src/mock_generator.py"""
import sys
import os
from datetime import datetime, timedelta

import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.mock_generator import generate_leads

REQUIRED_COLUMNS = [
    "comment_id", "lead_id", "username", "comment_text", "platform", "post_context",
    "lead_type", "competitor_mentioned", "pain_point_category", "writing_style",
    "comment_timestamp", "comment_age_days", "hour_of_day", "day_of_week",
    "is_recent", "lead_comment_number", "lead_total_comments", "days_since_first_comment",
]


@pytest.fixture(scope="module")
def df():
    return generate_leads(500)


def test_total_comment_count(df):
    assert len(df) == 500, f"Expected 500 comments, got {len(df)}"


def test_required_columns(df):
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_noise_rows_present(df):
    noise = df[df["lead_type"] == "noise"]
    assert len(noise) > 0, "No noise rows found"
    assert (noise["pain_point_category"] == "none").all(), "Noise rows should have pain_point = none"
    assert (noise["competitor_mentioned"] == "none").all(), "Noise rows should have no competitor"
    assert (noise["lead_total_comments"] == 1).all(), "Noise leads should have exactly 1 comment"


def test_lead_type_distribution(df):
    """Distribution measured against signal comments only."""
    leads = df[df["lead_type"] != "noise"]
    counts = leads["lead_type"].value_counts()
    total = len(leads)
    venter_pct    = counts.get("venter",    0) / total
    seeker_pct    = counts.get("seeker",    0) / total
    evaluator_pct = counts.get("evaluator", 0) / total
    assert abs(venter_pct    - 0.50) <= 0.12, f"Venter pct={venter_pct:.2f}"
    assert abs(seeker_pct    - 0.30) <= 0.12, f"Seeker pct={seeker_pct:.2f}"
    assert abs(evaluator_pct - 0.20) <= 0.12, f"Evaluator pct={evaluator_pct:.2f}"


def test_no_null_comment_text(df):
    assert df["comment_text"].isnull().sum() == 0


def test_no_duplicate_comment_ids(df):
    assert df["comment_id"].nunique() == len(df), "Duplicate comment_ids found"


def test_multiple_comments_per_lead_exist(df):
    """At least some signal leads should have more than one comment."""
    signal = df[df["lead_type"] != "noise"]
    multi = signal[signal["lead_total_comments"] > 1]
    assert len(multi) > 0, "No leads with multiple comments found"


def test_comment_arc_ordering(df):
    """lead_comment_number should increment correctly within each lead."""
    signal = df[df["lead_type"] != "noise"]
    for lead_id, group in signal.groupby("lead_id"):
        nums = sorted(group["lead_comment_number"].tolist())
        assert nums == list(range(1, len(nums) + 1)), \
            f"lead {lead_id} has bad comment numbering: {nums}"


def test_temporal_fields_valid(df):
    assert df["hour_of_day"].between(0, 23).all(), "hour_of_day out of range"
    assert df["day_of_week"].between(0, 6).all(), "day_of_week out of range"
    assert df["comment_age_days"].ge(0).all(), "negative comment_age_days"


def test_timestamps_within_90_days(df):
    now = datetime(2026, 6, 2, 12, 0, 0)
    cutoff = now - timedelta(days=90)
    df_copy = df.copy()
    df_copy["comment_timestamp"] = pd.to_datetime(df_copy["comment_timestamp"])
    assert (df_copy["comment_timestamp"] >= cutoff).all(), "Timestamps older than 90 days found"
    assert (df_copy["comment_timestamp"] <= now).all(), "Future timestamps found"


def test_is_recent_flag_correct(df):
    """is_recent should be True only for comments <= 7 days old."""
    assert (df.loc[df["comment_age_days"] <= 7, "is_recent"] == True).all()
    assert (df.loc[df["comment_age_days"] > 7,  "is_recent"] == False).all()


def test_days_since_first_comment_zero_for_first(df):
    """First comment of each lead should have days_since_first_comment = 0."""
    first_comments = df[df["lead_comment_number"] == 1]
    assert (first_comments["days_since_first_comment"] == 0.0).all(), \
        "First comments should have days_since_first_comment = 0"

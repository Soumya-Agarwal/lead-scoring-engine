"""Tests for src/scorer.py"""
import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.scorer import (
    score_row, assign_tier, apply_confidence_weight,
    temporal_urgency_boost, temporal_intent_boost,
    WEIGHTS, INTENT_SCORES, URGENCY_SCORES, FIT_SCORES,
)


def make_row(intent, urgency, pain, intent_conf=1.0, urgency_conf=1.0, pain_conf=1.0,
             is_recent=False, hour_of_day=14, lead_comment_number=1, lead_total_comments=1):
    return pd.Series({
        "intent_label":          intent,
        "urgency_label":         urgency,
        "pain_point_label":      pain,
        "intent_confidence":     intent_conf,
        "urgency_confidence":    urgency_conf,
        "pain_point_confidence": pain_conf,
        "is_recent":             is_recent,
        "hour_of_day":           hour_of_day,
        "lead_comment_number":   lead_comment_number,
        "lead_total_comments":   lead_total_comments,
    })


def test_composite_formula_correct():
    row = make_row("switching intent", "extremely urgent", "hidden fees or pricing complaints")
    result = score_row(row)
    expected = round(10 * 0.40 + 10 * 0.35 + 9 * 0.25, 2)
    assert result["composite_score"] == expected


def test_scores_clamp_minimum():
    row = make_row("no intent signal", "not urgent", "no pain point", 0.1, 0.1, 0.1)
    result = score_row(row)
    assert result["intent_score"] >= 1.0
    assert result["urgency_score"] >= 1.0
    assert result["fit_score"] >= 1.0
    assert result["composite_score"] >= 1.0


def test_high_composite_for_best_signals():
    row = make_row("switching intent", "extremely urgent", "hidden fees or pricing complaints")
    result = score_row(row)
    assert result["composite_score"] >= 9.0, f"Expected high score, got {result['composite_score']}"


def test_low_composite_for_no_signals():
    row = make_row("no intent signal", "not urgent", "no pain point")
    result = score_row(row)
    assert result["composite_score"] < 5.0, f"Expected low score, got {result['composite_score']}"


def test_lead_tier_hot():
    assert assign_tier(7.5) == "hot"
    assert assign_tier(9.0) == "hot"
    assert assign_tier(10.0) == "hot"


def test_lead_tier_warm():
    assert assign_tier(5.0) == "warm"
    assert assign_tier(6.5) == "warm"
    assert assign_tier(7.49) == "warm"


def test_lead_tier_cold():
    assert assign_tier(4.99) == "cold"
    assert assign_tier(1.0) == "cold"
    assert assign_tier(3.0) == "cold"


def test_confidence_weight_low():
    adjusted = apply_confidence_weight(10.0, 0.4)
    assert adjusted == 8.0


def test_confidence_weight_medium():
    adjusted = apply_confidence_weight(10.0, 0.6)
    assert adjusted == 9.0


def test_confidence_weight_high():
    adjusted = apply_confidence_weight(10.0, 0.8)
    assert adjusted == 10.0


def test_confidence_weight_minimum_floor():
    adjusted = apply_confidence_weight(1.0, 0.1)
    assert adjusted >= 1.0


# --- Temporal boost tests ---

def test_recent_comment_boosts_urgency():
    row_recent  = make_row("general complaint", "low urgency", "no pain point", is_recent=True)
    row_old     = make_row("general complaint", "low urgency", "no pain point", is_recent=False)
    assert score_row(row_recent)["urgency_score"] > score_row(row_old)["urgency_score"]


def test_late_night_boosts_urgency():
    row_night = make_row("general complaint", "low urgency", "no pain point", hour_of_day=22)
    row_day   = make_row("general complaint", "low urgency", "no pain point", hour_of_day=14)
    assert score_row(row_night)["urgency_score"] > score_row(row_day)["urgency_score"]


def test_escalating_comment_boosts_intent():
    row_3rd   = make_row("general complaint", "low urgency", "no pain point", lead_comment_number=3, lead_total_comments=3)
    row_1st   = make_row("general complaint", "low urgency", "no pain point", lead_comment_number=1, lead_total_comments=1)
    assert score_row(row_3rd)["intent_score"] > score_row(row_1st)["intent_score"]


def test_temporal_boosts_capped_at_10():
    """Scores should never exceed 10 even with maximum temporal boosts."""
    row = make_row("switching intent", "extremely urgent", "hidden fees or pricing complaints",
                   is_recent=True, hour_of_day=23, lead_comment_number=4, lead_total_comments=4)
    result = score_row(row)
    assert result["intent_score"]  <= 10.0
    assert result["urgency_score"] <= 10.0
    assert result["fit_score"]     <= 10.0


def test_no_temporal_data_does_not_crash():
    """Rows without temporal columns should score without error."""
    row = pd.Series({
        "intent_label": "switching intent",
        "urgency_label": "extremely urgent",
        "pain_point_label": "hidden fees or pricing complaints",
        "intent_confidence": 0.9,
        "urgency_confidence": 0.85,
        "pain_point_confidence": 0.92,
    })
    result = score_row(row)
    assert result["composite_score"] > 0

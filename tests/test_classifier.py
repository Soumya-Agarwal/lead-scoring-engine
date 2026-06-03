"""Tests for src/classifier.py — mocks all external API calls."""
import sys
import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# Patch env before importing classifier
os.environ.setdefault("HF_API_KEY", "hf_test_fake_key_for_testing")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.classifier as classifier


MOCK_ZSC_RESPONSE = {
    "labels": ["switching intent", "general complaint", "no intent signal"],
    "scores": [0.82, 0.12, 0.06],
}


def _make_classifiable_df():
    return pd.DataFrame([{
        "comment_id": "lead_001",
        "username": "test_user",
        "comment_text": "I am done with ShipMonk, lost 3 orders",
        "comment_text_clean": "I am done with ShipMonk, lost 3 orders",
        "is_classifiable": True,
        "is_english": True,
        "is_spam": False,
        "word_count": 9,
        "platform": "instagram",
        "post_context": "test",
        "timestamp": "2026-05-01",
        "lead_type": "venter",
        "competitor_mentioned": "ShipMonk",
        "pain_point_category": "errors",
        "writing_style": "raw",
    }])


def test_retry_logic_triggers_on_503():
    call_count = {"n": 0}

    def fake_post(*args, **kwargs):
        call_count["n"] += 1
        mock_resp = MagicMock()
        if call_count["n"] < 3:
            mock_resp.status_code = 503
        else:
            mock_resp.status_code = 200
            mock_resp.json.return_value = MOCK_ZSC_RESPONSE
        return mock_resp

    with patch("src.classifier.requests.post", side_effect=fake_post):
        with patch("src.classifier.time.sleep"):
            result = classifier.call_hf_api({"inputs": "test"}, retries=3, wait=1)

    assert call_count["n"] == 3
    assert result["labels"][0] == "switching intent"


def test_retry_exhausted_raises():
    mock_resp = MagicMock()
    mock_resp.status_code = 503

    with patch("src.classifier.requests.post", return_value=mock_resp):
        with patch("src.classifier.time.sleep"):
            with pytest.raises(RuntimeError, match="unavailable"):
                classifier.call_hf_api({"inputs": "test"}, retries=2, wait=1)


def test_output_columns_present_after_classification():
    expected_cols = [
        "intent_label", "intent_confidence",
        "urgency_label", "urgency_confidence",
        "pain_point_label", "pain_point_confidence",
        "classification_timestamp",
    ]

    def fake_post(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_ZSC_RESPONSE
        return mock_resp

    df = _make_classifiable_df()
    with patch("src.classifier.requests.post", side_effect=fake_post):
        with patch("src.classifier.time.sleep"):
            result = classifier.classify_dataframe(df)

    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_confidence_scores_between_0_and_1():
    def fake_post(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_ZSC_RESPONSE
        return mock_resp

    df = _make_classifiable_df()
    with patch("src.classifier.requests.post", side_effect=fake_post):
        with patch("src.classifier.time.sleep"):
            result = classifier.classify_dataframe(df)

    classifiable = result[result["is_classifiable"] == True]
    for col in ["intent_confidence", "urgency_confidence", "pain_point_confidence"]:
        vals = classifiable[col].astype(float)
        assert (vals >= 0.0).all() and (vals <= 1.0).all(), f"{col} out of [0,1] range"


def test_handles_empty_comment_text_gracefully():
    result = classifier.classify_single("")
    assert result["intent_label"] == "no intent signal"
    assert result["urgency_label"] == "not urgent"
    assert result["pain_point_label"] == "no pain point"

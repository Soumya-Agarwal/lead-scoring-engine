"""
Scoring engine — converts LLM classification output + temporal signals into composite lead scores.

Temporal urgency boosts (new in v2):
  - is_recent (posted <=7 days ago)     → +1.5 urgency bonus
  - late-night post (hour >= 21)         → +1.0 urgency bonus (emotional venting)
  - lead_comment_number >= 3             → +1.5 intent bonus (escalating lead)
  - lead_total_comments >= 3             → +1.0 intent bonus (persistent pain)
  All boosts are capped so final scores never exceed 10.
"""
import os
import pandas as pd

INTENT_SCORES = {
    "switching intent": 10,
    "seeking alternative": 8,
    "evaluating options": 6,
    "general complaint": 3,
    "no intent signal": 1,
}

URGENCY_SCORES = {
    "extremely urgent": 10,
    "moderately urgent": 7,
    "low urgency": 4,
    "not urgent": 1,
}

FIT_SCORES = {
    "hidden fees or pricing complaints": 9,
    "lost or damaged shipments": 8,
    "slow fulfillment or shipping delays": 9,
    "poor customer support": 6,
    "no pain point": 2,
}

WEIGHTS = {"intent": 0.40, "urgency": 0.35, "fit": 0.25}


def apply_confidence_weight(raw_score: float, confidence: float) -> float:
    if confidence < 0.5:
        adjusted = raw_score * 0.80
    elif confidence < 0.7:
        adjusted = raw_score * 0.90
    else:
        adjusted = raw_score
    return max(1.0, adjusted)


def temporal_urgency_boost(row: pd.Series) -> float:
    """
    Return an additive urgency boost based on temporal signals.
    Max possible boost: +2.5
    """
    boost = 0.0
    try:
        if bool(row.get("is_recent", False)):
            boost += 1.5   # posted in last 7 days — hot window
        hour = int(row.get("hour_of_day", 12))
        if hour >= 21 or hour <= 2:
            boost += 1.0   # late-night post — emotional, high urgency
    except (ValueError, TypeError):
        pass
    return boost


def temporal_intent_boost(row: pd.Series) -> float:
    """
    Return an additive intent boost based on comment escalation signals.
    Max possible boost: +2.5
    """
    boost = 0.0
    try:
        comment_num = int(row.get("lead_comment_number", 1))
        total = int(row.get("lead_total_comments", 1))
        if comment_num >= 3:
            boost += 1.5   # 3rd+ comment = escalating lead, strong intent
        if total >= 3:
            boost += 1.0   # lead has posted 3+ times = persistent pain
    except (ValueError, TypeError):
        pass
    return boost


def score_row(row: pd.Series) -> pd.Series:
    # Base LLM scores
    intent_raw  = INTENT_SCORES.get(str(row.get("intent_label",  "")).lower(), 1)
    urgency_raw = URGENCY_SCORES.get(str(row.get("urgency_label", "")).lower(), 1)
    fit_raw     = FIT_SCORES.get(str(row.get("pain_point_label", "")).lower(), 2)

    intent_conf  = float(row.get("intent_confidence",       1.0))
    urgency_conf = float(row.get("urgency_confidence",      1.0))
    pain_conf    = float(row.get("pain_point_confidence",   1.0))

    # Apply confidence weighting
    intent_score  = apply_confidence_weight(intent_raw,  intent_conf)
    urgency_score = apply_confidence_weight(urgency_raw, urgency_conf)
    fit_score     = apply_confidence_weight(fit_raw,     pain_conf)

    # Apply temporal boosts (capped at 10)
    urgency_score = min(10.0, urgency_score + temporal_urgency_boost(row))
    intent_score  = min(10.0, intent_score  + temporal_intent_boost(row))

    composite = round(
        intent_score  * WEIGHTS["intent"] +
        urgency_score * WEIGHTS["urgency"] +
        fit_score     * WEIGHTS["fit"],
        2,
    )

    return pd.Series({
        "intent_score":    round(intent_score, 2),
        "urgency_score":   round(urgency_score, 2),
        "fit_score":       round(fit_score, 2),
        "composite_score": composite,
        "lead_tier":       assign_tier(composite),
    })


def assign_tier(composite_score: float) -> str:
    if composite_score >= 7.5:
        return "hot"
    elif composite_score >= 5.0:
        return "warm"
    else:
        return "cold"


def score_leads(df: pd.DataFrame) -> pd.DataFrame:
    scores = df.apply(score_row, axis=1)
    df = pd.concat([df, scores], axis=1)
    print(f"\nScoring complete. {len(df)} comments scored.")
    print(df["lead_tier"].value_counts().to_string())
    return df


def run_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = score_leads(df)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return df


if __name__ == "__main__":
    import sys

    # Accept optional input path as CLI argument: python src/scorer.py my_file.csv
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        # Auto-select best available input file
        candidates = [
            "data/processed/classified_leads.csv",   # ideal — has LLM labels
            "data/processed/preprocessed_leads.csv", # fallback — no LLM labels yet
            "data/processed/mock_leads.csv",          # last resort — raw mock data
        ]
        input_path = next((p for p in candidates if os.path.exists(p)), None)
        if input_path is None:
            print("ERROR: No input data found. Run mock_generator.py first.")
            sys.exit(1)
        if input_path != candidates[0]:
            print(f"NOTE: classified_leads.csv not found — scoring {input_path} without LLM labels.")
            print("      Run classifier.py with a valid HF_API_KEY to get LLM-classified scores.\n")

    run_pipeline(
        input_path=input_path,
        output_path="data/processed/scored_leads.csv",
    )

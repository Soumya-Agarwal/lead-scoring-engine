"""
LLM classifier — uses HuggingFace Inference API (zero-shot) to classify each comment.
Uses raw requests — no HuggingFace SDK.
"""
import os
import time
import logging
from datetime import datetime

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise EnvironmentError("HF_API_KEY not set. Copy .env.example to .env and fill in your key.")

MODEL = os.getenv("HF_MODEL_CLASSIFICATION", "facebook/bart-large-mnli")
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"

INTENT_LABELS = [
    "switching intent",
    "seeking alternative",
    "evaluating options",
    "general complaint",
    "no intent signal",
]

URGENCY_LABELS = [
    "extremely urgent",
    "moderately urgent",
    "low urgency",
    "not urgent",
]

PAIN_LABELS = [
    "hidden fees or pricing complaints",
    "lost or damaged shipments",
    "slow fulfillment or shipping delays",
    "poor customer support",
    "no pain point",
]


def call_hf_api(payload: dict, retries: int = 3, wait: int = 20) -> dict:
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    for attempt in range(retries):
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 503:
            if attempt < retries - 1:
                logger.warning(f"Model loading (503). Waiting {wait}s before retry {attempt + 2}/{retries}...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"HuggingFace API unavailable after {retries} retries (503).")
        else:
            raise RuntimeError(f"HuggingFace API error {response.status_code}: {response.text[:200]}")
    return {}


def classify_single(text: str) -> dict:
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return {
            "intent_label": "no intent signal",
            "intent_confidence": 0.0,
            "urgency_label": "not urgent",
            "urgency_confidence": 0.0,
            "pain_point_label": "no pain point",
            "pain_point_confidence": 0.0,
        }

    results = {}

    for task_name, labels in [
        ("intent", INTENT_LABELS),
        ("urgency", URGENCY_LABELS),
        ("pain_point", PAIN_LABELS),
    ]:
        payload = {
            "inputs": text,
            "parameters": {
                "candidate_labels": labels,
                "hypothesis_template": "This text contains {}.",
            },
        }
        try:
            response = call_hf_api(payload)
            top_idx = 0
            results[f"{task_name}_label"] = response["labels"][top_idx]
            results[f"{task_name}_confidence"] = round(response["scores"][top_idx], 4)
        except Exception as e:
            logger.error(f"Classification failed for task={task_name}: {e}")
            if task_name == "intent":
                results["intent_label"] = "no intent signal"
                results["intent_confidence"] = 0.0
            elif task_name == "urgency":
                results["urgency_label"] = "not urgent"
                results["urgency_confidence"] = 0.0
            else:
                results["pain_point_label"] = "no pain point"
                results["pain_point_confidence"] = 0.0

        time.sleep(1)

    return results


def classify_dataframe(df: pd.DataFrame, test_mode: bool = False) -> pd.DataFrame:
    classifiable = df[df["is_classifiable"] == True].copy()
    skipped = df[df["is_classifiable"] != True].copy()

    if test_mode:
        classifiable = classifiable.head(5)
        logger.info("TEST MODE: classifying first 5 rows only")

    logger.info(f"Classifying {len(classifiable)} rows (skipping {len(df) - len(classifiable)} non-classifiable)...")

    classification_results = []
    batch_size = 10

    for i, (idx, row) in enumerate(classifiable.iterrows()):
        result = classify_single(row["comment_text_clean"])
        result["classification_timestamp"] = datetime.now().isoformat()
        classification_results.append((idx, result))

        if (i + 1) % batch_size == 0:
            logger.info(f"Progress: {i + 1}/{len(classifiable)} classified")

    logger.info(f"Classification complete: {len(classification_results)} rows processed")

    for idx, result in classification_results:
        for col, val in result.items():
            classifiable.loc[idx, col] = val

    # Add empty classification columns to skipped rows
    empty_cols = {
        "intent_label": "no intent signal",
        "intent_confidence": 0.0,
        "urgency_label": "not urgent",
        "urgency_confidence": 0.0,
        "pain_point_label": "no pain point",
        "pain_point_confidence": 0.0,
        "classification_timestamp": "",
    }
    for col, default in empty_cols.items():
        skipped[col] = default

    return pd.concat([classifiable, skipped], ignore_index=True)


def run_pipeline(input_path: str, output_path: str, test_mode: bool = False) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = classify_dataframe(df, test_mode=test_mode)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")
    return df


if __name__ == "__main__":
    import sys
    test_mode = "--test" in sys.argv
    run_pipeline(
        input_path="data/processed/preprocessed_leads.csv",
        output_path="data/processed/classified_leads.csv",
        test_mode=test_mode,
    )

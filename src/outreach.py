"""
Outreach generator — generates personalised DM drafts for hot and warm leads.
Uses mistralai/Mistral-7B-Instruct-v0.2 via HuggingFace Inference API.
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

MODEL = os.getenv("HF_MODEL_GENERATION", "mistralai/Mistral-7B-Instruct-v0.2")
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"

OUTREACH_PROMPT = """You are a sales development representative for a modern fulfillment platform.
A merchant posted the following comment online:
"{comment_text}"
Context:
- Their main pain point: {pain_point_label}
- Their intent signal: {intent_label}
- Competitor they mentioned: {competitor_mentioned}
Write a short, genuine LinkedIn or Instagram DM outreach message (max 3 sentences) that:
1. Acknowledges their specific pain point (do not copy their exact words)
2. Mentions one specific thing your platform does better
3. Ends with a low-pressure call to action (not "book a demo" — something softer)
Do not be salesy. Do not use jargon. Sound like a helpful human.
Output only the message text, nothing else."""


def call_hf_generation(prompt: str, retries: int = 3, wait: int = 20) -> str:
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "return_full_text": False,
        },
    }
    for attempt in range(retries):
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").strip()
            return ""
        if response.status_code == 503:
            if attempt < retries - 1:
                logger.warning(f"Model loading (503). Waiting {wait}s before retry {attempt + 2}/{retries}...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"HuggingFace API unavailable after {retries} retries.")
        else:
            raise RuntimeError(f"HuggingFace API error {response.status_code}")
    return ""


def generate_outreach(row: pd.Series) -> dict:
    prompt = OUTREACH_PROMPT.format(
        comment_text=str(row.get("comment_text", ""))[:500],
        pain_point_label=row.get("pain_point_label", "unknown"),
        intent_label=row.get("intent_label", "unknown"),
        competitor_mentioned=row.get("competitor_mentioned", "a competitor"),
    )
    try:
        draft = call_hf_generation(prompt)
        return {
            "outreach_draft": draft,
            "outreach_generated": True,
            "outreach_timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Outreach generation failed for lead_id={row.get('comment_id', '?')}: {type(e).__name__}")
        return {
            "outreach_draft": "",
            "outreach_generated": False,
            "outreach_timestamp": datetime.now().isoformat(),
        }


def generate_for_dataframe(df: pd.DataFrame, test_mode: bool = False) -> pd.DataFrame:
    eligible = df[df["composite_score"] >= 5.0].copy()
    ineligible = df[df["composite_score"] < 5.0].copy()

    if test_mode:
        eligible = eligible.head(3)
        logger.info("TEST MODE: generating outreach for first 3 hot/warm leads")

    logger.info(f"Generating outreach for {len(eligible)} hot/warm leads (skipping {len(ineligible)} cold)...")

    for idx, row in eligible.iterrows():
        result = generate_outreach(row)
        for col, val in result.items():
            eligible.loc[idx, col] = val
        time.sleep(1)

    ineligible["outreach_draft"] = ""
    ineligible["outreach_generated"] = False
    ineligible["outreach_timestamp"] = ""

    logger.info(f"Outreach generation complete.")
    return pd.concat([eligible, ineligible], ignore_index=True)


def run_pipeline(input_path: str, output_path: str, test_mode: bool = False) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = generate_for_dataframe(df, test_mode=test_mode)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")
    return df


if __name__ == "__main__":
    import sys
    test_mode = "--test" in sys.argv
    run_pipeline(
        input_path="data/processed/scored_leads.csv",
        output_path="data/processed/mock_leads_final.csv",
        test_mode=test_mode,
    )

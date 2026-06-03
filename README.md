# Lead Scoring Engine

AI pipeline that scores fulfillment leads by switching intent using prompt engineering and LLMs.

🚀 **[Live Demo → lead-scoring-engine-demo.streamlit.app](https://lead-scoring-engine-demo.streamlit.app/)**

## Architecture

```
Instagram Comments (mock data)
         │
         ▼
┌─────────────────────┐
│  mock_generator.py  │  → 250 synthetic merchant comments (Faker)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   preprocessor.py   │  → dedup · clean · language detect · spam flag
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│    classifier.py    │  → HuggingFace ZSC (bart-large-mnli)
│                     │    intent · urgency · pain point labels
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│      scorer.py      │  → composite score (intent×0.4 + urgency×0.35 + fit×0.25)
│                     │    hot / warm / cold tier assignment
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│     outreach.py     │  → personalised DM drafts (Mistral-7B-Instruct)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│       app.py        │  → Streamlit dashboard (public)
└─────────────────────┘
```

## Tech Stack

| Layer | Tool |
|---|---|
| Data generation | Python · Faker |
| LLM API | HuggingFace Inference API (raw HTTP) |
| Classification model | `facebook/bart-large-mnli` (zero-shot) |
| Generation model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Data processing | pandas |
| Dashboard | Streamlit · Plotly |
| Secret management | python-dotenv · st.secrets |
| Tests | pytest |

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Soumya-Agarwal/lead-scoring-engine.git
cd lead-scoring-engine

# 2. Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure secrets
cp .env.example .env
# Edit .env and add your HuggingFace API key

# 4. Generate mock data
python src/mock_generator.py

# 5. Run the preprocessing pipeline
python src/preprocessor.py

# 6. Run the classifier (add --test for 5-row test run first)
python src/classifier.py --test
python src/classifier.py

# 7. Score the leads
python src/scorer.py

# 8. Generate outreach drafts (add --test for 3-lead test run first)
python src/outreach.py --test
python src/outreach.py

# 9. Launch the dashboard
streamlit run app.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Notes on Data

Public demo uses synthetic data generated with Faker to avoid rate limits and respect platform ToS. No real user data is collected or stored.

## Security

No API keys are stored in this repository. See `.env.example` for required keys. The `.env` file and `secrets.toml` are gitignored and must be created locally.

## Phases

- **Phase 1** (this repo): Prompt engineering + LLM pipeline + Streamlit dashboard ✅
- **Phase 2** (coming): RAG — Fit score via ChromaDB product knowledge base retrieval
- **Phase 3** (coming): Agentic AI — LangGraph agent routes leads autonomously

## Live Demo

🚀 **[https://lead-scoring-engine-demo.streamlit.app/](https://lead-scoring-engine-demo.streamlit.app/)**

Built and deployed June 2026. Public demo runs on synthetic data (500 comments · 194 leads).

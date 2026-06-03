"""
Competitive Lead Scoring Engine — Streamlit Dashboard
Read-only display of pre-processed lead data. Never calls the HuggingFace API at runtime.
"""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HF_API_KEY = st.secrets.get("HF_API_KEY", os.getenv("HF_API_KEY", ""))

DATA_PATH    = "data/processed/mock_leads_final.csv"
FALLBACK_PATH = "data/processed/mock_leads.csv"

st.set_page_config(
    page_title="Competitive Lead Scoring Engine",
    page_icon="🎯",
    layout="wide",
)

TIER_EMOJI = {"hot": "🔴", "warm": "🟡", "cold": "🔵"}
DOW_NAMES  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@st.cache_data
def load_data() -> pd.DataFrame:
    path = DATA_PATH if os.path.exists(DATA_PATH) else FALLBACK_PATH
    df = pd.read_csv(path)

    # Ensure all expected columns exist with safe defaults
    defaults = {
        "intent_label": "no intent signal",     "urgency_label": "not urgent",
        "pain_point_label": "no pain point",     "intent_score": 1.0,
        "urgency_score": 1.0,                    "fit_score": 2.0,
        "composite_score": 1.5,                  "lead_tier": "cold",
        "outreach_draft": "",                    "outreach_generated": False,
        "intent_confidence": 0.0,               "urgency_confidence": 0.0,
        "pain_point_confidence": 0.0,
        # temporal
        "comment_age_days": 0.0,                "hour_of_day": 12,
        "day_of_week": 0,                        "is_recent": False,
        "lead_comment_number": 1,               "lead_total_comments": 1,
        "days_since_first_comment": 0.0,        "lead_id": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    for col in ["composite_score", "intent_score", "urgency_score", "fit_score",
                "comment_age_days", "days_since_first_comment"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in ["hour_of_day", "day_of_week", "lead_comment_number", "lead_total_comments"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["is_recent"] = df["is_recent"].astype(str).str.lower().isin(["true", "1"])

    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


df = load_data()

# ── Header metrics ────────────────────────────────────────────────────────────
st.title("🎯 Competitive Lead Scoring Engine")
st.markdown("**3PL / Fulfillment — AI-Powered Prospecting**")

signal_df   = df[df["lead_type"] != "noise"]
unique_leads = signal_df["lead_id"].nunique() if "lead_id" in df.columns else len(signal_df)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Comments",     len(df))
c2.metric("Unique Leads",       unique_leads)
c3.metric("Hot 🔴",             int((df["lead_tier"] == "hot").sum()))
c4.metric("Recent (≤7 days)",   int(df["is_recent"].sum()))
c5.metric("Avg Score",          f"{df['composite_score'].mean():.2f}")

st.divider()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    tier_filter = st.multiselect(
        "Lead Tier",
        options=["hot", "warm", "cold"],
        default=["hot", "warm", "cold"],
    )

    competitor_options = sorted(df["competitor_mentioned"].dropna().unique().tolist())
    competitor_filter = st.multiselect(
        "Competitor Mentioned",
        options=competitor_options,
        default=competitor_options,
    )

    pain_options = sorted(df["pain_point_category"].dropna().unique().tolist())
    pain_filter = st.multiselect(
        "Pain Point Category",
        options=pain_options,
        default=pain_options,
    )

    min_score = st.slider(
        "Min Composite Score",
        min_value=1.0, max_value=10.0, value=1.0, step=0.5,
    )

    type_options = sorted(df["lead_type"].dropna().unique().tolist())
    type_filter = st.multiselect(
        "Lead Type",
        options=type_options,
        default=[t for t in type_options if t != "noise"],
        help="'noise' rows are low-signal comments used to test classifier filtering — not real leads",
    )

    st.divider()
    st.subheader("Temporal Filters")

    recent_only = st.checkbox("Recent only (≤ 7 days)", value=False)

    max_age = st.slider(
        "Max comment age (days)",
        min_value=1, max_value=90, value=90, step=1,
    )

    min_comments = st.selectbox(
        "Min comments per lead",
        options=[1, 2, 3, 4],
        index=0,
        help="Show only leads who have posted this many times (escalation filter)",
    )

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = df[
    df["lead_tier"].isin(tier_filter) &
    df["competitor_mentioned"].isin(competitor_filter) &
    df["pain_point_category"].isin(pain_filter) &
    (df["composite_score"] >= min_score) &
    df["lead_type"].isin(type_filter) &
    (df["comment_age_days"] <= max_age) &
    (df["lead_total_comments"] >= min_comments)
].copy()

if recent_only:
    filtered = filtered[filtered["is_recent"] == True]

st.markdown(f"**{len(filtered)} comments** · **{filtered['lead_id'].nunique() if 'lead_id' in filtered.columns else '—'} unique leads** match your filters")

# ── Main table ────────────────────────────────────────────────────────────────
display_cols = [
    "rank", "username", "lead_type", "competitor_mentioned",
    "pain_point_category",
    "lead_comment_number", "lead_total_comments",
    "comment_age_days", "is_recent",
    "intent_score", "urgency_score", "fit_score",
    "composite_score", "lead_tier",
]

display_df = filtered[display_cols].copy()
display_df["lead_tier"] = display_df["lead_tier"].map(
    lambda t: f"{TIER_EMOJI.get(t, '')} {t}"
)
display_df["is_recent"] = display_df["is_recent"].map(lambda v: "✅" if v else "")

event = st.dataframe(
    display_df,
    width="stretch",
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "composite_score":      st.column_config.NumberColumn("Composite ⭐", format="%.2f"),
        "intent_score":         st.column_config.NumberColumn("Intent",        format="%.2f"),
        "urgency_score":        st.column_config.NumberColumn("Urgency",       format="%.2f"),
        "fit_score":            st.column_config.NumberColumn("Fit",           format="%.2f"),
        "comment_age_days":     st.column_config.NumberColumn("Age (days)",    format="%.1f"),
        "lead_comment_number":  st.column_config.NumberColumn("Comment #",     format="%d"),
        "lead_total_comments":  st.column_config.NumberColumn("Total posts",   format="%d"),
        "is_recent":            st.column_config.TextColumn("Recent?"),
    },
)

# ── Lead detail panel ─────────────────────────────────────────────────────────
selected_rows = event.selection.rows if event and event.selection else []

if selected_rows:
    lead = filtered.iloc[selected_rows[0]]

    st.divider()
    st.subheader(f"Comment Detail — @{lead['username']}")

    left, right = st.columns([1, 1])

    with left:
        # Identity + temporal summary
        tier_raw = str(lead["lead_tier"]).split()[-1]
        recent_badge = "🟢 Recent" if lead["is_recent"] else f"🕐 {float(lead['comment_age_days']):.0f} days ago"
        hour  = int(lead.get("hour_of_day", 12))
        dow   = DOW_NAMES[int(lead.get("day_of_week", 0))]
        arc   = int(lead.get("lead_comment_number", 1))
        total = int(lead.get("lead_total_comments", 1))
        since = float(lead.get("days_since_first_comment", 0))

        st.markdown(
            f"**Type:** `{lead['lead_type']}` &nbsp;|&nbsp; "
            f"**Tier:** {TIER_EMOJI.get(tier_raw,'')} `{tier_raw}` &nbsp;|&nbsp; "
            f"**Competitor:** `{lead['competitor_mentioned']}`"
        )

        st.markdown("##### 🕐 Temporal Signals")
        t1, t2, t3 = st.columns(3)
        t1.metric("Posted",     recent_badge)
        t2.metric("Time",       f"{dow} {hour:02d}:xx")
        t3.metric("Comment arc", f"{arc} of {total}")

        if since > 0:
            st.caption(f"First post was **{since:.0f} days ago** — this lead has been suffering that long.")
        if hour >= 21 or hour <= 2:
            st.caption("🌙 Late-night post — signals high emotional urgency.")
        if arc >= 3:
            st.caption(f"🔥 Comment #{arc} from this lead — strong escalation signal.")

        st.markdown("##### 💬 Comment")
        st.info(str(lead["comment_text"]))

        st.markdown("##### 🤖 Classification")
        st.markdown(f"- **Intent:** `{lead.get('intent_label','n/a')}` ({float(lead.get('intent_confidence',0)):.0%})")
        st.markdown(f"- **Urgency:** `{lead.get('urgency_label','n/a')}` ({float(lead.get('urgency_confidence',0)):.0%})")
        st.markdown(f"- **Pain Point:** `{lead.get('pain_point_label','n/a')}` ({float(lead.get('pain_point_confidence',0)):.0%})")

    with right:
        # Score breakdown bar chart
        fig = go.Figure(go.Bar(
            x=[float(lead["intent_score"]), float(lead["urgency_score"]), float(lead["fit_score"])],
            y=["Intent (×0.40)", "Urgency (×0.35)", "Fit (×0.25)"],
            orientation="h",
            marker_color=["#e74c3c", "#f39c12", "#2ecc71"],
            text=[f"{v:.2f}" for v in [float(lead["intent_score"]), float(lead["urgency_score"]), float(lead["fit_score"])]],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Score Breakdown — Composite: {float(lead['composite_score']):.2f}",
            xaxis=dict(range=[0, 12]),
            height=260,
            margin=dict(l=20, r=50, t=40, b=20),
        )
        st.plotly_chart(fig, width="stretch")

        # All comments from this lead
        if "lead_id" in df.columns and str(lead.get("lead_id", "")) != "":
            lead_history = df[df["lead_id"] == lead["lead_id"]].sort_values("lead_comment_number")
            if len(lead_history) > 1:
                st.markdown("##### 📜 Full comment history for this lead")
                for _, h in lead_history.iterrows():
                    age   = float(h["comment_age_days"])
                    cn    = int(h["lead_comment_number"])
                    score = float(h["composite_score"])
                    tier  = TIER_EMOJI.get(str(h["lead_tier"]), "")
                    with st.expander(f"Comment #{cn} — {age:.0f} days ago — score {score:.2f} {tier}"):
                        st.write(h["comment_text"])

    st.markdown("##### ✉️ Outreach Draft")
    draft = str(lead.get("outreach_draft", ""))
    if draft and draft.strip():
        st.info(draft)
        st.code(draft, language=None)
    else:
        st.caption("No outreach draft yet — run `src/outreach.py` with a real HF_API_KEY to generate drafts for hot/warm leads.")

else:
    st.caption("Select a row in the table above to view the full comment detail, temporal signals, and lead history.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "Built with **HuggingFace Inference API** · **Streamlit** · **Python** &nbsp;|&nbsp; "
    "[GitHub](https://github.com/your-username/lead-scoring-engine) &nbsp;|&nbsp; "
    "Phase 1 of 3"
)

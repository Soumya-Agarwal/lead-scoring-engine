"""
Competitive Lead Scoring Engine — Streamlit Dashboard
Read-only display of pre-processed lead data. Never calls the HuggingFace API at runtime.
"""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HF_API_KEY = st.secrets.get("HF_API_KEY", os.getenv("HF_API_KEY", ""))

DATA_PATH     = "data/processed/mock_leads_final.csv"
FALLBACK_PATH = "data/processed/mock_leads.csv"

st.set_page_config(
    page_title="Competitive Lead Scoring Engine",
    page_icon="🎯",
    layout="wide",
)

TIER_EMOJI  = {"hot": "🔴", "warm": "🟡", "cold": "🔵"}
TIER_ORDER  = {"hot": 0, "warm": 1, "cold": 2}
DOW_NAMES   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_raw() -> pd.DataFrame:
    path = DATA_PATH if os.path.exists(DATA_PATH) else FALLBACK_PATH
    df = pd.read_csv(path)

    defaults = {
        "intent_label": "no intent signal",    "urgency_label": "not urgent",
        "pain_point_label": "no pain point",   "intent_score": 1.0,
        "urgency_score": 1.0,                  "fit_score": 2.0,
        "composite_score": 1.5,                "lead_tier": "cold",
        "outreach_draft": "",                  "outreach_generated": False,
        "intent_confidence": 0.0,              "urgency_confidence": 0.0,
        "pain_point_confidence": 0.0,
        "comment_age_days": 0.0,               "hour_of_day": 12,
        "day_of_week": 0,                      "is_recent": False,
        "lead_comment_number": 1,              "lead_total_comments": 1,
        "days_since_first_comment": 0.0,       "lead_id": "",
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
    return df


@st.cache_data
def build_lead_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse comment-level rows into one row per lead.
    Best score, most recent comment, dominant pain point and competitor.
    """
    signal = df[df["lead_type"] != "noise"].copy()

    agg = signal.groupby("lead_id").agg(
        username            =("username",             "first"),
        lead_type           =("lead_type",            "first"),
        total_comments      =("lead_comment_number",  "max"),
        best_score          =("composite_score",      "max"),
        avg_score           =("composite_score",      "mean"),
        latest_age_days     =("comment_age_days",     "min"),   # smallest age = most recent
        any_recent          =("is_recent",            "any"),
        competitor_mentioned=("competitor_mentioned", lambda x: x.mode()[0] if len(x) else "none"),
        pain_point_category =("pain_point_category",  lambda x: x.mode()[0] if len(x) else "none"),
        first_comment_age   =("comment_age_days",     "max"),   # oldest comment
    ).reset_index()

    # Tier from best score
    agg["lead_tier"] = agg["best_score"].apply(
        lambda s: "hot" if s >= 7.5 else ("warm" if s >= 5.0 else "cold")
    )

    # Escalation flag
    agg["escalating"] = agg["total_comments"] >= 3

    # Days suffering = age of first comment
    agg["days_suffering"] = agg["first_comment_age"].round(0).astype(int)

    # Sort: tier order, then best score desc
    agg["_tier_order"] = agg["lead_tier"].map(TIER_ORDER)
    agg = agg.sort_values(["_tier_order", "best_score"], ascending=[True, False])
    agg = agg.drop(columns=["_tier_order"]).reset_index(drop=True)
    agg.insert(0, "rank", agg.index + 1)

    return agg


raw_df   = load_raw()
leads_df = build_lead_table(raw_df)
signal   = raw_df[raw_df["lead_type"] != "noise"]

# ── Header metrics ────────────────────────────────────────────────────────────
st.title("🎯 Competitive Lead Scoring Engine")
st.markdown("**3PL / Fulfillment — AI-Powered Prospecting**")

hot_this_week  = int(leads_df[leads_df["any_recent"] & (leads_df["lead_tier"] == "hot")].shape[0])
escalating     = int(leads_df["escalating"].sum())
top_competitor = raw_df[raw_df["competitor_mentioned"] != "none"]["competitor_mentioned"].value_counts()
top_comp_name  = top_competitor.index[0] if len(top_competitor) else "—"
top_comp_count = int(top_competitor.iloc[0]) if len(top_competitor) else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Leads",               len(leads_df))
c2.metric("Hot Leads This Week 🔴",    hot_this_week,    help="Hot leads with at least one comment in the last 7 days")
c3.metric("Leads with 3+ Comments 🔥", escalating,       help="Leads who have posted 3 or more times — highest switching intent")
c4.markdown(
    f"""
    <div style="padding:4px 0px">
        <p style="font-size:0.85rem;color:#888;margin:0">Top Competitor Mentioned</p>
        <p style="font-size:1.1rem;font-weight:600;margin:4px 0 0 0">{top_comp_name}
            <span style="font-size:0.85rem;font-weight:400;color:#888">({top_comp_count} mentions)</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    tier_filter = st.multiselect(
        "Lead Tier",
        options=["hot", "warm", "cold"],
        default=["hot", "warm", "cold"],
    )

    competitor_options = sorted(
        [c for c in leads_df["competitor_mentioned"].dropna().unique() if c != "none"]
    )
    competitor_filter = st.multiselect(
        "Competitor Mentioned",
        options=competitor_options + ["none"],
        default=competitor_options + ["none"],
    )

    pain_options = sorted(leads_df["pain_point_category"].dropna().unique().tolist())
    pain_filter = st.multiselect(
        "Pain Point",
        options=pain_options,
        default=pain_options,
    )

    type_options = sorted(leads_df["lead_type"].dropna().unique().tolist())
    type_filter = st.multiselect(
        "Lead Type",
        options=type_options,
        default=type_options,
    )

    min_score = st.slider(
        "Min Best Score",
        min_value=1.0, max_value=10.0, value=1.0, step=0.5,
    )

    st.divider()
    st.subheader("Escalation Filters")

    recent_only = st.checkbox("Hot this week only", value=False)
    escalating_only = st.checkbox("3+ comments only 🔥", value=False,
                                  help="Show only leads who have escalated to 3 or more posts")

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = leads_df[
    leads_df["lead_tier"].isin(tier_filter) &
    leads_df["competitor_mentioned"].isin(competitor_filter) &
    leads_df["pain_point_category"].isin(pain_filter) &
    leads_df["lead_type"].isin(type_filter) &
    (leads_df["best_score"] >= min_score)
].copy()

if recent_only:
    filtered = filtered[filtered["any_recent"] == True]
if escalating_only:
    filtered = filtered[filtered["escalating"] == True]

st.markdown(f"**{len(filtered)} leads** match your filters")

# ── Lead-centric table ────────────────────────────────────────────────────────
display = filtered[[
    "rank", "username", "lead_type", "competitor_mentioned",
    "pain_point_category", "total_comments", "days_suffering",
    "any_recent", "best_score", "avg_score", "lead_tier",
]].copy()

display["lead_tier"]   = display["lead_tier"].map(lambda t: f"{TIER_EMOJI.get(t,'')} {t}")
display["any_recent"]  = display["any_recent"].map(lambda v: "🟢 Yes" if v else "")
display["total_comments"] = display.apply(
    lambda r: f"{int(r['total_comments'])} 🔥" if r["total_comments"] >= 3 else str(int(r["total_comments"])),
    axis=1,
)

event = st.dataframe(
    display,
    width="stretch",
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "rank":                 st.column_config.NumberColumn("#",               format="%d",    width="small"),
        "username":             st.column_config.TextColumn("Lead"),
        "lead_type":            st.column_config.TextColumn("Type"),
        "competitor_mentioned": st.column_config.TextColumn("Competitor"),
        "pain_point_category":  st.column_config.TextColumn("Pain Point"),
        "total_comments":       st.column_config.TextColumn("Posts",            width="small"),
        "days_suffering":       st.column_config.NumberColumn("Days Active",     format="%d days", width="small"),
        "any_recent":           st.column_config.TextColumn("Recent?",          width="small"),
        "best_score":           st.column_config.NumberColumn("Best Score ⭐",  format="%.2f"),
        "avg_score":            st.column_config.NumberColumn("Avg Score",       format="%.2f"),
        "lead_tier":            st.column_config.TextColumn("Tier"),
    },
)

# ── Lead detail panel ─────────────────────────────────────────────────────────
selected_rows = event.selection.rows if event and event.selection else []

if selected_rows:
    lead_row   = filtered.iloc[selected_rows[0]]
    lead_id    = lead_row["lead_id"]
    username   = lead_row["username"]
    tier_raw   = lead_row["lead_tier"]

    # All comments for this lead, sorted by arc position
    all_comments = raw_df[raw_df["lead_id"] == lead_id].sort_values("lead_comment_number")
    best_comment = all_comments.loc[all_comments["composite_score"].idxmax()]

    st.divider()
    st.subheader(f"@{username}  {TIER_EMOJI.get(tier_raw,'')}  —  {tier_raw.upper()} Lead")

    # ── Lead summary bar ─────────────────────────────────────────────────────
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Best Score",      f"{lead_row['best_score']:.2f}")
    s2.metric("Total Posts",     f"{int(lead_row['total_comments'])}" + (" 🔥" if lead_row["escalating"] else ""))
    s3.metric("Competitor",      lead_row["competitor_mentioned"])
    s4.metric("Pain Point",      lead_row["pain_point_category"])
    s5.metric("Days Active",     f"{int(lead_row['days_suffering'])} days")

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.markdown("##### 📜 Comment History")
        for _, c in all_comments.iterrows():
            cn    = int(c["lead_comment_number"])
            age   = float(c["comment_age_days"])
            score = float(c["composite_score"])
            tier  = c["lead_tier"]
            hour  = int(c["hour_of_day"])
            recent_tag = "🟢" if c["is_recent"] else ""
            night_tag  = "🌙" if (hour >= 21 or hour <= 2) else ""
            label = f"Comment #{cn} · {age:.0f} days ago · score {score:.2f} {TIER_EMOJI.get(tier,'')} {recent_tag}{night_tag}"
            with st.expander(label, expanded=(cn == 1)):
                st.write(c["comment_text"])
                col_a, col_b, col_c = st.columns(3)
                col_a.caption(f"**Intent:** {c.get('intent_label','—')}")
                col_b.caption(f"**Urgency:** {c.get('urgency_label','—')}")
                col_c.caption(f"**Pain:** {c.get('pain_point_label','—')}")

    with right:
        # Score breakdown for best comment
        st.markdown(f"##### 📊 Score Breakdown (best comment)")
        fig = go.Figure(go.Bar(
            x=[float(best_comment["intent_score"]),
               float(best_comment["urgency_score"]),
               float(best_comment["fit_score"])],
            y=["Intent (×0.40)", "Urgency (×0.35)", "Fit (×0.25)"],
            orientation="h",
            marker_color=["#e74c3c", "#f39c12", "#2ecc71"],
            text=[f"{v:.2f}" for v in [
                float(best_comment["intent_score"]),
                float(best_comment["urgency_score"]),
                float(best_comment["fit_score"]),
            ]],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Composite: {float(best_comment['composite_score']):.2f}",
            xaxis=dict(range=[0, 12]),
            height=240,
            margin=dict(l=20, r=50, t=40, b=10),
        )
        st.plotly_chart(fig, width="stretch")

        # Score arc chart if multiple comments
        if len(all_comments) > 1:
            st.markdown("##### 📈 Score escalation across posts")
            fig2 = go.Figure(go.Scatter(
                x=all_comments["lead_comment_number"].tolist(),
                y=all_comments["composite_score"].tolist(),
                mode="lines+markers+text",
                text=[f"{s:.1f}" for s in all_comments["composite_score"]],
                textposition="top center",
                marker=dict(size=10, color="#e74c3c"),
                line=dict(width=2, color="#e74c3c"),
            ))
            fig2.update_layout(
                xaxis=dict(title="Comment #", tickmode="linear", dtick=1),
                yaxis=dict(title="Score", range=[0, 11]),
                height=220,
                margin=dict(l=20, r=20, t=20, b=40),
            )
            st.plotly_chart(fig2, width="stretch")

        # Outreach draft from best comment
        st.markdown("##### ✉️ Outreach Draft")
        draft = str(best_comment.get("outreach_draft", ""))
        if draft and draft.strip():
            st.info(draft)
            st.code(draft, language=None)
        else:
            st.caption("No outreach draft yet — run `src/outreach.py` with a real HF_API_KEY.")

else:
    st.caption("👆 Select a lead from the table above to see their full comment history and score breakdown.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "Built with **HuggingFace Inference API** · **Streamlit** · **Python** &nbsp;|&nbsp; "
    "[GitHub](https://github.com/Soumya-Agarwal/lead-scoring-engine) &nbsp;|&nbsp; "
    "Phase 1 of 3"
)

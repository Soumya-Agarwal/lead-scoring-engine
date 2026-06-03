"""
Mock data generator for the Lead Scoring Engine.

Data model:
  - A LEAD = a unique merchant (username + lead_id)
  - A COMMENT = one post by that merchant (comment_id)
  - One lead can have 1-4 comments posted at different times
  - Comments for the same lead follow a temporal arc:
      comment 1 → initial complaint / question
      comment 2 → follow-up / escalation
      comment 3 → actively seeking alternatives
      comment 4 → final decision / ultimatum

Temporal fields added per comment:
  - comment_timestamp      : exact datetime of the comment
  - comment_age_days       : how many days ago (vs. NOW = 2026-06-02)
  - hour_of_day            : 0-23 — late-night venting signals higher urgency
  - day_of_week            : 0=Mon … 6=Sun
  - is_recent              : True if posted within last 7 days
  - lead_comment_number    : 1st, 2nd, 3rd, 4th comment from this lead
  - lead_total_comments    : total comments this lead has posted
  - days_since_first_comment : escalation window (0 for first comment)
  - comment_velocity       : comments per week rate for this lead
"""

import random
import uuid
from datetime import datetime, timedelta

import pandas as pd
from faker import Faker

fake = Faker()
random.seed(42)
Faker.seed(42)

NOW = datetime(2026, 6, 2, 12, 0, 0)  # fixed "today" for reproducibility

COMPETITORS = [
    "ShipMonk", "Deliverr", "Red Stag Fulfillment", "EasyPost",
    "FedEx Fulfillment", "ShipBob", "Whiplash", "Rakuten Super Logistics",
]

PRODUCT_TYPES = [
    "candles", "skincare", "supplements", "apparel", "jewelry", "pet products",
    "home decor", "electronics accessories", "books", "toys", "food & beverage",
    "beauty products", "fitness gear", "baby products", "outdoor gear",
]

ORDER_VOLUMES = ["100", "200", "300", "400", "500", "750", "1000", "1500", "2000"]

# ---------------------------------------------------------------------------
# Comment templates — organised by lead_type AND comment_number (arc position)
# ---------------------------------------------------------------------------

# VENTER ARC
# comment 1: initial shock/frustration
VENTER_ARC_1_RAW = [
    "{competitor} LOST {num} of my orders this week. {num}!! My customers are going insane and I don't even get a response from support",
    "Just got hit with a surprise ${fee} invoice from {competitor}. No warning, no explanation. What even is this charge??",
    "{competitor} shipped wrong items to {num} customers in ONE DAY. My return rate is exploding right now",
    "Three weeks of emails to {competitor} support and NOTHING. My {product} inventory has been sitting 'under review' for 21 days",
    "{competitor} has had my {product} inventory 'in receiving' for 16 days. SIXTEEN. My entire holiday stock",
    "The audacity of {competitor} to charge me peak season surcharges they NEVER mentioned in the contract. Not ok",
    "{competitor} just told me a pallet of my {product} was 'misplaced'. {num} units. Gone. Their liability covers basically nothing",
    "Starting to think {competitor} has lost more of my packages than they've delivered correctly this month. Absolutely furious",
]

VENTER_ARC_1_PROFESSIONAL = [
    "We've been with {competitor} for {months} months and the hidden fees are becoming unsustainable. What started as $2.50/pick is now $4.{cents} with all the add-ons",
    "Our error rate with {competitor} sits at {pct}% this quarter. For {orders} orders/month that's unacceptable for a {product} brand",
    "{competitor}'s SLA adherence dropped from 96% to 81% over last quarter. We have contractual obligations we cannot meet at this level",
    "I've documented {num} billing discrepancies with {competitor} in 90 days totalling ~$3,{hundreds}. Their dispute process takes 30 days per claim",
    "Our chargeback rate increased 18% since moving to {competitor}. Root cause analysis points to fulfillment errors",
]

# comment 2: frustration deepens, still no resolution
VENTER_ARC_2 = [
    "Update on my {competitor} situation — still no resolution after {num} days. They offered a $15 credit. I lost $2,{hundreds} in sales",
    "Week {num} with {competitor} and the issues are getting WORSE. My {product} business is going backwards",
    "Just got off a call with {competitor} support. They told me to 'wait 7-10 business days'. I've been waiting {num} weeks already",
    "Following up on the {competitor} nightmare — third invoice dispute this month. This is becoming a full-time job",
    "Day {num} of the {competitor} saga. Still no inventory update. Still no response. Documenting everything at this point",
    "Another batch of {competitor} errors this week. {num} wrong orders shipped. My customer reviews are tanking",
    "Had a call with {competitor} account manager. Zero accountability, zero urgency. They don't care about small brands at all",
]

# comment 3: actively looking to leave
VENTER_ARC_3 = [
    "Officially done with {competitor}. Shopping for alternatives now. Anyone have 3PL recommendations for {orders} orders/month {product} brand?",
    "After {num} months of {competitor} issues we're formally opening an RFP for a new fulfillment partner. Done waiting",
    "Switching away from {competitor} ASAP. Need a 3PL that actually fulfills orders correctly. Recommendations?",
    "The {competitor} situation is now affecting our retail partnerships. We HAVE to move. Looking at alternatives this week",
    "Told my team we're transitioning away from {competitor} by end of month. The cost of staying is higher than the cost of switching",
]

# comment 4: final decision / ultimatum (rare, most urgent)
VENTER_ARC_4 = [
    "We've signed with a new 3PL. Leaving {competitor} after {months} months. The transition starts next week. Should have done this months ago",
    "Filed a formal complaint against {competitor} and starting migration to a new provider. If anyone needs more info on what to watch out for DM me",
    "CONTRACT CANCELLED with {competitor}. Moving everything out by end of the month. Happy to share what finally pushed us over the edge",
    "New 3PL contract signed. Done with {competitor}. For anyone still evaluating them — ask about their hidden fee structure and their actual SLA data",
]

# SEEKER ARC
SEEKER_ARC_1 = [
    "Anyone running a Shopify store doing ~{orders} orders/month actually happy with their 3PL? Been with {competitor} for {months} months and SLAs aren't being met",
    "Ready to leave {competitor}. What are people using for {product} fulfillment? We're at {orders} orders/month. DM me or drop recs below",
    "Looking to switch fulfillment providers by end of quarter. Not happy with {competitor}. Need {orders} orders/month for {product}",
    "Does anyone have experience moving FROM {competitor} to another 3PL? Planning the transition and want to know the timeline",
    "Raising my hand — looking for a new fulfillment partner. {competitor} has been a nightmare. {orders} orders/month, need 2-day domestic",
    "Help! Our {competitor} contract is up in 45 days. {orders} orders/month, {product}, mainly US domestic. Who's delivering right now?",
]

SEEKER_ARC_2 = [
    "Getting quotes from a few 3PLs after the {competitor} issues. Anyone used both {competitor} and {competitor2}? What was the comparison like?",
    "Had demos with 3 providers after leaving {competitor}. The pricing variance is wild. Happy to share notes with anyone else evaluating",
    "Narrowing down our {competitor} replacement to two options. One thing I've learned — always ask for their real error rate, not the marketing number",
    "Two weeks into the 3PL search post-{competitor}. Our volume ({orders} orders/month) is making some providers less responsive than I'd like",
]

SEEKER_ARC_3 = [
    "Update: we've shortlisted 2 providers to replace {competitor}. Making a decision by Friday. Will share how it goes",
    "Final stages of our 3PL transition from {competitor}. One provider really stood out on pricing transparency and tech integration",
    "Decision made on {competitor} replacement. Starting onboarding next week. The difference in transparency during the sales process was night and day",
]

# EVALUATOR ARC
EVALUATOR_ARC_1 = [
    "Trying to compare {competitor} vs {competitor2} for our {product} business. ~{orders} orders/month, mostly fragile items. Any experience switching?",
    "Doing a 3PL evaluation for our {product} brand ({orders} orders/month). Have proposals from {competitor} and {competitor2}. What questions should I be asking?",
    "Can anyone break down the real cost difference between {competitor} and {competitor2}? Building a comparison model and getting conflicting info on per-pick fees",
    "We're scaling our {product} brand to {orders} orders/month next year. Evaluating {competitor} — what should we know before signing?",
    "What metrics should I track in the first 90 days with a new 3PL? About to onboard with {competitor} and want clear benchmarks from day one",
]

EVALUATOR_ARC_2 = [
    "Follow up on my {competitor} evaluation — got their SLA data finally. Some of the numbers don't match what the sales rep quoted. Digging deeper",
    "Week 2 of our 3PL evaluation. {competitor} is solid on tech but their pricing for {product} SKUs surprised me on the high side",
    "Ran the numbers on {competitor} vs {competitor2} for our {orders} order/month volume. Happy to share the cost model if anyone wants it",
    "Had reference calls with 3 current {competitor} customers. Mixed feedback — great for high volume, rough for brands our size",
]

EVALUATOR_ARC_3 = [
    "Finishing up our {competitor} evaluation. Going to pilot with them for 90 days with a subset of our {product} SKUs before full commit",
    "After 3 weeks of evaluation, we're leaning toward {competitor} but still have questions on their peak season capacity guarantees",
]

# NOISE — generic engagement, no lead signal
NOISE_TEMPLATES = [
    "Love this post! Really helpful content as always 🙌",
    "Great insights here, thanks for sharing!",
    "Following for more content like this 👏",
    "This is exactly what I needed to read today",
    "Amazing post as always! You always deliver great value",
    "Saved this for later, so much good info here",
    "Tag a friend who needs to see this! 🔥",
    "Been following you for years, always love your content",
    "This community is so helpful, glad I found it",
    "Sharing this with my whole team!",
    "So much value in this post. Thank you",
    "First 🙌 Love what you're doing here",
    "This hits different today. Needed this reminder",
    "Great post! Love seeing this in my feed",
    "Bookmarking this for our next team meeting 📌",
    "100% agree with everything here",
    "Keep up the amazing work! 🚀",
    "Couldn't have said it better myself",
    "Exactly what we've been talking about internally",
    "Really appreciate you sharing this perspective",
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _pick(lst):
    return random.choice(lst)


def _fill(template: str, competitors=None) -> str:
    if competitors is None:
        competitors = random.sample(COMPETITORS, 2)
    return template.format(
        competitor=competitors[0],
        competitor2=competitors[1],
        num=random.randint(2, 15),
        num2=random.randint(2, 8),
        fee=random.randint(200, 800),
        hundreds=random.randint(100, 999),
        cents=random.randint(10, 99),
        months=random.randint(3, 18),
        orders=random.choice(ORDER_VOLUMES),
        product=_pick(PRODUCT_TYPES),
        pct=round(random.uniform(5, 25), 1),
    )


def _random_username():
    words = ["shop", "brand", "store", "co", "goods", "supply", "market",
             "boutique", "collective", "studio", "works", "lab", "hub"]
    adjectives = ["coastal", "golden", "modern", "urban", "wild", "tiny", "fresh",
                  "bright", "peak", "craft", "local", "daily", "swift", "bold"]
    return f"{_pick(adjectives)}_{_pick(words)}_{random.randint(1, 99)}"


def _competitor_from_text(text: str) -> str:
    for c in COMPETITORS:
        if c.lower() in text.lower():
            return c
    return "none"


def _pain_from_text(text: str) -> str:
    fee_words   = ["fee", "charge", "price", "cost", "invoice", "billing", "rate", "expensive", "money"]
    error_words = ["lost", "damaged", "wrong", "missing", "error", "incorrect", "misplace"]
    speed_words = ["slow", "delay", "late", "sla", "days", "shipping", "fulfillment time", "dispatch", "receiving"]
    tl = text.lower()
    scores = {
        "fees":   sum(1 for w in fee_words   if w in tl),
        "errors": sum(1 for w in error_words if w in tl),
        "speed":  sum(1 for w in speed_words if w in tl),
    }
    if max(scores.values()) == 0:
        return "general"
    return max(scores, key=scores.get)


def _post_context(competitor: str) -> str:
    slug = competitor.lower().replace(" ", "")
    contexts = [
        f"comment on @{slug} Instagram post",
        f"comment on fulfillment industry post mentioning {competitor}",
        f"comment on #3PLproblems hashtag post",
        f"comment on #ecommercefulfillment community post",
        f"reply thread on {competitor} pricing announcement post",
        f"comment on #3PL hashtag discussion post",
        f"comment on ecommerce operations community post",
    ]
    return _pick(contexts)


# ---------------------------------------------------------------------------
# Temporal helpers
# ---------------------------------------------------------------------------

def _comment_timestamp(age_days: float, hour: int, minute: int) -> datetime:
    """Build a datetime that is `age_days` days before NOW."""
    return NOW - timedelta(days=age_days, hours=(12 - hour), minutes=minute)


def _temporal_fields(ts: datetime, first_ts: datetime) -> dict:
    age_days = (NOW - ts).total_seconds() / 86400
    return {
        "comment_timestamp":         ts.strftime("%Y-%m-%d %H:%M:%S"),
        "comment_age_days":          round(age_days, 1),
        "hour_of_day":               ts.hour,
        "day_of_week":               ts.weekday(),        # 0=Mon, 6=Sun
        "day_of_week_name":          ts.strftime("%A"),
        "is_recent":                 age_days <= 7,
        "days_since_first_comment":  round((ts - first_ts).total_seconds() / 86400, 1),
    }


# ---------------------------------------------------------------------------
# Lead + comment arc generators
# ---------------------------------------------------------------------------

def _build_venter_comments(n_comments: int, competitors: list) -> list[dict]:
    """Return list of raw comment dicts for a venter lead (1-4 comments)."""
    comments = []
    styles = ["raw", "professional"]
    arc = [
        (VENTER_ARC_1_RAW if random.random() < 0.55 else VENTER_ARC_1_PROFESSIONAL, "raw" if random.random() < 0.55 else "professional"),
        (VENTER_ARC_2, random.choice(styles)),
        (VENTER_ARC_3, random.choice(styles)),
        (VENTER_ARC_4, "raw"),
    ]
    for i in range(min(n_comments, 4)):
        templates, style = arc[i]
        comments.append({
            "comment_text": _fill(_pick(templates), competitors),
            "writing_style": style,
            "lead_comment_number": i + 1,
        })
    return comments


def _build_seeker_comments(n_comments: int, competitors: list) -> list[dict]:
    arc = [
        (SEEKER_ARC_1, "neutral"),
        (SEEKER_ARC_2, "neutral"),
        (SEEKER_ARC_3, "professional"),
        (SEEKER_ARC_3, "professional"),  # repeat arc3 if 4 comments
    ]
    comments = []
    for i in range(min(n_comments, 4)):
        templates, style = arc[i]
        comments.append({
            "comment_text": _fill(_pick(templates), competitors),
            "writing_style": style,
            "lead_comment_number": i + 1,
        })
    return comments


def _build_evaluator_comments(n_comments: int, competitors: list) -> list[dict]:
    arc = [
        (EVALUATOR_ARC_1, "neutral"),
        (EVALUATOR_ARC_2, "neutral"),
        (EVALUATOR_ARC_3, "professional"),
        (EVALUATOR_ARC_3, "professional"),
    ]
    comments = []
    for i in range(min(n_comments, 4)):
        templates, style = arc[i]
        comments.append({
            "comment_text": _fill(_pick(templates), competitors),
            "writing_style": style,
            "lead_comment_number": i + 1,
        })
    return comments


def _assign_comment_timestamps(n_comments: int, lead_type: str) -> list[datetime]:
    """
    Assign timestamps to comments for a lead.

    Urgency patterns:
      - Venters: comments cluster in recent 30 days, rapid escalation
      - Seekers: spread over 60 days
      - Evaluators: methodical, spread over 45 days
      - Noise: random within 90 days
    """
    if lead_type == "noise":
        age = random.uniform(1, 89)
        hour = random.randint(8, 22)
        ts = NOW - timedelta(days=age, hours=random.randint(0, 4))
        return [ts]

    if lead_type == "venter":
        # First complaint: 7-60 days ago; escalation happens fast (2-10 days between)
        first_age = random.uniform(7, 60)
        gap_range = (2, 10)
    elif lead_type == "seeker":
        first_age = random.uniform(14, 75)
        gap_range = (3, 15)
    else:  # evaluator
        first_age = random.uniform(10, 60)
        gap_range = (5, 20)

    # Late-night venting hours for venters (8pm-2am), business hours for evaluators
    if lead_type == "venter":
        hour_choices = [20, 21, 22, 23, 0, 1, 9, 10, 14, 15]
    elif lead_type == "evaluator":
        hour_choices = [9, 10, 11, 14, 15, 16]
    else:
        hour_choices = list(range(8, 22))

    timestamps = []
    current_age = first_age
    for i in range(n_comments):
        hour = _pick(hour_choices)
        minute = random.randint(0, 59)
        ts = NOW - timedelta(days=current_age, hours=(12 - hour), minutes=minute)
        # Keep within bounds
        if ts > NOW:
            ts = NOW - timedelta(hours=2)
        timestamps.append(ts)
        # Next comment is more recent (smaller age = closer to now)
        gap = random.uniform(*gap_range)
        current_age = max(0.5, current_age - gap)

    return timestamps


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_leads(target_comments: int = 500) -> pd.DataFrame:
    """
    Generate leads with multiple comments each.

    Returns a flat DataFrame where each row = one comment.
    Use lead_id to group comments belonging to the same merchant.
    """
    rows = []
    comment_id_counter = 1
    lead_id_counter = 1

    # Comment count distribution per lead: 1-4, weighted toward 2-3
    comment_count_weights = [0.25, 0.40, 0.25, 0.10]   # 1, 2, 3, 4 comments

    # Lead type mix (applied to signal leads only)
    lead_type_weights = {"venter": 0.50, "seeker": 0.30, "evaluator": 0.20}

    # 15% of total comments come from noise "leads" (1 comment each)
    noise_comment_target = round(target_comments * 0.15)
    signal_comment_target = target_comments - noise_comment_target

    # --- Generate signal leads ---
    signal_comments_generated = 0
    while signal_comments_generated < signal_comment_target:
        lead_type = random.choices(
            list(lead_type_weights.keys()),
            weights=list(lead_type_weights.values()),
        )[0]

        n_comments = random.choices([1, 2, 3, 4], weights=comment_count_weights)[0]
        # Don't overshoot target
        remaining = signal_comment_target - signal_comments_generated
        n_comments = min(n_comments, remaining)

        competitors = random.sample(COMPETITORS, 2)
        username = _random_username()
        lead_id = f"lead_{str(lead_id_counter).zfill(3)}"
        lead_id_counter += 1

        # Build comment texts along arc
        if lead_type == "venter":
            comment_dicts = _build_venter_comments(n_comments, competitors)
        elif lead_type == "seeker":
            comment_dicts = _build_seeker_comments(n_comments, competitors)
        else:
            comment_dicts = _build_evaluator_comments(n_comments, competitors)

        # Assign timestamps
        timestamps = _assign_comment_timestamps(n_comments, lead_type)
        first_ts = min(timestamps)

        for i, (cd, ts) in enumerate(zip(comment_dicts, timestamps)):
            text = cd["comment_text"]
            competitor = _competitor_from_text(text)
            if competitor == "none":
                competitor = competitors[0]

            temporal = _temporal_fields(ts, first_ts)

            rows.append({
                "comment_id":          f"comment_{str(comment_id_counter).zfill(4)}",
                "lead_id":             lead_id,
                "username":            username,
                "comment_text":        text,
                "platform":            "instagram",
                "post_context":        _post_context(competitor),
                "lead_type":           lead_type,
                "competitor_mentioned": competitor,
                "pain_point_category": _pain_from_text(text),
                "writing_style":       cd["writing_style"],
                "lead_comment_number": cd["lead_comment_number"],
                "lead_total_comments": n_comments,
                **temporal,
            })
            comment_id_counter += 1

        signal_comments_generated += n_comments

    # --- Generate noise comments ---
    for _ in range(noise_comment_target):
        username = _random_username()
        lead_id = f"lead_{str(lead_id_counter).zfill(3)}"
        lead_id_counter += 1
        text = _pick(NOISE_TEMPLATES)
        ts = NOW - timedelta(days=random.uniform(1, 89), hours=random.randint(0, 23))
        temporal = _temporal_fields(ts, ts)

        rows.append({
            "comment_id":          f"comment_{str(comment_id_counter).zfill(4)}",
            "lead_id":             lead_id,
            "username":            username,
            "comment_text":        text,
            "platform":            "instagram",
            "post_context":        _post_context(_pick(COMPETITORS)),
            "lead_type":           "noise",
            "competitor_mentioned": "none",
            "pain_point_category": "none",
            "writing_style":       "neutral",
            "lead_comment_number": 1,
            "lead_total_comments": 1,
            **temporal,
        })
        comment_id_counter += 1

    df = pd.DataFrame(rows)

    # Shuffle rows so leads are interleaved (realistic ingestion order)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def main():
    df = generate_leads(500)
    output_path = "data/processed/mock_leads.csv"
    df.to_csv(output_path, index=False)

    leads_df = df[df["lead_type"] != "noise"]
    noise_df = df[df["lead_type"] == "noise"]
    unique_leads = df[df["lead_type"] != "noise"]["lead_id"].nunique()

    print("\nMock data generation complete.")
    print(f"Total comments  : {len(df)}")
    print(f"  Signal        : {len(leads_df)}  (from {unique_leads} unique leads)")
    print(f"  Noise         : {len(noise_df)}  (low-signal, 1 comment each)")
    print(f"\nBreakdown by lead_type:")
    print(df["lead_type"].value_counts().to_string())
    print(f"\nComments per lead (signal leads):")
    print(leads_df.groupby("lead_id")["comment_id"].count().value_counts().sort_index().to_string())
    print(f"\nBreakdown by pain_point_category:")
    print(df["pain_point_category"].value_counts().to_string())
    print(f"\nTemporal summary:")
    print(f"  Recent comments (<=7 days) : {df['is_recent'].sum()}")
    print(f"  Avg comment age (days)     : {df['comment_age_days'].mean():.1f}")
    print(f"  Late-night comments (9pm+) : {(df['hour_of_day'] >= 21).sum()}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

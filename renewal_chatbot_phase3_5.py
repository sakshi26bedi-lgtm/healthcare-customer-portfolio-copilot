import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from google import genai

# ============================================================
# PHASE 3.4 - HEALTHCARE RENEWAL AI ASSISTANT
# ============================================================
# This version preserves the renewal dashboard concept and adds:
# 1. Healthcare domain knowledge
# 2. Product/service knowledge
# 3. Strategic business reasoning
# 4. Portfolio-growth questions
# 5. Account-specific + portfolio-specific grounding
# 6. Healthcare-aware domain reasoning and manager guidance
#
# Files expected in the same folder:
#   renewal_chatbot.py
#   healthcare_knowledge.py
#   healthcare_renewal_predictions.csv
#   healthcare_provider_data.csv   (optional)
#
# IMPORTANT:
# Provider characteristics may come from public healthcare data.
# Commercial, renewal, product-adoption and opportunity attributes
# are POC/synthetic unless explicitly sourced from a real system.
# ============================================================

st.set_page_config(
    page_title="Healthcare Customer & Portfolio Intelligence Copilot",
    page_icon="🏥",
    layout="wide",
)

# ============================================================
# KNOWLEDGE LAYER
# ============================================================

try:
    from healthcare_knowledge import (
        HEALTHCARE_DOMAIN_KNOWLEDGE,
        PRODUCT_SERVICE_KNOWLEDGE,
        STRATEGIC_PLAYBOOKS,
        STRATEGIC_INTENT_KEYWORDS,
        build_knowledge_context,
    )
except Exception as e:
    st.error(
        "The knowledge layer could not be loaded. "
        "Please make sure healthcare_knowledge.py is in the same "
        "folder as renewal_chatbot.py."
    )
    st.code(str(e))
    st.stop()

# ============================================================
# CONSTANTS
# ============================================================

GEMINI_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "healthcare_renewal_predictions.csv"
PROVIDER_FILE = BASE_DIR / "healthcare_provider_data.csv"

# ============================================================
# PUBLIC POC CONTROLS
# ============================================================
# This app is designed for public demonstration. Commercial, renewal,
# adoption and opportunity attributes are synthetic POC data unless
# explicitly sourced from a connected production system.
PUBLIC_MAX_AI_QUESTIONS = 15
PUBLIC_MIN_SECONDS_BETWEEN_AI_CALLS = 2

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(CSV_FILE)


@st.cache_data
def load_provider_data():
    if PROVIDER_FILE.exists():
        return pd.read_csv(PROVIDER_FILE)
    return pd.DataFrame()


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "healthcare_renewal_predictions.csv was not found. "
        "Please keep it in the same folder as this app."
    )
    st.stop()
except Exception as e:
    st.error(f"Unable to load renewal data: {e}")
    st.stop()

provider_df = load_provider_data()

# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client():
    return genai.Client()

# ============================================================
# UI HELPERS
# ============================================================

def money(value):
    try:
        return f"${float(value):,.0f}"
    except (ValueError, TypeError):
        return "$0"


def get_risk_color(risk):
    if risk == "High Risk":
        return "🔴"
    if risk == "Medium Risk":
        return "🟠"
    return "🟢"


def get_priority(risk, arr):
    try:
        arr = float(arr)
    except (ValueError, TypeError):
        arr = 0

    if risk == "High Risk":
        return "Critical" if arr >= 1_000_000 else "High"
    if risk == "Medium Risk":
        return "High" if arr >= 1_000_000 else "Moderate"
    return "Low"


def get_risk_signals(row):
    signals = []

    probability = float(row["Renewal Probability (%)"])
    engagement = float(row["Engagement Score"])

    if probability < 40:
        signals.append(f"Low renewal probability ({probability:.1f}%)")
    elif probability < 70:
        signals.append(f"Moderate renewal probability ({probability:.1f}%)")

    if engagement < 50:
        signals.append(f"Low engagement score ({engagement:.1f})")

    product_columns = [
        ("Managed Services", "Has Managed Services"),
        ("Data Connect", "Has Data Connect"),
        ("Lumere", "Has Lumere"),
        ("Enterprise Exchange", "Has Enterprise Exchange"),
    ]

    for product_name, column in product_columns:
        if str(row.get(column, "No")).strip().lower() != "yes":
            signals.append(f"{product_name} is not currently used")

    if not signals:
        signals.append(
            "No major risk signals identified from the available account data."
        )

    return signals


def get_recommended_actions(row):
    actions = []

    risk = str(row["Renewal Risk"])
    engagement = float(row["Engagement Score"])
    discount = float(row["Discount (%)"])

    if risk == "High Risk":
        actions.append("Schedule a proactive renewal conversation.")
        actions.append(
            "Create an account-specific recovery plan around the identified risk signals."
        )

        if engagement < 70:
            actions.append(
                "Increase stakeholder engagement and executive touchpoints."
            )

        if discount >= 15:
            actions.append(
                "Review pricing and discount structure before renewal negotiations."
            )

        if str(row.get("Has Managed Services", "No")).lower() != "yes":
            actions.append(
                "Evaluate whether Managed Services can strengthen account value."
            )

        if str(row.get("Has Lumere", "No")).lower() != "yes":
            actions.append(
                "Evaluate whether Lumere could address additional account needs."
            )

    elif risk == "Medium Risk":
        actions.append(
            "Maintain proactive account engagement and monitor renewal signals."
        )
        actions.append(
            "Identify potential renewal blockers before contract discussions."
        )

        if engagement < 70:
            actions.append(
                "Increase stakeholder engagement with relevant stakeholders."
            )

        if discount >= 15:
            actions.append(
                "Review discount levels and overall account economics."
            )

    else:
        actions.append("Maintain regular account health and renewal engagement.")
        actions.append("Continue monitoring renewal probability and engagement.")
        actions.append(
            "Identify opportunities to increase product adoption and account value."
        )

    return actions


def classify_standard_question(question):
    q = str(question).strip().lower()

    if (
        "why is this account at risk" in q
        or "why is the account at risk" in q
        or "why at risk" in q
        or "risk drivers" in q
        or "main risk signals" in q
    ):
        return "risk"

    if (
        "what actions should we take" in q
        or "what should we do" in q
        or "recommended actions" in q
        or "next actions" in q
    ):
        return "actions"

    if (
        "summarize this account" in q
        or "summary of this account" in q
        or "account summary" in q
    ):
        return "summary"

    if (
        "arr exposure" in q
        or "business impact" in q
        or "financial impact" in q
    ):
        return "impact"

    # Portfolio-growth intent is split into three distinct business views:
    # 1) strategy, 2) account focus, 3) prioritization methodology.
    # This prevents different portfolio questions from returning the same answer.
    if (
        "what is the portfolio growth strategy" in q
        or "portfolio growth strategy" in q
        or "grow the portfolio" in q
        or "portfolio strategy" in q
        or "what is our portfolio growth approach" in q
        or "what should our portfolio growth strategy be" in q
    ):
        return "portfolio_growth_strategy"

    if (
        "which accounts should we focus on for growth" in q
        or "which accounts should we prioritize for growth" in q
        or "where should we focus for growth" in q
        or "which accounts are growth candidates" in q
        or "growth candidates" in q
        or "which accounts have growth potential" in q
        or "which accounts should we target for growth" in q
    ):
        return "growth_focus_accounts"

    if (
        "how should we prioritize the portfolio for growth" in q
        or "how do we prioritize the portfolio for growth" in q
        or "how should we prioritize portfolio growth" in q
        or "portfolio growth prioritization" in q
        or "how should the portfolio be prioritized for growth" in q
        or "how should we prioritize growth across the portfolio" in q
    ):
        return "portfolio_growth_prioritization"

    # Broader portfolio-growth wording still maps to the executive strategy view.
    if (
        "portfolio growth" in q
        or "growth opportunities across the portfolio" in q
        or "account expansion across the portfolio" in q
    ):
        return "portfolio_growth_strategy"

    # Product/service intent is deliberately split so the copilot does not
    # return the same answer for adoption gaps, discovery opportunities,
    # and adoption strategy.
    if (
        "how can we increase adoption" in q
        or "how can we improve adoption" in q
        or "how do we increase adoption" in q
        or "how do we improve adoption" in q
        or "increase product adoption" in q
        or "improve product adoption" in q
        or "increase service adoption" in q
        or "improve service adoption" in q
        or "adoption strategy" in q
    ):
        return "adoption_strategy"

    if (
        "what product/service opportunities should we investigate" in q
        or "what product opportunities should we investigate" in q
        or "what service opportunities should we investigate" in q
        or "which product should we investigate" in q
        or "which service should we investigate" in q
        or "what product/service opportunities" in q
        or "product opportunity" in q
        or "service opportunity" in q
        or ("what should we investigate" in q and ("product" in q or "service" in q))
    ):
        return "opportunity_discovery"

    if (
        "products or services are not being used" in q
        or "products not being used" in q
        or "which products are not" in q
        or "products not adopted" in q
        or "which services are not" in q
        or "services not adopted" in q
        or "what is not being used" in q
    ):
        return "adoption_gaps"

    if (
        "best business action" in q
        or "next best action" in q
        or "best action for this account" in q
        or "what should the account manager prioritize" in q
        or "what should account manager prioritize" in q
        or "what should we prioritize for this account" in q
    ):
        return "best_action"

    # Phase 3.5 - Final GenAI Strategic Synthesis.
    # This intent combines the intelligence layers into one account-level
    # recommendation instead of repeating any single prior phase.
    if (
        "overall strategic recommendation" in q
        or "overall strategy for this account" in q
        or "overall recommendation for this account" in q
        or "give me the overall strategic recommendation" in q
        or "give me the overall recommendation" in q
        or "overall strategic view" in q
        or "overall account strategy" in q
        or "executive recommendation for this account" in q
        or "what is the overall strategy for this account" in q
    ):
        return "strategic_synthesis"

    # Phase 3.4 - Healthcare Domain Reasoning.
    # This is intentionally checked before the generic priority fallback so
    # questions asking what the account manager should focus on are routed to
    # the healthcare-aware synthesis rather than the older renewal-priority view.
    if (
        "considering the healthcare context" in q
        or "healthcare context" in q
        or "healthcare-specific factors" in q
        or "healthcare specific factors" in q
        or "how does the healthcare context affect" in q
        or "healthcare domain" in q
        or "provider context" in q
        or "account footprint" in q and "focus" in q
        or ("healthcare" in q and "account manager" in q and "focus" in q)
    ):
        return "healthcare_domain_reasoning"

    if (
        "is this account a priority" in q
        or "what is the priority" in q
        or "renewal priority" in q
        or "how should we prioritize" in q
    ):
        return "priority"

    return None


def classify_strategic_question(question):
    q = str(question).strip().lower()

    # Directly use the keyword groups from the knowledge layer.
    for intent, keywords in STRATEGIC_INTENT_KEYWORDS.items():
        if any(keyword.lower() in q for keyword in keywords):
            return intent

    # Additional natural-language patterns.
    if (
        ("grow" in q or "growth" in q)
        and (
            "portfolio" in q
            or "hospital groups" in q
            or "accounts" in q
        )
    ):
        return "portfolio_growth"

    if (
        ("add" in q or "adding" in q or "new" in q or "expand" in q)
        and ("hospital" in q or "health system" in q)
    ):
        return "portfolio_growth"

    if (
        ("best business action" in q)
        or ("best action" in q and "business" in q)
        or ("strategic action" in q)
    ):
        return "renewal_strategy"

    return None


def product_adoption_from_row(row):
    mapping = [
        ("Managed Services", "Has Managed Services"),
        ("Data Connect", "Has Data Connect"),
        ("Lumere", "Has Lumere"),
        ("Enterprise Exchange", "Has Enterprise Exchange"),
    ]

    adopted = []
    not_adopted = []

    for name, column in mapping:
        value = str(row.get(column, "No")).strip().lower()
        if value == "yes":
            adopted.append(name)
        else:
            not_adopted.append(name)

    return adopted, not_adopted


def build_next_best_action(row):
    """
    Deterministic Phase 3.1 decision engine.

    The engine chooses the PRIMARY business action from structured
    account signals. Gemini can explain the recommendation later,
    but it does not decide the underlying action.

    Account facts remain facts.
    Renewal probability / risk remain model signals.
    Product gaps remain discovery opportunities, not confirmed fit.
    """
    risk = str(row.get("Renewal Risk", ""))
    probability = float(row.get("Renewal Probability (%)", 0) or 0)
    arr = float(row.get("Current ARR ($)", 0) or 0)
    engagement = float(row.get("Engagement Score", 0) or 0)
    discount = float(row.get("Discount (%)", 0) or 0)
    facilities = float(row.get("Facilities", 0) or 0)
    affiliates = float(row.get("Affiliates", 0) or 0)
    beds = float(row.get("Beds", 0) or 0)

    adopted, not_adopted = product_adoption_from_row(row)

    # Candidate action scores. These are POC business rules, not ML outputs.
    scores = {
        "Stabilize the renewal": 0,
        "Increase stakeholder engagement": 0,
        "Review pricing and value case": 0,
        "Investigate product/service whitespace": 0,
        "Prepare account expansion plan": 0,
        "Maintain and deepen account value": 0,
    }

    # 1. Renewal stabilization
    if risk == "High Risk":
        scores["Stabilize the renewal"] += 70
    elif risk == "Medium Risk":
        scores["Stabilize the renewal"] += 35

    if probability < 30:
        scores["Stabilize the renewal"] += 25
    elif probability < 50:
        scores["Stabilize the renewal"] += 15

    if arr >= 1_000_000:
        scores["Stabilize the renewal"] += 15
    elif arr >= 500_000:
        scores["Stabilize the renewal"] += 8

    # 2. Engagement intervention
    if engagement < 50:
        scores["Increase stakeholder engagement"] += 65
    elif engagement < 70:
        scores["Increase stakeholder engagement"] += 35

    if risk in {"High Risk", "Medium Risk"}:
        scores["Increase stakeholder engagement"] += 15

    # 3. Pricing / value case
    if discount >= 18:
        scores["Review pricing and value case"] += 60
    elif discount >= 15:
        scores["Review pricing and value case"] += 40
    elif discount >= 12:
        scores["Review pricing and value case"] += 20

    if risk == "High Risk":
        scores["Review pricing and value case"] += 10

    # 4. Product/service whitespace discovery
    if len(not_adopted) >= 3:
        scores["Investigate product/service whitespace"] += 50
    elif len(not_adopted) == 2:
        scores["Investigate product/service whitespace"] += 35
    elif len(not_adopted) == 1:
        scores["Investigate product/service whitespace"] += 20

    if engagement >= 65:
        scores["Investigate product/service whitespace"] += 15

    # 5. Expansion planning - only becomes primary when account is
    # relatively stable. Large footprint adds evidence for investigation.
    footprint_signal = (
        (facilities >= 10)
        or (affiliates >= 10)
        or (beds >= 1500)
    )

    if risk == "Low Risk":
        scores["Prepare account expansion plan"] += 55
    elif risk == "Medium Risk" and probability >= 55:
        scores["Prepare account expansion plan"] += 25

    if engagement >= 75:
        scores["Prepare account expansion plan"] += 20

    if footprint_signal:
        scores["Prepare account expansion plan"] += 15

    if not_adopted:
        scores["Prepare account expansion plan"] += 10

    # 6. Healthy account maintenance
    if risk == "Low Risk":
        scores["Maintain and deepen account value"] += 45

    if probability >= 75:
        scores["Maintain and deepen account value"] += 20

    if engagement >= 75:
        scores["Maintain and deepen account value"] += 15

    ordered = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    primary_action, primary_score = ordered[0]
    secondary_action, secondary_score = ordered[1]

    evidence = []
    if risk:
        evidence.append(f"Renewal risk: {risk}")
    evidence.append(f"Renewal probability: {probability:.1f}%")
    evidence.append(f"Current ARR exposure: {money(arr)}")
    evidence.append(f"Engagement score: {engagement:.1f}")

    if discount >= 12:
        evidence.append(f"Discount level: {discount:.1f}%")

    if not_adopted:
        evidence.append(
            "Product/service gaps to investigate: "
            + ", ".join(not_adopted)
        )

    if footprint_signal:
        evidence.append(
            f"Large operating footprint: {int(facilities)} facilities, "
            f"{int(affiliates)} affiliates, {int(beds)} beds"
        )

    # Immediate execution steps vary by selected primary action.
    steps_by_action = {
        "Stabilize the renewal": [
            "Confirm the specific commercial and stakeholder reasons behind the renewal risk.",
            "Build a focused recovery plan with owners, dates, and value proof points.",
            "Protect the current renewal before making expansion the primary conversation.",
        ],
        "Increase stakeholder engagement": [
            "Map decision-makers, users, and executive stakeholders.",
            "Increase purposeful touchpoints around value realization and unresolved concerns.",
            "Track engagement improvement before the renewal conversation.",
        ],
        "Review pricing and value case": [
            "Review discount history and current account economics.",
            "Prepare a clear value narrative before pricing discussions.",
            "Avoid treating further discounting as the default retention lever.",
        ],
        "Investigate product/service whitespace": [
            "Validate which unadopted capabilities address documented customer needs.",
            "Use discovery questions before positioning any additional product or service.",
            "Prioritize only the whitespace with clear business fit and stakeholder interest.",
        ],
        "Prepare account expansion plan": [
            "Identify whitespace across facilities, affiliates, or business units.",
            "Prioritize expansion opportunities by business fit, value, and stakeholder readiness.",
            "Sequence expansion after confirming current account health and value realization.",
        ],
        "Maintain and deepen account value": [
            "Maintain regular health checks and executive value communication.",
            "Monitor renewal probability and engagement for deterioration.",
            "Explore measured adoption or expansion opportunities without creating renewal risk.",
        ],
    }

    # Business objective is intentionally distinct from the action itself.
    objective_by_action = {
        "Stabilize the renewal": "Protect current ARR and improve renewal confidence.",
        "Increase stakeholder engagement": "Strengthen account health and decision-maker alignment.",
        "Review pricing and value case": "Protect account economics while reinforcing demonstrated value.",
        "Investigate product/service whitespace": "Identify validated adoption opportunities without assuming product fit.",
        "Prepare account expansion plan": "Grow account value through evidence-based whitespace expansion.",
        "Maintain and deepen account value": "Preserve healthy renewal momentum while increasing long-term account value.",
    }

    return {
        "primary_action": primary_action,
        "primary_score": int(primary_score),
        "secondary_action": secondary_action,
        "secondary_score": int(secondary_score),
        "business_objective": objective_by_action[primary_action],
        "evidence": evidence,
        "immediate_steps": steps_by_action[primary_action],
        "active_products": adopted,
        "product_gaps_to_investigate": not_adopted,
        "decision_note": (
            "This next-best-action is generated from deterministic POC "
            "business rules using the available account and model signals. "
            "It is a recommendation, not a customer fact."
        ),
    }


def build_product_service_intelligence(row):
    """
    Deterministic Phase 3.2 product/service intelligence layer.

    Purpose:
    - Separate what the account currently uses from what it does not use.
    - Translate generic product knowledge into discovery-oriented guidance.
    - Never treat non-adoption as proof that the customer needs to buy.

    The output is a recommendation layer, not customer fact discovery.
    """
    adopted, not_adopted = product_adoption_from_row(row)

    risk = str(row.get("Renewal Risk", ""))
    engagement = float(row.get("Engagement Score", 0) or 0)
    facilities = float(row.get("Facilities", 0) or 0)
    affiliates = float(row.get("Affiliates", 0) or 0)
    beds = float(row.get("Beds", 0) or 0)

    large_footprint = (
        facilities >= 10 or affiliates >= 10 or beds >= 1500
    )

    intelligence = []

    for product_name, knowledge in PRODUCT_SERVICE_KNOWLEDGE.items():
        is_active = product_name in adopted

        if is_active:
            posture = "Value realization"
            rationale = (
                "The account is already using this capability, so the immediate "
                "focus should be demonstrated value, usage health, and retention."
            )
            discovery_questions = [
                "What business outcome is this capability currently supporting?",
                "Is usage/value realization strong enough to reinforce renewal confidence?",
            ]
        else:
            posture = "Discovery candidate"
            rationale = (
                "The capability is not currently adopted. Its relevance should be "
                "validated through customer discovery before any expansion action."
            )
            discovery_questions = [
                "What customer need or workflow could this capability address?",
                "Is there stakeholder interest and a measurable business outcome?",
            ]

        fit_evidence = []

        # Evidence is deliberately phrased as a signal, not proof of fit.
        if product_name == "Managed Services" and large_footprint:
            fit_evidence.append(
                "Large operating footprint may indicate higher operational complexity."
            )
        elif product_name == "Data Connect" and large_footprint:
            fit_evidence.append(
                "Large organizational footprint may create data-connectivity or integration complexity."
            )
        elif product_name == "Enterprise Exchange" and large_footprint:
            fit_evidence.append(
                "Large footprint and multiple organizational entities may make scalable connectivity worth exploring."
            )
        elif product_name == "Lumere":
            fit_evidence.append(
                "Generic healthcare analytics capability; account-specific analytical need is not directly captured in the POC data."
            )

        if engagement >= 70 and not is_active:
            fit_evidence.append(
                "Relatively strong engagement can support a discovery conversation, but does not establish product fit."
            )

        if risk == "High Risk" and not is_active:
            fit_evidence.append(
                "Because renewal risk is high, any adoption discussion should support the renewal objective rather than distract from it."
            )

        intelligence.append(
            {
                "product_service": product_name,
                "status": "Active" if is_active else "Not currently adopted",
                "category": knowledge.get("category", ""),
                "capabilities": knowledge.get("capabilities", []),
                "business_value": knowledge.get("business_value", []),
                "generic_fit_signals": knowledge.get("fit_signals", []),
                "account_specific_signals": fit_evidence,
                "recommended_posture": posture,
                "rationale": rationale,
                "discovery_questions": discovery_questions,
            }
        )

    active_items = [
        item for item in intelligence if item["status"] == "Active"
    ]
    gap_items = [
        item for item in intelligence if item["status"] == "Not currently adopted"
    ]

    # Rank only for discovery sequencing. This is NOT a claim of product fit.
    def discovery_score(item):
        score = 0
        if item["status"] != "Not currently adopted":
            return -1
        score += len(item["account_specific_signals"]) * 20
        if engagement >= 70:
            score += 10
        if large_footprint:
            score += 10
        if risk == "High Risk":
            score -= 15
        return score

    gap_items = sorted(
        gap_items,
        key=discovery_score,
        reverse=True,
    )

    return {
        "active": active_items,
        "not_adopted": gap_items,
        "account_signals": {
            "renewal_risk": risk,
            "engagement_score": engagement,
            "facilities": facilities,
            "affiliates": affiliates,
            "beds": beds,
            "large_footprint_signal": large_footprint,
        },
        "decision_note": (
            "Product/service intelligence is a POC discovery layer. "
            "Non-adoption does not mean the customer needs the product or service. "
            "Validate business need, stakeholder interest, and measurable value first."
        ),
    }



def _product_item_map(row):
    intelligence = build_product_service_intelligence(row)
    return intelligence, {item["product_service"]: item for item in intelligence["active"] + intelligence["not_adopted"]}


def build_portfolio_growth_intelligence(dataframe):
    """
    Deterministic Phase 3.3 portfolio-growth intelligence.

    Segments accounts using available POC/model signals. This does not
    claim actual pipeline, customer intent, whitespace revenue, or sales
    readiness. It creates a prioritization framework from the available data.
    """
    rows = dataframe.copy()

    # Normalize numeric fields defensively.
    numeric_cols = [
        "Current ARR ($)",
        "Renewal Probability (%)",
        "Engagement Score",
        "Facilities",
        "Affiliates",
        "Beds",
    ]
    for col in numeric_cols:
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce").fillna(0)

    def adoption_gap_count(row):
        _, gaps = product_adoption_from_row(row)
        return len(gaps)

    rows["Adoption Gaps"] = rows.apply(adoption_gap_count, axis=1)

    # Growth score is a transparent POC prioritization signal, not an ML score.
    def growth_score(row):
        risk = str(row.get("Renewal Risk", ""))
        probability = float(row.get("Renewal Probability (%)", 0))
        engagement = float(row.get("Engagement Score", 0))
        arr = float(row.get("Current ARR ($)", 0))
        facilities = float(row.get("Facilities", 0))
        affiliates = float(row.get("Affiliates", 0))
        beds = float(row.get("Beds", 0))
        gaps = int(row.get("Adoption Gaps", 0))

        score = 0.0
        if risk == "Low Risk":
            score += 45
        elif risk == "Medium Risk" and probability >= 55:
            score += 20
        else:
            score -= 25

        if probability >= 80:
            score += 20
        elif probability >= 70:
            score += 12

        if engagement >= 80:
            score += 20
        elif engagement >= 70:
            score += 12

        # A larger footprint creates a discovery signal, not proof of demand.
        if facilities >= 20:
            score += 8
        elif facilities >= 10:
            score += 5
        if affiliates >= 15:
            score += 6
        elif affiliates >= 10:
            score += 4
        if beds >= 3000:
            score += 8
        elif beds >= 1500:
            score += 5

        # Adoption whitespace is useful only as a discovery signal.
        if gaps >= 2:
            score += 10
        elif gaps == 1:
            score += 5

        # ARR is used as exposure context, not as evidence of growth intent.
        if arr >= 1_000_000:
            score += 5

        return score

    rows["Growth Prioritization Signal"] = rows.apply(growth_score, axis=1)

    # Portfolio segments.
    growth_candidates = rows[
        (rows["Renewal Risk"] == "Low Risk")
        & (rows["Renewal Probability (%)"] >= 70)
        & (rows["Engagement Score"] >= 70)
    ].copy()

    protect_then_expand = rows[
        (rows["Renewal Risk"] == "Medium Risk")
        & (rows["Renewal Probability (%)"] >= 55)
        & (rows["Engagement Score"] >= 60)
    ].copy()

    renewal_first = rows[rows["Renewal Risk"] == "High Risk"].copy()

    growth_candidates = growth_candidates.sort_values(
        ["Growth Prioritization Signal", "Current ARR ($)"],
        ascending=[False, False],
    )
    protect_then_expand = protect_then_expand.sort_values(
        ["Growth Prioritization Signal", "Current ARR ($)"],
        ascending=[False, False],
    )
    renewal_first = renewal_first.sort_values(
        ["Current ARR ($)", "Renewal Probability (%)"],
        ascending=[False, True],
    )

    def account_preview(frame, limit=5):
        output = []
        for _, row in frame.head(limit).iterrows():
            output.append({
                "account": str(row.get("Account Name", "Unknown")),
                "arr": float(row.get("Current ARR ($)", 0)),
                "risk": str(row.get("Renewal Risk", "")),
                "probability": float(row.get("Renewal Probability (%)", 0)),
                "engagement": float(row.get("Engagement Score", 0)),
                "adoption_gaps": int(row.get("Adoption Gaps", 0)),
                "signal": float(row.get("Growth Prioritization Signal", 0)),
            })
        return output

    return {
        "portfolio_size": int(len(rows)),
        "total_arr": float(rows["Current ARR ($)"].sum()),
        "growth_candidate_count": int(len(growth_candidates)),
        "growth_candidate_arr": float(growth_candidates["Current ARR ($)"].sum()),
        "protect_then_expand_count": int(len(protect_then_expand)),
        "protect_then_expand_arr": float(protect_then_expand["Current ARR ($)"].sum()),
        "renewal_first_count": int(len(renewal_first)),
        "renewal_first_arr": float(renewal_first["Current ARR ($)"].sum()),
        "other_count": int(
            len(rows)
            - len(growth_candidates)
            - len(protect_then_expand)
            - len(renewal_first)
        ),
        "other_arr": float(
            rows["Current ARR ($)"].sum()
            - growth_candidates["Current ARR ($)"].sum()
            - protect_then_expand["Current ARR ($)"].sum()
            - renewal_first["Current ARR ($)"].sum()
        ),
        "growth_candidates": account_preview(growth_candidates, limit=8),
        "protect_then_expand": account_preview(protect_then_expand),
        "renewal_first": account_preview(renewal_first),
        "decision_note": (
            "Portfolio growth intelligence is a POC prioritization framework based on "
            "renewal, engagement, ARR, footprint, and adoption signals. It does not "
            "represent confirmed expansion intent, pipeline, whitespace revenue, or "
            "customer buying intent."
        ),
    }


def format_growth_focus_accounts(dataframe):
    """Account-focused portfolio view: identify growth candidates without repeating the full strategy."""
    intel = build_portfolio_growth_intelligence(dataframe)

    lines = [
        "### 🎯 Growth Focus Accounts",
        "",
        f"**{intel['growth_candidate_count']} accounts** meet the POC growth-candidate criteria, representing **{money(intel['growth_candidate_arr'])} ARR**.",
        "",
    ]

    if intel["growth_candidates"]:
        lines.append("**Growth-candidate accounts**")
        lines.append("")
        for idx, item in enumerate(intel["growth_candidates"], 1):
            lines.append(
                f"{idx}. **{item['account']}** — {money(item['arr'])} ARR | "
                f"{item['probability']:.1f}% renewal probability | "
                f"{item['engagement']:.1f} engagement | "
                f"{item['adoption_gaps']} adoption gap(s) to investigate"
            )
    else:
        lines.append("No accounts currently meet the configured growth-candidate criteria.")

    lines.extend([
        "",
        "### How to Use This List",
        "",
        "These accounts meet the POC growth-candidate criteria and can be used for targeted discovery.",
        "",
        "Before treating any account as an expansion opportunity, validate the business need, stakeholder interest, product fit, and measurable value.",
        "",
        "### Decision Guardrail",
        "",
        intel["decision_note"],
    ])
    return "\n".join(lines)


def format_portfolio_growth_prioritization(dataframe):
    """Methodology-focused portfolio view: explain how growth should be prioritized."""
    intel = build_portfolio_growth_intelligence(dataframe)

    lines = [
        "### 🧭 Portfolio Growth Prioritization",
        "",
        "**Objective:** Apply a consistent sequence for deciding where growth discovery belongs in the portfolio.",
        "",
        "### Prioritization Sequence",
        "",
        "1. **Renewal Health** — protect accounts with elevated renewal risk before making expansion the primary motion.",
        "2. **Account Engagement** — use stronger engagement as a signal that discovery can be supported by an active relationship.",
        "3. **Commercial Exposure** — use ARR to understand account importance and exposure, not as evidence of buying intent.",
        "4. **Adoption Whitespace** — use non-adopted products/services as discovery signals, not confirmed opportunities.",
        "5. **Business Need Validation** — confirm the operational, analytical, or workflow problem before recommending an expansion path.",
        "6. **Measurable Value** — define the expected business outcome before moving from discovery to an adoption/expansion proposal.",
        "",
        "### Portfolio POC Segmentation",
        "",
        f"- **Growth Candidates:** {intel['growth_candidate_count']} accounts | {money(intel['growth_candidate_arr'])} ARR",
        f"- **Protect Then Expand:** {intel['protect_then_expand_count']} accounts | {money(intel['protect_then_expand_arr'])} ARR",
        f"- **Renewal-First:** {intel['renewal_first_count']} accounts | {money(intel['renewal_first_arr'])} ARR",
        f"- **Other / Monitor:** {intel['other_count']} accounts | {money(intel['other_arr'])} ARR",
        "",
        "### Management Rule",
        "",
        "**Renewal health → engagement → whitespace signal → validated need → measurable value → expansion discovery**",
        "",
        "### Decision Guardrail",
        "",
        intel["decision_note"],
    ]
    return "\n".join(lines)


def format_portfolio_growth_strategy(dataframe):
    """Executive portfolio-level growth strategy using deterministic POC signals."""
    intel = build_portfolio_growth_intelligence(dataframe)

    lines = [
        "### 📈 Portfolio Growth Strategy",
        "",
        "**Objective:** Concentrate growth discovery where account health supports expansion, while protecting renewal value first.",
        "",
        "### 1. Growth Candidates",
        "",
        f"**{intel['growth_candidate_count']} accounts** meet the POC growth-candidate criteria, representing **{money(intel['growth_candidate_arr'])} ARR**.",
        "",
    ]

    if intel["growth_candidates"]:
        for item in intel["growth_candidates"]:
            lines.append(
                f"- **{item['account']}** — {money(item['arr'])} ARR | "
                f"{item['probability']:.1f}% renewal probability | "
                f"{item['engagement']:.1f} engagement | "
                f"{item['adoption_gaps']} adoption gap(s) to investigate"
            )
    else:
        lines.append("No accounts currently meet the configured growth-candidate criteria.")

    lines.extend([
        "",
        "### 2. Protect Then Expand",
        "",
        f"**{intel['protect_then_expand_count']} accounts** show a combination of moderate risk, reasonable renewal probability, and engagement that may support selective expansion after renewal blockers are addressed.",
        "",
    ])
    if intel["protect_then_expand"]:
        for item in intel["protect_then_expand"]:
            lines.append(
                f"- **{item['account']}** — {money(item['arr'])} ARR | "
                f"{item['probability']:.1f}% renewal probability | "
                f"{item['engagement']:.1f} engagement"
            )
    else:
        lines.append("No accounts currently meet the configured protect-then-expand criteria.")

    lines.extend([
        "",
        "### 3. Renewal-First Portfolio",
        "",
        f"**{intel['renewal_first_count']} high-risk accounts** should remain primarily renewal-focused before growth conversations become the main motion.",
        "",
        "### 4. Portfolio Execution Priorities",
        "",
        "1. Protect high-value renewals where risk is elevated.",
        "2. Focus expansion discovery on healthy, engaged accounts.",
        "3. Use adoption gaps as discovery signals rather than confirmed sales opportunities.",
        "4. Validate stakeholder need, business problem, and measurable value before expansion.",
        "5. Add CRM opportunity, contract, stakeholder, product-usage, and pipeline data before treating this as production growth prioritization.",
        "",
        "### Decision Guardrail",
        "",
        intel["decision_note"],
    ])
    return "\n".join(lines)


def format_product_service_gaps(row):
    """Factual adoption-gap view: answer only what is not currently adopted."""
    intelligence, items = _product_item_map(row)
    gaps = intelligence["not_adopted"]

    lines = [
        "### Product / Service Adoption Gaps",
        "",
    ]

    if gaps:
        lines.append("**Not Currently Adopted**")
        lines.append("")
        for item in gaps:
            lines.append(f"- **{item['product_service']}** — not currently adopted")
    else:
        lines.append("All configured products/services are currently marked as adopted.")

    lines.extend([
        "",
        "### Current Adoption Context",
        "",
        "Active: " + (", ".join(item["product_service"] for item in intelligence["active"]) or "None"),
        "",
        "This identifies adoption status from the available POC data. Non-adoption alone does not establish customer need or product fit.",
    ])
    return "\n".join(lines)


def format_product_service_opportunities(row):
    """Discovery view: turn non-adoption into ranked areas to validate, not upsell claims."""
    intelligence, items = _product_item_map(row)
    gaps = intelligence["not_adopted"]
    risk = intelligence["account_signals"]["renewal_risk"]

    lines = [
        "### Product / Service Opportunity Discovery",
        "",
        "The following are **discovery areas**, not confirmed sales opportunities.",
        "",
    ]

    if not gaps:
        lines.append("No configured adoption gaps are available for discovery.")
    else:
        for idx, item in enumerate(gaps, 1):
            signals = item["account_specific_signals"]
            signal_text = signals[0] if signals else "No direct account-specific fit signal is captured in the POC data."
            questions = item["discovery_questions"]
            lines.extend([
                f"### {idx}. {item['product_service']}",
                "",
                f"**Why investigate:** {signal_text}",
                f"**Discovery question:** {questions[0]}",
                f"**Validation question:** {questions[1]}",
                "",
            ])

    lines.extend([
        "### Recommended Discovery Posture",
        "",
        f"Because the current renewal risk is **{risk}**, validate whether any identified capability addresses a real customer problem before positioning it as an expansion conversation.",
        "",
        "**Decision rule:** business need → stakeholder interest → measurable value → then consider adoption/expansion.",
    ])
    return "\n".join(lines)


def format_product_adoption_strategy(row):
    """Action view: provide an adoption motion without repeating the product inventory."""
    intelligence, items = _product_item_map(row)
    active_names = [item["product_service"] for item in intelligence["active"]]
    gap_names = [item["product_service"] for item in intelligence["not_adopted"]]
    risk = intelligence["account_signals"]["renewal_risk"]
    engagement = intelligence["account_signals"]["engagement_score"]

    lines = [
        "### Product / Service Adoption Strategy",
        "",
        "**Objective:** Increase realized value first, then expand adoption where a validated business need exists.",
        "",
        "### 1. Strengthen Existing Adoption",
        "",
        f"Review usage, outcomes and value realization for **{', '.join(active_names) if active_names else 'the currently active capabilities'}**.",
        "",
        "### 2. Identify the Business Problem",
        "",
        "Start with operational, analytical or workflow pain points rather than starting with a product pitch.",
        "",
        "### 3. Run Targeted Discovery",
        "",
    ]

    if gap_names:
        lines.append("Validate whether " + ", ".join(gap_names) + " address a confirmed customer need.")
    else:
        lines.append("No current adoption gaps are captured in the POC data; focus on deeper utilization and value realization.")

    lines.extend([
        "",
        "### 4. Define Measurable Value",
        "",
        "Agree on a specific business outcome before moving from discovery to an adoption proposal.",
        "",
        "### 5. Protect the Renewal",
        "",
        f"With **{risk}** renewal risk and an Engagement Score of **{engagement:.1f}**, adoption activity should reinforce the renewal/value case rather than become a disconnected upsell campaign.",
        "",
        "### Adoption Guardrail",
        "",
        "Non-adoption is an opportunity to ask better discovery questions — not evidence that the customer should buy the capability.",
    ])
    return "\n".join(lines)

def format_product_service_intelligence(row):
    intelligence = build_product_service_intelligence(row)

    lines = [
        "### 🧩 Product & Service Intelligence",
        "",
        "### Currently Active",
        "",
    ]

    if intelligence["active"]:
        for item in intelligence["active"]:
            value = ", ".join(item["business_value"])
            lines.append(f"**{item['product_service']}** — {item['recommended_posture']}")
            lines.append(f"- Business value areas: {value}")
            lines.append(f"- Interpretation: {item['rationale']}")
            lines.append("")
    else:
        lines.append("No products/services are currently marked as active.")
        lines.append("")

    lines.extend([
        "### Not Currently Adopted",
        "",
    ])

    if intelligence["not_adopted"]:
        for item in intelligence["not_adopted"]:
            lines.append(f"**{item['product_service']}** — {item['recommended_posture']}")
            if item["account_specific_signals"]:
                for signal in item["account_specific_signals"]:
                    lines.append(f"- Signal to investigate: {signal}")
            else:
                lines.append(
                    "- Signal to investigate: No direct account-specific fit signal is captured in the POC data."
                )
            lines.append(
                "- Discovery focus: " + " / ".join(item["discovery_questions"])
            )
            lines.append("")
    else:
        lines.append("All configured products/services are currently marked as adopted.")
        lines.append("")

    lines.extend([
        "### Decision Guardrail",
        "",
        intelligence["decision_note"],
    ])

    return "\n".join(lines)


def format_next_best_action(row):
    nba = build_next_best_action(row)

    evidence_text = "\n".join(
        f"- {item}" for item in nba["evidence"]
    )

    step_text = "\n".join(
        f"{i}. {step}"
        for i, step in enumerate(nba["immediate_steps"], 1)
    )

    secondary = nba["secondary_action"]

    return (
        "### 🎯 Next Best Business Action\n\n"
        f"**Primary Action:** {nba['primary_action']}\n\n"
        f"**Business Objective:** {nba['business_objective']}\n\n"
        "### Why This Is the Priority\n"
        f"{evidence_text}\n\n"
        "### Immediate Next Steps\n"
        f"{step_text}\n\n"
        f"### Secondary Opportunity\n\n"
        f"**{secondary}** — consider this after the primary action "
        "or in parallel where it does not create renewal risk.\n\n"
        "### Decision Guardrail\n\n"
        f"{nba['decision_note']}"
    )


def build_dashboard_display_data(row, signals, actions):
    probability = float(row["Renewal Probability (%)"])
    engagement = float(row["Engagement Score"])
    arr = float(row["Current ARR ($)"])

    return {
        "Account Name": str(row["Account Name"]),
        "Renewal Risk": str(row["Renewal Risk"]),
        "Renewal Probability": f"{probability:.1f}%",
        "Current ARR": money(arr),
        "Engagement Score": f"{engagement:.1f}",
        "Renewed": str(row.get("Renewed", "")),
        "Contract Status": str(row.get("Contract Status", "")),
        "Sales Region": str(row.get("Sales Region", "")),
        "Facilities": str(row.get("Facilities", "")),
        "Affiliates": str(row.get("Affiliates", "")),
        "Beds": str(row.get("Beds", "")),
        "Discount": str(row.get("Discount (%)", "")),
        "Managed Services": str(row.get("Has Managed Services", "")),
        "Data Connect": str(row.get("Has Data Connect", "")),
        "Lumere": str(row.get("Has Lumere", "")),
        "Enterprise Exchange": str(row.get("Has Enterprise Exchange", "")),
        "Risk Signals": signals,
        "Recommended Actions": actions,
    }


def build_strategic_synthesis(row):
    """
    Phase 3.5 deterministic synthesis layer.

    Combines the outputs of Phases 3.1-3.4 into one grounded account-level
    recommendation. This layer does not invent new customer facts and does
    not replace the underlying decision engines; it synthesizes them.
    """
    account_name = str(row.get("Account Name", ""))
    risk = str(row.get("Renewal Risk", ""))
    probability = float(row.get("Renewal Probability (%)", 0) or 0)
    arr = float(row.get("Current ARR ($)", 0) or 0)
    engagement = float(row.get("Engagement Score", 0) or 0)
    facilities = float(row.get("Facilities", 0) or 0)
    affiliates = float(row.get("Affiliates", 0) or 0)
    beds = float(row.get("Beds", 0) or 0)

    adopted, not_adopted = product_adoption_from_row(row)
    nba = build_next_best_action(row)
    healthcare = build_healthcare_domain_reasoning(row)
    product_intel = build_product_service_intelligence(row)

    # Primary recommendation follows the deterministic Next Best Action.
    if risk == "High Risk":
        executive_recommendation = (
            f"Stabilize the renewal for {account_name} first, using the healthcare operating context "
            "to validate the drivers behind the risk and strengthen the value case before pursuing expansion."
        )
        business_objective = "Protect current ARR and improve renewal confidence."
    elif risk == "Medium Risk":
        executive_recommendation = (
            f"Stabilize renewal confidence for {account_name} while using validated healthcare business needs "
            "to guide selective adoption or expansion discovery."
        )
        business_objective = "Reduce renewal risk while creating validated growth pathways."
    else:
        executive_recommendation = (
            f"Maintain and deepen value for {account_name}, then use validated healthcare needs and adoption whitespace "
            "to guide selective expansion discovery."
        )
        business_objective = "Protect the healthy relationship while developing evidence-based growth opportunities."

    evidence = [
        f"Renewal signal: {risk} with {probability:.1f}% renewal probability.",
        f"Current commercial exposure: {money(arr)} ARR with engagement of {engagement:.1f}.",
        f"Healthcare footprint: {facilities:,.0f} facilities, {affiliates:,.0f} affiliates and {beds:,.0f} beds.",
    ]

    if adopted:
        evidence.append(
            "Existing adoption: " + ", ".join(adopted) + "."
        )
    if not_adopted:
        evidence.append(
            "Adoption whitespace to investigate: " + ", ".join(not_adopted) + "."
        )

    immediate_actions = list(nba["immediate_steps"])
    # Keep the synthesis concise and non-repetitive.
    if not immediate_actions:
        immediate_actions = [
            "Confirm the account's highest-priority business and renewal concerns.",
            "Validate realized value from currently adopted capabilities.",
            "Define a measurable outcome for any subsequent adoption or expansion discussion.",
        ]

    if risk == "High Risk":
        immediate_actions = [
            "Align with the customer on renewal blockers and the value case.",
            "Validate the healthcare workflow, data or coordination issues contributing to the renewal discussion.",
            "Confirm realized value from current capabilities before discussing additional scope.",
        ]
    elif risk == "Medium Risk":
        immediate_actions = [
            "Address the most material renewal blockers.",
            "Validate one or two healthcare business needs where current adoption or footprint creates a relevant signal.",
            "Define measurable value before progressing any expansion discussion.",
        ]
    else:
        immediate_actions = [
            "Maintain renewal health and reinforce realized value.",
            "Prioritize discovery around the strongest validated healthcare or adoption signal.",
            "Convert only validated needs into targeted expansion discussions.",
        ]

    product_considerations = []
    if adopted:
        product_considerations.append(
            "First validate usage depth and realized value of currently adopted capabilities: "
            + ", ".join(adopted) + "."
        )
    if not_adopted:
        product_considerations.append(
            "Treat non-adopted capabilities (" + ", ".join(not_adopted) + ") as discovery areas only; validate fit before recommending adoption."
        )
    if not product_considerations:
        product_considerations.append("No configured product/service adoption gap is available for this account.")

    healthcare_considerations = [
        item["theme"] for item in healthcare["themes_to_validate"]
    ]

    validation_points = [
        "Confirm the customer's actual business problem and renewal decision drivers.",
        "Validate stakeholder priorities and realized value; these are not captured in the POC dataset.",
        "Confirm product fit and measurable outcomes before treating whitespace as an expansion opportunity.",
    ]

    return {
        "account_name": account_name,
        "executive_recommendation": executive_recommendation,
        "business_objective": business_objective,
        "evidence": evidence,
        "immediate_actions": immediate_actions,
        "product_considerations": product_considerations,
        "healthcare_considerations": healthcare_considerations,
        "validation_points": validation_points,
        "next_best_action": nba["primary_action"],
        "decision_guardrail": (
            "This synthesis combines account facts, ML signals, deterministic business rules, product/service knowledge, "
            "portfolio context and healthcare-domain hypotheses. It is an account-management recommendation, not a customer fact, "
            "confirmed buying intent, pipeline opportunity, or causal explanation of renewal risk."
        ),
    }


def format_strategic_synthesis(row):
    s = build_strategic_synthesis(row)
    lines = [
        "### 🎯 Executive Strategic Recommendation",
        "",
        f"**Account:** {s['account_name']}",
        "",
        "### 1. Executive Recommendation",
        "",
        s["executive_recommendation"],
        "",
        "### 2. Why This Matters",
        "",
    ]
    for item in s["evidence"]:
        lines.append(f"- {item}")

    lines.extend([
        "",
        "### 3. Immediate Priorities",
        "",
    ])
    for idx, item in enumerate(s["immediate_actions"], 1):
        lines.append(f"{idx}. {item}")

    lines.extend([
        "",
        "### 4. Product / Service Considerations",
        "",
    ])
    for item in s["product_considerations"]:
        lines.append(f"- {item}")

    lines.extend([
        "",
        "### 5. Healthcare Considerations",
        "",
    ])
    for item in s["healthcare_considerations"]:
        lines.append(f"- {item}")

    lines.extend([
        "",
        "### 6. Business Objective",
        "",
        s["business_objective"],
        "",
        "### 7. Key Validation Points",
        "",
    ])
    for item in s["validation_points"]:
        lines.append(f"- {item}")

    lines.extend([
        "",
        "### Decision Guardrail",
        "",
        s["decision_guardrail"],
    ])
    return "\n".join(lines)


def build_healthcare_domain_reasoning(row):
    """
    Phase 3.4 healthcare-domain reasoning layer.

    Purpose:
    Interpret existing account, ML, product and portfolio signals through a
    healthcare-domain lens WITHOUT duplicating the Phase 3.2 product/service
    intelligence layer.

    Separation of responsibilities:
    - Phase 3.2 answers product adoption / whitespace questions.
    - Phase 3.3 answers portfolio growth questions.
    - Phase 3.4 explains how healthcare-provider context changes the
      interpretation of those signals.

    Healthcare themes are hypotheses to validate, not confirmed customer facts.
    """
    account_name = str(row.get("Account Name", ""))
    risk = str(row.get("Renewal Risk", ""))
    probability = float(row.get("Renewal Probability (%)", 0) or 0)
    arr = float(row.get("Current ARR ($)", 0) or 0)
    engagement = float(row.get("Engagement Score", 0) or 0)
    facilities = float(row.get("Facilities", 0) or 0)
    affiliates = float(row.get("Affiliates", 0) or 0)
    beds = float(row.get("Beds", 0) or 0)

    adopted, not_adopted = product_adoption_from_row(row)
    nba = build_next_best_action(row)

    facts = [
        f"Healthcare account: {account_name}",
        f"Operating footprint: {facilities:,.0f} facilities, {affiliates:,.0f} affiliates, {beds:,.0f} beds",
        f"Current ARR: {money(arr)}",
        f"Engagement score: {engagement:.1f}",
        "Currently active products/services: " + (", ".join(adopted) if adopted else "None captured"),
        "Not currently adopted: " + (", ".join(not_adopted) if not_adopted else "No configured gaps captured"),
    ]

    model_signals = [
        f"Renewal risk: {risk}",
        f"Renewal probability: {probability:.1f}%",
    ]

    # Keep Phase 3.4 focused on healthcare-domain interpretation rather than
    # repeating product-by-product recommendations already handled in Phase 3.2.
    themes = []

    if facilities >= 10 or affiliates >= 10:
        themes.append({
            "theme": "Multi-facility coordination and consistency",
            "reason": (
                f"The account spans {facilities:,.0f} facilities and {affiliates:,.0f} affiliates. "
                "That scale may warrant exploration of workflow consistency, coordination, "
                "data movement, and operational standardization across the provider network."
            ),
            "guardrail": "The footprint is a scale signal; it does not prove an operational problem.",
        })

    # Existing adoption is interpreted at a healthcare business-value level,
    # without repeating Phase 3.2's product-specific opportunity analysis.
    if adopted:
        themes.append({
            "theme": "Enterprise data and connectivity value realization",
            "reason": (
                "The account already uses configured data/connectivity capabilities. "
                "For a large healthcare organization, the relevant question is whether "
                "those capabilities are delivering consistent visibility, connectivity, "
                "and measurable value across the organization."
            ),
            "guardrail": "Active adoption does not establish utilization depth or realized business outcomes; the POC has no usage telemetry.",
        })

    if beds >= 1500:
        themes.append({
            "theme": "Analytics and decision-support relevance",
            "reason": (
                f"A {beds:,.0f}-bed provider footprint represents substantial healthcare scale. "
                "That makes consistent analytics, reporting, and decision-support themes "
                "worth validating where leaders need organization-wide insight."
            ),
            "guardrail": "Provider scale alone does not establish a specific analytics use case or unmet need.",
        })

    if "Managed Services" in not_adopted or facilities >= 10 or affiliates >= 10:
        themes.append({
            "theme": "Operational support and standardization discovery",
            "reason": (
                "The provider footprint may make operational support, process consistency, "
                "or cross-facility standardization relevant areas for discovery. If a validated "
                "need exists, the appropriate service or capability can then be evaluated."
            ),
            "guardrail": "Neither footprint nor non-adoption proves that an operational-support need exists or that a particular service is the right solution.",
        })

    if not themes:
        themes.append({
            "theme": "Healthcare operating-context validation",
            "reason": (
                "The available account data does not provide enough healthcare-specific scale "
                "signals to support a stronger domain hypothesis. Validate the customer's operating "
                "model, workflow priorities, and value drivers directly."
            ),
            "guardrail": "The POC should not infer a healthcare problem that is not represented in the available data.",
        })

    if risk == "High Risk":
        commercial_interpretation = (
            "Renewal protection should remain the primary commercial motion. Healthcare-domain "
            "reasoning should help the account team understand the operating context and strengthen "
            "the value case, not create a disconnected expansion campaign."
        )
    elif risk == "Medium Risk":
        commercial_interpretation = (
            "Balance renewal stabilization with selective healthcare-domain discovery. Any expansion "
            "path should follow understanding of renewal blockers and validation of a genuine business need."
        )
    else:
        commercial_interpretation = (
            "The healthier renewal position allows more room for healthcare-domain discovery, while "
            "expansion still requires validated need, stakeholder interest, and measurable value."
        )

    manager_focus = [
        f"Use **{nba['primary_action']}** as the primary business motion.",
        "Validate the healthcare operating or workflow issues behind the renewal signal instead of assuming a cause from account scale.",
        "Confirm realized value and usage health for capabilities already adopted across the organization.",
        "Use healthcare-domain themes as discovery hypotheses and connect any product/service discussion to a documented customer need.",
        "Define a measurable business outcome before moving from discovery to an expansion recommendation.",
    ]

    discovery_questions = [
        "Across the facilities and affiliates, where do leaders see the greatest workflow, data, or process inconsistency today?",
        "Which outcomes from the currently adopted capabilities are most important to the renewal decision?",
        "Where do leaders need more consistent analytics, reporting, or decision support across the organization?",
        "What measurable healthcare or operational outcome would justify a change in service scope or adoption?",
    ]

    return {
        "facts": facts,
        "model_signals": model_signals,
        "themes_to_validate": themes,
        "commercial_interpretation": commercial_interpretation,
        "manager_focus": manager_focus,
        "discovery_questions": discovery_questions,
        "next_best_action": nba,
        "decision_note": (
            "Healthcare-domain reasoning combines observed account facts, model signals, generic healthcare knowledge, "
            "and deterministic POC business rules. Healthcare themes are hypotheses to validate, not confirmed customer "
            "problems, product fit, expansion intent, or buying intent."
        ),
    }


def format_healthcare_domain_reasoning(row):
    reasoning = build_healthcare_domain_reasoning(row)

    lines = [
        "### 🏥 Healthcare-Aware Account Strategy",
        "",
        "**Objective:** Interpret the account through a healthcare-provider lens without turning scale, product gaps, or model signals into unsupported customer facts.",
        "",
        "### 1. What the Data Supports",
        "",
    ]

    for item in reasoning["facts"]:
        lines.append(f"- {item}")

    lines.extend(["", "**Model signals**", ""])
    for item in reasoning["model_signals"]:
        lines.append(f"- {item}")

    lines.extend(["", "### 2. Healthcare Themes to Validate", ""])
    if reasoning["themes_to_validate"]:
        for idx, item in enumerate(reasoning["themes_to_validate"], 1):
            lines.append(f"**{idx}. {item['theme']}**")
            lines.append(f"- Why it may matter: {item['reason']}")
            lines.append(f"- Evidence boundary: {item['guardrail']}")
            lines.append("")
    else:
        lines.append("No additional healthcare-domain theme is strongly indicated by the configured POC rules.")
        lines.append("")

    lines.extend([
        "### 3. Renewal & Commercial Implication",
        "",
        reasoning["commercial_interpretation"],
        "",
        "### 4. Account Manager Focus",
        "",
    ])
    for idx, item in enumerate(reasoning["manager_focus"], 1):
        lines.append(f"{idx}. {item}")

    lines.extend(["", "### 5. Suggested Discovery Questions", ""])
    for item in reasoning["discovery_questions"]:
        lines.append(f"- {item}")

    lines.extend([
        "",
        "### Decision Guardrail",
        "",
        reasoning["decision_note"],
    ])

    return "\n".join(lines)


def build_account_context(row, signals, actions, portfolio_summary):
    raw = {}

    for column in df.columns:
        value = row[column]
        try:
            if pd.isna(value):
                raw[column] = None
            elif hasattr(value, "item"):
                raw[column] = value.item()
            else:
                raw[column] = value
        except Exception:
            raw[column] = str(value)

    dashboard_display = build_dashboard_display_data(
        row,
        signals,
        actions,
    )

    adopted, not_adopted = product_adoption_from_row(row)

    knowledge_context = build_knowledge_context()

    context = {
        "dashboard_display_data": dashboard_display,
        "raw_source_data": raw,
        "portfolio_summary": portfolio_summary,
        "product_adoption_summary": {
            "active": adopted,
            "not_adopted": not_adopted,
        },
        "product_service_intelligence": build_product_service_intelligence(row),
        "next_best_action": build_next_best_action(row),
        "healthcare_domain_reasoning": build_healthcare_domain_reasoning(row),
        "strategic_synthesis": build_strategic_synthesis(row),
        "healthcare_knowledge": knowledge_context,
    }

    return json.dumps(context, indent=2, default=str)


def render_ai_markdown(content):
    if content is None:
        return

    text = str(content).strip()

    text = text.replace(r"\*\*", "**")
    text = text.replace(r"\_\_", "__")
    text = text.replace(r"\#", "#")
    text = text.replace(r"\- ", "- ")

    if text.startswith("```markdown") and text.endswith("```"):
        text = text[len("```markdown"):].strip()
        text = text[:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    st.markdown(text)


# ============================================================
# QUESTION-SPECIFIC DETERMINISTIC RESPONSES
# ============================================================

def build_renewal_manager_answer(row, signals, actions, question):
    intent = classify_standard_question(question)
    if intent == "portfolio_growth_strategy":
        return format_portfolio_growth_strategy(df)

    if intent == "growth_focus_accounts":
        return format_growth_focus_accounts(df)

    if intent == "portfolio_growth_prioritization":
        return format_portfolio_growth_prioritization(df)

    if intent == "strategic_synthesis":
        return format_strategic_synthesis(row)

    if intent == "healthcare_domain_reasoning":
        return format_healthcare_domain_reasoning(row)

    risk = str(row["Renewal Risk"])
    probability = float(row["Renewal Probability (%)"])
    engagement = float(row["Engagement Score"])
    arr = float(row["Current ARR ($)"])
    account_name = str(row["Account Name"])

    adopted, not_adopted = product_adoption_from_row(row)

    if intent == "risk":
        drivers = []

        if probability < 40:
            drivers.append(f"Low renewal probability ({probability:.1f}%)")
        elif probability < 70:
            drivers.append(
                f"Moderate renewal probability ({probability:.1f}%)"
            )

        if engagement < 50:
            drivers.append(f"Low engagement score ({engagement:.1f})")

        if not_adopted:
            drivers.append(
                f"{len(not_adopted)} product/service adoption gap(s): "
                + ", ".join(not_adopted)
            )

        if not drivers:
            drivers.append(
                "No major risk driver is identified from the available data."
            )

        return (
            f"### Why This Account Is at Risk\n\n"
            f"**{account_name} — {risk}**\n\n"
            f"**Renewal Probability:** {probability:.1f}%\n\n"
            f"### Key Risk Drivers\n"
            + "\n".join(
                f"{i}. {driver}" for i, driver in enumerate(drivers, 1)
            )
            + "\n\n### Management Interpretation\n\n"
            "The model signals indicate that this account deserves focused "
            "renewal attention. The signals should be treated as model "
            "indicators rather than proof of a specific customer cause."
        )

    if intent == "actions":
        action_text = "\n".join(
            f"{i}. {action}" for i, action in enumerate(actions, 1)
        )

        return (
            f"### Recommended Renewal Actions\n\n"
            f"**Account:** {account_name}\n\n"
            f"**Priority:** {get_priority(risk, arr)}\n\n"
            f"{action_text}\n\n"
            "### Business Objective\n\n"
            "Protect the renewal first, then use relevant adoption and "
            "value-expansion conversations to strengthen the account."
        )

    if intent == "summary":
        active_text = ", ".join(adopted) if adopted else "None"
        gap_text = ", ".join(not_adopted) if not_adopted else "None"

        return (
            f"### Executive Account Summary\n\n"
            f"**Account:** {account_name}\n"
            f"- **Risk:** {risk}\n"
            f"- **Renewal Probability:** {probability:.1f}%\n"
            f"- **Current ARR:** {money(arr)}\n"
            f"- **Engagement:** {engagement:.1f}\n"
            f"- **Footprint:** {row.get('Facilities', '')} facilities, "
            f"{row.get('Affiliates', '')} affiliates, "
            f"{row.get('Beds', '')} beds\n"
            f"- **Active Products/Services:** {active_text}\n"
            f"- **Not Currently Adopted:** {gap_text}\n\n"
            f"### Management Focus\n\n"
            f"Treat this as a {get_priority(risk, arr).lower()}-priority "
            "renewal and focus on protecting value while addressing the "
            "identified adoption gaps."
        )

    if intent == "impact":
        return (
            f"### Business Impact\n\n"
            f"**Current ARR:** {money(arr)}\n\n"
            f"**Renewal Probability:** {probability:.1f}%\n\n"
            f"**Risk:** {risk}\n\n"
            f"**Renewal Priority:** {get_priority(risk, arr)}\n\n"
            "The account represents meaningful current revenue exposure. "
            "The immediate commercial objective is to protect the renewal "
            "and strengthen demonstrated account value."
        )

    if intent == "adoption_gaps":
        return format_product_service_gaps(row)

    if intent == "opportunity_discovery":
        return format_product_service_opportunities(row)

    if intent == "adoption_strategy":
        return format_product_adoption_strategy(row)

    if intent == "best_action":
        return format_next_best_action(row)

    if intent == "priority":
        priority = get_priority(risk, arr)

        return (
            f"### Renewal Priority\n\n"
            f"**{priority} — {account_name}**\n\n"
            f"- **Risk:** {risk}\n"
            f"- **Renewal Probability:** {probability:.1f}%\n"
            f"- **Current ARR:** {money(arr)}\n"
            f"- **Engagement:** {engagement:.1f}\n\n"
            "### Management Focus\n\n"
            "Prioritize this account according to its risk classification "
            "and ARR exposure."
        )

    return None


# ============================================================
# STRATEGIC FALLBACK / KNOWLEDGE RESPONSE
# ============================================================

def build_strategic_prompt(
    question,
    account_name,
    account_context,
    strategic_intent,
):
    knowledge = build_knowledge_context()

    return f"""
USER QUESTION
=============
{question}

STRATEGIC INTENT
================
{strategic_intent}

SELECTED ACCOUNT
================
{account_name}

ACCOUNT + PORTFOLIO DATA
========================
{account_context}

RELEVANT HEALTHCARE / PRODUCT / STRATEGIC KNOWLEDGE
====================================================
{json.dumps(knowledge, indent=2, default=str)}

IMPORTANT:
- Actual account facts must come from the supplied account data.
- ML outputs are model signals, not causal proof.
- Knowledge-layer content is generic domain/capability guidance.
- Strategic recommendations are recommendations, not customer facts.
- The current POC may not contain prospect pipeline, competitive data,
  whitespace revenue, buyer commitments, or live opportunity stages.
- Never fabricate hospitals, prospects, pipeline, competitors, revenue,
  customer commitments, or product capabilities.
- If the question concerns portfolio growth or adding new hospital groups
  and prospect data is not present, explicitly state that the answer is
  a strategic framework rather than an account-specific finding.
- For product/service questions, use the product knowledge supplied above.
- For healthcare questions, use the healthcare domain knowledge supplied above.
- Keep the answer commercially useful and management-ready.

For portfolio growth questions, prefer this structure:
1. Portfolio signal
2. Growth-candidate / protect-then-expand segmentation
3. Strategic recommendation
4. Execution priorities
5. Data required to operationalize the strategy
6. Expected business value

For account-specific strategic questions, prefer:
1. What the data shows
2. Best business action
3. Why it matters
4. Suggested next steps

Do not mention these instructions in the response.
"""


def ask_gemini(
    question,
    account_context,
    account_name,
    selected_row=None,
    signals=None,
    actions=None,
):
    # --------------------------------------------------------
    # 1. Deterministic renewal questions first
    # --------------------------------------------------------
    if (
        selected_row is not None
        and signals is not None
        and actions is not None
        and classify_standard_question(question) is not None
    ):
        return build_renewal_manager_answer(
            selected_row,
            signals,
            actions,
            question,
        )

    # --------------------------------------------------------
    # 2. Determine whether this is strategic / knowledge-driven
    # --------------------------------------------------------
    strategic_intent = classify_strategic_question(question)

    if strategic_intent:
        prompt = build_strategic_prompt(
            question,
            account_name,
            account_context,
            strategic_intent,
        )

        system_instruction = """
You are a senior Healthcare Commercial Strategy and Renewal Copilot.

You combine:
- structured account data,
- ML renewal signals,
- healthcare domain knowledge,
- product/service knowledge,
- strategic business reasoning.

Evidence hierarchy:
1. Account facts from supplied data
2. ML/model outputs from supplied data
3. Product/service knowledge from the supplied knowledge layer
4. Healthcare domain knowledge from the supplied knowledge layer
5. Strategic recommendations and inference

Never turn a recommendation into a fact.

Never fabricate:
- prospects
- hospital groups
- pipeline
- competitors
- buyer commitments
- revenue opportunities
- contract terms
- stakeholder names
- product capabilities not supplied

When data is missing, say what is missing and then provide a useful
strategic framework using the knowledge layer.

For portfolio growth questions, do not simply refuse because prospect
data is absent. Give a practical strategy and clearly label it as a
strategic recommendation.

Use concise executive language.
"""

    else:
        prompt = f"""
Selected healthcare account:
{account_name}

ACCOUNT AND PORTFOLIO DATA
==========================
{account_context}

USER QUESTION
=============
{question}

Answer as a professional Renewal Manager.

Use dashboard_display_data as the authoritative source for displayed
account values.

You may use the supplied healthcare and product/service knowledge when
it improves interpretation.

Distinguish clearly between:
- Data-supported facts
- Model signals
- Recommendations

Do not invent customer facts or product capabilities.
"""

        system_instruction = """
You are an AI Renewal Manager for a healthcare customer-renewal and
account-management dashboard.

Rules:
1. Use supplied account data as the source of truth.
2. dashboard_display_data is authoritative for displayed values.
3. Never invent customer facts, dates, stakeholders, competitors,
   contract terms, pipeline, or product usage.
4. ML outputs are signals, not causal proof.
5. Product/service knowledge is generic guidance unless account data
   explicitly says the product is adopted.
6. Healthcare-domain knowledge is contextual guidance, not proof of an
   account-specific problem. Treat footprint, facilities, affiliates and beds
   as signals that may justify discovery, never as causal evidence.
7. Strategic recommendations must be labelled as recommendations.
8. If requested data is unavailable, say so clearly.
9. Keep responses concise, commercially relevant, and action-oriented.
"""

    try:
        client = get_gemini_client()
    except Exception as e:
        return (
            "### AI connection error\n\n"
            "The Gemini client could not be initialized.\n\n"
            f"**Error:** `{e}`"
        )

    errors = []

    for model_name in GEMINI_MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "system_instruction": system_instruction,
                        "max_output_tokens": 1600,
                    },
                )

                if response is None:
                    raise RuntimeError("Empty Gemini response.")

                answer = getattr(response, "text", None)

                if answer and answer.strip():
                    return answer.strip()

                raise RuntimeError("Gemini returned no readable text.")

            except Exception as e:
                error_text = str(e)
                errors.append(
                    f"{model_name} attempt {attempt + 1}: {error_text}"
                )

                temporary_error = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text.upper()
                    or "high demand" in error_text.lower()
                    or "temporarily" in error_text.lower()
                    or "timeout" in error_text.lower()
                    or "timed out" in error_text.lower()
                )

                if temporary_error and attempt == 0:
                    time.sleep(2)
                    continue

                break

    return (
        "### AI temporarily unavailable\n\n"
        "Gemini is currently unavailable or experiencing high demand.\n\n"
        "I tried multiple Gemini models automatically. "
        "Please wait a moment and try again."
    )


# ============================================================
# PORTFOLIO METRICS
# ============================================================

total_accounts = len(df)
total_arr = df["Current ARR ($)"].sum()

high_risk = df[df["Renewal Risk"] == "High Risk"]
medium_risk = df[df["Renewal Risk"] == "Medium Risk"]
low_risk = df[df["Renewal Risk"] == "Low Risk"]

high_risk_arr = high_risk["Current ARR ($)"].sum()

high_risk_pct = (
    high_risk_arr / total_arr * 100
    if total_arr > 0
    else 0
)

portfolio_summary = {
    "Total Accounts": int(total_accounts),
    "Total ARR": float(total_arr),
    "High-Risk Accounts": int(len(high_risk)),
    "Medium-Risk Accounts": int(len(medium_risk)),
    "Low-Risk Accounts": int(len(low_risk)),
    "High-Risk ARR": float(high_risk_arr),
    "High-Risk ARR Percentage": float(high_risk_pct),
}

# ============================================================
# PAGE HEADER - MANAGEMENT VIEW
# ============================================================

st.title("🏥 Healthcare Customer & Portfolio Intelligence Copilot")
st.write(
    "AI-powered renewal risk, account intelligence, healthcare context, "
    "and strategic decision support."
)
st.caption("From renewal risk to actionable account strategy.")

st.info(
    "**Public POC Disclosure:** This demonstration uses synthetic commercial, "
    "renewal, adoption and opportunity data, together with public healthcare "
    "provider information. It does not use confidential customer data. AI output "
    "is decision-support guidance and should be validated before business use."
)

# ============================================================
# PORTFOLIO OVERVIEW
# ============================================================

st.divider()
st.subheader("📊 Portfolio Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total Accounts", f"{total_accounts:,}")

with c2:
    st.metric("Total ARR", f"${total_arr / 1_000_000:.1f}M")

with c3:
    st.metric("High-Risk ARR", f"${high_risk_arr / 1_000_000:.1f}M")

with c4:
    st.metric("High-Risk ARR %", f"{high_risk_pct:.1f}%")

r1, r2, r3 = st.columns(3)

with r1:
    st.metric("🔴 High-Risk Accounts", f"{len(high_risk):,}")

with r2:
    st.metric("🟠 Medium-Risk Accounts", f"{len(medium_risk):,}")

with r3:
    st.metric("🟢 Low-Risk Accounts", f"{len(low_risk):,}")

# ============================================================
# PORTFOLIO GROWTH SNAPSHOT
# ============================================================

st.divider()
st.subheader("📈 Portfolio Growth Snapshot")
st.caption(
    "Portfolio-level prioritization using renewal health, engagement, ARR, "
    "operating footprint, and product/service adoption signals."
)

portfolio_growth = build_portfolio_growth_intelligence(df)

g1, g2, g3 = st.columns(3)

with g1:
    st.metric(
        "Growth Candidates",
        f"{portfolio_growth['growth_candidate_count']:,}",
        f"${portfolio_growth['growth_candidate_arr'] / 1_000_000:.2f}M ARR",
    )

with g2:
    st.metric(
        "Protect Then Expand",
        f"{portfolio_growth['protect_then_expand_count']:,}",
        f"${portfolio_growth['protect_then_expand_arr'] / 1_000_000:.2f}M ARR",
    )

with g3:
    st.metric(
        "Renewal-First Accounts",
        f"{portfolio_growth['renewal_first_count']:,}",
        f"${portfolio_growth['renewal_first_arr'] / 1_000_000:.2f}M ARR",
    )

with st.expander("View Portfolio Growth Strategy"):
    render_ai_markdown(format_portfolio_growth_strategy(df))

# ============================================================
# ACCOUNT RISK EXPLORER
# ============================================================

st.divider()
st.subheader("🔎 Account Risk Explorer")

account_options = df["Account Name"].astype(str).tolist()

default_index = 0
if "Hospital Group 321" in account_options:
    default_index = account_options.index("Hospital Group 321")

selected_account = st.selectbox(
    "Select healthcare account",
    account_options,
    index=default_index,
)

selected = df[
    df["Account Name"].astype(str) == selected_account
].iloc[0]

selected_risk = str(selected["Renewal Risk"])
selected_arr = float(selected["Current ARR ($)"])
selected_probability = float(selected["Renewal Probability (%)"])

signals = get_risk_signals(selected)
actions = get_recommended_actions(selected)

# ============================================================
# COMPACT ACCOUNT SUMMARY
# ============================================================

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

with summary_col1:
    st.caption("Account")
    st.markdown(f"### {selected_account}")

with summary_col2:
    st.caption("Current ARR")
    st.markdown(f"### {money(selected_arr)}")

with summary_col3:
    st.caption("Renewal Probability")
    st.markdown(f"### {selected_probability:.1f}%")

with summary_col4:
    st.caption("Renewal Risk")
    st.markdown(f"### {get_risk_color(selected_risk)} {selected_risk}")

st.caption(
    f"{int(float(selected.get('Facilities', 0))):,} Facilities  •  "
    f"{int(float(selected.get('Affiliates', 0))):,} Affiliates  •  "
    f"{int(float(selected.get('Beds', 0))):,} Beds  •  "
    f"Engagement {float(selected.get('Engagement Score', 0)):.1f}"
)

# ============================================================
# ACCOUNT DETAILS - COLLAPSED SUPPORTING DATA
# ============================================================

with st.expander("View Account Details"):
    d1, d2 = st.columns(2)

    with d1:
        st.write(f"**Sales Region:** {selected.get('Sales Region', '')}")
        st.write(f"**Facilities:** {selected.get('Facilities', '')}")
        st.write(f"**Affiliates:** {selected.get('Affiliates', '')}")
        st.write(f"**Beds:** {selected.get('Beds', '')}")
        st.write(f"**Engagement Score:** {selected.get('Engagement Score', '')}")
        st.write(f"**Discount:** {selected.get('Discount (%)', '')}")

    with d2:
        st.write(f"**Contract Status:** {selected.get('Contract Status', '')}")
        st.write(f"**Managed Services:** {selected.get('Has Managed Services', '')}")
        st.write(f"**Data Connect:** {selected.get('Has Data Connect', '')}")
        st.write(f"**Lumere:** {selected.get('Has Lumere', '')}")
        st.write(f"**Enterprise Exchange:** {selected.get('Has Enterprise Exchange', '')}")

    provider_name = selected.get("Public Provider Name", "")
    provider_id = selected.get("Provider ID", "")

    if provider_name or provider_id:
        st.markdown("#### Public Provider Context")
        p1, p2 = st.columns(2)
        with p1:
            st.write(f"**Public Provider:** {provider_name}")
            st.write(f"**Provider ID:** {provider_id}")
        with p2:
            st.write(f"**Hospital Type:** {selected.get('Hospital Type', 'Not available')}")
            st.write(f"**Hospital Ownership:** {selected.get('Hospital Ownership', 'Not available')}")

# ============================================================
# RENEWAL RISK
# ============================================================

st.divider()
st.subheader("🎯 Renewal Risk")

st.info(
    f"**{get_risk_color(selected_risk)} {selected_risk}** — "
    f"{money(selected_arr)} ARR exposure with "
    f"{selected_probability:.1f}% renewal probability."
)

st.markdown("#### Why does this matter?")
for signal in signals:
    st.write(f"• {signal}")

st.caption(
    "These are model and account signals; they indicate where to focus "
    "renewal attention and do not establish a specific customer cause."
)

# ============================================================
# NEXT BEST BUSINESS ACTION
# ============================================================

st.subheader("🎯 Next Best Business Action")

nba = build_next_best_action(selected)

st.success(f"**{nba['primary_action']}** — {nba['business_objective']}")

with st.expander("Why this action?"):
    for item in nba["evidence"]:
        st.write(f"• {item}")
    st.markdown("**Immediate next steps**")
    for step in nba["immediate_steps"]:
        st.write(f"• {step}")
    st.caption(nba["decision_note"])

# ============================================================
# PRODUCT / SERVICE INTELLIGENCE
# ============================================================

st.divider()
st.subheader("🧩 Product / Service Intelligence")
st.caption(
    "Use adoption status as a value-realization and discovery signal; "
    "non-adoption is not automatic evidence of an upsell opportunity."
)

product_intelligence = build_product_service_intelligence(selected)

p_active, p_gap = st.columns(2)

with p_active:
    st.markdown("#### ✅ Currently Active")
    active_names = [item["product_service"] for item in product_intelligence["active"]]
    if active_names:
        for name in active_names:
            st.markdown(f"**{name}**")
    else:
        st.info("No configured products/services are currently marked as active.")

with p_gap:
    st.markdown("#### 🔎 Discovery Areas")
    gap_names = [item["product_service"] for item in product_intelligence["not_adopted"]]
    if gap_names:
        for name in gap_names:
            st.markdown(f"**{name}** — not currently adopted")
    else:
        st.info("No configured adoption gaps are available.")

with st.expander("View Product / Service Intelligence Detail"):
    for item in product_intelligence["active"] + product_intelligence["not_adopted"]:
        st.markdown(f"### {item['product_service']}  •  {item['status']}")
        st.write(f"**Category:** {item['category']}")
        st.write("**Capabilities:** " + ", ".join(item["capabilities"]))
        st.write("**Business value:** " + ", ".join(item["business_value"]))
        st.write("**Generic fit signals:** " + ", ".join(item["generic_fit_signals"]))
        if item["account_specific_signals"]:
            st.write("**Account signals to investigate:** " + " ".join(item["account_specific_signals"]))
        st.write("**Discovery questions:** " + " / ".join(item["discovery_questions"]))
        st.divider()

# ============================================================
# HEALTHCARE CONTEXT
# ============================================================

st.divider()
st.subheader("🏥 Healthcare Context")
st.caption(
    "Healthcare context helps interpret account signals and identify "
    "areas that require customer validation."
)

healthcare_reasoning = build_healthcare_domain_reasoning(selected)

st.info(
    f"**Primary manager focus:** "
    f"{healthcare_reasoning['next_best_action']['primary_action']} — "
    f"{healthcare_reasoning['next_best_action']['business_objective']}"
)

with st.expander("View Healthcare Context"):
    render_ai_markdown(format_healthcare_domain_reasoning(selected))

st.caption(healthcare_reasoning["decision_note"])

# ============================================================
# AI CONTEXT
# ============================================================

account_context = build_account_context(
    selected,
    signals,
    actions,
    portfolio_summary,
)

# ============================================================
# RENEWAL MANAGER COPILOT - HERO INTERACTION
# ============================================================

st.divider()
st.subheader("🤖 Ask the Renewal Manager Copilot")
st.caption(
    "Bring together account facts, ML signals, product/service intelligence, "
    "healthcare context, and business rules into one actionable recommendation."
)

if "ai_selected_account" not in st.session_state:
    st.session_state.ai_selected_account = selected_account

if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []

if st.session_state.ai_selected_account != selected_account:
    st.session_state.ai_selected_account = selected_account
    st.session_state.ai_messages = []

for message in st.session_state.ai_messages:
    with st.chat_message(message["role"]):
        render_ai_markdown(message["content"])

# Lightweight public-demo protection. Limits AI calls per browser session
# and prevents rapid repeated calls. Deterministic dashboard responses remain available.
if "public_ai_question_count" not in st.session_state:
    st.session_state.public_ai_question_count = 0
if "public_last_ai_call_time" not in st.session_state:
    st.session_state.public_last_ai_call_time = 0.0

question = st.chat_input(
    "Ask about this account, renewal, products, healthcare context, or strategy..."
)

if question:
    import time as _time

    now = _time.time()
    elapsed = now - st.session_state.public_last_ai_call_time

    if st.session_state.public_ai_question_count >= PUBLIC_MAX_AI_QUESTIONS:
        st.warning(
            f"This public POC allows up to {PUBLIC_MAX_AI_QUESTIONS} AI questions per session. "
            "Please refresh the session to continue the demonstration."
        )
        st.stop()

    if elapsed < PUBLIC_MIN_SECONDS_BETWEEN_AI_CALLS:
        st.warning("Please wait a moment before sending another AI question.")
        st.stop()

    st.session_state.public_ai_question_count += 1
    st.session_state.public_last_ai_call_time = now
    st.session_state.ai_messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing account, portfolio and business context..."):
            answer = ask_gemini(
                question=question,
                account_context=account_context,
                account_name=selected_account,
                selected_row=selected,
                signals=signals,
                actions=actions,
            )
        render_ai_markdown(answer)

    st.session_state.ai_messages.append({"role": "assistant", "content": answer})

# ============================================================
# SUGGESTED QUESTIONS
# ============================================================
# Keep these visible even after the user asks a question so the
# remaining prompts are always available for copying into the Copilot.

st.markdown("#### Suggested questions")
st.caption("Copy any question below and paste it into the Copilot input. These options remain available after every response.")

q1, q2 = st.columns(2)

with q1:
    st.markdown("• Why is this account at risk?")
    st.markdown("• What is the best business action for this account?")
    st.markdown("• What products or services are not being used?")

with q2:
    st.markdown("• How can we increase adoption?")
    st.markdown("• Give me the overall strategic recommendation for this account.")
    st.markdown("• Considering the healthcare context, what should the account manager focus on?")

# ============================================================
# SUPPORTING DATA
# ============================================================

with st.expander("View Complete Account Record"):
    st.dataframe(
        pd.DataFrame([selected]),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Healthcare Customer & Portfolio Intelligence Copilot • Account facts are "
    "grounded in available data. ML outputs are model signals. Strategic "
    "guidance is explicitly treated as recommendation rather than customer fact."
)
st.caption(
    "Public POC: commercial/renewal/adoption data is synthetic; healthcare provider "
    "context is based on public data. Validate all AI-generated recommendations before use."
)

# Healthcare Intelligence Knowledge Layer
# Client-agnostic foundation for the Healthcare Renewal AI Assistant.

HEALTHCARE_DOMAIN_KNOWLEDGE = {
    "industry_context": [
        "Healthcare organizations operate across provider, supplier, procurement, clinical, financial, and administrative ecosystems.",
        "Large hospital groups and health systems may manage multiple facilities, affiliates, service lines, suppliers, contracts, and stakeholders.",
        "Useful strategic dimensions can include facility footprint, beds, affiliates, geography, procurement complexity, technology adoption, stakeholder engagement, data availability, and demonstrated business value.",
    ],
    "provider_context": [
        "Healthcare providers can include hospitals, health systems, hospital groups, integrated delivery networks, and specialty organizations.",
        "Provider-oriented analysis can consider facility footprint, beds, affiliates, geography, operational complexity, technology adoption, procurement maturity, and stakeholder engagement.",
    ],
    "procurement_context": [
        "Healthcare procurement and supply-chain functions may involve sourcing, contract management, supplier relationships, purchasing, utilization, product standardization, data integration, and value analysis.",
        "Potential objectives include improving visibility, contract compliance, sourcing effectiveness, purchasing standardization, data connectivity, and operational efficiency.",
    ],
    "renewal_context": [
        "Renewal decisions should combine retention with value demonstration.",
        "Useful considerations include account health, engagement, demonstrated value, product/service adoption, unresolved issues, stakeholder alignment, contract timing, customer priorities, and expansion whitespace.",
    ],
}

PRODUCT_SERVICE_KNOWLEDGE = {
    "Managed Services": {
        "category": "Service",
        "capabilities": ["ongoing operational support", "process management", "specialized operational expertise"],
        "business_value": ["reduce operational burden", "improve process consistency", "provide additional operational capacity"],
        "fit_signals": ["high operational complexity", "limited internal capacity", "need for standardization"],
    },
    "Data Connect": {
        "category": "Data / Integration",
        "capabilities": ["data connectivity", "data integration", "connected information access"],
        "business_value": ["improve data visibility", "reduce manual data movement", "support analytics and decision-making"],
        "fit_signals": ["multiple data sources", "integration requirements", "need for better data visibility"],
    },
    "Lumere": {
        "category": "Healthcare Analytics / Intelligence",
        "capabilities": ["healthcare analytics capabilities", "insight generation", "support for data-driven decisions"],
        "business_value": ["support evidence-based decisions", "identify opportunities through data", "strengthen analytical capabilities"],
        "fit_signals": ["need for analytics", "data-driven decision-making", "opportunity identification"],
    },
    "Enterprise Exchange": {
        "category": "Network / Commerce / Connectivity",
        "capabilities": ["enterprise connectivity", "digital business interaction", "information exchange"],
        "business_value": ["improve connectivity", "reduce manual interaction", "support scalable enterprise processes"],
        "fit_signals": ["large organizational footprint", "many business relationships", "need for scalable connectivity"],
    },
}

STRATEGIC_PLAYBOOKS = {
    "renewal_retention": [
        "Prioritize accounts using risk, ARR exposure, and account health.",
        "Understand the evidence behind renewal risk.",
        "Address adoption or engagement gaps.",
        "Align stakeholders around demonstrated business value.",
        "Create focused recovery or renewal plans for higher-risk accounts.",
    ],
    "adoption_growth": [
        "Identify products or services not currently adopted.",
        "Assess relevance to the organization's needs.",
        "Connect capabilities to concrete business objectives.",
        "Validate opportunities through stakeholder engagement.",
        "Prioritize adoption opportunities using business impact and fit.",
    ],
    "portfolio_growth": [
        "Segment organizations by size, footprint, geography, and strategic fit.",
        "Prioritize organizations where solution capabilities address clear business needs.",
        "Assess market and account whitespace.",
        "Use engagement and fit signals to prioritize outreach.",
        "Develop differentiated value propositions for priority segments.",
        "Track opportunity stage, competitive context, and next-best action.",
    ],
    "account_expansion": [
        "Identify product and service whitespace.",
        "Assess expansion across facilities, affiliates, or business units.",
        "Link expansion opportunities to documented customer needs.",
        "Prioritize opportunities using value, fit, and engagement.",
        "Coordinate expansion with the broader account strategy.",
    ],
}

STRATEGIC_INTENT_KEYWORDS = {
    "portfolio_growth": [
        "grow portfolio", "grow the portfolio", "add hospital", "add hospitals",
        "new hospital", "new hospitals", "new hospital group", "new hospital groups",
        "new customers", "new accounts", "acquisition", "prospecting", "new logo", "new logos",
    ],
    "account_expansion": [
        "expand this account", "account expansion", "cross sell", "cross-sell",
        "upsell", "whitespace", "expansion opportunity", "increase account value",
    ],
    "adoption_growth": [
        "increase adoption", "improve adoption", "adoption opportunity",
        "product adoption", "service adoption",
    ],
    "renewal_strategy": [
        "renewal strategy", "retention strategy", "best action for renewal",
        "best business action", "how should we approach",
    ],
}

def build_knowledge_context():
    return {
        "healthcare_domain": HEALTHCARE_DOMAIN_KNOWLEDGE,
        "product_service_knowledge": PRODUCT_SERVICE_KNOWLEDGE,
        "strategic_playbooks": STRATEGIC_PLAYBOOKS,
    }

def knowledge_as_text():
    context = build_knowledge_context()
    return str(context)

# Healthcare Customer & Portfolio Intelligence Copilot

An AI-powered healthcare customer intelligence POC designed to bring together renewal risk, account intelligence, product/service adoption, healthcare context, and strategic decision support in one experience.

## 🚀 Live Demo

[Open the Healthcare Customer & Portfolio Intelligence Copilot](https://healthcare-customer-portfolio-copilot-kbyrb775a2lh8kykjqbi9w.streamlit.app)

## What the Copilot Does

The solution combines multiple intelligence layers:

- Renewal risk prediction
- Account-level risk exploration
- Next Best Business Action
- Product & Service Intelligence
- Portfolio Growth Strategy
- Healthcare Domain Context
- GenAI-powered strategic synthesis

## Solution Flow

Data → ML Risk Signals → Business Rules → Product Intelligence → Healthcare Context → GenAI Copilot

## Key Capabilities

### 1. Renewal Risk Intelligence
Uses ML-based renewal probability and risk segmentation to identify accounts requiring focused attention.

### 2. Next Best Business Action
Uses deterministic business rules to translate account and model signals into an actionable manager recommendation.

### 3. Product & Service Intelligence
Highlights current adoption and areas for discovery while explicitly treating non-adoption as a discovery signal rather than automatic upsell evidence.

### 4. Portfolio Growth Intelligence
Provides portfolio-level prioritization using renewal health, engagement, ARR, operating footprint, and product/service adoption signals.

### 5. Healthcare Context
Uses generic healthcare-domain knowledge to frame relevant discovery themes and manager questions.

### 6. GenAI Strategic Synthesis
Brings the intelligence layers together into a grounded, management-ready account recommendation.

## Data & POC Disclosure

This is a demonstration POC.

Commercial, renewal, adoption, and opportunity attributes are synthetically generated for demonstration purposes. Public healthcare provider information is used to create realistic provider context.

The solution does not use confidential customer data.

AI-generated outputs are decision-support guidance and should be validated before business use.

## ML Approach

The POC demonstrates a Logistic Regression-based renewal probability model using account-level signals including:

- Current ARR
- Facilities
- Affiliates
- Beds
- Engagement Score
- Discount

A Random Forest model is also used for comparison and feature-importance analysis.

The ML outputs are treated as signals rather than causal explanations.

## Technology

- Python
- Streamlit
- Pandas
- Scikit-learn
- Google Gemini
- Healthcare domain knowledge layer
- Deterministic business rules

## Important Guardrails

The Copilot does not claim:

- confirmed customer problems
- confirmed buying intent
- confirmed expansion opportunities
- live pipeline
- competitive intelligence
- causal explanations for renewal risk

Recommendations are intended as account-management decision support.

## Future Evolution

Potential next steps include:

- Validation on historical production renewal data
- Integration with CRM and customer activity data
- Production-grade monitoring
- Outcome tracking
- Feedback-driven recommendation improvement
- Enterprise deployment controls

---

## Public POC

**From data to prediction, from prediction to action, and from action to intelligent conversation.**

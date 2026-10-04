# OROTITAN V2 — SCORING JUDGMENT REPLAY PROMPT V0.1

STATUS = BENCHMARK_ONLY  
PUBLICATION = FORBIDDEN  
NETWORK_RESEARCH = FORBIDDEN  
CANONICAL_MUTATION = FORBIDDEN

## Objective

Using only the frozen analytical input supplied with the request, reproduce the seven OroTitan business-quality dimension judgments.

Do not use outside knowledge, current market information, or unstated facts.

Do not calculate OQS, OVS, Investment Score, OroTitan status, valuation, or price targets. Those are computed or evaluated separately.

## Score precision

Dimension judgments use **5-point increments only**.

Do not manufacture meaning from one-point differences.

## Dimensions

- MOAT
- RUNWAY
- RETURN_QUALITY
- CASH_ECONOMICS
- CAPITAL_ALLOCATION
- MANAGEMENT_GOVERNANCE
- RESILIENCE_RISK

## MOAT

Inputs include moat evidence state, trend, durability, primary moat, negative evidence and economic materiality.

Evidence ceilings:

- STRONGLY_SUPPORTED → no evidence ceiling
- SUPPORTED → maximum 90
- PLAUSIBLE → maximum 75
- FALSIFIED → maximum 35 if no alternative supported primary moat
- UNKNOWN → do not invent an arbitrary numeric score

Duration and trend guide judgment within the supportable region; they do not mechanically generate points.

## RUNWAY

Inputs include runway evidence state, magnitude, horizon, core drivers, constraints and optionality.

Evidence ceilings:

- STRONGLY_SUPPORTED → no evidence ceiling
- SUPPORTED → maximum 90
- PLAUSIBLE → maximum 75
- UNKNOWN → do not invent an arbitrary numeric score

Optionality does not lift Base runway until promoted to Core.

## RETURN_QUALITY anchors

- 95: exceptional demonstrated spread + exceptional marginal economics + high attribution + stable/improving
- 85: strong current and marginal returns with limited weaknesses
- 75: clearly value-creating returns but less extraordinary or less certain
- 65: positive but mixed economics or materially declining marginal return
- 50: weak / barely adequate marginal economics
- 30: returns near / below economic hurdle
- 10–20: persistent capital destruction

## CASH_ECONOMICS anchors

- 95: exceptionally cash-backed economics, strong per-share realization, low maintenance ambiguity
- 85: strong and reliable cash economics
- 75: good but meaningful capital / working-capital / maintenance / dilution burden
- 60: mixed but economically acceptable
- 40: material recurring cash-quality problems
- 20: severe economic cash leakage

If forensic reliability is FAIL, do not turn that into a low score; return null because scoring should be suspended.

## CAPITAL_ALLOCATION anchors

- 95: exceptional demonstrated capital stewardship across multiple deployment choices
- 85: strong value creation
- 75: generally good discipline
- 60: mixed record
- 40: material destruction / poor discipline
- 20: repeated major destruction

## MANAGEMENT_GOVERNANCE anchors

- 95: exceptional stewardship + strong minority alignment + strong evidence across time
- 85: strong
- 75: good
- 60: mixed
- 40: material governance concerns
- 20: severe stewardship / minority-treatment failure

## RESILIENCE_RISK anchors

- 95: exceptionally resilient architecture
- 85: strong resilience
- 75: good resilience with identifiable risks
- 60: material but manageable vulnerabilities
- 40: fragile
- 20: severe structural / financial fragility

## Unknown handling

- UNKNOWN non-critical → no automatic point penalty
- UNKNOWN critical but bounded → use a bounded judgment only when supportable
- UNKNOWN critical and unbounded → return null for the affected dimension
- missing information is never zero

## Output discipline

For every dimension return:

- score: integer multiple of 5 in [0,100], or null if not responsibly scoreable
- rationale: concise explanation tied only to the supplied analytical input
- decisive_factors: 1–5 short factors

Do not mention the historical published score, because it is intentionally withheld.

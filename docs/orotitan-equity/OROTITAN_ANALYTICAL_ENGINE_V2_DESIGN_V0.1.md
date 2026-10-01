# OROTITAN ANALYTICAL ENGINE V2 — DESIGN SPEC V0.1

Status: DESIGN DRAFT  
Program: OroTitan Equity Research vNext  
Post-C7 direction: HUMAN-CENTERED / AI-ASSISTED / PROVIDER-AGNOSTIC  
Target language: French-first UI, canonical machine semantics preserved in English  
Production mutation authority: NONE  
Routing freeze authority: NONE

---

# 1. PURPOSE

OroTitan Analytical Engine V2 is the post-C7 analytical architecture for OroTitan Equity Research.

Its objective is not to make the current local LLM autonomous.

Its objective is to make the overall research system materially better at:

1. understanding the true economics of a company;
2. adapting the analysis to sector, business model and economic archetype;
3. understanding cyclicality and normalization;
4. understanding technology, disruption and technical barriers;
5. separating evidence from inference;
6. making causal reasoning explicit;
7. detecting weak links, contradictions and failure modes;
8. producing more robust valuation inputs;
9. preventing research loops and redundant analysis;
10. allowing human / external-LLM assistance without compromising traceability;
11. giving the user a fluid, premium and intelligible research workflow.

The design principle is:

> MAXIMUM ANALYTICAL DEPTH BEHIND THE SYSTEM  
> MINIMUM COGNITIVE FRICTION IN FRONT OF THE USER

---

# 2. FOUNDATIONAL BOUNDARY

V2 does not replace the frozen OroTitan methodology.

The following remain canonical:

- Discovery semantics;
- Research Stage Contract;
- Deep Dive Stage Contract;
- Integration Stage Contract;
- Analysis Standard V1;
- frozen evidence / conflict / calculation semantics;
- Certification;
- scoring formulas;
- OroTitan terminal gate;
- Dossier Readiness;
- price-ladder semantics;
- run / stage artifact authority.

V2 adds execution intelligence around those contracts.

New analytical engines must map into canonical outputs.

Example:

CYCLICALITY ENGINE
→ informs BUSINESS MODEL / RUNWAY / RETURN QUALITY / RISK / VALUATION

TECHNOLOGY ENGINE
→ informs MOAT / RUNWAY / RISK / OUTSIDE VIEW / VALUATION

SECTOR INTELLIGENCE
→ selects sector-valid methods and evidence requirements

CAUSAL GRAPH
→ strengthens cross-block reconciliation and evidence traceability

No parallel scoring ontology is created.

---

# 3. PRODUCT THESIS

OroTitan V2 should behave like an integrated equity-research workstation.

The user should not have to manually reconstruct:
- what has already been researched;
- which questions are still open;
- why a block is blocked;
- which source was already searched;
- which evidence supports which claim;
- which analytical method applies to the sector;
- what changed since the prior canonical snapshot;
- what the next action is.

The system must persist and expose:

CURRENT_STATE
NEXT_ACTION
WHY
BLOCKER
EVIDENCE
OPEN_QUESTIONS
INVALIDATION_TRIGGERS

Every stage should be resumable.

---

# 4. ANALYTICAL QUALITY TARGET

An OroTitan V2 analysis is not considered high quality merely because it is long.

It should demonstrate, where material:

- business-model comprehension;
- industry structure;
- sector-specific economics;
- cyclicality;
- technology;
- customer behavior;
- competitive behavior;
- capital intensity;
- reinvestment requirements;
- unit/cohort economics;
- return on incremental capital;
- cash conversion;
- accounting quality;
- capital allocation;
- governance;
- base rates;
- structural risks;
- disruption;
- valuation;
- market-implied expectations;
- uncertainty;
- disconfirming evidence.

The system should be able to distinguish:

GOOD COMPANY
from
GOOD BUSINESS ECONOMICS

and:

GOOD BUSINESS
from
GOOD INVESTMENT AT CURRENT PRICE.

---

# 5. COMPANY ECONOMIC DNA

Every Research run should construct a persistent COMPANY_ECONOMIC_DNA object before full Deep Dive.

Minimum representation:

COMPANY_ECONOMIC_DNA
- PRIMARY_SECTOR
- SUBSECTORS[]
- BUSINESS_MODEL_ARCHETYPES[]
- REVENUE_MODEL[]
- PRICING_MODEL[]
- DEMAND_DRIVERS[]
- COST_STRUCTURE
- CAPITAL_INTENSITY
- WORKING_CAPITAL_PROFILE
- FIXED_COST_INTENSITY
- REINVESTMENT_MODEL
- ORGANIC_VS_ACQUIRED_GROWTH
- CYCLICALITY_PROFILE
- TECHNOLOGY_EXPOSURE
- REGULATORY_EXPOSURE
- GEOGRAPHIC_EXPOSURE
- CUSTOMER_CONCENTRATION
- SUPPLIER_CONCENTRATION
- DISTRIBUTION_MODEL
- NETWORK_EFFECT_EXPOSURE
- INSTALLED_BASE_EXPOSURE
- INTANGIBLE_INTENSITY
- M&A_DEPENDENCE
- COMMODITY_EXPOSURE
- FINANCIAL_LEVERAGE_MODEL
- KEY_ECONOMIC_BOTTLENECKS[]
- KEY_VALUE_DRIVERS[]
- KEY_FAILURE_MODES[]

The DNA is not a score.

It determines which analytical overlays activate.

---

# 6. COMPOSABLE ANALYTICAL OVERLAYS

Sector adaptation is composable.

A company can receive multiple overlays simultaneously.

Examples:

SOFTWARE + SERIAL_ACQUIRER

MARKETPLACE + SOFTWARE

INSTALLED_BASE + MEDTECH

INDUSTRIAL + CYCLICAL

SEMICONDUCTOR_EQUIPMENT + TECHNOLOGY_BOTTLENECK + CYCLICAL

LUXURY + BRAND + GEOGRAPHIC_CHINA_EXPOSURE

BANK + CREDIT_CYCLE + REGULATED

Overlays modify:
- evidence requirements;
- relevant KPIs;
- normalization logic;
- failure modes;
- valuation methods;
- Red Team questions.

Overlays do not change the frozen meaning of canonical blocks.

---

# 7. SECTOR INTELLIGENCE LIBRARY

Create a versioned Sector Intelligence Library.

Initial families:

- SOFTWARE
- MARKETPLACE
- PAYMENTS
- DATA_INFORMATION_SERVICES
- SEMICONDUCTORS
- SEMICONDUCTOR_EQUIPMENT
- INDUSTRIAL_AUTOMATION
- INDUSTRIAL_DISTRIBUTION
- AEROSPACE
- DEFENSE
- MEDTECH
- PHARMA
- LIFE_SCIENCE_TOOLS
- CHEMICALS
- COMMODITIES
- UTILITIES
- BANK
- INSURANCE
- ASSET_MANAGER
- EXCHANGE
- SERIAL_ACQUIRER
- LUXURY
- CONSUMER_BRAND
- RETAIL
- LOGISTICS
- TRANSPORT
- TELECOM
- REAL_ESTATE
- CONSTRUCTION_MATERIALS
- PROFESSIONAL_SERVICES

Each sector profile should contain:

SECTOR_PROFILE
- ECONOMIC_MODEL
- REVENUE_DRIVERS
- COST_DRIVERS
- CAPITAL_REQUIREMENTS
- WORKING_CAPITAL_MECHANICS
- KEY_KPIS[]
- CYCLE_DRIVERS[]
- COMMON_MOAT_MECHANISMS[]
- COMMON_FALSE_MOAT_SIGNALS[]
- TECHNOLOGY_QUESTIONS[]
- COMMON_ACCOUNTING_TRAPS[]
- REGULATORY_STRUCTURE
- INDUSTRY_STRUCTURE
- CAPACITY_DYNAMICS
- FAILURE_MODES[]
- BASE_RATE_REFERENCE_CLASSES[]
- VALUATION_METHODS[]
- NORMALIZATION_RULES[]
- RED_TEAM_QUESTIONS[]
- REQUIRED_EXTERNAL_EVIDENCE_TYPES[]

Sector knowledge must be versioned and auditable.

---

# 8. CYCLICALITY ENGINE

Cyclicality becomes a first-class engine.

The system should classify the relevant cycle mechanisms rather than merely tag a company CYCLICAL.

CYCLE_TYPE may include:

- DEMAND_CYCLE
- INVENTORY_CYCLE
- CAPACITY_CYCLE
- COMMODITY_PRICE_CYCLE
- CREDIT_CYCLE
- HOUSING_CYCLE
- ADVERTISING_CYCLE
- SEMICONDUCTOR_CYCLE
- FREIGHT_CYCLE
- INSURANCE_CYCLE
- INDUSTRIAL_EQUIPMENT_CYCLE
- CAPITAL_MARKETS_CYCLE
- REGULATORY_CYCLE
- COMPANY_SPECIFIC_REPLACEMENT_CYCLE

For each material cycle:

CYCLE_ANALYSIS
- CYCLE_TYPE
- PRIMARY_DRIVERS[]
- LEADING_INDICATORS[]
- LAGGING_INDICATORS[]
- CURRENT_REGIME
- HISTORICAL_AMPLITUDE
- HISTORICAL_DURATION
- CAPACITY_STATE
- INVENTORY_STATE
- UTILIZATION_STATE
- ORDER_BOOK_STATE
- PRICING_STATE
- VOLUME_STATE
- MARGIN_STATE
- WORKING_CAPITAL_STATE
- CAPITAL_SPENDING_STATE
- SUPPLY_RESPONSE
- DEMAND_RESPONSE
- NORMALIZATION_RANGE
- PEAK_RISK
- TROUGH_UPSIDE
- EVIDENCE_IDS[]
- CONFIDENCE
- INVALIDATION_TRIGGERS[]

CURRENT_REGIME:
- PEAK
- LATE_CYCLE
- MID_CYCLE
- DOWNTURN
- TROUGH
- EARLY_RECOVERY
- EXPANSION
- UNKNOWN

Core rule:

REPORTED_EARNINGS
≠
NORMALIZED_EARNINGS

when material cyclicality exists.

The cycle engine feeds valuation and expected return.

---

# 9. TECHNOLOGY ENGINE

Technology analysis becomes a first-class engine when technology is economically material.

TECHNOLOGY_MAP
- CORE_TECHNOLOGIES[]
- PRODUCT_ARCHITECTURE
- TECHNICAL_BOTTLENECKS[]
- PROPRIETARY_ASSETS[]
- IP_POSITION
- STANDARDS_DEPENDENCE
- INTEROPERABILITY
- WORKFLOW_EMBEDDING
- TECHNICAL_SWITCHING_COSTS
- DATA_ADVANTAGES
- R&D_MODEL
- R&D_INTENSITY
- R&D_PRODUCTIVITY_EVIDENCE
- ENGINEERING_TALENT_DEPENDENCE
- SUPPLIER_TECH_DEPENDENCIES[]
- CUSTOMER_TECH_DEPENDENCIES[]
- CURRENT_GENERATION
- NEXT_GENERATION
- COMPETING_TECHNOLOGIES[]
- SUBSTITUTE_TECHNOLOGIES[]
- OPEN_SOURCE_THREAT
- COMMODITIZATION_RISK
- OBSOLESCENCE_RISK
- TECHNICAL_REPLICATION_DIFFICULTY
- TIME_TO_REPLICATE
- CAPEX_TO_REPLICATE
- REGULATORY_OR_CERTIFICATION_BARRIERS
- TECHNOLOGY_ROADMAP_CONFIDENCE
- INVALIDATION_TRIGGERS[]

Technology claims must connect to economic consequences.

Example:

TECHNICAL_BOTTLENECK
→ CUSTOMER_DEPENDENCE
→ LOW_SUBSTITUTABILITY
→ PRICING_POWER
→ MARGIN / RETURNS

Do not infer moat merely from high R&D spending or technical complexity.

---

# 10. INDUSTRY STRUCTURE ENGINE

Analyze industry economics explicitly.

INDUSTRY_STRUCTURE
- MARKET_STRUCTURE
- COMPETITOR_SET[]
- MARKET_SHARE_DISTRIBUTION
- ENTRY_RATE
- EXIT_RATE
- CAPACITY_DISCIPLINE
- PRICING_DISCIPLINE
- CUSTOMER_BARGAINING_POWER
- SUPPLIER_BARGAINING_POWER
- DISTRIBUTOR_POWER
- REGULATORY_BARRIERS
- SWITCHING_FRICTION
- MULTIHOMING
- VERTICAL_INTEGRATION
- CONSOLIDATION_TREND
- DISRUPTION_VECTOR[]
- PROFIT_POOL_LOCATION
- VALUE_CHAIN_POSITION
- HISTORICAL_RETURN_DISTRIBUTION

This is not a generic Porter paragraph.

It must explain how industry structure transmits into company economics.

---

# 11. CAUSAL GRAPH ENGINE

Material claims should be represented through explicit causal chains.

Example:

INSTALLED_BASE
→ REPLACEMENT_FRICTION
→ RETENTION
→ PRICING_POWER
→ MARGIN_STABILITY
→ FCF

Each causal node / edge stores:

CAUSAL_NODE
- NODE_ID
- CLAIM
- TYPE
- EVIDENCE_IDS[]
- CONTRADICTING_EVIDENCE_IDS[]
- CONFIDENCE

CAUSAL_EDGE
- FROM_NODE
- TO_NODE
- MECHANISM
- SUPPORTING_EVIDENCE_IDS[]
- COUNTEREVIDENCE_IDS[]
- STATUS
- CONFIDENCE

Allowed edge status:
- SUPPORTED
- MIXED
- NOT_SUPPORTED
- LOW_CONFIDENCE
- NOT_ASSESSABLE

A conclusion with an unsupported material causal edge cannot be promoted as strongly proven.

---

# 12. MOAT PROOF V2

MOAT analysis should follow:

MOAT_MECHANISM
→ INDEPENDENT_EVIDENCE
→ CUSTOMER_BEHAVIOR
→ COMPETITOR_BEHAVIOR
→ ECONOMIC_CONSEQUENCE
→ REPLICATION_DIFFICULTY
→ SUBSTITUTION_RISK
→ TREND
→ DURABILITY

For each moat mechanism:

- claim;
- mechanism;
- issuer evidence;
- independent evidence;
- customer evidence;
- competitor evidence;
- behavioral evidence;
- economic consequence;
- evidence against;
- alternative explanation;
- replication path;
- substitution path;
- durability horizon;
- moat trend.

False positives to detect:

- scale with no cost advantage;
- brand awareness with no pricing power;
- regulation that benefits all competitors equally;
- mandatory demand with no differential barrier;
- high switching claims without observed retention/behavior;
- high market share caused by temporary cycle position;
- high ROIC caused by denominator artifacts;
- technical complexity with no economic consequence.

---

# 13. RUNWAY ENGINE V2

Runway should decompose future growth into causal sources.

RUNWAY_BRIDGE
- CURRENT_REVENUE
- PRICE
- VOLUME
- MIX
- SHARE_GAIN
- GEOGRAPHIC_EXPANSION
- PRODUCT_EXPANSION
- CROSS_SELL
- INSTALLED_BASE_GROWTH
- MARKET_GROWTH
- ACQUISITIONS
- CYCLICAL_RECOVERY
- OPTIONALITY

Every major growth source should classify:

- CORE
- OPTIONALITY
- ACQUIRED
- CYCLICAL
- PRICE
- VOLUME
- SHARE
- NEW_MARKET

Runway analysis must distinguish:

TAM
≠
SERVICEABLE MARKET
≠
REALISTIC CAPTURE POOL

and test:
- penetration;
- saturation;
- capacity;
- capital needs;
- sales capacity;
- organizational complexity;
- competition;
- regulation;
- cannibalization;
- marginal economics.

---

# 14. RETURN QUALITY ENGINE V2

Return analysis must adapt to business model.

Core outputs:

- STANDARD_ROIC
- ALL_IN_ROIC where M&A material
- ROIIC
- MARGINAL_RETURN
- COHORT_RETURN where applicable
- GROWTH_SPEND_ECONOMICS
- INTERPRETABILITY
- ATTRIBUTABILITY

Do not reward mathematically explosive ROIC created by near-zero invested capital.

For asset-light businesses use, where appropriate:
- incremental margin;
- CAC / payback;
- cohort economics;
- growth-spend returns;
- unit economics;
- cash conversion.

For serial acquirers:
- acquisition deployment;
- purchase multiple;
- post-acquisition economics;
- cohort returns;
- organic / acquired split;
- leverage;
- share issuance.

---

# 15. FCF / FORENSIC ENGINE V2

The engine should systematically inspect:

- working capital;
- receivables;
- inventory;
- payables;
- factoring;
- supplier finance;
- provisions;
- capitalization;
- R&D capitalization;
- software capitalization;
- restructuring;
- acquisition-related adjustments;
- SBC;
- dilution;
- leases;
- cash taxes;
- maintenance capex;
- deferred maintenance;
- goodwill;
- impairments;
- one-offs;
- pension;
- related parties;
- discontinued operations.

Output bridge:

REPORTED_CASH_FLOW
→ ACCOUNTING_ADJUSTMENTS
→ ECONOMIC_ADJUSTMENTS
→ STANDARDIZED_FCF
→ OWNER_EARNINGS_RANGE

Each adjustment requires:
- rationale;
- source;
- recurring/non-recurring;
- cash/non-cash;
- amount/range;
- uncertainty.

---

# 16. CAPITAL ALLOCATION ENGINE V2

Reconstruct multi-year capital deployment.

CAPITAL_SOURCES
- OPERATING_CASH
- DEBT
- EQUITY_ISSUANCE
- ASSET_SALES

CAPITAL_USES
- ORGANIC_REINVESTMENT
- R&D
- CAPEX
- M&A
- DIVIDENDS
- BUYBACKS
- DEBT_REPAYMENT
- CASH_ACCUMULATION

For each material deployment:
- amount;
- timing;
- rationale;
- funding;
- realized outcome;
- per-share consequence;
- estimated return;
- alternative use.

Question:

> What happened to each retained euro/dollar of shareholder capital?

---

# 17. MANAGEMENT / GOVERNANCE ENGINE V2

Separate narrative from stewardship.

Assess:
- ownership;
- incentives;
- compensation metrics;
- board;
- succession;
- capital-allocation accountability;
- disclosure quality;
- related parties;
- minority treatment;
- dilution;
- target setting;
- guidance behavior;
- claims vs realized outcomes.

Founder status or confidence is not evidence of quality.

---

# 18. OUTSIDE VIEW / BASE-RATE ENGINE V2

Use:

REFERENCE CLASS
→ PRIOR
→ COMPANY-SPECIFIC EVIDENCE
→ UPDATED JUDGMENT

Reference classes may include:
- growth persistence;
- margin persistence;
- acquisition-return persistence;
- credit-cycle loss normalization;
- underwriting-cycle returns;
- industrial capacity returns;
- R&D productivity;
- market-share durability;
- valuation mean reversion.

The engine should make explicit:
- prior;
- sample quality;
- comparability;
- differences;
- posterior judgment.

No base rate automatically overrides company-specific evidence.

---

# 19. VARIANT PERCEPTION ENGINE

Explicitly represent:

MARKET_EXPECTATION
vs
OROTITAN_FUNDAMENTAL_VIEW

Fields:

VARIANT_PERCEPTION
- MARKET_EXPECTATION
- OROTITAN_EXPECTATION
- DIFFERENCE
- WHY_MARKET_MAY_BELIEVE_IT
- WHY_OROTITAN_DIFFERS
- SUPPORTING_EVIDENCE_IDS[]
- DISCONFIRMING_EVIDENCE_IDS[]
- WHAT_WOULD_PROVE_OROTITAN_WRONG
- TIME_HORIZON
- CATALYST_NOT_REQUIRED
- CONFIDENCE

This should feed valuation, not become a narrative opinion module.

---

# 20. RISK / RESILIENCE ENGINE V2

Preserve frozen risk taxonomy:

- TEMPORARY
- CYCLICAL
- STRUCTURAL
- FINANCIAL
- DISRUPTION
- REGULATORY
- GOVERNANCE
- CONCENTRATION
- EXECUTION

Each risk needs:

EVIDENCE
TRANSMISSION_MECHANISM
SEVERITY
DETECTABILITY
REVERSIBILITY
THESIS_IMPACT
EARLY_INDICATOR
MITIGATION
INVALIDATION_TRIGGER

Risk must be causal.

Avoid generic risk lists.

---

# 21. RED TEAM V2

Mandatory perspectives:

- BEAR_CASE
- SHORT_SELLER_CASE
- COMPETITOR_CASE
- CUSTOMER_CASE
- TECHNOLOGIST_CASE
- REGULATOR_CASE
- ACCOUNTING_FORENSIC_CASE
- CAPITAL_ALLOCATION_CASE
- CYCLE_PEAK_CASE
- RUNWAY_FAILURE_CASE
- ROIIC_FAILURE_CASE
- REVERSE_VALUATION_CASE
- PRE_MORTEM

The Red Team may reopen:
- moat;
- runway;
- return quality;
- forensic reliability;
- risk;
- valuation.

A reopened material issue blocks normal certification until resolved or validly classified.

---

# 22. VALUATION ENGINE V2

Valuation must consume normalized economics.

Questions remain distinct:

INTRINSIC_VALUE
≠
EXPECTED_SHAREHOLDER_RETURN
≠
MARKET_IMPLIED_EXPECTATIONS

Valuation model selection should adapt to:
- cyclicality;
- capital intensity;
- growth duration;
- financial business model;
- acquisition dependence;
- terminal-value reliability;
- unit/cohort economics.

Outputs should include:

- PRIMARY_VALUATION_METHOD
- CROSS_CHECKS[]
- NORMALIZED_BASE
- BULL / BASE / BEAR
- NO_MULTIPLE_EXPANSION_RETURN
- MATURE_NORMALIZATION_RETURN
- REVERSE_DCF / IMPLIED_EXPECTATIONS
- SENSITIVITIES
- TERMINAL_DEPENDENCE
- VALUATION_RELIABILITY
- MARGIN_OF_SAFETY

For cyclicals, valuation must use normalized rather than peak/trough economics unless explicitly scenario-based.

---

# 23. EVIDENCE ENGINE V2

Every material conclusion should map to:
- supporting evidence;
- contradicting evidence;
- calculations;
- assumptions;
- source quality;
- freshness;
- independence.

Evidence roles:

FACT
MANAGEMENT_CLAIM
CUSTOMER_EVIDENCE
COMPETITOR_EVIDENCE
INDUSTRY_EVIDENCE
REGULATORY_EVIDENCE
ACADEMIC_TECHNICAL_EVIDENCE
CALCULATION
ASSUMPTION
INFERENCE

The UI must never visually present an inference as if it were a fact.

---

# 24. OPEN QUESTIONS ENGINE

Each dossier maintains explicit unresolved questions.

OPEN_QUESTION
- QUESTION_ID
- BLOCK
- QUESTION
- WHY_MATERIAL
- CURRENT_EVIDENCE
- MISSING_EVIDENCE
- SEARCHES_ALREADY_PERFORMED[]
- BEST_NEXT_SOURCE
- STATUS
- IMPACT_IF_UNRESOLVED

Statuses:
- OPEN
- RESOLVED
- EXHAUSTED
- NOT_ASSESSABLE
- NON_MATERIAL

This is a user-facing representation of the canonical Gap Register.

---

# 25. ANTI-LOOP PROCESS ENGINE RULES

No analysis may repeat indefinitely.

Every execution unit stores:

EXECUTION_FINGERPRINT
- RUN_ID
- BLOCK_ID
- INPUT_VERSION
- EVIDENCE_SET_HASH
- PROMPT_OR_METHOD_VERSION
- OUTPUT_SCHEMA_VERSION

If the exact fingerprint was already executed:

NO_NEW_INFORMATION
→ DO NOT REPEAT

A retry requires at least one:
- new evidence;
- resolved conflict;
- changed method version;
- explicit forensic reason;
- corrected deterministic bug;
- user-supplied critical input.

Retry budget should be bounded.

After repeated unresolved failure:

BLOCKED
→ DETERMINISTIC_DIAGNOSTIC
→ HUMAN_REVIEW
or
→ NOT_ASSESSABLE / INSUFFICIENT

Research dead ends must use the Gap Register.

---

# 26. HUMAN / AI EXECUTION MODES

The post-C7 operating model is now governed by:

`docs/orotitan-equity/OROTITAN_CHATGPT_OPERATING_PROTOCOL_V0.1.md`

Current analytical authority model:

```text
CHATGPT
= PRIMARY NON-DETERMINISTIC ANALYTICAL ENGINE

OROTITAN / SUPABASE / GITHUB / VERCEL
= INFRASTRUCTURE / SYSTEM OF RECORD / PRODUCT SURFACE
```

Deterministic code remains responsible for:

- formulas;
- schema validation;
- reconciliations;
- evidence-ID validation;
- freshness checks;
- state transitions;
- integration;
- publication controls.

ChatGPT remains responsible for:

- research;
- evidence interpretation;
- causal reasoning;
- hypothesis testing;
- sector analysis;
- cyclicality;
- technology analysis;
- moat proof;
- runway;
- forensic interpretation;
- capital allocation;
- Outside View;
- Red Team;
- valuation judgment;
- dossier synthesis.

The current V2 plan does not require a local model, cloud LLM API, or ChatGPT Work.

Future local inference may replace or supplement ChatGPT only after qualification under the then-current frozen contract.

---

# 27. CHATGPT DIRECT INFRASTRUCTURE PROTOCOL

The earlier manual copy/paste bridge is no longer the target operating model.

ChatGPT should work directly with authorized infrastructure connectors where available:

```text
CHATGPT
↔ SUPABASE
↔ GITHUB
↔ VERCEL
```

Normal company-analysis writes must use controlled registry / stage-finalization primitives rather than ad-hoc SQL.

The detailed rules for:

- LOAD;
- STATUS;
- CHECKPOINT;
- SAVE;
- REFRESH;
- PUBLISH;
- context retrieval;
- source ingestion;
- evidence handling;
- checkpoints;
- stage finalization;
- concurrency;
- failure recovery;

are defined by the ChatGPT Operating Protocol.

ChatGPT Work is optional and currently out of scope.

---

# 28. PROCESS STATE MODEL

User-facing workflow:

SHORTLIST
→ RESEARCH
→ DEEP DIVE
→ INTEGRATION
→ PUBLISHED / READY / MONITORING

Internal stages preserve frozen contracts.

Each stage exposes:

STAGE_STATUS
CURRENT_BLOCK
BLOCK_STATUS
BLOCKER
OPEN_QUESTIONS_COUNT
NEXT_ACTION
LAST_ACTIVITY
DATA_CUTOFF

No fake percentage completion.

---

# 29. RESEARCH QUEUE / SHORTLIST

Separate:

RESEARCH_QUEUE
ACTIVE_DOSSIERS
CANONICAL_UNIVERSE

Research Queue fields:

- COMPANY
- TICKER
- SOURCE_METHOD
- ARCHETYPE_HINT
- PRIORITY
- WHY_INTERESTING
- DATE_ADDED
- PRICE
- RESEARCH_STATUS
- NEXT_ACTION

Queue status is organizational only.

It cannot imply analytical quality or OroTitan status.

---

# 30. FRENCH-FIRST PRODUCT LAYER

UI default language: French.

Canonical machine states remain unchanged.

Example:

READY_FOR_DEEP_DIVE
→ "Prêt pour Deep Dive"

NOT_ASSESSABLE
→ "Non évaluable"

SCORE_PERMISSION
→ "Autorisation de scoring"

This avoids breaking frozen semantics.

Future i18n must keep canonical IDs stable.

---

# 31. MARKET DATA / PRICE ENGINE

Market data becomes a first-class product function.

For every security show:

- latest price;
- timestamp;
- source;
- freshness;
- currency;
- market status;
- sync state;
- last successful update;
- data health.

Price-only deltas must not trigger full business reanalysis.

PRICE_ONLY_DELTA
→ valuation / activation path only

unless a market event contains material fundamental information.

---

# 32. EXPORT ENGINE

Required export targets:

PDF
- human-readable final research report

MARKDOWN
- analytical working copy / portability

JSON
- canonical machine artifact

CSV
- screener / shortlist / selected tables

Future:
XLSX

Exports must declare:
- data cutoff;
- artifact versions;
- certification status;
- snapshot time;
- limitations.

---

# 33. UI / UX DESIGN PRINCIPLES

Target aesthetic:

PREMIUM RESEARCH TERMINAL
+
MODERN INVESTMENT WORKBENCH

Not:
- generic SaaS dashboard;
- spreadsheet clone;
- neon trading terminal;
- admin panel;
- wall of cards.

Design principles:

1. high information density without visual noise;
2. strong hierarchy;
3. typography-first;
4. restrained color;
5. semantic status color only;
6. wide analytical workspace;
7. persistent company header;
8. progressive disclosure;
9. fast navigation;
10. keyboard-friendly;
11. mobile is secondary;
12. desktop research workflow is primary.

---

# 34. COMPANY WORKSPACE INFORMATION ARCHITECTURE

Persistent header:

COMPANY
TICKER
PRICE
PRICE FRESHNESS
RUN STATUS
CERTIFICATION
OQS / OVS where valid
OROTITAN STATUS
NEXT ACTION

Primary navigation:

OVERVIEW

ANALYSIS
- Business
- Industry
- Technology
- Cycle
- Moat
- Runway
- Returns
- Cash
- Capital Allocation
- Management
- Outside View
- Risk
- Red Team
- Valuation

EVIDENCE

FINANCIALS

OPEN QUESTIONS

HISTORY

MONITORING

Integration / internal technical states should be visible but not dominate the main investment view.

---

# 35. MODULE PRESENTATION CONTRACT

Each module initially shows:

VERDICT
CONFIDENCE
STRONGEST_EVIDENCE
STRONGEST_COUNTEREVIDENCE
WEAKEST_LINK
KEY_UNCERTAINTY
OPEN_QUESTIONS
INVALIDATION_TRIGGERS
NEXT_ACTION

Then expandable sections show full research.

The user should not need to read a 50-page report to understand the current thesis.

---

# 36. ANALYSIS MAP

Provide a visual dependency map.

Example:

BUSINESS MODEL
↓
ECONOMIC QUALITY
├── MOAT
├── RUNWAY
└── RETURN QUALITY
     ↓
FCF / CAPITAL ALLOCATION
     ↓
OUTSIDE VIEW / RISK
     ↓
RED TEAM
     ↓
VALUATION
     ↓
CERTIFICATION
     ↓
TERMINAL GATE

Block states:
- COMPLETE
- IN_PROGRESS
- BLOCKED
- NOT_APPLICABLE
- NOT_ASSESSABLE
- REOPENED

Clicking a node opens:
- why;
- evidence;
- blocker;
- next action.

---

# 37. COCKPIT

Home cockpit should prioritize action.

Sections:

ACTION REQUIRED
- blocked runs;
- human validation;
- conflicts;
- missing critical input.

IN PROGRESS
- current Research / Deep Dive / Integration dossiers.

RESEARCH QUEUE
- shortlist waiting for work.

READY / MONITORING
- completed high-quality dossiers waiting for price or catalyst.

RECENT CHANGES
- price moves;
- new filings;
- updated canonical snapshots;
- reopened risks.

DATA HEALTH
- stale prices;
- source failures;
- incomplete sync.

---

# 38. FLUIDITY TARGET

Ideal normal-path interaction:

OPEN COMPANY
→ SEE CURRENT STATE
→ CLICK NEXT ACTION
→ OROTITAN PREPARES EXACT WORK
→ USER / AI COMPLETES TASK
→ IMPORT
→ VALIDATE
→ STATE ADVANCES
→ NEXT ACTION

Target:

ONE-CLICK-NEXT

The user should not manually decide:
- which prompt;
- which evidence;
- which schema;
- which stage;
- which artifact;
- which next step.

The system decides from canonical state.

---

# 39. QUALITY BEFORE AUTOMATION

Automation priority:

1. deterministic correctness;
2. evidence traceability;
3. analytical quality;
4. process fluidity;
5. automation rate.

Never invert the order.

A slower assisted analysis with materially better evidence and judgment is preferred over a fully autonomous low-quality analysis.

---

# 40. IMPLEMENTATION PROGRAM

## P0 — Analytical Engine contracts

Deliver:
- COMPANY_ECONOMIC_DNA schema;
- Sector Overlay schema;
- Cyclicality schema;
- Technology schema;
- Industry Structure schema;
- Causal Graph schema;
- Open Questions schema;
- module presentation contract;
- provider-agnostic task contract.

No UI redesign before these contracts are coherent.

## P1 — Process Engine

Deliver:
- run/stage/block state model;
- execution fingerprints;
- retry / loop guards;
- blocker system;
- next-action resolver;
- queue;
- resume;
- history.

## P2 — Analytical Engines

Implement first:
1. Company DNA
2. Sector selection/composition
3. Cyclicality
4. Technology
5. Causal graph
6. Moat Proof V2
7. Runway V2
8. Return Quality V2
9. Forensic FCF
10. Capital Allocation
11. Outside View
12. Red Team
13. Valuation adaptation

## P3 — Workbench UI

Deliver:
- cockpit;
- company workspace;
- analysis map;
- module views;
- blockers;
- open questions;
- evidence browser;
- history.

## P4 — Product functions

Deliver:
- French-first labels;
- market data freshness;
- price refresh;
- export;
- monitoring;
- shortlist;
- activation.

## P5 — AI bridge

Deliver:
- prepare analysis packet;
- copy/export task;
- import result;
- deterministic validation;
- provider abstraction.

## P6 — Future local automation

When hardware permits:
- plug approved local provider into existing analysis task contract;
- rerun qualification under current contract;
- do not redesign analytical engine around a specific model.

---

# 41. ACCEPTANCE CRITERIA

V2 design is acceptable only if:

A. ANALYTICAL DEPTH

- cyclicality can alter normalization and valuation;
- technology analysis can prove or disprove a moat mechanism;
- sector overlays change methods and evidence requirements;
- causal chains expose unsupported links;
- forensic FCF distinguishes reported vs economic cash;
- capital allocation reconstructs actual deployment;
- Outside View influences priors without replacing bottom-up work;
- Red Team can reopen blocks;
- valuation consumes normalized economics.

B. PROCESS QUALITY

- same analysis fingerprint cannot loop indefinitely;
- blockers are explicit;
- gaps record prior searches;
- next action is deterministic where possible;
- resume works without reconstructing context;
- human intervention is requested only when materially necessary.

C. UI QUALITY

- current thesis is understandable in under one minute;
- deepest evidence remains accessible;
- status and blockers are obvious;
- no fake progress percentages;
- user can move to next valid action with minimal friction;
- visual design feels like an equity-research product, not an admin tool.

D. PROVIDER INDEPENDENCE

- manual ChatGPT assistance works;
- future API provider works;
- future local model works;
- canonical analysis artifacts remain provider-independent.

---

# 42. NON-GOALS

V2 is not:

- a chatbot wrapper;
- an autonomous agent that can silently alter canonical conclusions;
- a generic financial dashboard;
- a black-box investment score;
- a replacement for evidence;
- a guarantee of investment performance;
- an attempt to automate every judgment immediately.

---

# 43. POST-C7 PROGRAM DECISION

The current Phase C local-model campaign concluded:

LOCAL_CANDIDATE_REJECTED

for the current tested local candidate set.

Therefore the immediate product strategy is:

BUILD HIGH-QUALITY HUMAN-CENTERED ANALYTICAL ENGINE
+
SIMPLIFY THE RESEARCH WORKFLOW
+
IMPROVE UI/UX
+
KEEP AI PROVIDER-AGNOSTIC
+
DEFER FULL LOCAL AUTOMATION UNTIL HARDWARE / MODEL CAPABILITY IMPROVES

This direction does not reopen Gate 18 Phase C historical results.

---

# 44. NEXT DESIGN ARTIFACTS

After this design spec, prepare in order:

1. ANALYTICAL_ENGINE_V2_DATA_CONTRACTS
2. SECTOR_INTELLIGENCE_LIBRARY_V0.1
3. PROCESS_ENGINE_V2_STATE_MACHINE
4. ANALYSIS_TASK_PROVIDER_CONTRACT
5. WORKBENCH_INFORMATION_ARCHITECTURE
6. UI_DESIGN_SYSTEM_V0.1
7. IMPLEMENTATION_BACKLOG_V0.1

Coding should begin only after items 1-4 are sufficiently stable to avoid rework.

---

# 45. DESIGN STATUS

OROTITAN_ANALYTICAL_ENGINE_V2_DESIGN_V0.1

= POST-C7 TARGET ARCHITECTURE

= NOT FROZEN

= READY FOR CONTRACT DECOMPOSITION

No production mutation is authorized by this document.

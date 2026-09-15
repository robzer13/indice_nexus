# OROTITAN_RESEARCH_STAGE_CONTRACT_V2 — FREEZE V2.0

**Status:** FROZEN DESIGN — V2.0  
**Depends on:** V2 Process + V2 Pilotage  
**Methodology change:** NO

## 0. Purpose

Research builds the authoritative evidence base required for later Fundamentals and Valuation. It asks whether enough reliable, current and sufficiently independent evidence exists to analyze the required blocks honestly.

```text
RESEARCH
= EVIDENCE ACQUISITION
+ NORMALIZATION
+ GAP CLOSURE
+ SOURCE DIVERSIFICATION
+ CONFLICT REGISTRATION
+ MATERIAL RESEARCH HYPOTHESES
+ INPUT SUFFICIENCY
+ CLASSIFICATION INPUTS
```

Research is not final analysis, scoring, valuation or terminal judgment.

## 1. Explicit prohibitions

Research MUST NOT emit final:

```text
MOAT VERDICT
RUNWAY VERDICT
BUSINESS QUALITY VERDICT
DCF / FAIR VALUE
EXPECTED RETURN
OQS / OVS / INVESTMENT SCORE
OROTITAN STATUS
INVESTMENT THESIS
```

## 2. Pre-flight

Before work:

```text
resolve exact RUN_ID / issuer / security / dossier
verify CANONICAL_MODE / RUN_TYPE
verify immutable DATA_CUTOFF
load exact V2 Research contract
verify V2 process / pilotage compatibility
resolve baseline snapshot for REFRESH
inventory user-provided sources
verify no blocking upstream condition
```

Failure -> stop and emit resolution/Pilotage prompt.

## 3. Point-in-time discipline

Material run evidence must respect:

```text
SOURCE_DATE <= DATA_CUTOFF
```

Post-cutoff evidence cannot silently alter the run. It may be flagged for future refresh only.

## 4. Evidence authority

There is one authoritative Evidence Ledger lineage per run. Research normalizes source identity, source date, evidence date/as-of date, claim, source type, evidence grade, block relevance, conflict status and limitations under the frozen ledger semantics.

Research must preserve the frozen Conflict Ledger, Calculation Ledger and Material Assumption semantics. It does not invent parallel ledgers.

## 5. Required coverage

Research must gather enough evidence for later analysis of, as applicable:

```text
identity / data lock
business model
economic quality
moat
runway
return quality
FCF / Owner Earnings / forensic
capital allocation
management / governance
outside view
risk / resilience
valuation inputs
sector-specific overlays
```

Sufficiency means mandatory coverage, evidence adequacy and no material blocking gap.

## 6. Taxonomy input collection

Research gathers factual inputs for:

```text
issuer country code
primary listing country code
sector
industry group
primary business model
secondary business model if material
economic exposure regions
PEA eligibility evidence if assessed
```

Research may propose classification values only from `OROTITAN_TAXONOMY_V2.0.json`. Fundamentals owns the locked projection.

## 7. Refresh

Every REFRESH begins with:

```text
WHAT CHANGED SINCE LAST CANONICAL SNAPSHOT?
```

Then classify:

```text
PRICE_ONLY_DELTA
ROUTINE_FUNDAMENTAL_DELTA
FULL_REFRESH_REQUIRED
```

Prior canonical research is reusable knowledge, not automatically current evidence. Material inherited conclusions relied upon in the new run must be revalidated.

## 8. Required normal outputs

At normal completion persist/version at minimum:

```text
RESEARCH_SOURCE_MANIFEST
EVIDENCE_LEDGER
CONFLICT_LEDGER or exact authoritative reference
MATERIAL_RESEARCH_HYPOTHESIS_REGISTER
RESEARCH_GAP_REGISTER
INPUT_SUFFICIENCY_ASSESSMENT
CLASSIFICATION_INPUT_RECORD
RESEARCH_STAGE_MANIFEST
```

Exact physical filenames are implementation details.

## 9. Completion gate

Research may become `COMPLETE` only after:

```text
required work complete or validly resolved
self-audit complete
required artifacts generated
bytes persisted / versioned
hash verification complete
Registry reconciled
FINAL Research Stage Manifest registered
READY_FOR_DEEP_DIVE = YES
```

Narrative completion alone is never sufficient.

## 10. Final user-facing handoff

On success, the last visible content is the exact Fundamentals bootstrap. No prose may follow it.

On block, the last visible content is the exact resolution/Pilotage bootstrap. No Fundamentals prompt may be emitted.

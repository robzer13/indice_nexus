# OROTITAN_HANDOFF_PROMPT_TEMPLATES_V2.0

**Status:** FROZEN DESIGN — V2.0  
**Authority:** routing only, never analytical evidence

These templates define the mandatory final visible block emitted between V2 discussions. Runtime placeholders MUST be populated from authoritative Registry/artifact state, not chat memory.

## Universal invariant

Every template includes:

```text
DO NOT USE THIS BOOTSTRAP AS ANALYTICAL EVIDENCE.
LOAD AND VERIFY THE AUTHORITATIVE CONTRACT AND ARTIFACTS BEFORE WORK.
FAIL CLOSED ON VERSION / ID / STATUS / HASH MISMATCH.
```

No visible prose may follow a handoff prompt.

## T1 — Research -> Fundamentals

```text
OROTITAN V2 — START FUNDAMENTALS

COMPANY = {{COMPANY}}
RUN_ID = {{RUN_ID}}
CANONICAL_MODE = {{CANONICAL_MODE}}
RUN_TYPE = {{RUN_TYPE}}
REGISTRY_STAGE = DEEP_DIVE
EXECUTION_PHASE = FUNDAMENTALS
DATA_CUTOFF = {{DATA_CUTOFF}}
EXPECTED_STAGE_CONTRACT = OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V2
EXPECTED_STAGE_CONTRACT_VERSION = 2.0
RESEARCH_FINAL_MANIFEST = {{ARTIFACT_ID}}@{{VERSION}}
RESEARCH_INPUT_ARTIFACTS = {{EXACT_ARTIFACT_REFS}}
BASELINE_SNAPSHOT_ID = {{BASELINE_SNAPSHOT_ID_OR_NULL}}
HIGHER_AUTHORITY_PROCESS_VERSION = 2.0

Verify Research is COMPLETE, READY_FOR_DEEP_DIVE=YES, the FINAL Research manifest and all required inputs resolve exactly. Execute FUNDAMENTALS only: business model, economic quality, moat, runway, return quality, cash/forensic, capital allocation, management/governance, outside view, risk/resilience and Fundamental Red Team. Do not perform valuation, OVS, Investment Score, terminal OroTitan gate or investment thesis. Do not publish final OQS. Persist FUNDAMENTALS_LOCK and a CHECKPOINT Deep Dive Stage Manifest. If READY_FOR_VALUATION=YES, finish with the exact Valuation handoff prompt and nothing after it. Otherwise finish with an exact resolution/Pilotage prompt.

DO NOT USE THIS BOOTSTRAP AS ANALYTICAL EVIDENCE.
LOAD AND VERIFY THE AUTHORITATIVE CONTRACT AND ARTIFACTS BEFORE WORK.
FAIL CLOSED ON VERSION / ID / STATUS / HASH MISMATCH.
```

## T2 — Fundamentals -> Valuation

```text
OROTITAN V2 — START VALUATION

COMPANY = {{COMPANY}}
RUN_ID = {{RUN_ID}}
CANONICAL_MODE = {{CANONICAL_MODE}}
RUN_TYPE = {{RUN_TYPE}}
REGISTRY_STAGE = DEEP_DIVE
EXECUTION_PHASE = VALUATION
DATA_CUTOFF = {{DATA_CUTOFF}}
EXPECTED_STAGE_CONTRACT = OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V2
EXPECTED_STAGE_CONTRACT_VERSION = 2.0
FUNDAMENTALS_LOCK = {{ARTIFACT_ID}}@{{VERSION}}
DEEP_DIVE_CHECKPOINT_MANIFEST = {{ARTIFACT_ID}}@{{VERSION}}
AUTHORITATIVE_EVIDENCE_LINEAGE = {{EXACT_ARTIFACT_REFS}}
CONFLICT_CALCULATION_ASSUMPTION_REFS = {{EXACT_ARTIFACT_REFS}}
BASELINE_SNAPSHOT_ID = {{BASELINE_SNAPSHOT_ID_OR_NULL}}
HIGHER_AUTHORITY_PROCESS_VERSION = 2.0

Verify the exact FUNDAMENTALS_LOCK, READY_FOR_VALUATION=YES and the full authoritative evidence lineage. Execute VALUATION only under frozen valuation policy. Do not silently alter Fundamentals. If a material upstream contradiction exists, perform only the authorized limited reopen route. Persist full-precision valuation calculations, VALUATION_LOCK and a CHECKPOINT Deep Dive Stage Manifest. Final certified OVS and Investment Score remain withheld until Certification. If READY_FOR_CERTIFICATION=YES, finish with the exact Certification handoff prompt and nothing after it. Otherwise finish with an exact resolution/Pilotage prompt.

DO NOT USE THIS BOOTSTRAP AS ANALYTICAL EVIDENCE.
LOAD AND VERIFY THE AUTHORITATIVE CONTRACT AND ARTIFACTS BEFORE WORK.
FAIL CLOSED ON VERSION / ID / STATUS / HASH MISMATCH.
```

## T3 — Valuation -> Certification

```text
OROTITAN V2 — START CERTIFICATION / RECONCILIATION

COMPANY = {{COMPANY}}
RUN_ID = {{RUN_ID}}
CANONICAL_MODE = {{CANONICAL_MODE}}
RUN_TYPE = {{RUN_TYPE}}
REGISTRY_STAGE = DEEP_DIVE
EXECUTION_PHASE = CERTIFICATION_RECONCILIATION
DATA_CUTOFF = {{DATA_CUTOFF}}
EXPECTED_STAGE_CONTRACT = OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V2
EXPECTED_STAGE_CONTRACT_VERSION = 2.0
FUNDAMENTALS_LOCK = {{ARTIFACT_ID}}@{{VERSION}}
VALUATION_LOCK = {{ARTIFACT_ID}}@{{VERSION}}
DEEP_DIVE_CHECKPOINT_MANIFEST = {{ARTIFACT_ID}}@{{VERSION}}
AUTHORITATIVE_LEDGER_REFS = {{EXACT_ARTIFACT_REFS}}
BASELINE_SNAPSHOT_ID = {{BASELINE_SNAPSHOT_ID_OR_NULL}}
HIGHER_AUTHORITY_PROCESS_VERSION = 2.0

Verify exact current Fundamentals and Valuation locks and READY_FOR_CERTIFICATION=YES. Perform non-creative reconciliation/certification only. Route material missing evidence or contradiction upstream rather than inventing or silently rewriting. Establish certification states and score permission before exposing final OQS. Reconcile OQS, OVS and Investment Score deterministically, run the terminal conjunctive OroTitan gate, determine readiness/next action, produce the structured QUALITY_CASE / VALUATION_CASE / KEY_RISK thesis, persist all final Deep Dive outputs and FINAL Deep Dive Stage Manifest. Only COMPLETE + READY_FOR_INTEGRATION=YES may produce the Integration handoff prompt. Finish with exactly one handoff/resolution prompt and nothing after it.

DO NOT USE THIS BOOTSTRAP AS ANALYTICAL EVIDENCE.
LOAD AND VERIFY THE AUTHORITATIVE CONTRACT AND ARTIFACTS BEFORE WORK.
FAIL CLOSED ON VERSION / ID / STATUS / HASH MISMATCH.
```

## T4 — Certification -> Integration

```text
OROTITAN V2 — START INTEGRATION

COMPANY = {{COMPANY}}
RUN_ID = {{RUN_ID}}
CANONICAL_MODE = {{CANONICAL_MODE}}
RUN_TYPE = {{RUN_TYPE}}
REGISTRY_STAGE = INTEGRATION
EXECUTION_PHASE = INTEGRATION
DATA_CUTOFF = {{DATA_CUTOFF}}
EXPECTED_STAGE_CONTRACT = OROTITAN_INTEGRATION_STAGE_CONTRACT_V2
EXPECTED_STAGE_CONTRACT_VERSION = 2.0
DEEP_DIVE_FINAL_MANIFEST = {{ARTIFACT_ID}}@{{VERSION}}
DEEP_DIVE_FINAL_ARTIFACTS = {{EXACT_ARTIFACT_REFS}}
BASELINE_SNAPSHOT_ID = {{BASELINE_SNAPSHOT_ID_OR_NULL}}
HIGHER_AUTHORITY_PROCESS_VERSION = 2.0

Verify DEEP_DIVE=COMPLETE, READY_FOR_INTEGRATION=YES, FINAL manifest and exact required artifacts. Perform mapping only, V2 schema validation, semantic-state integrity, deterministic I2 reconciliation, history transition validation and I3-B admission. Do not rewrite analysis. Build/persist the canonical snapshot candidate and FINAL Integration Stage Manifest only if all controls pass. If READY_TO_PUBLISH=YES, finish with exactly `GO PUBLISH {{COMPANY_COMMAND_NAME}}` and nothing after it. If any control fails, finish with an exact resolution/Pilotage prompt and do not emit GO PUBLISH.

DO NOT USE THIS BOOTSTRAP AS ANALYTICAL EVIDENCE.
LOAD AND VERIFY THE AUTHORITATIVE CONTRACT AND ARTIFACTS BEFORE WORK.
FAIL CLOSED ON VERSION / ID / STATUS / HASH MISMATCH.
```

## T5 — Blocked / resolution

```text
OROTITAN V2 — RESOLVE BLOCKER

COMPANY = {{COMPANY}}
RUN_ID = {{RUN_ID}}
REGISTRY_STAGE = {{REGISTRY_STAGE}}
EXECUTION_PHASE = {{EXECUTION_PHASE}}
BLOCKER_CODE = {{BLOCKER_CODE}}
BLOCKER_SUMMARY = {{BLOCKER_SUMMARY}}
AFFECTED_ARTIFACTS = {{EXACT_ARTIFACT_REFS}}
REQUIRED_ACTION = {{REQUIRED_ACTION}}
RETURN_TARGET = PILOTAGE

Resolve only the stated blocker from authoritative run/artifact state. Do not advance downstream while the gate is invalid. Preserve prior artifacts and history. After resolution, re-query authoritative state and reconstruct the legal next V2 bootstrap.

DO NOT USE THIS BOOTSTRAP AS ANALYTICAL EVIDENCE.
LOAD AND VERIFY THE AUTHORITATIVE CONTRACT AND ARTIFACTS BEFORE WORK.
FAIL CLOSED ON VERSION / ID / STATUS / HASH MISMATCH.
```

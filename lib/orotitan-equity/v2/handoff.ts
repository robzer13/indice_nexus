export type ArtifactRef = { artifact_id: string; version: number; content_sha256?: string };

type Common = {
  company: string;
  companyCommandName: string;
  runId: string;
  canonicalMode: string;
  runType: "INITIAL" | "REFRESH";
  dataCutoff: string;
  baselineSnapshotId: string | null;
};

const INVARIANT = `DO NOT USE THIS BOOTSTRAP AS ANALYTICAL EVIDENCE.\nLOAD AND VERIFY THE AUTHORITATIVE CONTRACT AND ARTIFACTS BEFORE WORK.\nFAIL CLOSED ON VERSION / ID / STATUS / HASH MISMATCH.`;

function ref(value: ArtifactRef): string {
  return `${value.artifact_id}@${value.version}`;
}

function refs(values: ArtifactRef[]): string {
  return values.map(ref).join(", ");
}

function baseline(value: string | null): string {
  return value ?? "NULL";
}

export function buildResearchToFundamentals(input: Common & {
  researchFinalManifest: ArtifactRef;
  researchInputs: ArtifactRef[];
}): string {
  return `OROTITAN V2 — START FUNDAMENTALS

COMPANY = ${input.company}
RUN_ID = ${input.runId}
CANONICAL_MODE = ${input.canonicalMode}
RUN_TYPE = ${input.runType}
REGISTRY_STAGE = DEEP_DIVE
EXECUTION_PHASE = FUNDAMENTALS
DATA_CUTOFF = ${input.dataCutoff}
EXPECTED_STAGE_CONTRACT = OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V2
EXPECTED_STAGE_CONTRACT_VERSION = 2.0
RESEARCH_FINAL_MANIFEST = ${ref(input.researchFinalManifest)}
RESEARCH_INPUT_ARTIFACTS = ${refs(input.researchInputs)}
BASELINE_SNAPSHOT_ID = ${baseline(input.baselineSnapshotId)}
HIGHER_AUTHORITY_PROCESS_VERSION = 2.0

Verify Research is COMPLETE, READY_FOR_DEEP_DIVE=YES, the FINAL Research manifest and all required inputs resolve exactly. Execute FUNDAMENTALS only: business model, economic quality, moat, runway, return quality, cash/forensic, capital allocation, management/governance, outside view, risk/resilience and Fundamental Red Team. Do not perform valuation, OVS, Investment Score, terminal OroTitan gate or investment thesis. Do not publish final OQS. Persist FUNDAMENTALS_LOCK and a CHECKPOINT Deep Dive Stage Manifest. If READY_FOR_VALUATION=YES, finish with the exact Valuation handoff prompt and nothing after it. Otherwise finish with an exact resolution/Pilotage prompt.

${INVARIANT}`;
}

export function buildFundamentalsToValuation(input: Common & {
  fundamentalsLock: ArtifactRef;
  checkpointManifest: ArtifactRef;
  evidenceLineage: ArtifactRef[];
  conflictCalculationAssumptionRefs: ArtifactRef[];
}): string {
  return `OROTITAN V2 — START VALUATION

COMPANY = ${input.company}
RUN_ID = ${input.runId}
CANONICAL_MODE = ${input.canonicalMode}
RUN_TYPE = ${input.runType}
REGISTRY_STAGE = DEEP_DIVE
EXECUTION_PHASE = VALUATION
DATA_CUTOFF = ${input.dataCutoff}
EXPECTED_STAGE_CONTRACT = OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V2
EXPECTED_STAGE_CONTRACT_VERSION = 2.0
FUNDAMENTALS_LOCK = ${ref(input.fundamentalsLock)}
DEEP_DIVE_CHECKPOINT_MANIFEST = ${ref(input.checkpointManifest)}
AUTHORITATIVE_EVIDENCE_LINEAGE = ${refs(input.evidenceLineage)}
CONFLICT_CALCULATION_ASSUMPTION_REFS = ${refs(input.conflictCalculationAssumptionRefs)}
BASELINE_SNAPSHOT_ID = ${baseline(input.baselineSnapshotId)}
HIGHER_AUTHORITY_PROCESS_VERSION = 2.0

Verify the exact FUNDAMENTALS_LOCK, READY_FOR_VALUATION=YES and the full authoritative evidence lineage. Execute VALUATION only under frozen valuation policy. Do not silently alter Fundamentals. If a material upstream contradiction exists, perform only the authorized limited reopen route. Persist full-precision valuation calculations, VALUATION_LOCK and a CHECKPOINT Deep Dive Stage Manifest. Final certified OVS and Investment Score remain withheld until Certification. If READY_FOR_CERTIFICATION=YES, finish with the exact Certification handoff prompt and nothing after it. Otherwise finish with an exact resolution/Pilotage prompt.

${INVARIANT}`;
}

export function buildValuationToCertification(input: Common & {
  fundamentalsLock: ArtifactRef;
  valuationLock: ArtifactRef;
  checkpointManifest: ArtifactRef;
  authoritativeLedgerRefs: ArtifactRef[];
}): string {
  return `OROTITAN V2 — START CERTIFICATION / RECONCILIATION

COMPANY = ${input.company}
RUN_ID = ${input.runId}
CANONICAL_MODE = ${input.canonicalMode}
RUN_TYPE = ${input.runType}
REGISTRY_STAGE = DEEP_DIVE
EXECUTION_PHASE = CERTIFICATION_RECONCILIATION
DATA_CUTOFF = ${input.dataCutoff}
EXPECTED_STAGE_CONTRACT = OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V2
EXPECTED_STAGE_CONTRACT_VERSION = 2.0
FUNDAMENTALS_LOCK = ${ref(input.fundamentalsLock)}
VALUATION_LOCK = ${ref(input.valuationLock)}
DEEP_DIVE_CHECKPOINT_MANIFEST = ${ref(input.checkpointManifest)}
AUTHORITATIVE_LEDGER_REFS = ${refs(input.authoritativeLedgerRefs)}
BASELINE_SNAPSHOT_ID = ${baseline(input.baselineSnapshotId)}
HIGHER_AUTHORITY_PROCESS_VERSION = 2.0

Verify exact current Fundamentals and Valuation locks and READY_FOR_CERTIFICATION=YES. Perform non-creative reconciliation/certification only. Route material missing evidence or contradiction upstream rather than inventing or silently rewriting. Establish certification states and score permission before exposing final OQS. Reconcile OQS, OVS and Investment Score deterministically, run the terminal conjunctive OroTitan gate, determine readiness/next action, produce the structured QUALITY_CASE / VALUATION_CASE / KEY_RISK thesis, persist all final Deep Dive outputs and FINAL Deep Dive Stage Manifest. Only COMPLETE + READY_FOR_INTEGRATION=YES may produce the Integration handoff prompt. Finish with exactly one handoff/resolution prompt and nothing after it.

${INVARIANT}`;
}

export function buildCertificationToIntegration(input: Common & {
  deepDiveFinalManifest: ArtifactRef;
  deepDiveFinalArtifacts: ArtifactRef[];
}): string {
  return `OROTITAN V2 — START INTEGRATION

COMPANY = ${input.company}
RUN_ID = ${input.runId}
CANONICAL_MODE = ${input.canonicalMode}
RUN_TYPE = ${input.runType}
REGISTRY_STAGE = INTEGRATION
EXECUTION_PHASE = INTEGRATION
DATA_CUTOFF = ${input.dataCutoff}
EXPECTED_STAGE_CONTRACT = OROTITAN_INTEGRATION_STAGE_CONTRACT_V2
EXPECTED_STAGE_CONTRACT_VERSION = 2.0
DEEP_DIVE_FINAL_MANIFEST = ${ref(input.deepDiveFinalManifest)}
DEEP_DIVE_FINAL_ARTIFACTS = ${refs(input.deepDiveFinalArtifacts)}
BASELINE_SNAPSHOT_ID = ${baseline(input.baselineSnapshotId)}
HIGHER_AUTHORITY_PROCESS_VERSION = 2.0

Verify DEEP_DIVE=COMPLETE, READY_FOR_INTEGRATION=YES, FINAL manifest and exact required artifacts. Perform mapping only, V2 schema validation, semantic-state integrity, deterministic I2 reconciliation, history transition validation and I3-B admission. Do not rewrite analysis. Build/persist the canonical snapshot candidate and FINAL Integration Stage Manifest only if all controls pass. If READY_TO_PUBLISH=YES, finish with exactly \`GO PUBLISH ${input.companyCommandName}\` and nothing after it. If any control fails, finish with an exact resolution/Pilotage prompt and do not emit GO PUBLISH.

${INVARIANT}`;
}

export function buildBlockedResolution(input: Common & {
  registryStage: "RESEARCH" | "DEEP_DIVE" | "INTEGRATION";
  executionPhase: string;
  blockerCode: string;
  blockerSummary: string;
  affectedArtifacts: ArtifactRef[];
  requiredAction: string;
}): string {
  return `OROTITAN V2 — RESOLVE BLOCKER

COMPANY = ${input.company}
RUN_ID = ${input.runId}
REGISTRY_STAGE = ${input.registryStage}
EXECUTION_PHASE = ${input.executionPhase}
BLOCKER_CODE = ${input.blockerCode}
BLOCKER_SUMMARY = ${input.blockerSummary}
AFFECTED_ARTIFACTS = ${refs(input.affectedArtifacts)}
REQUIRED_ACTION = ${input.requiredAction}
RETURN_TARGET = PILOTAGE

Resolve only the stated blocker from authoritative run/artifact state. Do not advance downstream while the gate is invalid. Preserve prior artifacts and history. After resolution, re-query authoritative state and reconstruct the legal next V2 bootstrap.

${INVARIANT}`;
}

export type V2RoutingState = Common & {
  registryStage: "RESEARCH" | "DEEP_DIVE" | "INTEGRATION";
  executionPhase: "RESEARCH" | "FUNDAMENTALS" | "VALUATION" | "CERTIFICATION_RECONCILIATION" | "INTEGRATION";
  blocker?: { code: string; summary: string; requiredAction: string; affectedArtifacts: ArtifactRef[] };
  readyForDeepDive?: boolean;
  readyForValuation?: boolean;
  readyForCertification?: boolean;
  readyForIntegration?: boolean;
  readyToPublish?: boolean;
  researchFinalManifest?: ArtifactRef;
  researchInputs?: ArtifactRef[];
  fundamentalsLock?: ArtifactRef;
  valuationLock?: ArtifactRef;
  checkpointManifest?: ArtifactRef;
  evidenceLineage?: ArtifactRef[];
  conflictCalculationAssumptionRefs?: ArtifactRef[];
  authoritativeLedgerRefs?: ArtifactRef[];
  deepDiveFinalManifest?: ArtifactRef;
  deepDiveFinalArtifacts?: ArtifactRef[];
};

export function reconstructV2Handoff(state: V2RoutingState): string {
  if (state.blocker) return buildBlockedResolution({
    ...state,
    blockerCode: state.blocker.code,
    blockerSummary: state.blocker.summary,
    affectedArtifacts: state.blocker.affectedArtifacts,
    requiredAction: state.blocker.requiredAction,
  });
  if (state.registryStage === "RESEARCH" && state.readyForDeepDive && state.researchFinalManifest && state.researchInputs) {
    return buildResearchToFundamentals({ ...state, researchFinalManifest: state.researchFinalManifest, researchInputs: state.researchInputs });
  }
  if (state.registryStage === "DEEP_DIVE" && state.executionPhase === "FUNDAMENTALS" && state.readyForValuation
      && state.fundamentalsLock && state.checkpointManifest && state.evidenceLineage && state.conflictCalculationAssumptionRefs) {
    return buildFundamentalsToValuation({ ...state, fundamentalsLock: state.fundamentalsLock, checkpointManifest: state.checkpointManifest,
      evidenceLineage: state.evidenceLineage, conflictCalculationAssumptionRefs: state.conflictCalculationAssumptionRefs });
  }
  if (state.registryStage === "DEEP_DIVE" && state.executionPhase === "VALUATION" && state.readyForCertification
      && state.fundamentalsLock && state.valuationLock && state.checkpointManifest && state.authoritativeLedgerRefs) {
    return buildValuationToCertification({ ...state, fundamentalsLock: state.fundamentalsLock, valuationLock: state.valuationLock,
      checkpointManifest: state.checkpointManifest, authoritativeLedgerRefs: state.authoritativeLedgerRefs });
  }
  if (state.registryStage === "DEEP_DIVE" && state.executionPhase === "CERTIFICATION_RECONCILIATION" && state.readyForIntegration
      && state.deepDiveFinalManifest && state.deepDiveFinalArtifacts) {
    return buildCertificationToIntegration({ ...state, deepDiveFinalManifest: state.deepDiveFinalManifest,
      deepDiveFinalArtifacts: state.deepDiveFinalArtifacts });
  }
  if (state.registryStage === "INTEGRATION" && state.executionPhase === "INTEGRATION" && state.readyToPublish) {
    return `GO PUBLISH ${state.companyCommandName}`;
  }
  return buildBlockedResolution({
    ...state,
    blockerCode: "NO_LEGAL_HANDOFF",
    blockerSummary: "Authoritative state does not satisfy a V2 downstream gate.",
    affectedArtifacts: [],
    requiredAction: "Re-query Registry state and resolve the missing gate or exact artifact reference.",
  });
}

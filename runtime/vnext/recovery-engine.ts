export const RECOVERY_CLASSES = [
  "AUTO_RETRY",
  "DETERMINISTIC_AUTO_REPAIR",
  "VERIFIABLE_RECOVERY",
  "HUMAN_REQUIRED",
] as const;

export type RecoveryClass = (typeof RECOVERY_CLASSES)[number];

export const RECOVERY_INCIDENT_KINDS = [
  "TRANSIENT_EXECUTION_FAILURE",
  "MANIFEST_POINTER_MISMATCH",
  "ARTIFACT_LOCATOR_UNRESOLVED",
  "CONTRACT_DRIFT",
  "DATA_CUTOFF_CONFLICT",
  "IDENTITY_AMBIGUITY",
  "ARTIFACT_HASH_MISMATCH",
  "ANALYTICAL_AMBIGUITY",
  "UNKNOWN",
] as const;

export type RecoveryIncidentKind =
  (typeof RECOVERY_INCIDENT_KINDS)[number];

export type RecoveryAction =
  | "RETRY_SAME_REQUEST"
  | "REBIND_PROVEN_MANIFEST_POINTER"
  | "RESOLVE_AND_VERIFY_LOCATOR"
  | "ESCALATE_HUMAN";

export type RecoveryMutationScope =
  | "NONE"
  | "EXECUTION_RETRY_ONLY"
  | "CURRENT_ROUTING_METADATA_ONLY"
  | "VERIFIED_LOCATOR_BINDING_ONLY";

export interface RecoveryProofs {
  sameRequestFingerprint: boolean;
  operationIdempotent: boolean;
  retryBudgetAvailable: boolean;
  noCommittedMutation: boolean;
  expectedStateVersionFresh: boolean;

  repairTargetUnique: boolean;
  authoritativeTargetResolved: boolean;
  targetSealed: boolean;
  targetAvailable: boolean;
  targetHashVerified: boolean;
  targetContractPinsMatch: boolean;

  expectedHashKnown: boolean;
  candidateLocatorDurable: boolean;
  candidateBytesHashVerified: boolean;
  artifactIdentityExact: boolean;

  changesAnalyticalMeaning: boolean;
  rewritesHistoricalArtifact: boolean;
  rewritesHistoricalEvent: boolean;
  requiresAnalystJudgment: boolean;
  identityAmbiguous: boolean;
  contractDriftUnresolved: boolean;
  dataCutoffChangeRequired: boolean;
}

export interface RecoveryIncident {
  incidentId: string;
  runId: string;
  kind: RecoveryIncidentKind;
  operationId: string | null;
  requestFingerprint: string | null;
  attempt: number;
  maxAttempts: number;
  proofs: RecoveryProofs;
}

export interface RecoveryPlan {
  incidentId: string;
  runId: string;
  classification: RecoveryClass;
  action: RecoveryAction;
  mutationScope: RecoveryMutationScope;
  automatic: boolean;
  requiresFreshReread: boolean;
  reasonCodes: readonly string[];
}

export interface RecoveryHistoryEntry {
  sequence: number;
  eventId: string;
  fingerprint: string;
}

export interface RecoveryHistorySnapshot {
  runId: string;
  entries: readonly RecoveryHistoryEntry[];
}

export interface RecoveryActionReceipt {
  incidentId: string;
  action: RecoveryAction;
  applied: boolean;
}

export interface RecoveryStore {
  readHistory(runId: string): Promise<RecoveryHistorySnapshot>;
  applyRecovery(
    incident: RecoveryIncident,
    plan: RecoveryPlan,
  ): Promise<RecoveryActionReceipt>;
}

export interface RecoveryExecutionResult {
  plan: RecoveryPlan;
  receipt: RecoveryActionReceipt | null;
  historyBefore: RecoveryHistorySnapshot;
  historyAfter: RecoveryHistorySnapshot;
}

function humanRequired(
  incident: RecoveryIncident,
  ...reasonCodes: string[]
): RecoveryPlan {
  return {
    incidentId: incident.incidentId,
    runId: incident.runId,
    classification: "HUMAN_REQUIRED",
    action: "ESCALATE_HUMAN",
    mutationScope: "NONE",
    automatic: false,
    requiresFreshReread: true,
    reasonCodes,
  };
}

function hasSemanticOrHistoricalRisk(proofs: RecoveryProofs): boolean {
  return (
    proofs.changesAnalyticalMeaning ||
    proofs.rewritesHistoricalArtifact ||
    proofs.rewritesHistoricalEvent ||
    proofs.requiresAnalystJudgment ||
    proofs.identityAmbiguous ||
    proofs.contractDriftUnresolved ||
    proofs.dataCutoffChangeRequired
  );
}

export function classifyRecoveryIncident(
  incident: RecoveryIncident,
): RecoveryPlan {
  const { proofs } = incident;

  if (hasSemanticOrHistoricalRisk(proofs)) {
    return humanRequired(
      incident,
      "SEMANTIC_OR_HISTORICAL_AUTHORITY_RISK",
    );
  }

  if (incident.kind === "TRANSIENT_EXECUTION_FAILURE") {
    const retryAllowed =
      incident.operationId !== null &&
      incident.requestFingerprint !== null &&
      proofs.sameRequestFingerprint &&
      proofs.operationIdempotent &&
      proofs.retryBudgetAvailable &&
      incident.attempt < incident.maxAttempts &&
      proofs.noCommittedMutation &&
      proofs.expectedStateVersionFresh;

    if (!retryAllowed) {
      return humanRequired(
        incident,
        "AUTO_RETRY_PRECONDITIONS_NOT_PROVEN",
      );
    }

    return {
      incidentId: incident.incidentId,
      runId: incident.runId,
      classification: "AUTO_RETRY",
      action: "RETRY_SAME_REQUEST",
      mutationScope: "EXECUTION_RETRY_ONLY",
      automatic: true,
      requiresFreshReread: true,
      reasonCodes: ["TRANSIENT_IDEMPOTENT_RETRY_PROVEN"],
    };
  }

  if (incident.kind === "MANIFEST_POINTER_MISMATCH") {
    const repairProven =
      proofs.repairTargetUnique &&
      proofs.authoritativeTargetResolved &&
      proofs.targetSealed &&
      proofs.targetAvailable &&
      proofs.targetHashVerified &&
      proofs.targetContractPinsMatch &&
      proofs.expectedStateVersionFresh;

    if (!repairProven) {
      return humanRequired(
        incident,
        "DETERMINISTIC_MANIFEST_REPAIR_NOT_PROVEN",
      );
    }

    return {
      incidentId: incident.incidentId,
      runId: incident.runId,
      classification: "DETERMINISTIC_AUTO_REPAIR",
      action: "REBIND_PROVEN_MANIFEST_POINTER",
      mutationScope: "CURRENT_ROUTING_METADATA_ONLY",
      automatic: true,
      requiresFreshReread: true,
      reasonCodes: ["UNIQUE_AUTHORITATIVE_MANIFEST_TARGET_PROVEN"],
    };
  }

  if (incident.kind === "ARTIFACT_LOCATOR_UNRESOLVED") {
    const verifiable =
      proofs.expectedHashKnown &&
      proofs.candidateLocatorDurable &&
      proofs.candidateBytesHashVerified &&
      proofs.artifactIdentityExact &&
      proofs.expectedStateVersionFresh;

    if (!verifiable) {
      return humanRequired(
        incident,
        "VERIFIABLE_LOCATOR_RECOVERY_NOT_PROVEN",
      );
    }

    return {
      incidentId: incident.incidentId,
      runId: incident.runId,
      classification: "VERIFIABLE_RECOVERY",
      action: "RESOLVE_AND_VERIFY_LOCATOR",
      mutationScope: "VERIFIED_LOCATOR_BINDING_ONLY",
      automatic: true,
      requiresFreshReread: true,
      reasonCodes: ["EXACT_IDENTITY_AND_BYTES_HASH_VERIFIED"],
    };
  }

  return humanRequired(
    incident,
    "NO_FROZEN_AUTOMATIC_RECOVERY_RULE",
  );
}

function historyPrefixIsImmutable(
  before: RecoveryHistorySnapshot,
  after: RecoveryHistorySnapshot,
): boolean {
  if (before.runId !== after.runId) return false;
  if (after.entries.length < before.entries.length) return false;

  for (let index = 0; index < before.entries.length; index += 1) {
    const oldEntry = before.entries[index];
    const newEntry = after.entries[index];

    if (
      oldEntry.sequence !== newEntry.sequence ||
      oldEntry.eventId !== newEntry.eventId ||
      oldEntry.fingerprint !== newEntry.fingerprint
    ) {
      return false;
    }
  }

  return true;
}

/**
 * Gate 11 recovery execution.
 *
 * Historical entries are snapshotted before any automatic action and must
 * remain byte/fingerprint-identical as a prefix afterward.
 */
export async function executeRecovery(
  store: RecoveryStore,
  incident: RecoveryIncident,
): Promise<RecoveryExecutionResult> {
  const plan = classifyRecoveryIncident(incident);
  const historyBefore = await store.readHistory(incident.runId);

  if (plan.classification === "HUMAN_REQUIRED") {
    return {
      plan,
      receipt: null,
      historyBefore,
      historyAfter: historyBefore,
    };
  }

  const receipt = await store.applyRecovery(incident, plan);
  const historyAfter = await store.readHistory(incident.runId);

  if (!receipt.applied) {
    throw new Error("VNEXT_RECOVERY_ACTION_NOT_APPLIED");
  }

  if (
    receipt.incidentId !== incident.incidentId ||
    receipt.action !== plan.action
  ) {
    throw new Error("VNEXT_RECOVERY_RECEIPT_MISMATCH");
  }

  if (!historyPrefixIsImmutable(historyBefore, historyAfter)) {
    throw new Error("VNEXT_RECOVERY_HISTORY_REWRITE_DETECTED");
  }

  return {
    plan,
    receipt,
    historyBefore,
    historyAfter,
  };
}

export function assertRecoveryPlanAutomatic(plan: RecoveryPlan): void {
  if (!plan.automatic || plan.classification === "HUMAN_REQUIRED") {
    throw new Error(
      "VNEXT_RECOVERY_REQUIRES_HUMAN: " + plan.reasonCodes.join(","),
    );
  }
}

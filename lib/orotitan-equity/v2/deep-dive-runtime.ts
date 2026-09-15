import {
  validateStageManifest,
  type StageManifestValidationContext,
  type StageManifestValidationResult,
} from "../v1/stage-manifest";
import type { ArtifactRef } from "./handoff";

export type FundamentalsLock = {
  lock_type: "FUNDAMENTALS_LOCK";
  artifact: ArtifactRef;
  run_id: string;
  data_cutoff: string;
  stage_revision: number;
  research_final_manifest: ArtifactRef;
  evidence_lineage: ArtifactRef[];
  red_team_status: "COMPLETE";
  taxonomy_record: ArtifactRef;
  business_description_artifact: ArtifactRef;
  scoring_inputs_artifact: ArtifactRef;
  limitations: string[];
  invalidation_triggers: string[];
  ready_for_valuation: "YES" | "NO";
  supersedes?: ArtifactRef | null;
};

export type ValuationLock = {
  lock_type: "VALUATION_LOCK";
  artifact: ArtifactRef;
  run_id: string;
  data_cutoff: string;
  stage_revision: number;
  fundamentals_lock: ArtifactRef;
  evidence_lineage: ArtifactRef[];
  calculation_ledger: ArtifactRef;
  assumption_register: ArtifactRef;
  valuation_artifact: ArtifactRef;
  full_precision_inputs_artifact: ArtifactRef;
  ready_for_certification: "YES" | "NO";
  supersedes?: ArtifactRef | null;
};

function exactRef(left: ArtifactRef, right: ArtifactRef): boolean {
  return left.artifact_id === right.artifact_id
    && left.version === right.version
    && (right.content_sha256 === undefined || left.content_sha256 === right.content_sha256);
}

function validRef(value: ArtifactRef | undefined): boolean {
  return Boolean(value && /^[0-9a-f-]{36}$/i.test(value.artifact_id) && Number.isInteger(value.version) && value.version > 0);
}

export function validateFundamentalsLock(lock: FundamentalsLock, expected: {
  runId: string;
  dataCutoff: string;
  researchFinalManifest: ArtifactRef;
}): string[] {
  const errors: string[] = [];
  if (lock.lock_type !== "FUNDAMENTALS_LOCK") errors.push("lock_type must be FUNDAMENTALS_LOCK");
  if (lock.run_id !== expected.runId) errors.push("FUNDAMENTALS_LOCK run_id mismatch");
  if (lock.data_cutoff !== expected.dataCutoff) errors.push("FUNDAMENTALS_LOCK DATA_CUTOFF mismatch");
  if (!exactRef(lock.research_final_manifest, expected.researchFinalManifest)) errors.push("FUNDAMENTALS_LOCK research manifest mismatch");
  if (lock.red_team_status !== "COMPLETE") errors.push("Fundamental Red Team must be COMPLETE before lock");
  if (!validRef(lock.artifact) || !validRef(lock.taxonomy_record) || !validRef(lock.business_description_artifact) || !validRef(lock.scoring_inputs_artifact)) {
    errors.push("FUNDAMENTALS_LOCK contains an invalid required artifact reference");
  }
  if (!Array.isArray(lock.evidence_lineage) || lock.evidence_lineage.length === 0 || lock.evidence_lineage.some((item) => !validRef(item))) {
    errors.push("FUNDAMENTALS_LOCK requires exact evidence lineage");
  }
  return errors;
}

export function validateValuationLock(lock: ValuationLock, expected: {
  runId: string;
  dataCutoff: string;
  fundamentalsLock: ArtifactRef;
}): string[] {
  const errors: string[] = [];
  if (lock.lock_type !== "VALUATION_LOCK") errors.push("lock_type must be VALUATION_LOCK");
  if (lock.run_id !== expected.runId) errors.push("VALUATION_LOCK run_id mismatch");
  if (lock.data_cutoff !== expected.dataCutoff) errors.push("VALUATION_LOCK DATA_CUTOFF mismatch");
  if (!exactRef(lock.fundamentals_lock, expected.fundamentalsLock)) errors.push("VALUATION_LOCK must consume the exact current FUNDAMENTALS_LOCK");
  if (!Array.isArray(lock.evidence_lineage) || lock.evidence_lineage.length === 0 || lock.evidence_lineage.some((item) => !validRef(item))) {
    errors.push("VALUATION_LOCK requires the authoritative Evidence Ledger lineage");
  }
  for (const required of [lock.artifact, lock.calculation_ledger, lock.assumption_register, lock.valuation_artifact, lock.full_precision_inputs_artifact]) {
    if (!validRef(required)) errors.push("VALUATION_LOCK contains an invalid required artifact reference");
  }
  return errors;
}

export function validateDeepDiveCheckpoint(
  manifest: unknown,
  context: StageManifestValidationContext,
  phase: "FUNDAMENTALS" | "VALUATION",
  lockArtifactType: "FUNDAMENTALS_LOCK" | "VALUATION_LOCK",
): StageManifestValidationResult {
  const result = validateStageManifest(manifest, context);
  if (!result.ok) return result;
  const value = result.manifest as Record<string, unknown>;
  if (value.manifest_kind !== "CHECKPOINT") return { ok: false, stage: "contract", errors: [`${phase} must emit CHECKPOINT manifest`] };
  if (result.downstreamAdmission) return { ok: false, stage: "contract", errors: [`${phase} CHECKPOINT cannot admit Integration`] };
  const outputs = Array.isArray(value.output_artifacts) ? value.output_artifacts : [];
  const hasLock = outputs.some((item) => {
    if (typeof item !== "object" || item === null) return false;
    const output = item as Record<string, unknown>;
    return output.artifact_type === lockArtifactType && output.authority_class === "CHECKPOINT_STAGE_OUTPUT";
  });
  if (!hasLock) return { ok: false, stage: "contract", errors: [`${phase} CHECKPOINT is missing ${lockArtifactType}`] };
  return result;
}

export type LimitedReopenRequest = {
  material: boolean;
  reason: string;
  affectedScopes: string[];
  priorFundamentalsLock: ArtifactRef;
  priorValuationLock?: ArtifactRef | null;
};

export type LimitedReopenPlan = {
  action: "REOPEN_FUNDAMENTALS_LIMITED";
  reason: string;
  affectedScopes: string[];
  preserve: ArtifactRef[];
  invalidate: Array<"VALUATION_ELIGIBILITY" | "CERTIFICATION_ELIGIBILITY" | "INTEGRATION_ELIGIBILITY">;
  requireNewFundamentalsLock: true;
};

export function planLimitedFundamentalsReopen(request: LimitedReopenRequest): LimitedReopenPlan {
  if (!request.material) throw new Error("Automatic reopen requires a material contradiction");
  if (request.reason.trim().length === 0) throw new Error("Automatic reopen requires a persisted reason");
  const scopes = [...new Set(request.affectedScopes.map((scope) => scope.trim()).filter(Boolean))];
  if (scopes.length === 0) throw new Error("Automatic reopen requires exact affected scope");
  return {
    action: "REOPEN_FUNDAMENTALS_LIMITED",
    reason: request.reason,
    affectedScopes: scopes,
    preserve: [request.priorFundamentalsLock, ...(request.priorValuationLock ? [request.priorValuationLock] : [])],
    invalidate: ["VALUATION_ELIGIBILITY", "CERTIFICATION_ELIGIBILITY", "INTEGRATION_ELIGIBILITY"],
    requireNewFundamentalsLock: true,
  };
}

export type RefreshClass = "PRICE_ONLY_DELTA" | "ROUTINE_FUNDAMENTAL_DELTA" | "FULL_REFRESH_REQUIRED";

export function v2RefreshRoute(refreshClass: RefreshClass): {
  research: "MINIMAL_REVALIDATION" | "TARGETED_DELTA" | "FULL";
  fundamentals: "REVALIDATE_PRIOR_LOCK" | "REOPEN_AFFECTED_BLOCKS" | "FULL";
  valuation: true;
  certification: true;
  integration: true;
  oqsMayChangeWithoutFundamentalReopen: false;
} {
  if (refreshClass === "PRICE_ONLY_DELTA") return {
    research: "MINIMAL_REVALIDATION",
    fundamentals: "REVALIDATE_PRIOR_LOCK",
    valuation: true,
    certification: true,
    integration: true,
    oqsMayChangeWithoutFundamentalReopen: false,
  };
  if (refreshClass === "ROUTINE_FUNDAMENTAL_DELTA") return {
    research: "TARGETED_DELTA",
    fundamentals: "REOPEN_AFFECTED_BLOCKS",
    valuation: true,
    certification: true,
    integration: true,
    oqsMayChangeWithoutFundamentalReopen: false,
  };
  return {
    research: "FULL",
    fundamentals: "FULL",
    valuation: true,
    certification: true,
    integration: true,
    oqsMayChangeWithoutFundamentalReopen: false,
  };
}

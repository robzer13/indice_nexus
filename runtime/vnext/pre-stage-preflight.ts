import type { RunStatus, StageLifecycle } from "./state-machine";

export const PREFLIGHT_SCOPES = [
  "RESEARCH",
  "DEEP_DIVE",
  "FUNDAMENTALS",
  "VALUATION",
  "CERTIFICATION",
  "INTEGRATION",
] as const;

export type PreflightScope = (typeof PREFLIGHT_SCOPES)[number];

export type CanonicalMode =
  | "DISCOVER"
  | "ANALYZE"
  | "DISCOVER + ANALYZE"
  | "REFRESH"
  | "ACTIVATION CHECK";

export type RunType = "INITIAL" | "REFRESH" | null;

export interface PreflightIdentity {
  issuerId: string | null;
  securityId: string | null;
  dossierId: string | null;
}

export interface PreflightRunRecord {
  runId: string;
  runStatus: RunStatus;
  currentStage: "RESEARCH" | "DEEP_DIVE" | "INTEGRATION" | null;
  canonicalMode: CanonicalMode;
  runType: RunType;
  dataCutoff: string;
  processVersion: string;
  pilotageContractVersion: string;
  contractSetSha256: string;
  stateVersion: number;
  baselineSnapshotId: string | null;
  identity: PreflightIdentity;
}

export interface PreflightStageRecord {
  runId: string;
  stageCode: "RESEARCH" | "DEEP_DIVE" | "INTEGRATION";
  stageRevision: number;
  stageContractName: string;
  stageContractVersion: string;
  stageContractSha256: string;
  lifecycleStatus: StageLifecycle;
  handoffGateName:
    | "READY_FOR_DEEP_DIVE"
    | "READY_FOR_INTEGRATION"
    | "READY_TO_PUBLISH";
  handoffGateState: "NOT_EVALUATED" | "YES" | "NO";
  activeManifestArtifactId: string | null;
  activeManifestVersion: number | null;
  activeManifestKind: "CHECKPOINT" | "FINAL" | null;
  stateVersion: number;
  blockerCount: number;
}

export interface ExpectedContractPin {
  name: string;
  version: string;
  contentSha256: string;
}

export interface ResolvedContractPin extends ExpectedContractPin {
  durableLocatorAvailable: boolean;
}

export interface ExpectedArtifactRef {
  artifactId: string;
  version: number;
  runId: string;
  artifactType: string;
  contentSha256: string;
  expectedAuthorityState:
    | "AUTHORITATIVE"
    | "CHECKPOINT"
    | "SUPERSEDED"
    | "NON_AUTHORITATIVE";
}

export interface ResolvedArtifactRef extends ExpectedArtifactRef {
  artifactStatus: "SEALED" | "INVALIDATED";
  authorityState:
    | "AUTHORITATIVE"
    | "CHECKPOINT"
    | "SUPERSEDED"
    | "NON_AUTHORITATIVE";
  availabilityState: "AVAILABLE" | "WITHDRAWN" | "MISSING";
  durableLocatorAvailable: boolean;
}

export interface ExpectedUpstreamManifest {
  artifactId: string;
  version: number;
  runId: string;
  stageCode: "RESEARCH" | "DEEP_DIVE";
  contentSha256: string;
  handoffGateName: "READY_FOR_DEEP_DIVE" | "READY_FOR_INTEGRATION";
}

export interface ResolvedUpstreamManifest extends ExpectedUpstreamManifest {
  artifactStatus: "SEALED" | "INVALIDATED";
  authorityState:
    | "AUTHORITATIVE"
    | "CHECKPOINT"
    | "SUPERSEDED"
    | "NON_AUTHORITATIVE";
  availabilityState: "AVAILABLE" | "WITHDRAWN" | "MISSING";
  manifestKind: "CHECKPOINT" | "FINAL";
  stageLifecycleStatus: StageLifecycle;
  handoffGateState: "NOT_EVALUATED" | "YES" | "NO";
  durableLocatorAvailable: boolean;
}

export interface PreStagePreflightRequest {
  scope: PreflightScope;

  expectedRunId: string;
  expectedRunStateVersion: number;
  expectedStageStateVersion: number;
  expectedIdentity: PreflightIdentity;
  expectedCanonicalMode: CanonicalMode;
  expectedRunType: RunType;
  expectedDataCutoff: string;
  expectedProcessVersion: string;
  expectedPilotageContractVersion: string;
  expectedContractSetSha256: string;
  expectedStageContractName: string;
  expectedStageContractVersion: string;
  expectedStageContractSha256: string;

  run: PreflightRunRecord;
  stage: PreflightStageRecord;

  expectedContracts: readonly ExpectedContractPin[];
  resolvedContracts: readonly ResolvedContractPin[];

  expectedArtifacts: readonly ExpectedArtifactRef[];
  resolvedArtifacts: readonly ResolvedArtifactRef[];

  expectedUpstreamManifest: ExpectedUpstreamManifest | null;
  resolvedUpstreamManifest: ResolvedUpstreamManifest | null;

  priorAssuranceGatePassed: boolean;
  authorityStateCompatible: boolean;
  noBlockingExecutionDefect: boolean;
}

export type PreflightFailureCode =
  | "RUN_ID_MISMATCH"
  | "RUN_TERMINAL_OR_UNSTARTABLE"
  | "RUN_STATE_VERSION_STALE"
  | "STAGE_RUN_ID_MISMATCH"
  | "STAGE_STATE_VERSION_STALE"
  | "STAGE_CODE_MISMATCH"
  | "STAGE_LIFECYCLE_NOT_ADMISSIBLE"
  | "STAGE_CONTRACT_NAME_MISMATCH"
  | "STAGE_CONTRACT_VERSION_MISMATCH"
  | "STAGE_CONTRACT_HASH_MISMATCH"
  | "IDENTITY_ISSUER_MISMATCH"
  | "IDENTITY_SECURITY_MISMATCH"
  | "IDENTITY_SECURITY_REQUIRED"
  | "IDENTITY_DOSSIER_MISMATCH"
  | "CANONICAL_MODE_MISMATCH"
  | "RUN_TYPE_MISMATCH"
  | "DATA_CUTOFF_MISMATCH"
  | "PROCESS_VERSION_MISMATCH"
  | "PILOTAGE_CONTRACT_VERSION_MISMATCH"
  | "CONTRACT_SET_HASH_MISMATCH"
  | "CONTRACT_PIN_MISSING"
  | "CONTRACT_VERSION_MISMATCH"
  | "CONTRACT_HASH_MISMATCH"
  | "CONTRACT_LOCATOR_MISSING"
  | "BASELINE_SNAPSHOT_REQUIRED"
  | "PRIOR_ASSURANCE_GATE_FAILED"
  | "AUTHORITY_STATE_INCOMPATIBLE"
  | "BLOCKING_EXECUTION_DEFECT"
  | "UPSTREAM_MANIFEST_REQUIRED"
  | "UPSTREAM_MANIFEST_UNEXPECTED"
  | "UPSTREAM_MANIFEST_ID_MISMATCH"
  | "UPSTREAM_MANIFEST_VERSION_MISMATCH"
  | "UPSTREAM_MANIFEST_RUN_MISMATCH"
  | "UPSTREAM_MANIFEST_STAGE_MISMATCH"
  | "UPSTREAM_MANIFEST_HASH_MISMATCH"
  | "UPSTREAM_MANIFEST_NOT_FINAL"
  | "UPSTREAM_MANIFEST_NOT_AUTHORITATIVE"
  | "UPSTREAM_MANIFEST_NOT_AVAILABLE"
  | "UPSTREAM_MANIFEST_NOT_SEALED"
  | "UPSTREAM_MANIFEST_LOCATOR_MISSING"
  | "UPSTREAM_STAGE_NOT_COMPLETE"
  | "UPSTREAM_HANDOFF_GATE_MISMATCH"
  | "UPSTREAM_HANDOFF_NOT_YES"
  | "ARTIFACT_MISSING"
  | "ARTIFACT_VERSION_MISMATCH"
  | "ARTIFACT_RUN_MISMATCH"
  | "ARTIFACT_TYPE_MISMATCH"
  | "ARTIFACT_HASH_MISMATCH"
  | "ARTIFACT_NOT_SEALED"
  | "ARTIFACT_AUTHORITY_MISMATCH"
  | "ARTIFACT_NOT_AVAILABLE"
  | "ARTIFACT_LOCATOR_MISSING";

export interface PreflightFailure {
  code: PreflightFailureCode;
  detail: string;
}

export interface PreStagePreflightResult {
  status: "PASS" | "FAIL";
  scope: PreflightScope;
  failures: readonly PreflightFailure[];
}

const SHA256_RE = /^[0-9a-f]{64}$/;

function fail(
  failures: PreflightFailure[],
  code: PreflightFailureCode,
  detail: string,
): void {
  failures.push({ code, detail });
}

function expectedStageCode(
  scope: PreflightScope,
): "RESEARCH" | "DEEP_DIVE" | "INTEGRATION" {
  switch (scope) {
    case "RESEARCH":
      return "RESEARCH";
    case "INTEGRATION":
      return "INTEGRATION";
    default:
      return "DEEP_DIVE";
  }
}

function requiresSecurity(scope: PreflightScope): boolean {
  return scope !== "RESEARCH";
}

function admissibleLifecycle(
  scope: PreflightScope,
  status: StageLifecycle,
): boolean {
  if (
    scope === "FUNDAMENTALS" ||
    scope === "VALUATION" ||
    scope === "CERTIFICATION"
  ) {
    return status === "IN_PROGRESS";
  }

  return status === "NOT_STARTED" || status === "IN_PROGRESS";
}

function requiresBaseline(run: PreflightRunRecord): boolean {
  return (
    run.runType === "REFRESH" ||
    run.canonicalMode === "REFRESH" ||
    run.canonicalMode === "ACTIVATION CHECK"
  );
}

function requiresUpstreamManifest(scope: PreflightScope): boolean {
  return scope === "DEEP_DIVE" || scope === "INTEGRATION";
}

function validateRunAndStage(
  request: PreStagePreflightRequest,
  failures: PreflightFailure[],
): void {
  const run = request.run;
  const stage = request.stage;

  if (run.runId !== request.expectedRunId) {
    fail(failures, "RUN_ID_MISMATCH", "expected=" + request.expectedRunId + " actual=" + run.runId);
  }

  if (
    run.runStatus === "PUBLISHED" ||
    run.runStatus === "CANCELLED" ||
    run.runStatus === "READY_TO_PUBLISH"
  ) {
    fail(failures, "RUN_TERMINAL_OR_UNSTARTABLE", "run_status=" + run.runStatus);
  }

  if (run.stateVersion !== request.expectedRunStateVersion) {
    fail(
      failures,
      "RUN_STATE_VERSION_STALE",
      "expected=" + request.expectedRunStateVersion + " actual=" + run.stateVersion,
    );
  }

  if (stage.runId !== request.expectedRunId) {
    fail(failures, "STAGE_RUN_ID_MISMATCH", "expected=" + request.expectedRunId + " actual=" + stage.runId);
  }

  if (stage.stateVersion !== request.expectedStageStateVersion) {
    fail(
      failures,
      "STAGE_STATE_VERSION_STALE",
      "expected=" + request.expectedStageStateVersion + " actual=" + stage.stateVersion,
    );
  }

  const stageCode = expectedStageCode(request.scope);
  if (stage.stageCode !== stageCode) {
    fail(
      failures,
      "STAGE_CODE_MISMATCH",
      "scope=" + request.scope + " expected=" + stageCode + " actual=" + stage.stageCode,
    );
  }

  if (!admissibleLifecycle(request.scope, stage.lifecycleStatus)) {
    fail(
      failures,
      "STAGE_LIFECYCLE_NOT_ADMISSIBLE",
      "scope=" + request.scope + " lifecycle=" + stage.lifecycleStatus,
    );
  }

  if (stage.stageContractName !== request.expectedStageContractName) {
    fail(
      failures,
      "STAGE_CONTRACT_NAME_MISMATCH",
      "expected=" + request.expectedStageContractName + " actual=" + stage.stageContractName,
    );
  }

  if (stage.stageContractVersion !== request.expectedStageContractVersion) {
    fail(
      failures,
      "STAGE_CONTRACT_VERSION_MISMATCH",
      "expected=" + request.expectedStageContractVersion + " actual=" + stage.stageContractVersion,
    );
  }

  if (
    !SHA256_RE.test(stage.stageContractSha256) ||
    stage.stageContractSha256 !== request.expectedStageContractSha256
  ) {
    fail(
      failures,
      "STAGE_CONTRACT_HASH_MISMATCH",
      "expected=" + request.expectedStageContractSha256 + " actual=" + stage.stageContractSha256,
    );
  }

  if (run.identity.issuerId !== request.expectedIdentity.issuerId) {
    fail(
      failures,
      "IDENTITY_ISSUER_MISMATCH",
      "expected=" + String(request.expectedIdentity.issuerId) + " actual=" + String(run.identity.issuerId),
    );
  }

  if (requiresSecurity(request.scope) && !run.identity.securityId) {
    fail(failures, "IDENTITY_SECURITY_REQUIRED", "scope=" + request.scope);
  } else if (run.identity.securityId !== request.expectedIdentity.securityId) {
    fail(
      failures,
      "IDENTITY_SECURITY_MISMATCH",
      "expected=" + String(request.expectedIdentity.securityId) + " actual=" + String(run.identity.securityId),
    );
  }

  if (run.identity.dossierId !== request.expectedIdentity.dossierId) {
    fail(
      failures,
      "IDENTITY_DOSSIER_MISMATCH",
      "expected=" + String(request.expectedIdentity.dossierId) + " actual=" + String(run.identity.dossierId),
    );
  }

  if (run.canonicalMode !== request.expectedCanonicalMode) {
    fail(
      failures,
      "CANONICAL_MODE_MISMATCH",
      "expected=" + request.expectedCanonicalMode + " actual=" + run.canonicalMode,
    );
  }

  if (run.runType !== request.expectedRunType) {
    fail(
      failures,
      "RUN_TYPE_MISMATCH",
      "expected=" + String(request.expectedRunType) + " actual=" + String(run.runType),
    );
  }

  if (run.dataCutoff !== request.expectedDataCutoff) {
    fail(
      failures,
      "DATA_CUTOFF_MISMATCH",
      "expected=" + request.expectedDataCutoff + " actual=" + run.dataCutoff,
    );
  }

  if (run.processVersion !== request.expectedProcessVersion) {
    fail(
      failures,
      "PROCESS_VERSION_MISMATCH",
      "expected=" + request.expectedProcessVersion + " actual=" + run.processVersion,
    );
  }

  if (run.pilotageContractVersion !== request.expectedPilotageContractVersion) {
    fail(
      failures,
      "PILOTAGE_CONTRACT_VERSION_MISMATCH",
      "expected=" + request.expectedPilotageContractVersion + " actual=" + run.pilotageContractVersion,
    );
  }

  if (
    !SHA256_RE.test(run.contractSetSha256) ||
    run.contractSetSha256 !== request.expectedContractSetSha256
  ) {
    fail(
      failures,
      "CONTRACT_SET_HASH_MISMATCH",
      "expected=" + request.expectedContractSetSha256 + " actual=" + run.contractSetSha256,
    );
  }

  if (requiresBaseline(run) && !run.baselineSnapshotId) {
    fail(
      failures,
      "BASELINE_SNAPSHOT_REQUIRED",
      "canonical_mode=" + run.canonicalMode + " run_type=" + String(run.runType),
    );
  }

  if (!request.priorAssuranceGatePassed) {
    fail(failures, "PRIOR_ASSURANCE_GATE_FAILED", "scope=" + request.scope);
  }

  if (!request.authorityStateCompatible) {
    fail(failures, "AUTHORITY_STATE_INCOMPATIBLE", "scope=" + request.scope);
  }

  if (!request.noBlockingExecutionDefect || stage.blockerCount > 0) {
    fail(
      failures,
      "BLOCKING_EXECUTION_DEFECT",
      "declared_clear=" + request.noBlockingExecutionDefect + " blocker_count=" + stage.blockerCount,
    );
  }
}

function validateContracts(
  request: PreStagePreflightRequest,
  failures: PreflightFailure[],
): void {
  for (const expected of request.expectedContracts) {
    const actual = request.resolvedContracts.find(
      (candidate) => candidate.name === expected.name,
    );

    if (!actual) {
      fail(failures, "CONTRACT_PIN_MISSING", "contract=" + expected.name);
      continue;
    }

    if (actual.version !== expected.version) {
      fail(
        failures,
        "CONTRACT_VERSION_MISMATCH",
        "contract=" + expected.name + " expected=" + expected.version + " actual=" + actual.version,
      );
    }

    if (
      !SHA256_RE.test(actual.contentSha256) ||
      actual.contentSha256 !== expected.contentSha256
    ) {
      fail(failures, "CONTRACT_HASH_MISMATCH", "contract=" + expected.name);
    }

    if (!actual.durableLocatorAvailable) {
      fail(failures, "CONTRACT_LOCATOR_MISSING", "contract=" + expected.name);
    }
  }
}

function validateUpstreamManifest(
  request: PreStagePreflightRequest,
  failures: PreflightFailure[],
): void {
  const required = requiresUpstreamManifest(request.scope);
  const expected = request.expectedUpstreamManifest;
  const actual = request.resolvedUpstreamManifest;

  if (required && (!expected || !actual)) {
    fail(failures, "UPSTREAM_MANIFEST_REQUIRED", "scope=" + request.scope);
    return;
  }

  if (!required && (expected || actual)) {
    fail(failures, "UPSTREAM_MANIFEST_UNEXPECTED", "scope=" + request.scope);
    return;
  }

  if (!expected || !actual) {
    return;
  }

  if (actual.artifactId !== expected.artifactId) {
    fail(
      failures,
      "UPSTREAM_MANIFEST_ID_MISMATCH",
      "expected=" + expected.artifactId + " actual=" + actual.artifactId,
    );
  }

  if (actual.version !== expected.version) {
    fail(
      failures,
      "UPSTREAM_MANIFEST_VERSION_MISMATCH",
      "expected=" + expected.version + " actual=" + actual.version,
    );
  }

  if (actual.runId !== request.expectedRunId || actual.runId !== expected.runId) {
    fail(
      failures,
      "UPSTREAM_MANIFEST_RUN_MISMATCH",
      "expected=" + request.expectedRunId + " actual=" + actual.runId,
    );
  }

  if (actual.stageCode !== expected.stageCode) {
    fail(
      failures,
      "UPSTREAM_MANIFEST_STAGE_MISMATCH",
      "expected=" + expected.stageCode + " actual=" + actual.stageCode,
    );
  }

  if (
    !SHA256_RE.test(actual.contentSha256) ||
    actual.contentSha256 !== expected.contentSha256
  ) {
    fail(failures, "UPSTREAM_MANIFEST_HASH_MISMATCH", "artifact=" + actual.artifactId);
  }

  if (actual.manifestKind !== "FINAL") {
    fail(failures, "UPSTREAM_MANIFEST_NOT_FINAL", "kind=" + actual.manifestKind);
  }

  if (actual.authorityState !== "AUTHORITATIVE") {
    fail(
      failures,
      "UPSTREAM_MANIFEST_NOT_AUTHORITATIVE",
      "authority=" + actual.authorityState,
    );
  }

  if (actual.availabilityState !== "AVAILABLE") {
    fail(
      failures,
      "UPSTREAM_MANIFEST_NOT_AVAILABLE",
      "availability=" + actual.availabilityState,
    );
  }

  if (actual.artifactStatus !== "SEALED") {
    fail(
      failures,
      "UPSTREAM_MANIFEST_NOT_SEALED",
      "status=" + actual.artifactStatus,
    );
  }

  if (!actual.durableLocatorAvailable) {
    fail(failures, "UPSTREAM_MANIFEST_LOCATOR_MISSING", "artifact=" + actual.artifactId);
  }

  if (actual.stageLifecycleStatus !== "COMPLETE") {
    fail(
      failures,
      "UPSTREAM_STAGE_NOT_COMPLETE",
      "status=" + actual.stageLifecycleStatus,
    );
  }

  if (actual.handoffGateName !== expected.handoffGateName) {
    fail(
      failures,
      "UPSTREAM_HANDOFF_GATE_MISMATCH",
      "expected=" + expected.handoffGateName + " actual=" + actual.handoffGateName,
    );
  }

  if (actual.handoffGateState !== "YES") {
    fail(
      failures,
      "UPSTREAM_HANDOFF_NOT_YES",
      "state=" + actual.handoffGateState,
    );
  }
}

function validateArtifacts(
  request: PreStagePreflightRequest,
  failures: PreflightFailure[],
): void {
  for (const expected of request.expectedArtifacts) {
    const actual = request.resolvedArtifacts.find(
      (candidate) => candidate.artifactId === expected.artifactId,
    );

    if (!actual) {
      fail(failures, "ARTIFACT_MISSING", "artifact=" + expected.artifactId);
      continue;
    }

    if (actual.version !== expected.version) {
      fail(
        failures,
        "ARTIFACT_VERSION_MISMATCH",
        "artifact=" + expected.artifactId + " expected=" + expected.version + " actual=" + actual.version,
      );
    }

    if (actual.runId !== expected.runId || actual.runId !== request.expectedRunId) {
      fail(
        failures,
        "ARTIFACT_RUN_MISMATCH",
        "artifact=" + expected.artifactId + " expected_run=" + request.expectedRunId + " actual_run=" + actual.runId,
      );
    }

    if (actual.artifactType !== expected.artifactType) {
      fail(
        failures,
        "ARTIFACT_TYPE_MISMATCH",
        "artifact=" + expected.artifactId + " expected=" + expected.artifactType + " actual=" + actual.artifactType,
      );
    }

    if (
      !SHA256_RE.test(actual.contentSha256) ||
      actual.contentSha256 !== expected.contentSha256
    ) {
      fail(failures, "ARTIFACT_HASH_MISMATCH", "artifact=" + expected.artifactId);
    }

    if (actual.artifactStatus !== "SEALED") {
      fail(
        failures,
        "ARTIFACT_NOT_SEALED",
        "artifact=" + expected.artifactId + " status=" + actual.artifactStatus,
      );
    }

    if (actual.authorityState !== expected.expectedAuthorityState) {
      fail(
        failures,
        "ARTIFACT_AUTHORITY_MISMATCH",
        "artifact=" + expected.artifactId + " expected=" + expected.expectedAuthorityState + " actual=" + actual.authorityState,
      );
    }

    if (actual.availabilityState !== "AVAILABLE") {
      fail(
        failures,
        "ARTIFACT_NOT_AVAILABLE",
        "artifact=" + expected.artifactId + " availability=" + actual.availabilityState,
      );
    }

    if (!actual.durableLocatorAvailable) {
      fail(failures, "ARTIFACT_LOCATOR_MISSING", "artifact=" + expected.artifactId);
    }
  }
}

/**
 * Gate 9 deterministic admission check.
 *
 * Read-only by construction. It does not create, start, resume, complete,
 * reopen, or publish any run/stage.
 */
export function runPreStagePreflight(
  request: PreStagePreflightRequest,
): PreStagePreflightResult {
  const failures: PreflightFailure[] = [];

  validateRunAndStage(request, failures);
  validateContracts(request, failures);
  validateUpstreamManifest(request, failures);
  validateArtifacts(request, failures);

  return {
    status: failures.length === 0 ? "PASS" : "FAIL",
    scope: request.scope,
    failures,
  };
}

export function assertPreStagePreflightPass(
  request: PreStagePreflightRequest,
): void {
  const result = runPreStagePreflight(request);

  if (result.status === "FAIL") {
    const detail = result.failures
      .map((failure) => failure.code + ":" + failure.detail)
      .join("|");
    throw new Error("VNEXT_PRE_STAGE_PREFLIGHT_FAIL: " + detail);
  }
}

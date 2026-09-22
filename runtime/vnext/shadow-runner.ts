export const AUDIT_CORPUS_VNEXT_V1_ID =
  "AUDIT_CORPUS_VNEXT_V1" as const;

export const AUDIT_CORPUS_VNEXT_V1_SIZE = 33 as const;

const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

const DIVERGENCE_KEYS = [
  "CHANGE_ID",
  "V2_STATE",
  "VNEXT_STATE",
  "CAUSE",
  "MODULE",
  "EVIDENCE",
  "ECONOMIC_INTERPRETATION",
] as const;

export interface ProductionFingerprint {
  research_dossiers: number;
  research_snapshots: number;
  orotitan_runs: number;
  orotitan_artifacts: number;
  dossier_pointer_sha256: string;
  snapshots_sha256: string;
}

export interface AuditCorpusMember {
  display_name: string;
  issuer_id: string;
  dossier_id: string;
  security_id: string;
  v2_snapshot_id: string;
  v2_source_run_id: string;
  ticker: string;
  exchange: string;
  data_cutoff: string;
  v2_payload_sha256: string;
}

export interface AuditCorpusManifest {
  format: typeof AUDIT_CORPUS_VNEXT_V1_ID;
  version: "1.0.0";
  status: "PINNED_GATE17_INPUT";
  pinned_at_date: string;
  source: {
    production_project_ref: string;
    production_project_name: string;
    production_freeze_tag: string;
    production_freeze_commit_sha: string;
    vnext_shadow_project_ref: string;
  };
  expected_member_count: typeof AUDIT_CORPUS_VNEXT_V1_SIZE;
  production_before_fingerprint: ProductionFingerprint;
  members: readonly AuditCorpusMember[];
}

export interface VnextShadowExecutionInput {
  corpusId: typeof AUDIT_CORPUS_VNEXT_V1_ID;
  memberIndex: number;
  displayName: string;
  issuerId: string;
  dossierId: string;
  securityId: string;
  sourceRunId: string;
  ticker: string;
  exchange: string;
  dataCutoff: string;
}

export interface VnextShadowExecutionResult<TState = unknown> {
  executionId: string;
  state: TState;
  evidenceRefs: readonly string[];
  moduleRefs: readonly string[];
}

export interface V2Baseline<TState = unknown> {
  snapshotId: string;
  payloadSha256: string;
  state: TState;
}

export interface ShadowDivergence {
  CHANGE_ID: string;
  V2_STATE: unknown;
  VNEXT_STATE: unknown;
  CAUSE: string;
  MODULE: string;
  EVIDENCE: readonly string[];
  ECONOMIC_INTERPRETATION: string;
}

export interface ShadowComparisonInput<
  TV2State = unknown,
  TVnextState = unknown,
> {
  member: AuditCorpusMember;
  v2: V2Baseline<TV2State>;
  vnext: VnextShadowExecutionResult<TVnextState>;
}

export interface ShadowRunnerDependencies<
  TV2State = unknown,
  TVnextState = unknown,
> {
  executeVNext(
    input: VnextShadowExecutionInput,
  ): Promise<VnextShadowExecutionResult<TVnextState>>;
  loadV2Baseline(
    member: AuditCorpusMember,
  ): Promise<V2Baseline<TV2State>>;
  compare(
    input: ShadowComparisonInput<TV2State, TVnextState>,
  ): Promise<readonly ShadowDivergence[]>;
}

export interface ShadowMemberCompleteResult {
  status: "COMPLETE";
  displayName: string;
  issuerId: string;
  v2SnapshotId: string;
  v2SourceRunId: string;
  dataCutoff: string;
  vnextExecutionId: string;
  changeCount: number;
  changes: readonly ShadowDivergence[];
}

export interface ShadowMemberFailedResult {
  status: "FAILED";
  displayName: string;
  issuerId: string;
  v2SnapshotId: string;
  v2SourceRunId: string;
  dataCutoff: string;
  errorCode: string;
  errorMessage: string;
}

export type ShadowMemberResult =
  | ShadowMemberCompleteResult
  | ShadowMemberFailedResult;

export interface ShadowCorpusRunResult {
  corpusId: typeof AUDIT_CORPUS_VNEXT_V1_ID;
  expectedMemberCount: typeof AUDIT_CORPUS_VNEXT_V1_SIZE;
  attemptedMemberCount: number;
  completedMemberCount: number;
  failedMemberCount: number;
  status: "COMPLETE" | "FAILED";
  members: readonly ShadowMemberResult[];
}

export interface Gate17ExitAssessment {
  gate: 17;
  corpusId: typeof AUDIT_CORPUS_VNEXT_V1_ID;
  status: "PASS" | "FAIL";
  expectedMemberCount: typeof AUDIT_CORPUS_VNEXT_V1_SIZE;
  attemptedMemberCount: number;
  completedMemberCount: number;
  failedMemberCount: number;
  productionUnchanged: boolean;
}

function assertNonBlank(value: string, errorCode: string): void {
  if (value.trim().length === 0) {
    throw new Error(errorCode);
  }
}

function assertSha256(value: string, errorCode: string): void {
  if (!SHA256_PATTERN.test(value)) {
    throw new Error(errorCode);
  }
}

function assertDate(value: string, errorCode: string): void {
  if (!DATE_PATTERN.test(value)) {
    throw new Error(errorCode);
  }
}

function assertNonNegativeInteger(
  value: number,
  errorCode: string,
): void {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(errorCode);
  }
}

function assertFingerprint(
  fingerprint: ProductionFingerprint,
  prefix: string,
): void {
  assertNonNegativeInteger(
    fingerprint.research_dossiers,
    `${prefix}_RESEARCH_DOSSIERS_INVALID`,
  );
  assertNonNegativeInteger(
    fingerprint.research_snapshots,
    `${prefix}_RESEARCH_SNAPSHOTS_INVALID`,
  );
  assertNonNegativeInteger(
    fingerprint.orotitan_runs,
    `${prefix}_OROTITAN_RUNS_INVALID`,
  );
  assertNonNegativeInteger(
    fingerprint.orotitan_artifacts,
    `${prefix}_OROTITAN_ARTIFACTS_INVALID`,
  );
  assertSha256(
    fingerprint.dossier_pointer_sha256,
    `${prefix}_DOSSIER_POINTER_SHA256_INVALID`,
  );
  assertSha256(
    fingerprint.snapshots_sha256,
    `${prefix}_SNAPSHOTS_SHA256_INVALID`,
  );
}

function assertUnique(
  values: readonly string[],
  errorCode: string,
): void {
  if (new Set(values).size !== values.length) {
    throw new Error(errorCode);
  }
}

export function assertValidAuditCorpus(
  manifest: AuditCorpusManifest,
): void {
  if (manifest.format !== AUDIT_CORPUS_VNEXT_V1_ID) {
    throw new Error("VNEXT_SHADOW_CORPUS_ID_MISMATCH");
  }

  if (manifest.version !== "1.0.0") {
    throw new Error("VNEXT_SHADOW_CORPUS_VERSION_MISMATCH");
  }

  if (manifest.status !== "PINNED_GATE17_INPUT") {
    throw new Error("VNEXT_SHADOW_CORPUS_STATUS_INVALID");
  }

  if (
    manifest.expected_member_count !==
    AUDIT_CORPUS_VNEXT_V1_SIZE
  ) {
    throw new Error("VNEXT_SHADOW_CORPUS_EXPECTED_SIZE_MISMATCH");
  }

  if (manifest.members.length !== AUDIT_CORPUS_VNEXT_V1_SIZE) {
    throw new Error("VNEXT_SHADOW_CORPUS_MEMBER_COUNT_MISMATCH");
  }

  assertDate(
    manifest.pinned_at_date,
    "VNEXT_SHADOW_CORPUS_PIN_DATE_INVALID",
  );
  assertNonBlank(
    manifest.source.production_project_ref,
    "VNEXT_SHADOW_CORPUS_PRODUCTION_PROJECT_REQUIRED",
  );
  assertNonBlank(
    manifest.source.production_freeze_tag,
    "VNEXT_SHADOW_CORPUS_PRODUCTION_FREEZE_TAG_REQUIRED",
  );
  assertNonBlank(
    manifest.source.production_freeze_commit_sha,
    "VNEXT_SHADOW_CORPUS_PRODUCTION_FREEZE_SHA_REQUIRED",
  );
  assertNonBlank(
    manifest.source.vnext_shadow_project_ref,
    "VNEXT_SHADOW_CORPUS_SHADOW_PROJECT_REQUIRED",
  );

  if (
    manifest.source.production_project_ref ===
    manifest.source.vnext_shadow_project_ref
  ) {
    throw new Error("VNEXT_SHADOW_CORPUS_ENVIRONMENT_COLLISION");
  }

  assertFingerprint(
    manifest.production_before_fingerprint,
    "VNEXT_SHADOW_CORPUS_BASELINE",
  );

  for (const member of manifest.members) {
    assertNonBlank(
      member.display_name,
      "VNEXT_SHADOW_CORPUS_DISPLAY_NAME_REQUIRED",
    );
    assertNonBlank(
      member.issuer_id,
      "VNEXT_SHADOW_CORPUS_ISSUER_ID_REQUIRED",
    );
    assertNonBlank(
      member.dossier_id,
      "VNEXT_SHADOW_CORPUS_DOSSIER_ID_REQUIRED",
    );
    assertNonBlank(
      member.security_id,
      "VNEXT_SHADOW_CORPUS_SECURITY_ID_REQUIRED",
    );
    assertNonBlank(
      member.v2_snapshot_id,
      "VNEXT_SHADOW_CORPUS_SNAPSHOT_ID_REQUIRED",
    );
    assertNonBlank(
      member.v2_source_run_id,
      "VNEXT_SHADOW_CORPUS_SOURCE_RUN_ID_REQUIRED",
    );
    assertNonBlank(
      member.ticker,
      "VNEXT_SHADOW_CORPUS_TICKER_REQUIRED",
    );
    assertNonBlank(
      member.exchange,
      "VNEXT_SHADOW_CORPUS_EXCHANGE_REQUIRED",
    );
    assertDate(
      member.data_cutoff,
      "VNEXT_SHADOW_CORPUS_DATA_CUTOFF_INVALID",
    );
    assertSha256(
      member.v2_payload_sha256,
      "VNEXT_SHADOW_CORPUS_V2_PAYLOAD_SHA256_INVALID",
    );
  }

  assertUnique(
    manifest.members.map((member) => member.issuer_id),
    "VNEXT_SHADOW_CORPUS_DUPLICATE_ISSUER",
  );
  assertUnique(
    manifest.members.map((member) => member.dossier_id),
    "VNEXT_SHADOW_CORPUS_DUPLICATE_DOSSIER",
  );
  assertUnique(
    manifest.members.map((member) => member.security_id),
    "VNEXT_SHADOW_CORPUS_DUPLICATE_SECURITY",
  );
  assertUnique(
    manifest.members.map((member) => member.v2_snapshot_id),
    "VNEXT_SHADOW_CORPUS_DUPLICATE_SNAPSHOT",
  );
  assertUnique(
    manifest.members.map((member) => member.v2_source_run_id),
    "VNEXT_SHADOW_CORPUS_DUPLICATE_SOURCE_RUN",
  );
}

function buildExecutionInput(
  member: AuditCorpusMember,
  memberIndex: number,
): VnextShadowExecutionInput {
  return {
    corpusId: AUDIT_CORPUS_VNEXT_V1_ID,
    memberIndex,
    displayName: member.display_name,
    issuerId: member.issuer_id,
    dossierId: member.dossier_id,
    securityId: member.security_id,
    sourceRunId: member.v2_source_run_id,
    ticker: member.ticker,
    exchange: member.exchange,
    dataCutoff: member.data_cutoff,
  };
}

function assertVnextExecutionResult(
  result: VnextShadowExecutionResult,
): void {
  assertNonBlank(
    result.executionId,
    "VNEXT_SHADOW_EXECUTION_ID_REQUIRED",
  );

  for (const ref of result.evidenceRefs) {
    assertNonBlank(
      ref,
      "VNEXT_SHADOW_EXECUTION_EVIDENCE_REF_REQUIRED",
    );
  }

  for (const ref of result.moduleRefs) {
    assertNonBlank(
      ref,
      "VNEXT_SHADOW_EXECUTION_MODULE_REF_REQUIRED",
    );
  }
}

function assertV2Baseline(
  member: AuditCorpusMember,
  baseline: V2Baseline,
): void {
  if (baseline.snapshotId !== member.v2_snapshot_id) {
    throw new Error("VNEXT_SHADOW_V2_SNAPSHOT_ID_MISMATCH");
  }

  if (baseline.payloadSha256 !== member.v2_payload_sha256) {
    throw new Error("VNEXT_SHADOW_V2_PAYLOAD_HASH_MISMATCH");
  }
}

function assertDivergence(
  change: ShadowDivergence,
): void {
  const keys = Object.keys(change).sort();
  const expectedKeys = [...DIVERGENCE_KEYS].sort();

  if (
    keys.length !== expectedKeys.length ||
    keys.some((key, index) => key !== expectedKeys[index])
  ) {
    throw new Error("VNEXT_SHADOW_CHANGE_FIELD_SET_INVALID");
  }

  assertNonBlank(
    change.CHANGE_ID,
    "VNEXT_SHADOW_CHANGE_ID_REQUIRED",
  );
  assertNonBlank(
    change.CAUSE,
    "VNEXT_SHADOW_CHANGE_CAUSE_REQUIRED",
  );
  assertNonBlank(
    change.MODULE,
    "VNEXT_SHADOW_CHANGE_MODULE_REQUIRED",
  );
  assertNonBlank(
    change.ECONOMIC_INTERPRETATION,
    "VNEXT_SHADOW_CHANGE_ECONOMIC_INTERPRETATION_REQUIRED",
  );

  if (change.EVIDENCE.length === 0) {
    throw new Error("VNEXT_SHADOW_CHANGE_EVIDENCE_REQUIRED");
  }

  for (const evidence of change.EVIDENCE) {
    assertNonBlank(
      evidence,
      "VNEXT_SHADOW_CHANGE_EVIDENCE_REF_REQUIRED",
    );
  }
}

function toFailure(error: unknown): {
  errorCode: string;
  errorMessage: string;
} {
  if (error instanceof Error) {
    const code = /^VNEXT_[A-Z0-9_]+$/.test(error.message)
      ? error.message
      : "VNEXT_SHADOW_MEMBER_EXECUTION_FAILED";

    return {
      errorCode: code,
      errorMessage: error.message,
    };
  }

  return {
    errorCode: "VNEXT_SHADOW_MEMBER_EXECUTION_FAILED",
    errorMessage: String(error),
  };
}

export async function runVNextShadowComparison<
  TV2State = unknown,
  TVnextState = unknown,
>(
  manifest: AuditCorpusManifest,
  dependencies: ShadowRunnerDependencies<TV2State, TVnextState>,
): Promise<ShadowCorpusRunResult> {
  assertValidAuditCorpus(manifest);

  const members: ShadowMemberResult[] = [];
  const seenChangeIds = new Set<string>();

  for (
    let memberIndex = 0;
    memberIndex < manifest.members.length;
    memberIndex += 1
  ) {
    const member = manifest.members[memberIndex];

    try {
      const vnext = await dependencies.executeVNext(
        buildExecutionInput(member, memberIndex),
      );

      assertVnextExecutionResult(vnext);

      const v2 = await dependencies.loadV2Baseline(member);
      assertV2Baseline(member, v2);

      const changes = await dependencies.compare({
        member,
        v2,
        vnext,
      });

      for (const change of changes) {
        assertDivergence(change);

        if (seenChangeIds.has(change.CHANGE_ID)) {
          throw new Error(
            "VNEXT_SHADOW_DUPLICATE_CHANGE_ID",
          );
        }

        seenChangeIds.add(change.CHANGE_ID);
      }

      members.push({
        status: "COMPLETE",
        displayName: member.display_name,
        issuerId: member.issuer_id,
        v2SnapshotId: member.v2_snapshot_id,
        v2SourceRunId: member.v2_source_run_id,
        dataCutoff: member.data_cutoff,
        vnextExecutionId: vnext.executionId,
        changeCount: changes.length,
        changes: [...changes],
      });
    } catch (error) {
      const failure = toFailure(error);

      members.push({
        status: "FAILED",
        displayName: member.display_name,
        issuerId: member.issuer_id,
        v2SnapshotId: member.v2_snapshot_id,
        v2SourceRunId: member.v2_source_run_id,
        dataCutoff: member.data_cutoff,
        ...failure,
      });
    }
  }

  const completedMemberCount = members.filter(
    (member) => member.status === "COMPLETE",
  ).length;
  const failedMemberCount =
    members.length - completedMemberCount;

  return {
    corpusId: AUDIT_CORPUS_VNEXT_V1_ID,
    expectedMemberCount: AUDIT_CORPUS_VNEXT_V1_SIZE,
    attemptedMemberCount: members.length,
    completedMemberCount,
    failedMemberCount,
    status: failedMemberCount === 0 ? "COMPLETE" : "FAILED",
    members,
  };
}

function fingerprintsEqual(
  left: ProductionFingerprint,
  right: ProductionFingerprint,
): boolean {
  return (
    left.research_dossiers === right.research_dossiers &&
    left.research_snapshots === right.research_snapshots &&
    left.orotitan_runs === right.orotitan_runs &&
    left.orotitan_artifacts === right.orotitan_artifacts &&
    left.dossier_pointer_sha256 === right.dossier_pointer_sha256 &&
    left.snapshots_sha256 === right.snapshots_sha256
  );
}

export function assertProductionBaselineMatchesCorpus(
  manifest: AuditCorpusManifest,
  observed: ProductionFingerprint,
): void {
  assertFingerprint(
    observed,
    "VNEXT_SHADOW_PRODUCTION_BASELINE_OBSERVED",
  );

  if (
    !fingerprintsEqual(
      observed,
      manifest.production_before_fingerprint,
    )
  ) {
    throw new Error(
      "VNEXT_SHADOW_PRODUCTION_BASELINE_DRIFT",
    );
  }
}

export function assertNoProductionMutation(
  before: ProductionFingerprint,
  after: ProductionFingerprint,
): void {
  assertFingerprint(before, "VNEXT_SHADOW_PRODUCTION_BEFORE");
  assertFingerprint(after, "VNEXT_SHADOW_PRODUCTION_AFTER");

  if (!fingerprintsEqual(before, after)) {
    throw new Error("VNEXT_SHADOW_PRODUCTION_POLLUTION_DETECTED");
  }
}

export function assessGate17Exit(
  manifest: AuditCorpusManifest,
  run: ShadowCorpusRunResult,
  before: ProductionFingerprint,
  after: ProductionFingerprint,
): Gate17ExitAssessment {
  assertValidAuditCorpus(manifest);
  assertProductionBaselineMatchesCorpus(manifest, before);

  let productionUnchanged = true;

  try {
    assertNoProductionMutation(before, after);
  } catch {
    productionUnchanged = false;
  }

  const pass =
    run.corpusId === AUDIT_CORPUS_VNEXT_V1_ID &&
    run.expectedMemberCount === AUDIT_CORPUS_VNEXT_V1_SIZE &&
    run.attemptedMemberCount === AUDIT_CORPUS_VNEXT_V1_SIZE &&
    run.completedMemberCount === AUDIT_CORPUS_VNEXT_V1_SIZE &&
    run.failedMemberCount === 0 &&
    run.status === "COMPLETE" &&
    productionUnchanged;

  return {
    gate: 17,
    corpusId: AUDIT_CORPUS_VNEXT_V1_ID,
    status: pass ? "PASS" : "FAIL",
    expectedMemberCount: AUDIT_CORPUS_VNEXT_V1_SIZE,
    attemptedMemberCount: run.attemptedMemberCount,
    completedMemberCount: run.completedMemberCount,
    failedMemberCount: run.failedMemberCount,
    productionUnchanged,
  };
}

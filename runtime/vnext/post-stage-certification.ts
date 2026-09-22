import {
  HANDOFF_GATE_BY_STAGE,
  type RunStatus,
  type StageCode,
  type StageLifecycle,
} from "./state-machine";

export type FinalArtifactAuthorityClass =
  | "AUTHORITATIVE_STAGE_OUTPUT"
  | "CHECKPOINT_STAGE_OUTPUT"
  | "ROUTING_ONLY"
  | "HUMAN_SUMMARY"
  | "SOURCE_ATTACHMENT"
  | "UPSTREAM_DISCOVERY_OUTPUT"
  | "IMPLEMENTATION_DIAGNOSTIC";

export interface FinalizationRunState {
  runId: string;
  runStatus: RunStatus;
  issuerId: string | null;
  securityId: string | null;
  dossierId: string | null;
  canonicalMode: string;
  runType: string | null;
  dataCutoff: string;
  baselineSnapshotId: string | null;
  processVersion: string;
  pilotageContractVersion: string;
  contractSetSha256: string;
  stateVersion: number;
}

export interface FinalizationStageState {
  runId: string;
  stageCode: StageCode;
  stageRevision: number;
  stageContractName: string;
  stageContractVersion: string;
  stageContractSha256: string;
  lifecycleStatus: StageLifecycle;
  handoffGateName: string;
  handoffGateState: "NOT_EVALUATED" | "YES" | "NO";
  activeManifestArtifactId: string | null;
  activeManifestVersion: number | null;
  activeManifestKind: "CHECKPOINT" | "FINAL" | null;
  blockerCount: number;
  stateVersion: number;
}

export interface FinalizationArtifact {
  artifactId: string;
  version: number;
  runId: string;
  stageCode: StageCode;
  artifactType: string;
  authorityClass: FinalArtifactAuthorityClass;
  artifactStatus: "SEALED" | "INVALIDATED";
  authorityState:
    | "AUTHORITATIVE"
    | "CHECKPOINT"
    | "SUPERSEDED"
    | "NON_AUTHORITATIVE";
  availabilityState: "AVAILABLE" | "WITHDRAWN" | "MISSING";
  contentSha256: string;
  retrievedContentSha256: string;
  mediaType: string;
  sizeBytes: number;
  durableLocatorAvailable: boolean;
  schemaValidationRequired: boolean;
  schemaValidationPassed: boolean;
}

export interface ManifestArtifactRef {
  artifactId: string;
  version: number;
  artifactType: string;
  contentSha256: string;
}

export interface ManifestOutputRef extends ManifestArtifactRef {
  authorityClass: FinalArtifactAuthorityClass;
  mediaType: string;
  sizeBytes: number;
}

export interface FinalStageManifestBody {
  manifestSchemaVersion: string;
  manifestId: string;
  manifestKind: "FINAL";
  runId: string;
  stage: StageCode;
  stageRevision: number;
  issuerId: string | null;
  securityId: string | null;
  canonicalMode: string;
  runType: string | null;
  dataCutoff: string;
  baselineSnapshotId: string | null;
  processVersion: string;
  pilotageContractVersion: string;
  stageContract: {
    name: string;
    version: string;
    contentSha256: string;
  };
  contractSetSha256: string;
  inputArtifacts: readonly ManifestArtifactRef[];
  outputArtifacts: readonly ManifestOutputRef[];
  stageStatus: "COMPLETE";
  handoffGate: {
    name: string;
    state: "YES";
  };
  criticalBlockers: readonly string[];
  parentManifests: readonly ManifestArtifactRef[];
}

export interface FinalStageManifest {
  artifact: FinalizationArtifact;
  body: FinalStageManifestBody;
}

export type LineageRelation =
  | "CONSUMES"
  | "DERIVED_FROM"
  | "SUPERSEDES"
  | "BASELINE_OF"
  | "REVALIDATES";

export interface FinalizationLineageEdge {
  childRunId: string;
  childArtifactId: string;
  childVersion: number;
  parentRunId: string;
  parentArtifactId: string;
  parentVersion: number;
  relationType: LineageRelation;
}

export interface RequiredLineageEdge {
  childArtifactId: string;
  childVersion: number;
  parentArtifactId: string;
  parentVersion: number;
  relationType: LineageRelation;
}

export interface PostStageCertificationRequest {
  runId: string;
  stageCode: StageCode;
  expectedRunStateVersion: number;
  expectedStageStateVersion: number;
  requiredOutputTypes: readonly string[];
  requiredLineageEdges: readonly RequiredLineageEdge[];
  outputs: readonly FinalizationArtifact[];
  resolvedInputs: readonly FinalizationArtifact[];
  manifest: FinalStageManifest;
  lineageEdges: readonly FinalizationLineageEdge[];
  selfAuditPassed: boolean;
  contractPinsVerified: boolean;
  forbiddenMutationCheckPassed: boolean;
}

export type PostStageCertificationFailureCode =
  | "RUN_ID_MISMATCH"
  | "RUN_STATE_VERSION_STALE"
  | "RUN_NOT_FINALIZABLE"
  | "STAGE_RUN_ID_MISMATCH"
  | "STAGE_CODE_MISMATCH"
  | "STAGE_STATE_VERSION_STALE"
  | "STAGE_NOT_IN_PROGRESS"
  | "STAGE_BLOCKER_PRESENT"
  | "SELF_AUDIT_FAILED"
  | "CONTRACT_PINS_NOT_VERIFIED"
  | "FORBIDDEN_MUTATION_CHECK_FAILED"
  | "REQUIRED_OUTPUT_POLICY_EMPTY"
  | "REQUIRED_OUTPUT_MISSING"
  | "OUTPUT_DUPLICATE"
  | "OUTPUT_RUN_MISMATCH"
  | "OUTPUT_STAGE_MISMATCH"
  | "OUTPUT_NOT_SEALED"
  | "OUTPUT_NOT_AUTHORITATIVE"
  | "OUTPUT_NOT_AVAILABLE"
  | "OUTPUT_HASH_MISMATCH"
  | "OUTPUT_LOCATOR_MISSING"
  | "OUTPUT_SCHEMA_INVALID"
  | "MANIFEST_SCHEMA_UNSUPPORTED"
  | "MANIFEST_ID_MISMATCH"
  | "MANIFEST_ARTIFACT_TYPE_MISMATCH"
  | "MANIFEST_RUN_MISMATCH"
  | "MANIFEST_STAGE_MISMATCH"
  | "MANIFEST_REVISION_MISMATCH"
  | "MANIFEST_IDENTITY_MISMATCH"
  | "MANIFEST_MODE_MISMATCH"
  | "MANIFEST_CUTOFF_MISMATCH"
  | "MANIFEST_BASELINE_MISMATCH"
  | "MANIFEST_PROCESS_VERSION_MISMATCH"
  | "MANIFEST_PILOTAGE_VERSION_MISMATCH"
  | "MANIFEST_STAGE_CONTRACT_MISMATCH"
  | "MANIFEST_CONTRACT_SET_MISMATCH"
  | "MANIFEST_STATUS_INVALID"
  | "MANIFEST_HANDOFF_GATE_MISMATCH"
  | "MANIFEST_BLOCKER_MISMATCH"
  | "MANIFEST_SELF_REFERENCE"
  | "MANIFEST_OUTPUT_DUPLICATE"
  | "MANIFEST_OUTPUT_MISMATCH"
  | "MANIFEST_INPUT_UNRESOLVED"
  | "MANIFEST_ARTIFACT_NOT_SEALED"
  | "MANIFEST_ARTIFACT_NOT_AUTHORITATIVE"
  | "MANIFEST_ARTIFACT_NOT_AVAILABLE"
  | "MANIFEST_ARTIFACT_HASH_MISMATCH"
  | "MANIFEST_ARTIFACT_LOCATOR_MISSING"
  | "LINEAGE_SELF_EDGE"
  | "LINEAGE_DUPLICATE_EDGE"
  | "LINEAGE_ENDPOINT_UNRESOLVED"
  | "REQUIRED_LINEAGE_EDGE_MISSING"
  | "FRESH_REREAD_RUN_IMMUTABLE_DRIFT"
  | "FRESH_REREAD_STAGE_IMMUTABLE_DRIFT"
  | "FRESH_REREAD_STAGE_NOT_COMPLETE"
  | "FRESH_REREAD_HANDOFF_NOT_YES"
  | "FRESH_REREAD_MANIFEST_POINTER_MISMATCH"
  | "FRESH_REREAD_STAGE_VERSION_MISMATCH";

export interface PostStageCertificationFailure {
  code: PostStageCertificationFailureCode;
  detail: string;
}

export interface PostStageCertificationReport {
  status: "PASS" | "FAIL";
  failures: readonly PostStageCertificationFailure[];
}

export interface PostFinalizeContext {
  run: FinalizationRunState;
  stage: FinalizationStageState;
}

export interface PostStageCertificationStore {
  readContext(
    runId: string,
    stageCode: StageCode,
  ): Promise<PostFinalizeContext>;

  finalizeCertifiedStage(input: {
    runId: string;
    stageCode: StageCode;
    expectedRunStateVersion: number;
    expectedStageStateVersion: number;
    manifestArtifactId: string;
    manifestVersion: number;
    handoffGateName: string;
  }): Promise<void>;
}

export interface CertifiedStageFinalizationResult {
  certification: PostStageCertificationReport;
  before: PostFinalizeContext;
  after: PostFinalizeContext;
}

const SHA256_RE = /^[0-9a-f]{64}$/;
const SUPPORTED_MANIFEST_SCHEMA = "1.0.0";

function fail(
  failures: PostStageCertificationFailure[],
  code: PostStageCertificationFailureCode,
  detail: string,
): void {
  failures.push({ code, detail });
}

function artifactKey(
  artifact: Pick<FinalizationArtifact, "artifactId" | "version">,
): string {
  return artifact.artifactId + ":" + artifact.version;
}

function refKey(
  artifact: Pick<ManifestArtifactRef, "artifactId" | "version">,
): string {
  return artifact.artifactId + ":" + artifact.version;
}

function lineageKey(edge: FinalizationLineageEdge): string {
  return [
    edge.childRunId,
    edge.childArtifactId,
    edge.childVersion,
    edge.parentRunId,
    edge.parentArtifactId,
    edge.parentVersion,
    edge.relationType,
  ].join(":");
}

function requiredLineageKey(
  runId: string,
  edge: RequiredLineageEdge,
): string {
  return [
    runId,
    edge.childArtifactId,
    edge.childVersion,
    runId,
    edge.parentArtifactId,
    edge.parentVersion,
    edge.relationType,
  ].join(":");
}

function validateBaseState(
  request: PostStageCertificationRequest,
  context: PostFinalizeContext,
  failures: PostStageCertificationFailure[],
): void {
  const { run, stage } = context;

  if (run.runId !== request.runId) {
    fail(
      failures,
      "RUN_ID_MISMATCH",
      "expected=" + request.runId + " actual=" + run.runId,
    );
  }

  if (run.stateVersion !== request.expectedRunStateVersion) {
    fail(
      failures,
      "RUN_STATE_VERSION_STALE",
      "expected=" +
        request.expectedRunStateVersion +
        " actual=" +
        run.stateVersion,
    );
  }

  if (
    run.runStatus === "READY_TO_PUBLISH" ||
    run.runStatus === "PUBLISHED" ||
    run.runStatus === "CANCELLED"
  ) {
    fail(
      failures,
      "RUN_NOT_FINALIZABLE",
      "run_status=" + run.runStatus,
    );
  }

  if (stage.runId !== request.runId) {
    fail(
      failures,
      "STAGE_RUN_ID_MISMATCH",
      "expected=" + request.runId + " actual=" + stage.runId,
    );
  }

  if (stage.stageCode !== request.stageCode) {
    fail(
      failures,
      "STAGE_CODE_MISMATCH",
      "expected=" + request.stageCode + " actual=" + stage.stageCode,
    );
  }

  if (stage.stateVersion !== request.expectedStageStateVersion) {
    fail(
      failures,
      "STAGE_STATE_VERSION_STALE",
      "expected=" +
        request.expectedStageStateVersion +
        " actual=" +
        stage.stateVersion,
    );
  }

  if (stage.lifecycleStatus !== "IN_PROGRESS") {
    fail(
      failures,
      "STAGE_NOT_IN_PROGRESS",
      "lifecycle=" + stage.lifecycleStatus,
    );
  }

  if (stage.blockerCount > 0) {
    fail(
      failures,
      "STAGE_BLOCKER_PRESENT",
      "blocker_count=" + stage.blockerCount,
    );
  }

  if (!request.selfAuditPassed) {
    fail(failures, "SELF_AUDIT_FAILED", "stage self-audit did not pass");
  }

  if (!request.contractPinsVerified) {
    fail(
      failures,
      "CONTRACT_PINS_NOT_VERIFIED",
      "exact contract pins were not verified",
    );
  }

  if (!request.forbiddenMutationCheckPassed) {
    fail(
      failures,
      "FORBIDDEN_MUTATION_CHECK_FAILED",
      "forbidden-mutation assurance did not pass",
    );
  }
}

function validateOutputs(
  request: PostStageCertificationRequest,
  failures: PostStageCertificationFailure[],
): void {
  if (request.requiredOutputTypes.length === 0) {
    fail(
      failures,
      "REQUIRED_OUTPUT_POLICY_EMPTY",
      "at least one frozen required output must be declared",
    );
  }

  const outputKeys = new Set<string>();
  const outputTypeCounts = new Map<string, number>();

  for (const output of request.outputs) {
    const key = artifactKey(output);

    if (outputKeys.has(key)) {
      fail(failures, "OUTPUT_DUPLICATE", "artifact=" + key);
    }
    outputKeys.add(key);

    outputTypeCounts.set(
      output.artifactType,
      (outputTypeCounts.get(output.artifactType) ?? 0) + 1,
    );

    if (output.runId !== request.runId) {
      fail(
        failures,
        "OUTPUT_RUN_MISMATCH",
        "artifact=" + key + " run=" + output.runId,
      );
    }

    if (output.stageCode !== request.stageCode) {
      fail(
        failures,
        "OUTPUT_STAGE_MISMATCH",
        "artifact=" + key + " stage=" + output.stageCode,
      );
    }

    if (output.artifactStatus !== "SEALED") {
      fail(
        failures,
        "OUTPUT_NOT_SEALED",
        "artifact=" + key + " status=" + output.artifactStatus,
      );
    }

    if (
      output.authorityClass !== "AUTHORITATIVE_STAGE_OUTPUT" ||
      output.authorityState !== "AUTHORITATIVE"
    ) {
      fail(
        failures,
        "OUTPUT_NOT_AUTHORITATIVE",
        "artifact=" +
          key +
          " class=" +
          output.authorityClass +
          " state=" +
          output.authorityState,
      );
    }

    if (output.availabilityState !== "AVAILABLE") {
      fail(
        failures,
        "OUTPUT_NOT_AVAILABLE",
        "artifact=" + key + " availability=" + output.availabilityState,
      );
    }

    if (
      !SHA256_RE.test(output.contentSha256) ||
      output.retrievedContentSha256 !== output.contentSha256
    ) {
      fail(
        failures,
        "OUTPUT_HASH_MISMATCH",
        "artifact=" + key,
      );
    }

    if (!output.durableLocatorAvailable) {
      fail(
        failures,
        "OUTPUT_LOCATOR_MISSING",
        "artifact=" + key,
      );
    }

    if (
      output.schemaValidationRequired &&
      !output.schemaValidationPassed
    ) {
      fail(
        failures,
        "OUTPUT_SCHEMA_INVALID",
        "artifact=" + key + " type=" + output.artifactType,
      );
    }
  }

  for (const requiredType of request.requiredOutputTypes) {
    if ((outputTypeCounts.get(requiredType) ?? 0) < 1) {
      fail(
        failures,
        "REQUIRED_OUTPUT_MISSING",
        "artifact_type=" + requiredType,
      );
    }
  }
}

function validateManifest(
  request: PostStageCertificationRequest,
  context: PostFinalizeContext,
  failures: PostStageCertificationFailure[],
): void {
  const { run, stage } = context;
  const { artifact, body } = request.manifest;
  const expectedManifestType =
    request.stageCode === "RESEARCH"
      ? "RESEARCH_STAGE_MANIFEST"
      : request.stageCode === "DEEP_DIVE"
        ? "DEEP_DIVE_STAGE_MANIFEST"
        : "INTEGRATION_STAGE_MANIFEST";

  if (body.manifestSchemaVersion !== SUPPORTED_MANIFEST_SCHEMA) {
    fail(
      failures,
      "MANIFEST_SCHEMA_UNSUPPORTED",
      "schema=" + body.manifestSchemaVersion,
    );
  }

  if (body.manifestId !== artifact.artifactId) {
    fail(
      failures,
      "MANIFEST_ID_MISMATCH",
      "body=" + body.manifestId + " artifact=" + artifact.artifactId,
    );
  }

  if (artifact.artifactType !== expectedManifestType) {
    fail(
      failures,
      "MANIFEST_ARTIFACT_TYPE_MISMATCH",
      "expected=" + expectedManifestType + " actual=" + artifact.artifactType,
    );
  }

  if (body.runId !== request.runId || artifact.runId !== request.runId) {
    fail(
      failures,
      "MANIFEST_RUN_MISMATCH",
      "manifest run must equal request run",
    );
  }

  if (body.stage !== request.stageCode || artifact.stageCode !== request.stageCode) {
    fail(
      failures,
      "MANIFEST_STAGE_MISMATCH",
      "manifest stage must equal target stage",
    );
  }

  if (body.stageRevision !== stage.stageRevision) {
    fail(
      failures,
      "MANIFEST_REVISION_MISMATCH",
      "expected=" + stage.stageRevision + " actual=" + body.stageRevision,
    );
  }

  if (
    body.issuerId !== run.issuerId ||
    body.securityId !== run.securityId
  ) {
    fail(
      failures,
      "MANIFEST_IDENTITY_MISMATCH",
      "manifest identity differs from run identity",
    );
  }

  if (
    body.canonicalMode !== run.canonicalMode ||
    body.runType !== run.runType
  ) {
    fail(
      failures,
      "MANIFEST_MODE_MISMATCH",
      "manifest canonical mode/run type differs from run",
    );
  }

  if (body.dataCutoff !== run.dataCutoff) {
    fail(
      failures,
      "MANIFEST_CUTOFF_MISMATCH",
      "expected=" + run.dataCutoff + " actual=" + body.dataCutoff,
    );
  }

  if (body.baselineSnapshotId !== run.baselineSnapshotId) {
    fail(
      failures,
      "MANIFEST_BASELINE_MISMATCH",
      "manifest baseline differs from run",
    );
  }

  if (body.processVersion !== run.processVersion) {
    fail(
      failures,
      "MANIFEST_PROCESS_VERSION_MISMATCH",
      "manifest process version differs from run",
    );
  }

  if (body.pilotageContractVersion !== run.pilotageContractVersion) {
    fail(
      failures,
      "MANIFEST_PILOTAGE_VERSION_MISMATCH",
      "manifest Pilotage contract differs from run",
    );
  }

  if (
    body.stageContract.name !== stage.stageContractName ||
    body.stageContract.version !== stage.stageContractVersion ||
    body.stageContract.contentSha256 !== stage.stageContractSha256
  ) {
    fail(
      failures,
      "MANIFEST_STAGE_CONTRACT_MISMATCH",
      "manifest stage-contract pin differs from stage row",
    );
  }

  if (body.contractSetSha256 !== run.contractSetSha256) {
    fail(
      failures,
      "MANIFEST_CONTRACT_SET_MISMATCH",
      "manifest contract-set hash differs from run",
    );
  }

  if (body.stageStatus !== "COMPLETE") {
    fail(
      failures,
      "MANIFEST_STATUS_INVALID",
      "manifest stage status must be COMPLETE",
    );
  }

  const expectedGate = HANDOFF_GATE_BY_STAGE[request.stageCode];
  if (
    body.handoffGate.name !== expectedGate ||
    body.handoffGate.state !== "YES"
  ) {
    fail(
      failures,
      "MANIFEST_HANDOFF_GATE_MISMATCH",
      "expected=" + expectedGate + ":YES",
    );
  }

  if (body.criticalBlockers.length > 0) {
    fail(
      failures,
      "MANIFEST_BLOCKER_MISMATCH",
      "FINAL + handoff YES cannot contain critical blockers",
    );
  }

  if (
    artifact.artifactStatus !== "SEALED"
  ) {
    fail(
      failures,
      "MANIFEST_ARTIFACT_NOT_SEALED",
      "status=" + artifact.artifactStatus,
    );
  }

  if (
    artifact.authorityClass !== "AUTHORITATIVE_STAGE_OUTPUT" ||
    artifact.authorityState !== "AUTHORITATIVE"
  ) {
    fail(
      failures,
      "MANIFEST_ARTIFACT_NOT_AUTHORITATIVE",
      "manifest artifact must be authoritative",
    );
  }

  if (artifact.availabilityState !== "AVAILABLE") {
    fail(
      failures,
      "MANIFEST_ARTIFACT_NOT_AVAILABLE",
      "availability=" + artifact.availabilityState,
    );
  }

  if (
    !SHA256_RE.test(artifact.contentSha256) ||
    artifact.retrievedContentSha256 !== artifact.contentSha256
  ) {
    fail(
      failures,
      "MANIFEST_ARTIFACT_HASH_MISMATCH",
      "manifest bytes did not hash-verify",
    );
  }

  if (!artifact.durableLocatorAvailable) {
    fail(
      failures,
      "MANIFEST_ARTIFACT_LOCATOR_MISSING",
      "manifest durable locator missing",
    );
  }

  const manifestOutputKeys = new Set<string>();

  for (const outputRef of body.outputArtifacts) {
    const key = refKey(outputRef);

    if (outputRef.artifactId === body.manifestId) {
      fail(
        failures,
        "MANIFEST_SELF_REFERENCE",
        "manifest cannot list itself in output_artifacts",
      );
    }

    if (manifestOutputKeys.has(key)) {
      fail(
        failures,
        "MANIFEST_OUTPUT_DUPLICATE",
        "artifact=" + key,
      );
    }
    manifestOutputKeys.add(key);

    const output = request.outputs.find(
      (candidate) =>
        candidate.artifactId === outputRef.artifactId &&
        candidate.version === outputRef.version,
    );

    if (
      !output ||
      output.artifactType !== outputRef.artifactType ||
      output.contentSha256 !== outputRef.contentSha256 ||
      output.authorityClass !== outputRef.authorityClass ||
      output.mediaType !== outputRef.mediaType ||
      output.sizeBytes !== outputRef.sizeBytes
    ) {
      fail(
        failures,
        "MANIFEST_OUTPUT_MISMATCH",
        "artifact=" + key,
      );
    }
  }

  for (const output of request.outputs) {
    if (!manifestOutputKeys.has(artifactKey(output))) {
      fail(
        failures,
        "MANIFEST_OUTPUT_MISMATCH",
        "registered output absent from FINAL manifest: " + artifactKey(output),
      );
    }
  }

  for (const inputRef of body.inputArtifacts) {
    const input = request.resolvedInputs.find(
      (candidate) =>
        candidate.artifactId === inputRef.artifactId &&
        candidate.version === inputRef.version,
    );

    if (
      !input ||
      input.runId !== request.runId ||
      input.artifactType !== inputRef.artifactType ||
      input.contentSha256 !== inputRef.contentSha256 ||
      input.artifactStatus !== "SEALED" ||
      input.availabilityState !== "AVAILABLE" ||
      input.retrievedContentSha256 !== input.contentSha256
    ) {
      fail(
        failures,
        "MANIFEST_INPUT_UNRESOLVED",
        "artifact=" + refKey(inputRef),
      );
    }
  }
}

function validateLineage(
  request: PostStageCertificationRequest,
  failures: PostStageCertificationFailure[],
): void {
  const endpointKeys = new Set<string>();

  for (const artifact of [...request.outputs, ...request.resolvedInputs]) {
    endpointKeys.add(
      artifact.runId + ":" + artifact.artifactId + ":" + artifact.version,
    );
  }

  const edgeKeys = new Set<string>();

  for (const edge of request.lineageEdges) {
    const key = lineageKey(edge);

    if (
      edge.childRunId === edge.parentRunId &&
      edge.childArtifactId === edge.parentArtifactId &&
      edge.childVersion === edge.parentVersion
    ) {
      fail(failures, "LINEAGE_SELF_EDGE", "edge=" + key);
    }

    if (edgeKeys.has(key)) {
      fail(failures, "LINEAGE_DUPLICATE_EDGE", "edge=" + key);
    }
    edgeKeys.add(key);

    const childKey =
      edge.childRunId +
      ":" +
      edge.childArtifactId +
      ":" +
      edge.childVersion;
    const parentKey =
      edge.parentRunId +
      ":" +
      edge.parentArtifactId +
      ":" +
      edge.parentVersion;

    if (!endpointKeys.has(childKey) || !endpointKeys.has(parentKey)) {
      fail(
        failures,
        "LINEAGE_ENDPOINT_UNRESOLVED",
        "edge=" + key,
      );
    }
  }

  for (const required of request.requiredLineageEdges) {
    const key = requiredLineageKey(request.runId, required);
    if (!edgeKeys.has(key)) {
      fail(
        failures,
        "REQUIRED_LINEAGE_EDGE_MISSING",
        "edge=" + key,
      );
    }
  }
}

export function certifyPostStageBundle(
  request: PostStageCertificationRequest,
  context: PostFinalizeContext,
): PostStageCertificationReport {
  const failures: PostStageCertificationFailure[] = [];

  validateBaseState(request, context, failures);
  validateOutputs(request, failures);
  validateManifest(request, context, failures);
  validateLineage(request, failures);

  return {
    status: failures.length === 0 ? "PASS" : "FAIL",
    failures,
  };
}

function runImmutableFingerprint(run: FinalizationRunState): string {
  return JSON.stringify({
    runId: run.runId,
    issuerId: run.issuerId,
    securityId: run.securityId,
    dossierId: run.dossierId,
    canonicalMode: run.canonicalMode,
    runType: run.runType,
    dataCutoff: run.dataCutoff,
    baselineSnapshotId: run.baselineSnapshotId,
    processVersion: run.processVersion,
    pilotageContractVersion: run.pilotageContractVersion,
    contractSetSha256: run.contractSetSha256,
  });
}

function stageImmutableFingerprint(stage: FinalizationStageState): string {
  return JSON.stringify({
    runId: stage.runId,
    stageCode: stage.stageCode,
    stageRevision: stage.stageRevision,
    stageContractName: stage.stageContractName,
    stageContractVersion: stage.stageContractVersion,
    stageContractSha256: stage.stageContractSha256,
    handoffGateName: stage.handoffGateName,
  });
}

function verifyFreshReread(
  request: PostStageCertificationRequest,
  before: PostFinalizeContext,
  after: PostFinalizeContext,
): PostStageCertificationFailure[] {
  const failures: PostStageCertificationFailure[] = [];

  if (runImmutableFingerprint(before.run) !== runImmutableFingerprint(after.run)) {
    fail(
      failures,
      "FRESH_REREAD_RUN_IMMUTABLE_DRIFT",
      "run immutable fields changed during stage finalization",
    );
  }

  if (
    stageImmutableFingerprint(before.stage) !==
    stageImmutableFingerprint(after.stage)
  ) {
    fail(
      failures,
      "FRESH_REREAD_STAGE_IMMUTABLE_DRIFT",
      "stage immutable fields changed during finalization",
    );
  }

  if (after.stage.lifecycleStatus !== "COMPLETE") {
    fail(
      failures,
      "FRESH_REREAD_STAGE_NOT_COMPLETE",
      "lifecycle=" + after.stage.lifecycleStatus,
    );
  }

  if (after.stage.handoffGateState !== "YES") {
    fail(
      failures,
      "FRESH_REREAD_HANDOFF_NOT_YES",
      "handoff=" + after.stage.handoffGateState,
    );
  }

  if (
    after.stage.activeManifestArtifactId !==
      request.manifest.artifact.artifactId ||
    after.stage.activeManifestVersion !==
      request.manifest.artifact.version ||
    after.stage.activeManifestKind !== "FINAL"
  ) {
    fail(
      failures,
      "FRESH_REREAD_MANIFEST_POINTER_MISMATCH",
      "active manifest pointer does not match certified FINAL manifest",
    );
  }

  if (after.stage.stateVersion !== before.stage.stateVersion + 1) {
    fail(
      failures,
      "FRESH_REREAD_STAGE_VERSION_MISMATCH",
      "before=" +
        before.stage.stateVersion +
        " after=" +
        after.stage.stateVersion,
    );
  }

  return failures;
}

/**
 * Gate 10 execution boundary.
 *
 * READ -> CERTIFY PREPARED FINAL BUNDLE -> ATOMIC STORE FINALIZATION -> REREAD.
 *
 * Invalid bundles never reach the mutation call.
 */
export async function executeCertifiedStageFinalization(
  store: PostStageCertificationStore,
  request: PostStageCertificationRequest,
): Promise<CertifiedStageFinalizationResult> {
  const before = await store.readContext(request.runId, request.stageCode);
  const certification = certifyPostStageBundle(request, before);

  if (certification.status !== "PASS") {
    throw new Error(
      "VNEXT_POST_STAGE_CERTIFICATION_FAIL: " +
        certification.failures.map((failure) => failure.code).join(","),
    );
  }

  await store.finalizeCertifiedStage({
    runId: request.runId,
    stageCode: request.stageCode,
    expectedRunStateVersion: request.expectedRunStateVersion,
    expectedStageStateVersion: request.expectedStageStateVersion,
    manifestArtifactId: request.manifest.artifact.artifactId,
    manifestVersion: request.manifest.artifact.version,
    handoffGateName: HANDOFF_GATE_BY_STAGE[request.stageCode],
  });

  const after = await store.readContext(request.runId, request.stageCode);
  const rereadFailures = verifyFreshReread(request, before, after);

  if (rereadFailures.length > 0) {
    throw new Error(
      "VNEXT_POST_STAGE_FRESH_REREAD_FAIL: " +
        rereadFailures.map((failure) => failure.code).join(","),
    );
  }

  return {
    certification,
    before,
    after,
  };
}

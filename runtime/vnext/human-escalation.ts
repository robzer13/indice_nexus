import type {
  RunStatus,
  StageCode,
  StageLifecycle,
} from "./state-machine";

export const HUMAN_ESCALATION_REASONS = [
  "ANALYTICAL_AMBIGUITY",
  "EVIDENCE_ACCESS_REQUIRED",
  "MATERIAL_CONFLICT",
  "MODEL_DISAGREEMENT",
  "IDENTITY_AMBIGUITY",
  "DATA_LICENSE_EXCEPTION",
  "DATA_RESIDENCY_EXCEPTION",
  "BUDGET_OVERRIDE_REQUIRED",
  "UNCLASSIFIED_HUMAN_JUDGMENT",
] as const;

export type HumanEscalationReason =
  (typeof HUMAN_ESCALATION_REASONS)[number];

export interface EscalationArtifactRef {
  artifactId: string;
  version: number;
  contentSha256: string;
  artifactType: string;
}

export interface HumanEscalationContext {
  runId: string;
  stageCode: StageCode;
  runStatus: RunStatus;
  stageLifecycle: StageLifecycle;
  runStateVersion: number;
  stageStateVersion: number;
  stageRevision: number;
  dataCutoff: string;
  contractSetSha256: string;
  activeManifestArtifactId: string | null;
  activeManifestVersion: number | null;
  activeManifestKind: "CHECKPOINT" | "FINAL" | null;
  blockerCodes: readonly string[];
}

export interface CheckpointAndPauseReceipt {
  runId: string;
  stageCode: StageCode;
  checkpointManifestArtifactId: string;
  checkpointManifestVersion: number;
  checkpointManifestSha256: string;
  priorRunStateVersion: number;
  priorStageStateVersion: number;
  nextRunStateVersion: number;
  nextStageStateVersion: number;
}

export interface HumanEscalationPackage {
  schemaVersion: "0.1.0";
  escalationId: string;
  runId: string;
  stageCode: StageCode;
  reason: HumanEscalationReason;
  requestedHumanAction: string;
  createdFromCheckpoint: {
    artifactId: string;
    version: number;
    contentSha256: string;
  };
  frozenContext: {
    dataCutoff: string;
    contractSetSha256: string;
    stageRevision: number;
    runStateVersion: number;
    stageStateVersion: number;
  };
  exactArtifacts: readonly EscalationArtifactRef[];
  blockerCodes: readonly string[];
  workBoundary: {
    invocationMode: "HUMAN_OPENED_CHATGPT_WORK";
    runtimeApiInvocationAllowed: false;
    productionMutationAllowed: false;
    registryMutationAllowed: false;
    publicationAllowed: false;
  };
  resumeRequirements: {
    responseArtifactRequired: true;
    responseHashRequired: true;
    checkpointMustStillBeActive: true;
    exactStateVersionRequired: true;
  };
}

export interface HumanEscalationResponse {
  escalationId: string;
  runId: string;
  stageCode: StageCode;
  responseArtifactId: string;
  responseArtifactVersion: number;
  responseContentSha256: string;
  acceptedByHuman: boolean;
  expectedRunStateVersion: number;
  expectedStageStateVersion: number;
  checkpointManifestArtifactId: string;
  checkpointManifestVersion: number;
}

export interface HumanEscalationStore {
  readContext(
    runId: string,
    stageCode: StageCode,
  ): Promise<HumanEscalationContext>;

  checkpointAndPauseForHuman(input: {
    runId: string;
    stageCode: StageCode;
    expectedRunStateVersion: number;
    expectedStageStateVersion: number;
    reason: HumanEscalationReason;
  }): Promise<CheckpointAndPauseReceipt>;

  persistEscalationPackage(
    escalation: HumanEscalationPackage,
  ): Promise<void>;
}

export interface HumanEscalationRequest {
  escalationId: string;
  runId: string;
  stageCode: StageCode;
  reason: HumanEscalationReason;
  requestedHumanAction: string;
  exactArtifacts: readonly EscalationArtifactRef[];
  expectedRunStateVersion: number;
  expectedStageStateVersion: number;
}

export interface HumanEscalationExecutionResult {
  before: HumanEscalationContext;
  afterPause: HumanEscalationContext;
  checkpoint: CheckpointAndPauseReceipt;
  escalation: HumanEscalationPackage;
}

const SHA256_RE = /^[0-9a-f]{64}$/;

function assertSha256(value: string, code: string): void {
  if (!SHA256_RE.test(value)) {
    throw new Error(code);
  }
}

function assertEscalationRequest(
  request: HumanEscalationRequest,
  context: HumanEscalationContext,
): void {
  if (request.runId !== context.runId) {
    throw new Error("VNEXT_HUMAN_ESCALATION_RUN_MISMATCH");
  }
  if (request.stageCode !== context.stageCode) {
    throw new Error("VNEXT_HUMAN_ESCALATION_STAGE_MISMATCH");
  }
  if (request.expectedRunStateVersion !== context.runStateVersion) {
    throw new Error("VNEXT_HUMAN_ESCALATION_RUN_STATE_STALE");
  }
  if (request.expectedStageStateVersion !== context.stageStateVersion) {
    throw new Error("VNEXT_HUMAN_ESCALATION_STAGE_STATE_STALE");
  }
  if (
    context.runStatus === "PUBLISHED" ||
    context.runStatus === "CANCELLED"
  ) {
    throw new Error("VNEXT_HUMAN_ESCALATION_TERMINAL_RUN");
  }
  if (context.stageLifecycle === "COMPLETE") {
    throw new Error("VNEXT_HUMAN_ESCALATION_STAGE_COMPLETE");
  }
  if (request.requestedHumanAction.trim().length === 0) {
    throw new Error("VNEXT_HUMAN_ESCALATION_ACTION_REQUIRED");
  }
  if (request.exactArtifacts.length === 0) {
    throw new Error("VNEXT_HUMAN_ESCALATION_ARTIFACT_CONTEXT_REQUIRED");
  }

  const seen = new Set<string>();
  for (const ref of request.exactArtifacts) {
    const key = `${ref.artifactId}:${ref.version}`;
    if (seen.has(key)) {
      throw new Error("VNEXT_HUMAN_ESCALATION_DUPLICATE_ARTIFACT");
    }
    seen.add(key);
    assertSha256(
      ref.contentSha256,
      "VNEXT_HUMAN_ESCALATION_ARTIFACT_HASH_INVALID",
    );
  }
}

function assertPausedCheckpointState(
  before: HumanEscalationContext,
  after: HumanEscalationContext,
  receipt: CheckpointAndPauseReceipt,
): void {
  if (after.runStatus !== "PAUSED") {
    throw new Error("VNEXT_HUMAN_ESCALATION_RUN_NOT_PAUSED");
  }
  if (after.stageLifecycle !== "PAUSED") {
    throw new Error("VNEXT_HUMAN_ESCALATION_STAGE_NOT_PAUSED");
  }
  if (after.activeManifestKind !== "CHECKPOINT") {
    throw new Error("VNEXT_HUMAN_ESCALATION_CHECKPOINT_NOT_ACTIVE");
  }
  if (
    after.activeManifestArtifactId !==
      receipt.checkpointManifestArtifactId ||
    after.activeManifestVersion !==
      receipt.checkpointManifestVersion
  ) {
    throw new Error("VNEXT_HUMAN_ESCALATION_CHECKPOINT_POINTER_MISMATCH");
  }
  if (after.runStateVersion !== before.runStateVersion + 1) {
    throw new Error("VNEXT_HUMAN_ESCALATION_RUN_VERSION_MISMATCH");
  }
  if (after.stageStateVersion !== before.stageStateVersion + 1) {
    throw new Error("VNEXT_HUMAN_ESCALATION_STAGE_VERSION_MISMATCH");
  }
  if (after.dataCutoff !== before.dataCutoff) {
    throw new Error("VNEXT_HUMAN_ESCALATION_DATA_CUTOFF_DRIFT");
  }
  if (after.contractSetSha256 !== before.contractSetSha256) {
    throw new Error("VNEXT_HUMAN_ESCALATION_CONTRACT_DRIFT");
  }
  if (after.stageRevision !== before.stageRevision) {
    throw new Error("VNEXT_HUMAN_ESCALATION_STAGE_REVISION_DRIFT");
  }
}

function buildPackage(
  request: HumanEscalationRequest,
  paused: HumanEscalationContext,
  checkpoint: CheckpointAndPauseReceipt,
): HumanEscalationPackage {
  assertSha256(
    checkpoint.checkpointManifestSha256,
    "VNEXT_HUMAN_ESCALATION_CHECKPOINT_HASH_INVALID",
  );

  return {
    schemaVersion: "0.1.0",
    escalationId: request.escalationId,
    runId: request.runId,
    stageCode: request.stageCode,
    reason: request.reason,
    requestedHumanAction: request.requestedHumanAction,
    createdFromCheckpoint: {
      artifactId: checkpoint.checkpointManifestArtifactId,
      version: checkpoint.checkpointManifestVersion,
      contentSha256: checkpoint.checkpointManifestSha256,
    },
    frozenContext: {
      dataCutoff: paused.dataCutoff,
      contractSetSha256: paused.contractSetSha256,
      stageRevision: paused.stageRevision,
      runStateVersion: paused.runStateVersion,
      stageStateVersion: paused.stageStateVersion,
    },
    exactArtifacts: request.exactArtifacts,
    blockerCodes: paused.blockerCodes,
    workBoundary: {
      invocationMode: "HUMAN_OPENED_CHATGPT_WORK",
      runtimeApiInvocationAllowed: false,
      productionMutationAllowed: false,
      registryMutationAllowed: false,
      publicationAllowed: false,
    },
    resumeRequirements: {
      responseArtifactRequired: true,
      responseHashRequired: true,
      checkpointMustStillBeActive: true,
      exactStateVersionRequired: true,
    },
  };
}

export async function executeHumanEscalation(
  store: HumanEscalationStore,
  request: HumanEscalationRequest,
): Promise<HumanEscalationExecutionResult> {
  const before = await store.readContext(
    request.runId,
    request.stageCode,
  );

  assertEscalationRequest(request, before);

  const checkpoint = await store.checkpointAndPauseForHuman({
    runId: request.runId,
    stageCode: request.stageCode,
    expectedRunStateVersion: request.expectedRunStateVersion,
    expectedStageStateVersion: request.expectedStageStateVersion,
    reason: request.reason,
  });

  const afterPause = await store.readContext(
    request.runId,
    request.stageCode,
  );

  assertPausedCheckpointState(before, afterPause, checkpoint);

  const escalation = buildPackage(
    request,
    afterPause,
    checkpoint,
  );

  await store.persistEscalationPackage(escalation);

  return {
    before,
    afterPause,
    checkpoint,
    escalation,
  };
}

export function assertHumanEscalationResumeAllowed(
  context: HumanEscalationContext,
  response: HumanEscalationResponse,
): void {
  if (response.runId !== context.runId) {
    throw new Error("VNEXT_HUMAN_RESUME_RUN_MISMATCH");
  }
  if (response.stageCode !== context.stageCode) {
    throw new Error("VNEXT_HUMAN_RESUME_STAGE_MISMATCH");
  }
  if (!response.acceptedByHuman) {
    throw new Error("VNEXT_HUMAN_RESUME_RESPONSE_NOT_ACCEPTED");
  }
  if (context.runStatus !== "PAUSED") {
    throw new Error("VNEXT_HUMAN_RESUME_RUN_NOT_PAUSED");
  }
  if (context.stageLifecycle !== "PAUSED") {
    throw new Error("VNEXT_HUMAN_RESUME_STAGE_NOT_PAUSED");
  }
  if (context.activeManifestKind !== "CHECKPOINT") {
    throw new Error("VNEXT_HUMAN_RESUME_CHECKPOINT_REQUIRED");
  }
  if (
    context.activeManifestArtifactId !==
      response.checkpointManifestArtifactId ||
    context.activeManifestVersion !==
      response.checkpointManifestVersion
  ) {
    throw new Error("VNEXT_HUMAN_RESUME_CHECKPOINT_MISMATCH");
  }
  if (context.runStateVersion !== response.expectedRunStateVersion) {
    throw new Error("VNEXT_HUMAN_RESUME_RUN_STATE_STALE");
  }
  if (context.stageStateVersion !== response.expectedStageStateVersion) {
    throw new Error("VNEXT_HUMAN_RESUME_STAGE_STATE_STALE");
  }
  if (
    response.responseArtifactId.trim().length === 0 ||
    !Number.isInteger(response.responseArtifactVersion) ||
    response.responseArtifactVersion < 1
  ) {
    throw new Error("VNEXT_HUMAN_RESUME_RESPONSE_ARTIFACT_INVALID");
  }
  assertSha256(
    response.responseContentSha256,
    "VNEXT_HUMAN_RESUME_RESPONSE_HASH_INVALID",
  );
}

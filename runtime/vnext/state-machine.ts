export type RunStatus =
  | "CREATED"
  | "ACTIVE"
  | "PAUSED"
  | "BLOCKED"
  | "READY_TO_PUBLISH"
  | "PUBLISHED"
  | "CANCELLED";

export type StageCode = "RESEARCH" | "DEEP_DIVE" | "INTEGRATION";

export type StageLifecycle =
  | "NOT_STARTED"
  | "IN_PROGRESS"
  | "PAUSED"
  | "BLOCKED"
  | "COMPLETE";

export type HandoffState = "NOT_EVALUATED" | "YES" | "NO";

export type ManifestKind = "CHECKPOINT" | "FINAL";

export type StageTransitionMode = "NORMAL" | "REOPEN";

export type StageState = {
  stageCode: StageCode;
  lifecycle: StageLifecycle;
  handoff: HandoffState;
  manifestKind: ManifestKind | null;
  stageRevision: number;
  criticalBlockerCount?: number;
  contractStatusCode?: string | null;
};

export type ReadyToPublishGuards = {
  snapshotCandidateResolved: boolean;
  schemaValidationPass: boolean;
  i2ReconciliationPass: boolean;
  i3bAdmissionPass: boolean;
  noPublicationBlocker: boolean;
};

export type PublishResult = "SUCCEEDED" | "FAILED";

export const RUN_STATUSES: readonly RunStatus[] = [
  "CREATED",
  "ACTIVE",
  "PAUSED",
  "BLOCKED",
  "READY_TO_PUBLISH",
  "PUBLISHED",
  "CANCELLED",
];

export const STAGE_LIFECYCLES: readonly StageLifecycle[] = [
  "NOT_STARTED",
  "IN_PROGRESS",
  "PAUSED",
  "BLOCKED",
  "COMPLETE",
];

export const RUN_TRANSITIONS: Record<RunStatus, readonly RunStatus[]> = {
  CREATED: ["ACTIVE", "PAUSED", "BLOCKED", "CANCELLED"],
  ACTIVE: ["PAUSED", "BLOCKED", "READY_TO_PUBLISH", "CANCELLED"],
  PAUSED: ["ACTIVE", "BLOCKED", "CANCELLED"],
  BLOCKED: ["ACTIVE", "PAUSED", "CANCELLED"],
  READY_TO_PUBLISH: ["PUBLISHED", "BLOCKED", "CANCELLED"],
  PUBLISHED: [],
  CANCELLED: [],
};

export const NORMAL_STAGE_TRANSITIONS: Record<
  StageLifecycle,
  readonly StageLifecycle[]
> = {
  NOT_STARTED: ["IN_PROGRESS"],
  IN_PROGRESS: ["PAUSED", "BLOCKED", "COMPLETE"],
  PAUSED: ["IN_PROGRESS", "BLOCKED"],
  BLOCKED: ["IN_PROGRESS", "PAUSED"],
  COMPLETE: [],
};

export const HANDOFF_GATE_BY_STAGE: Record<StageCode, string> = {
  RESEARCH: "READY_FOR_DEEP_DIVE",
  DEEP_DIVE: "READY_FOR_INTEGRATION",
  INTEGRATION: "READY_TO_PUBLISH",
};

export function isLegalRunTransition(
  from: RunStatus,
  to: RunStatus,
): boolean {
  return RUN_TRANSITIONS[from].includes(to);
}

export function assertRunTransition(
  from: RunStatus,
  to: RunStatus,
): void {
  if (!isLegalRunTransition(from, to)) {
    throw new Error(`VNEXT_ILLEGAL_RUN_TRANSITION: ${from} -> ${to}`);
  }
}

export function isTerminalRunStatus(status: RunStatus): boolean {
  return status === "PUBLISHED" || status === "CANCELLED";
}

export function isLegalStageTransition(
  from: StageLifecycle,
  to: StageLifecycle,
  mode: StageTransitionMode = "NORMAL",
): boolean {
  if (mode === "REOPEN") {
    return from === "COMPLETE" && (to === "IN_PROGRESS" || to === "BLOCKED");
  }

  return NORMAL_STAGE_TRANSITIONS[from].includes(to);
}

export function assertStageTransition(
  from: StageLifecycle,
  to: StageLifecycle,
  mode: StageTransitionMode = "NORMAL",
): void {
  if (!isLegalStageTransition(from, to, mode)) {
    throw new Error(
      `VNEXT_ILLEGAL_STAGE_TRANSITION: ${from} -> ${to} [${mode}]`,
    );
  }
}

export function assertStageMutable(runStatus: RunStatus): void {
  if (isTerminalRunStatus(runStatus)) {
    throw new Error(
      `VNEXT_TERMINAL_RUN_STAGE_MUTATION_FORBIDDEN: ${runStatus}`,
    );
  }
}

export function assertExpectedStateVersion(
  stored: number,
  expected: number,
): void {
  if (stored !== expected) {
    throw new Error(
      `VNEXT_CONCURRENT_STATE_CHANGE: stored=${stored} expected=${expected}`,
    );
  }
}

export function assertIdempotencyFingerprint(
  priorFingerprint: string,
  requestFingerprint: string,
): void {
  if (priorFingerprint !== requestFingerprint) {
    throw new Error("VNEXT_IDEMPOTENCY_CONFLICT");
  }
}

export function assertHandoffGateName(
  stage: StageCode,
  gateName: string,
): void {
  const expected = HANDOFF_GATE_BY_STAGE[stage];
  if (gateName !== expected) {
    throw new Error(
      `VNEXT_HANDOFF_GATE_MISMATCH: stage=${stage} expected=${expected} actual=${gateName}`,
    );
  }
}

function assertFinalCompleteYes(state: StageState, expectedStage: StageCode): void {
  if (state.stageCode !== expectedStage) {
    throw new Error(
      `VNEXT_UPSTREAM_STAGE_MISMATCH: expected=${expectedStage} actual=${state.stageCode}`,
    );
  }

  if (state.lifecycle !== "COMPLETE") {
    throw new Error(
      `VNEXT_UPSTREAM_NOT_COMPLETE: ${state.stageCode}=${state.lifecycle}`,
    );
  }

  if (state.handoff !== "YES") {
    throw new Error(
      `VNEXT_HANDOFF_NOT_YES: ${state.stageCode}=${state.handoff}`,
    );
  }

  if (state.manifestKind !== "FINAL") {
    throw new Error(
      `VNEXT_FINAL_MANIFEST_REQUIRED: ${state.stageCode}=${state.manifestKind ?? "NONE"}`,
    );
  }

  if ((state.criticalBlockerCount ?? 0) > 0) {
    throw new Error(
      `VNEXT_CRITICAL_BLOCKER_PRESENT: ${state.stageCode}`,
    );
  }
}

export function assertDownstreamAdmission(
  targetStage: Exclude<StageCode, "RESEARCH">,
  upstreamState: StageState,
): void {
  if (targetStage === "DEEP_DIVE") {
    assertFinalCompleteYes(upstreamState, "RESEARCH");
    return;
  }

  assertFinalCompleteYes(upstreamState, "DEEP_DIVE");
}

export function assertReadyToPublish(
  integrationState: StageState,
  guards: ReadyToPublishGuards,
): void {
  assertFinalCompleteYes(integrationState, "INTEGRATION");

  const failed: string[] = [];

  if (!guards.snapshotCandidateResolved) failed.push("SNAPSHOT_CANDIDATE_RESOLVED");
  if (!guards.schemaValidationPass) failed.push("SCHEMA_VALIDATION_PASS");
  if (!guards.i2ReconciliationPass) failed.push("I2_RECONCILIATION_PASS");
  if (!guards.i3bAdmissionPass) failed.push("I3B_ADMISSION_PASS");
  if (!guards.noPublicationBlocker) failed.push("NO_PUBLICATION_BLOCKER");

  if (failed.length > 0) {
    throw new Error(
      `VNEXT_READY_TO_PUBLISH_GUARD_FAILED: ${failed.join(",")}`,
    );
  }
}

export function assertPublicationAuthorized(
  runStatus: RunStatus,
  explicitAuthorization: boolean,
): void {
  if (runStatus !== "READY_TO_PUBLISH") {
    throw new Error(
      `VNEXT_PUBLISH_WRONG_RUN_STATUS: ${runStatus}`,
    );
  }

  if (!explicitAuthorization) {
    throw new Error("VNEXT_EXPLICIT_PUBLISH_AUTHORIZATION_REQUIRED");
  }
}

export function assertPauseAllowed(stage: StageState): void {
  if (stage.lifecycle !== "IN_PROGRESS") {
    throw new Error(
      `VNEXT_PAUSE_REQUIRES_IN_PROGRESS: ${stage.lifecycle}`,
    );
  }

  if (stage.manifestKind !== "CHECKPOINT") {
    throw new Error("VNEXT_PAUSE_REQUIRES_CHECKPOINT");
  }
}

export function assertResumeAllowed(stage: StageState): void {
  if (stage.lifecycle !== "PAUSED" && stage.lifecycle !== "BLOCKED") {
    throw new Error(
      `VNEXT_RESUME_REQUIRES_PAUSED_OR_BLOCKED: ${stage.lifecycle}`,
    );
  }

  if (stage.manifestKind !== "CHECKPOINT") {
    throw new Error("VNEXT_RESUME_REQUIRES_CHECKPOINT");
  }
}

export function runStatusForCheckpoint(
  targetLifecycle: Extract<
    StageLifecycle,
    "IN_PROGRESS" | "PAUSED" | "BLOCKED"
  >,
): Extract<RunStatus, "ACTIVE" | "PAUSED" | "BLOCKED"> {
  if (targetLifecycle === "PAUSED") return "PAUSED";
  if (targetLifecycle === "BLOCKED") return "BLOCKED";
  return "ACTIVE";
}

export function nextReopenedStageState(
  prior: StageState,
  targetLifecycle: Extract<StageLifecycle, "IN_PROGRESS" | "BLOCKED">,
): StageState {
  if (prior.lifecycle !== "COMPLETE") {
    throw new Error(
      `VNEXT_REOPEN_REQUIRES_COMPLETE: ${prior.lifecycle}`,
    );
  }

  assertStageTransition(prior.lifecycle, targetLifecycle, "REOPEN");

  return {
    ...prior,
    lifecycle: targetLifecycle,
    handoff: "NOT_EVALUATED",
    manifestKind: null,
    stageRevision: prior.stageRevision + 1,
    criticalBlockerCount:
      targetLifecycle === "BLOCKED"
        ? Math.max(prior.criticalBlockerCount ?? 0, 1)
        : 0,
    contractStatusCode:
      targetLifecycle === "BLOCKED" ? "REOPENED_BLOCKED" : null,
  };
}

export function downstreamStateAfterUpstreamReopen(
  prior: StageState,
): StageState {
  return {
    ...prior,
    lifecycle: "BLOCKED",
    handoff: "NOT_EVALUATED",
    manifestKind: null,
    stageRevision:
      prior.lifecycle === "COMPLETE"
        ? prior.stageRevision + 1
        : prior.stageRevision,
    criticalBlockerCount: Math.max(prior.criticalBlockerCount ?? 0, 1),
    contractStatusCode: "UPSTREAM_STAGE_REOPENED",
  };
}

export function runStatusAfterStageReopen(
  currentStatus: RunStatus,
  targetLifecycle: Extract<StageLifecycle, "IN_PROGRESS" | "BLOCKED">,
): Extract<RunStatus, "ACTIVE" | "BLOCKED"> {
  if (isTerminalRunStatus(currentStatus)) {
    throw new Error(
      `VNEXT_TERMINAL_RUN_REOPEN_FORBIDDEN: ${currentStatus}`,
    );
  }

  return targetLifecycle === "BLOCKED" ? "BLOCKED" : "ACTIVE";
}

export function runStatusAfterPublishResult(
  result: PublishResult,
  recoverable: boolean,
): Extract<RunStatus, "PUBLISHED" | "READY_TO_PUBLISH" | "BLOCKED"> {
  if (result === "SUCCEEDED") return "PUBLISHED";
  return recoverable ? "READY_TO_PUBLISH" : "BLOCKED";
}

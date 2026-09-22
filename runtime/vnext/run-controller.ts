import {
  assertExpectedStateVersion,
  assertIdempotencyFingerprint,
  assertRunTransition,
  type RunStatus,
} from "./state-machine";

export type RunControllerEnvironment = "VNEXT_SHADOW";

export interface RunControllerRunRecord {
  runId: string;
  status: RunStatus;
  stateVersion: number;
}

export interface RunControllerOperationReceipt {
  operationId: string;
  runId: string;
  requestFingerprint: string;
  fromStatus: RunStatus;
  toStatus: RunStatus;
  beforeStateVersion: number;
  afterStateVersion: number;
}

export interface RunControllerReadContext {
  run: RunControllerRunRecord;
  priorOperation: RunControllerOperationReceipt | null;
}

export interface RunControllerTransitionMutation {
  runId: string;
  operationId: string;
  requestFingerprint: string;
  expectedStateVersion: number;
  fromStatus: RunStatus;
  toStatus: RunStatus;
}

export interface RunControllerStore {
  readContext(
    runId: string,
    operationId: string,
  ): Promise<RunControllerReadContext>;

  compareAndSetTransition(
    mutation: RunControllerTransitionMutation,
  ): Promise<void>;
}

export interface RunControllerTransitionRequest {
  runId: string;
  operationId: string;
  requestFingerprint: string;
  expectedStateVersion: number;
  targetStatus: RunStatus;
}

export type RunControllerTransitionDisposition =
  | "APPLIED"
  | "IDEMPOTENT_REPLAY";

export interface RunControllerTransitionResult {
  disposition: RunControllerTransitionDisposition;
  operationId: string;
  before: RunControllerRunRecord;
  after: RunControllerRunRecord;
}

export interface RunControllerOptions {
  environment: RunControllerEnvironment;
}

function copyRun(run: RunControllerRunRecord): RunControllerRunRecord {
  return { ...run };
}

function assertShadowPublicationFirewall(
  environment: RunControllerEnvironment,
  targetStatus: RunStatus,
): void {
  if (environment === "VNEXT_SHADOW" && targetStatus === "PUBLISHED") {
    throw new Error("VNEXT_SHADOW_PUBLICATION_FORBIDDEN");
  }
}

function assertReceiptMatchesRequest(
  receipt: RunControllerOperationReceipt,
  request: RunControllerTransitionRequest,
): void {
  assertIdempotencyFingerprint(
    receipt.requestFingerprint,
    request.requestFingerprint,
  );

  if (receipt.runId !== request.runId) {
    throw new Error(
      `VNEXT_IDEMPOTENCY_RUN_MISMATCH: stored=${receipt.runId} requested=${request.runId}`,
    );
  }

  if (receipt.toStatus !== request.targetStatus) {
    throw new Error(
      `VNEXT_IDEMPOTENCY_TARGET_MISMATCH: stored=${receipt.toStatus} requested=${request.targetStatus}`,
    );
  }
}

function assertPostMutationState(
  before: RunControllerRunRecord,
  after: RunControllerRunRecord,
  receipt: RunControllerOperationReceipt | null,
  request: RunControllerTransitionRequest,
): asserts receipt is RunControllerOperationReceipt {
  if (after.runId !== before.runId) {
    throw new Error("VNEXT_RUN_CONTROLLER_REREAD_RUN_MISMATCH");
  }

  if (after.status !== request.targetStatus) {
    throw new Error(
      `VNEXT_RUN_CONTROLLER_REREAD_STATUS_MISMATCH: expected=${request.targetStatus} actual=${after.status}`,
    );
  }

  if (after.stateVersion !== before.stateVersion + 1) {
    throw new Error(
      `VNEXT_RUN_CONTROLLER_REREAD_VERSION_MISMATCH: before=${before.stateVersion} after=${after.stateVersion}`,
    );
  }

  if (!receipt) {
    throw new Error("VNEXT_RUN_CONTROLLER_RECEIPT_MISSING_AFTER_MUTATION");
  }

  assertReceiptMatchesRequest(receipt, request);

  if (
    receipt.fromStatus !== before.status ||
    receipt.beforeStateVersion !== before.stateVersion ||
    receipt.afterStateVersion !== after.stateVersion
  ) {
    throw new Error("VNEXT_RUN_CONTROLLER_RECEIPT_STATE_MISMATCH");
  }
}

/**
 * Gate 8 deterministic transition controller.
 *
 * Its execution protocol is intentionally narrow:
 *
 * READ
 * -> determine legal transition
 * -> execute one deterministic compare-and-set mutation
 * -> REREAD
 *
 * No AI provider, chat context, analytical judgment, or publication authority
 * participates in this function.
 */
export async function executeRunTransition(
  store: RunControllerStore,
  request: RunControllerTransitionRequest,
  options: RunControllerOptions = { environment: "VNEXT_SHADOW" },
): Promise<RunControllerTransitionResult> {
  const initial = await store.readContext(
    request.runId,
    request.operationId,
  );
  const before = copyRun(initial.run);

  if (initial.priorOperation) {
    assertReceiptMatchesRequest(initial.priorOperation, request);

    const replayRead = await store.readContext(
      request.runId,
      request.operationId,
    );

    return {
      disposition: "IDEMPOTENT_REPLAY",
      operationId: request.operationId,
      before,
      after: copyRun(replayRead.run),
    };
  }

  assertExpectedStateVersion(
    initial.run.stateVersion,
    request.expectedStateVersion,
  );
  assertShadowPublicationFirewall(options.environment, request.targetStatus);
  assertRunTransition(initial.run.status, request.targetStatus);

  await store.compareAndSetTransition({
    runId: request.runId,
    operationId: request.operationId,
    requestFingerprint: request.requestFingerprint,
    expectedStateVersion: request.expectedStateVersion,
    fromStatus: initial.run.status,
    toStatus: request.targetStatus,
  });

  const final = await store.readContext(
    request.runId,
    request.operationId,
  );

  assertPostMutationState(
    before,
    final.run,
    final.priorOperation,
    request,
  );

  return {
    disposition: "APPLIED",
    operationId: request.operationId,
    before,
    after: copyRun(final.run),
  };
}

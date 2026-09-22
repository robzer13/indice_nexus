import assert from "node:assert/strict";
import test from "node:test";

import {
  assertHumanEscalationResumeAllowed,
  executeHumanEscalation,
  type CheckpointAndPauseReceipt,
  type HumanEscalationContext,
  type HumanEscalationPackage,
  type HumanEscalationRequest,
  type HumanEscalationResponse,
  type HumanEscalationStore,
} from "../runtime/vnext/human-escalation";

const RUN_ID = "run-gate14";
const HASH_A = "a".repeat(64);
const HASH_B = "b".repeat(64);
const HASH_C = "c".repeat(64);

function context(): HumanEscalationContext {
  return {
    runId: RUN_ID,
    stageCode: "DEEP_DIVE",
    runStatus: "ACTIVE",
    stageLifecycle: "IN_PROGRESS",
    runStateVersion: 5,
    stageStateVersion: 9,
    stageRevision: 2,
    dataCutoff: "2026-09-21",
    contractSetSha256: HASH_A,
    activeManifestArtifactId: null,
    activeManifestVersion: null,
    activeManifestKind: null,
    blockerCodes: ["MATERIAL_CONFLICT"],
  };
}

function request(): HumanEscalationRequest {
  return {
    escalationId: "esc-gate14-001",
    runId: RUN_ID,
    stageCode: "DEEP_DIVE",
    reason: "MATERIAL_CONFLICT",
    requestedHumanAction:
      "Review the unresolved material evidence conflict and determine the acceptable next analytical action.",
    exactArtifacts: [
      {
        artifactId: "evidence-ledger",
        version: 3,
        contentSha256: HASH_B,
        artifactType: "EVIDENCE_LEDGER",
      },
    ],
    expectedRunStateVersion: 5,
    expectedStageStateVersion: 9,
  };
}

class InMemoryStore implements HumanEscalationStore {
  current = context();
  checkpointCalls = 0;
  packages: HumanEscalationPackage[] = [];
  corruptAfterPause = false;

  async readContext(): Promise<HumanEscalationContext> {
    return structuredClone(this.current);
  }

  async checkpointAndPauseForHuman(): Promise<CheckpointAndPauseReceipt> {
    this.checkpointCalls += 1;

    this.current = {
      ...this.current,
      runStatus: "PAUSED",
      stageLifecycle: "PAUSED",
      runStateVersion: this.current.runStateVersion + 1,
      stageStateVersion: this.current.stageStateVersion + 1,
      activeManifestArtifactId: "checkpoint-g14",
      activeManifestVersion: 1,
      activeManifestKind: "CHECKPOINT",
      dataCutoff: this.corruptAfterPause
        ? "2026-09-22"
        : this.current.dataCutoff,
    };

    return {
      runId: RUN_ID,
      stageCode: "DEEP_DIVE",
      checkpointManifestArtifactId: "checkpoint-g14",
      checkpointManifestVersion: 1,
      checkpointManifestSha256: HASH_C,
      priorRunStateVersion: 5,
      priorStageStateVersion: 9,
      nextRunStateVersion: 6,
      nextStageStateVersion: 10,
    };
  }

  async persistEscalationPackage(
    escalation: HumanEscalationPackage,
  ): Promise<void> {
    this.packages.push(structuredClone(escalation));
  }
}

test("Gate 14 checkpoints and pauses before creating a Work handoff", async () => {
  const store = new InMemoryStore();

  const result = await executeHumanEscalation(store, request());

  assert.equal(store.checkpointCalls, 1);
  assert.equal(result.afterPause.runStatus, "PAUSED");
  assert.equal(result.afterPause.stageLifecycle, "PAUSED");
  assert.equal(result.afterPause.activeManifestKind, "CHECKPOINT");
  assert.equal(store.packages.length, 1);

  assert.deepEqual(result.escalation.workBoundary, {
    invocationMode: "HUMAN_OPENED_CHATGPT_WORK",
    runtimeApiInvocationAllowed: false,
    productionMutationAllowed: false,
    registryMutationAllowed: false,
    publicationAllowed: false,
  });
});

test("Gate 14 handoff is anchored to exact checkpoint and artifact hashes", async () => {
  const store = new InMemoryStore();
  const result = await executeHumanEscalation(store, request());

  assert.deepEqual(result.escalation.createdFromCheckpoint, {
    artifactId: "checkpoint-g14",
    version: 1,
    contentSha256: HASH_C,
  });

  assert.deepEqual(result.escalation.exactArtifacts, [
    {
      artifactId: "evidence-ledger",
      version: 3,
      contentSha256: HASH_B,
      artifactType: "EVIDENCE_LEDGER",
    },
  ]);
});

test("Gate 14 refuses stale state before checkpoint mutation", async () => {
  const store = new InMemoryStore();
  const stale = request();
  stale.expectedStageStateVersion = 8;

  await assert.rejects(
    executeHumanEscalation(store, stale),
    /VNEXT_HUMAN_ESCALATION_STAGE_STATE_STALE/,
  );

  assert.equal(store.checkpointCalls, 0);
  assert.equal(store.current.stageLifecycle, "IN_PROGRESS");
});

test("Gate 14 refuses terminal runs and completed stages", async () => {
  const terminal = new InMemoryStore();
  terminal.current.runStatus = "PUBLISHED";

  await assert.rejects(
    executeHumanEscalation(terminal, request()),
    /VNEXT_HUMAN_ESCALATION_TERMINAL_RUN/,
  );

  const completed = new InMemoryStore();
  completed.current.stageLifecycle = "COMPLETE";

  await assert.rejects(
    executeHumanEscalation(completed, request()),
    /VNEXT_HUMAN_ESCALATION_STAGE_COMPLETE/,
  );
});

test("Gate 14 detects immutable drift introduced during checkpoint pause", async () => {
  const store = new InMemoryStore();
  store.corruptAfterPause = true;

  await assert.rejects(
    executeHumanEscalation(store, request()),
    /VNEXT_HUMAN_ESCALATION_DATA_CUTOFF_DRIFT/,
  );

  assert.equal(store.packages.length, 0);
});

test("Gate 14 remains durably paused if external Work never returns", async () => {
  const store = new InMemoryStore();

  await executeHumanEscalation(store, request());

  assert.equal(store.current.runStatus, "PAUSED");
  assert.equal(store.current.stageLifecycle, "PAUSED");
  assert.equal(store.current.activeManifestKind, "CHECKPOINT");
  assert.equal(
    store.current.activeManifestArtifactId,
    "checkpoint-g14",
  );
});

function acceptedResponse(
  ctx: HumanEscalationContext,
): HumanEscalationResponse {
  return {
    escalationId: "esc-gate14-001",
    runId: RUN_ID,
    stageCode: "DEEP_DIVE",
    responseArtifactId: "human-response-1",
    responseArtifactVersion: 1,
    responseContentSha256: HASH_B,
    acceptedByHuman: true,
    expectedRunStateVersion: ctx.runStateVersion,
    expectedStageStateVersion: ctx.stageStateVersion,
    checkpointManifestArtifactId:
      ctx.activeManifestArtifactId ?? "",
    checkpointManifestVersion:
      ctx.activeManifestVersion ?? 0,
  };
}

test("Gate 14 resume requires human-accepted hashed response and exact checkpoint", async () => {
  const store = new InMemoryStore();
  await executeHumanEscalation(store, request());

  const response = acceptedResponse(store.current);

  assert.doesNotThrow(() =>
    assertHumanEscalationResumeAllowed(
      store.current,
      response,
    ),
  );
});

test("Gate 14 resume fails closed on stale checkpoint or unaccepted response", async () => {
  const store = new InMemoryStore();
  await executeHumanEscalation(store, request());

  const wrongCheckpoint = acceptedResponse(store.current);
  wrongCheckpoint.checkpointManifestArtifactId = "old-checkpoint";

  assert.throws(
    () =>
      assertHumanEscalationResumeAllowed(
        store.current,
        wrongCheckpoint,
      ),
    /VNEXT_HUMAN_RESUME_CHECKPOINT_MISMATCH/,
  );

  const rejected = acceptedResponse(store.current);
  rejected.acceptedByHuman = false;

  assert.throws(
    () =>
      assertHumanEscalationResumeAllowed(
        store.current,
        rejected,
      ),
    /VNEXT_HUMAN_RESUME_RESPONSE_NOT_ACCEPTED/,
  );
});

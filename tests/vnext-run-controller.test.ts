import assert from "node:assert/strict";
import test from "node:test";

import {
  executeRunTransition,
  type RunControllerReadContext,
  type RunControllerRunRecord,
  type RunControllerStore,
  type RunControllerTransitionMutation,
} from "../runtime/vnext/run-controller";
import type { RunStatus } from "../runtime/vnext/state-machine";

class InMemoryRunControllerStore implements RunControllerStore {
  private run: RunControllerRunRecord;
  private readonly receipts = new Map<
    string,
    RunControllerReadContext["priorOperation"]
  >();

  mutationCount = 0;
  mutateBeforeNextCas:
    | ((run: RunControllerRunRecord) => RunControllerRunRecord)
    | null = null;

  constructor(
    runId = "fake-run-gate8",
    status: RunStatus = "CREATED",
    stateVersion = 1,
  ) {
    this.run = { runId, status, stateVersion };
  }

  snapshot(): RunControllerRunRecord {
    return { ...this.run };
  }

  async readContext(
    runId: string,
    operationId: string,
  ): Promise<RunControllerReadContext> {
    if (runId !== this.run.runId) {
      throw new Error(`VNEXT_FAKE_RUN_NOT_FOUND: ${runId}`);
    }

    return {
      run: { ...this.run },
      priorOperation: this.receipts.get(operationId) ?? null,
    };
  }

  async compareAndSetTransition(
    mutation: RunControllerTransitionMutation,
  ): Promise<void> {
    if (this.mutateBeforeNextCas) {
      this.run = this.mutateBeforeNextCas({ ...this.run });
      this.mutateBeforeNextCas = null;
    }

    const existing = this.receipts.get(mutation.operationId);
    if (existing) {
      if (existing.requestFingerprint !== mutation.requestFingerprint) {
        throw new Error("VNEXT_FAKE_IDEMPOTENCY_CONFLICT");
      }
      return;
    }

    if (this.run.runId !== mutation.runId) {
      throw new Error("VNEXT_FAKE_RUN_MISMATCH");
    }

    if (
      this.run.stateVersion !== mutation.expectedStateVersion ||
      this.run.status !== mutation.fromStatus
    ) {
      throw new Error("VNEXT_FAKE_CAS_CONFLICT");
    }

    const beforeStateVersion = this.run.stateVersion;
    this.run = {
      ...this.run,
      status: mutation.toStatus,
      stateVersion: beforeStateVersion + 1,
    };
    this.mutationCount += 1;

    this.receipts.set(mutation.operationId, {
      operationId: mutation.operationId,
      runId: mutation.runId,
      requestFingerprint: mutation.requestFingerprint,
      fromStatus: mutation.fromStatus,
      toStatus: mutation.toStatus,
      beforeStateVersion,
      afterStateVersion: this.run.stateVersion,
    });
  }
}

let opCounter = 0;

function request(
  store: InMemoryRunControllerStore,
  targetStatus: RunStatus,
) {
  opCounter += 1;
  const run = store.snapshot();
  return {
    runId: run.runId,
    operationId: `gate8-op-${opCounter}`,
    requestFingerprint: `sha256:gate8-${opCounter}-${targetStatus}`,
    expectedStateVersion: run.stateVersion,
    targetStatus,
  };
}

test("Gate 8 simulates a complete fake run-control lifecycle without ChatGPT", async () => {
  const store = new InMemoryRunControllerStore();

  const path: RunStatus[] = [
    "ACTIVE",
    "PAUSED",
    "ACTIVE",
    "BLOCKED",
    "ACTIVE",
    "READY_TO_PUBLISH",
  ];

  for (const target of path) {
    const result = await executeRunTransition(
      store,
      request(store, target),
    );

    assert.equal(result.disposition, "APPLIED");
    assert.equal(result.after.status, target);
    assert.equal(
      result.after.stateVersion,
      result.before.stateVersion + 1,
    );
  }

  assert.deepEqual(store.snapshot(), {
    runId: "fake-run-gate8",
    status: "READY_TO_PUBLISH",
    stateVersion: 7,
  });
  assert.equal(store.mutationCount, 6);
});

test("Gate 8 hard-blocks publication in VNEXT_SHADOW", async () => {
  const store = new InMemoryRunControllerStore(
    "shadow-ready",
    "READY_TO_PUBLISH",
    9,
  );

  await assert.rejects(
    executeRunTransition(store, request(store, "PUBLISHED")),
    /VNEXT_SHADOW_PUBLICATION_FORBIDDEN/,
  );

  assert.deepEqual(store.snapshot(), {
    runId: "shadow-ready",
    status: "READY_TO_PUBLISH",
    stateVersion: 9,
  });
  assert.equal(store.mutationCount, 0);
});

test("Gate 8 rejects an illegal run transition before mutation", async () => {
  const store = new InMemoryRunControllerStore();

  await assert.rejects(
    executeRunTransition(
      store,
      request(store, "READY_TO_PUBLISH"),
    ),
    /VNEXT_ILLEGAL_RUN_TRANSITION/,
  );

  assert.equal(store.snapshot().status, "CREATED");
  assert.equal(store.mutationCount, 0);
});

test("Gate 8 rejects stale expected state_version before mutation", async () => {
  const store = new InMemoryRunControllerStore();
  const req = request(store, "ACTIVE");
  req.expectedStateVersion = 0;

  await assert.rejects(
    executeRunTransition(store, req),
    /VNEXT_CONCURRENT_STATE_CHANGE/,
  );

  assert.equal(store.mutationCount, 0);
});

test("Gate 8 compare-and-set closes a race after initial READ", async () => {
  const store = new InMemoryRunControllerStore();
  const req = request(store, "ACTIVE");

  store.mutateBeforeNextCas = (run) => ({
    ...run,
    stateVersion: run.stateVersion + 1,
  });

  await assert.rejects(
    executeRunTransition(store, req),
    /VNEXT_FAKE_CAS_CONFLICT/,
  );

  assert.equal(store.mutationCount, 0);
});

test("Gate 8 same operation + same fingerprint is idempotent", async () => {
  const store = new InMemoryRunControllerStore();
  const req = request(store, "ACTIVE");

  const first = await executeRunTransition(store, req);
  const replay = await executeRunTransition(store, req);

  assert.equal(first.disposition, "APPLIED");
  assert.equal(replay.disposition, "IDEMPOTENT_REPLAY");
  assert.equal(store.mutationCount, 1);
  assert.equal(store.snapshot().stateVersion, 2);
  assert.equal(store.snapshot().status, "ACTIVE");
});

test("Gate 8 same operation + different fingerprint fails closed", async () => {
  const store = new InMemoryRunControllerStore();
  const req = request(store, "ACTIVE");

  await executeRunTransition(store, req);

  await assert.rejects(
    executeRunTransition(store, {
      ...req,
      requestFingerprint: "sha256:conflicting-request",
    }),
    /VNEXT_IDEMPOTENCY_CONFLICT/,
  );

  assert.equal(store.mutationCount, 1);
});

test("Gate 8 same operation cannot be replayed against another target", async () => {
  const store = new InMemoryRunControllerStore();
  const req = request(store, "ACTIVE");

  await executeRunTransition(store, req);

  await assert.rejects(
    executeRunTransition(store, {
      ...req,
      targetStatus: "PAUSED",
    }),
    /VNEXT_IDEMPOTENCY_TARGET_MISMATCH/,
  );

  assert.equal(store.mutationCount, 1);
});

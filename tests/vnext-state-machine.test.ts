import assert from "node:assert/strict";
import test from "node:test";

import {
  HANDOFF_GATE_BY_STAGE,
  assertDownstreamAdmission,
  assertExpectedStateVersion,
  assertHandoffGateName,
  assertIdempotencyFingerprint,
  assertPublicationAuthorized,
  assertReadyToPublish,
  assertRunTransition,
  assertStageMutable,
  assertStageTransition,
  isLegalRunTransition,
  isLegalStageTransition,
  isTerminalRunStatus,
  nextReopenedStageState,
  runStatusAfterUpstreamReopen,
  type RunStatus,
  type StageLifecycle,
  type StageState,
} from "../runtime/vnext/state-machine";

const RUN_LEGAL: Record<RunStatus, readonly RunStatus[]> = {
  CREATED: ["ACTIVE", "PAUSED", "BLOCKED", "CANCELLED"],
  ACTIVE: ["PAUSED", "BLOCKED", "READY_TO_PUBLISH", "CANCELLED"],
  PAUSED: ["ACTIVE", "BLOCKED", "CANCELLED"],
  BLOCKED: ["ACTIVE", "PAUSED", "CANCELLED"],
  READY_TO_PUBLISH: ["PUBLISHED", "BLOCKED", "CANCELLED"],
  PUBLISHED: [],
  CANCELLED: [],
};

const RUN_STATUSES = Object.keys(RUN_LEGAL) as RunStatus[];

test("G6-01/G6-02 run transition matrix is exact", () => {
  for (const from of RUN_STATUSES) {
    for (const to of RUN_STATUSES) {
      assert.equal(
        isLegalRunTransition(from, to),
        RUN_LEGAL[from].includes(to),
        `${from} -> ${to}`,
      );
    }
  }
});

test("G6-03/G6-04 terminal run states reject all transitions", () => {
  for (const terminal of ["PUBLISHED", "CANCELLED"] as const) {
    assert.equal(isTerminalRunStatus(terminal), true);
    for (const to of RUN_STATUSES) {
      assert.equal(isLegalRunTransition(terminal, to), false);
      assert.throws(() => assertRunTransition(terminal, to));
    }
    assert.throws(() => assertStageMutable(terminal));
  }
});

const STAGE_LEGAL: Record<StageLifecycle, readonly StageLifecycle[]> = {
  NOT_STARTED: ["IN_PROGRESS"],
  IN_PROGRESS: ["PAUSED", "BLOCKED", "COMPLETE"],
  PAUSED: ["IN_PROGRESS", "BLOCKED"],
  BLOCKED: ["IN_PROGRESS", "PAUSED"],
  COMPLETE: [],
};

const STAGE_STATUSES = Object.keys(STAGE_LEGAL) as StageLifecycle[];

test("G6-05 normal stage transition matrix is exact", () => {
  for (const from of STAGE_STATUSES) {
    for (const to of STAGE_STATUSES) {
      assert.equal(
        isLegalStageTransition(from, to, "NORMAL"),
        STAGE_LEGAL[from].includes(to),
        `${from} -> ${to}`,
      );
    }
  }
});

test("G6-06/G6-07 COMPLETE can reopen only through REOPEN", () => {
  assert.equal(isLegalStageTransition("COMPLETE", "IN_PROGRESS"), false);
  assert.equal(
    isLegalStageTransition("COMPLETE", "IN_PROGRESS", "REOPEN"),
    true,
  );
  assert.equal(
    isLegalStageTransition("COMPLETE", "BLOCKED", "REOPEN"),
    true,
  );
  assert.throws(() =>
    assertStageTransition("COMPLETE", "IN_PROGRESS", "NORMAL"),
  );
});

const completeResearch: StageState = {
  stageCode: "RESEARCH",
  lifecycle: "COMPLETE",
  handoff: "YES",
  manifestKind: "FINAL",
  stageRevision: 1,
  criticalBlockerCount: 0,
};

const completeDeepDive: StageState = {
  stageCode: "DEEP_DIVE",
  lifecycle: "COMPLETE",
  handoff: "YES",
  manifestKind: "FINAL",
  stageRevision: 1,
  criticalBlockerCount: 0,
};

const completeIntegration: StageState = {
  stageCode: "INTEGRATION",
  lifecycle: "COMPLETE",
  handoff: "YES",
  manifestKind: "FINAL",
  stageRevision: 1,
  criticalBlockerCount: 0,
};

test("G6-08 CHECKPOINT cannot admit downstream", () => {
  assert.throws(() =>
    assertDownstreamAdmission("DEEP_DIVE", {
      ...completeResearch,
      manifestKind: "CHECKPOINT",
    }),
  );
});

test("G6-09/G6-10 Research admission to Deep Dive is FINAL + COMPLETE + YES", () => {
  assert.doesNotThrow(() =>
    assertDownstreamAdmission("DEEP_DIVE", completeResearch),
  );

  assert.throws(() =>
    assertDownstreamAdmission("DEEP_DIVE", {
      ...completeResearch,
      handoff: "NO",
    }),
  );
  assert.throws(() =>
    assertDownstreamAdmission("DEEP_DIVE", {
      ...completeResearch,
      lifecycle: "BLOCKED",
    }),
  );
});

test("G6-11/G6-12 Deep Dive admission to Integration is FINAL + COMPLETE + YES", () => {
  assert.doesNotThrow(() =>
    assertDownstreamAdmission("INTEGRATION", completeDeepDive),
  );

  assert.throws(() =>
    assertDownstreamAdmission("INTEGRATION", {
      ...completeDeepDive,
      handoff: "NO",
    }),
  );
});

test("G6-13 Integration must satisfy every READY_TO_PUBLISH guard", () => {
  const allPass = {
    snapshotCandidateResolved: true,
    schemaValidationPass: true,
    i2ReconciliationPass: true,
    i3bAdmissionPass: true,
    noPublicationBlocker: true,
  };

  assert.doesNotThrow(() =>
    assertReadyToPublish(completeIntegration, allPass),
  );

  for (const key of Object.keys(allPass) as Array<keyof typeof allPass>) {
    assert.throws(() =>
      assertReadyToPublish(completeIntegration, {
        ...allPass,
        [key]: false,
      }),
    );
  }
});

test("G6-14 publication requires READY_TO_PUBLISH plus explicit authorization", () => {
  assert.doesNotThrow(() =>
    assertPublicationAuthorized("READY_TO_PUBLISH", true),
  );
  assert.throws(() =>
    assertPublicationAuthorized("READY_TO_PUBLISH", false),
  );
  assert.throws(() => assertPublicationAuthorized("ACTIVE", true));
});

test("G6-15/G6-16 reopen increments revision and resets handoff", () => {
  const reopened = nextReopenedStageState(completeDeepDive, "IN_PROGRESS");
  assert.equal(reopened.stageRevision, 2);
  assert.equal(reopened.lifecycle, "IN_PROGRESS");
  assert.equal(reopened.handoff, "NOT_EVALUATED");
  assert.equal(reopened.manifestKind, "FINAL");
});

test("G6-17 reopen preserves prior FINAL identity in pure transition output", () => {
  const reopened = nextReopenedStageState(completeResearch, "BLOCKED");
  assert.equal(reopened.manifestKind, "FINAL");
  assert.equal(reopened.stageCode, "RESEARCH");
});

test("G6-19 READY_TO_PUBLISH becomes BLOCKED on upstream reopen", () => {
  assert.equal(
    runStatusAfterUpstreamReopen("READY_TO_PUBLISH"),
    "BLOCKED",
  );
  assert.equal(runStatusAfterUpstreamReopen("ACTIVE"), "ACTIVE");
  assert.throws(() => runStatusAfterUpstreamReopen("PUBLISHED"));
  assert.throws(() => runStatusAfterUpstreamReopen("CANCELLED"));
});

test("G6-20 optimistic concurrency fails closed", () => {
  assert.doesNotThrow(() => assertExpectedStateVersion(7, 7));
  assert.throws(() => assertExpectedStateVersion(8, 7));
});

test("G6-21 conflicting idempotency fingerprint fails closed", () => {
  assert.doesNotThrow(() =>
    assertIdempotencyFingerprint("abc", "abc"),
  );
  assert.throws(() =>
    assertIdempotencyFingerprint("abc", "def"),
  );
});

test("G6-22 active run permits stage mutation", () => {
  assert.doesNotThrow(() => assertStageMutable("ACTIVE"));
  assert.doesNotThrow(() => assertStageMutable("BLOCKED"));
});

test("G6-25 stage-specific handoff gate vocabulary is exact", () => {
  assert.deepEqual(HANDOFF_GATE_BY_STAGE, {
    RESEARCH: "READY_FOR_DEEP_DIVE",
    DEEP_DIVE: "READY_FOR_INTEGRATION",
    INTEGRATION: "READY_TO_PUBLISH",
  });

  assert.doesNotThrow(() =>
    assertHandoffGateName("RESEARCH", "READY_FOR_DEEP_DIVE"),
  );
  assert.throws(() =>
    assertHandoffGateName("RESEARCH", "READY_FOR_INTEGRATION"),
  );
});

test("critical blocker invalidates otherwise complete upstream handoff", () => {
  assert.throws(() =>
    assertDownstreamAdmission("DEEP_DIVE", {
      ...completeResearch,
      criticalBlockerCount: 1,
    }),
  );
});

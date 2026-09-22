import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  HANDOFF_GATE_BY_STAGE,
  RUN_STATUSES,
  STAGE_LIFECYCLES,
  assertDownstreamAdmission,
  assertExpectedStateVersion,
  assertHandoffGateName,
  assertIdempotencyFingerprint,
  assertPauseAllowed,
  assertPublicationAuthorized,
  assertReadyToPublish,
  assertResumeAllowed,
  assertRunTransition,
  assertStageMutable,
  assertStageTransition,
  downstreamStateAfterUpstreamReopen,
  isLegalRunTransition,
  isLegalStageTransition,
  isTerminalRunStatus,
  nextReopenedStageState,
  runStatusAfterPublishResult,
  runStatusAfterStageReopen,
  runStatusForCheckpoint,
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

test("G6-05 normal stage transition matrix is exact", () => {
  for (const from of STAGE_LIFECYCLES) {
    for (const to of STAGE_LIFECYCLES) {
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

test("G6-15/G6-16 reopen increments revision resets gate and clears active manifest", () => {
  const reopened = nextReopenedStageState(completeDeepDive, "IN_PROGRESS");
  assert.equal(reopened.stageRevision, 2);
  assert.equal(reopened.lifecycle, "IN_PROGRESS");
  assert.equal(reopened.handoff, "NOT_EVALUATED");
  assert.equal(reopened.manifestKind, null);
  assert.equal(reopened.contractStatusCode, null);
});

test("G6-17 blocked reopen clears active manifest and records reopen blocker", () => {
  const reopened = nextReopenedStageState(completeResearch, "BLOCKED");
  assert.equal(reopened.manifestKind, null);
  assert.equal(reopened.stageCode, "RESEARCH");
  assert.equal(reopened.contractStatusCode, "REOPENED_BLOCKED");
  assert.equal(reopened.criticalBlockerCount, 1);
});

test("G6-18 upstream reopen blocks and invalidates downstream state", () => {
  const invalidated = downstreamStateAfterUpstreamReopen(completeIntegration);
  assert.equal(invalidated.lifecycle, "BLOCKED");
  assert.equal(invalidated.handoff, "NOT_EVALUATED");
  assert.equal(invalidated.manifestKind, null);
  assert.equal(invalidated.contractStatusCode, "UPSTREAM_STAGE_REOPENED");
  assert.equal(invalidated.stageRevision, 2);

  const alreadyBlocked = downstreamStateAfterUpstreamReopen({
    ...completeDeepDive,
    lifecycle: "BLOCKED",
    handoff: "NOT_EVALUATED",
    manifestKind: "CHECKPOINT",
    stageRevision: 3,
  });
  assert.equal(alreadyBlocked.stageRevision, 3);
  assert.equal(alreadyBlocked.manifestKind, null);
});

test("G6-19 reopen maps run status from target lifecycle", () => {
  assert.equal(
    runStatusAfterStageReopen("READY_TO_PUBLISH", "IN_PROGRESS"),
    "ACTIVE",
  );
  assert.equal(
    runStatusAfterStageReopen("ACTIVE", "BLOCKED"),
    "BLOCKED",
  );
  assert.throws(() =>
    runStatusAfterStageReopen("PUBLISHED", "IN_PROGRESS"),
  );
  assert.throws(() =>
    runStatusAfterStageReopen("CANCELLED", "BLOCKED"),
  );
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

test("G6-22 active non-terminal runs permit stage mutation", () => {
  assert.doesNotThrow(() => assertStageMutable("ACTIVE"));
  assert.doesNotThrow(() => assertStageMutable("BLOCKED"));
});

test("G6-25 machine-readable vocabulary matches runtime vocabulary", () => {
  const model = JSON.parse(
    readFileSync(
      "schemas/vnext/orotitan-vnext-state-model.v0.1.json",
      "utf8",
    ),
  ) as {
    run_statuses: string[];
    stage_lifecycle: string[];
    handoff_gate_by_stage: Record<string, string>;
  };

  assert.deepEqual(model.run_statuses, RUN_STATUSES);
  assert.deepEqual(model.stage_lifecycle, STAGE_LIFECYCLES);
  assert.deepEqual(model.handoff_gate_by_stage, HANDOFF_GATE_BY_STAGE);
});

test("G6-26 pause requires IN_PROGRESS plus CHECKPOINT", () => {
  assert.doesNotThrow(() =>
    assertPauseAllowed({
      stageCode: "RESEARCH",
      lifecycle: "IN_PROGRESS",
      handoff: "NOT_EVALUATED",
      manifestKind: "CHECKPOINT",
      stageRevision: 1,
    }),
  );

  assert.throws(() =>
    assertPauseAllowed({
      stageCode: "RESEARCH",
      lifecycle: "IN_PROGRESS",
      handoff: "NOT_EVALUATED",
      manifestKind: null,
      stageRevision: 1,
    }),
  );
});

test("G6-27 resume requires PAUSED or BLOCKED plus CHECKPOINT", () => {
  for (const lifecycle of ["PAUSED", "BLOCKED"] as const) {
    assert.doesNotThrow(() =>
      assertResumeAllowed({
        stageCode: "RESEARCH",
        lifecycle,
        handoff: "NOT_EVALUATED",
        manifestKind: "CHECKPOINT",
        stageRevision: 1,
      }),
    );
  }

  assert.throws(() =>
    assertResumeAllowed({
      stageCode: "RESEARCH",
      lifecycle: "PAUSED",
      handoff: "NOT_EVALUATED",
      manifestKind: null,
      stageRevision: 1,
    }),
  );
});

test("checkpoint lifecycle maps to the Registry V1.11 run status", () => {
  assert.equal(runStatusForCheckpoint("IN_PROGRESS"), "ACTIVE");
  assert.equal(runStatusForCheckpoint("PAUSED"), "PAUSED");
  assert.equal(runStatusForCheckpoint("BLOCKED"), "BLOCKED");
});

test("G6-28/G6-29 publish result mapping matches Registry V1.11", () => {
  assert.equal(
    runStatusAfterPublishResult("SUCCEEDED", false),
    "PUBLISHED",
  );
  assert.equal(
    runStatusAfterPublishResult("FAILED", true),
    "READY_TO_PUBLISH",
  );
  assert.equal(
    runStatusAfterPublishResult("FAILED", false),
    "BLOCKED",
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

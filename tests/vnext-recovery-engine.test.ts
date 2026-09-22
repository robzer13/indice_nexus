import assert from "node:assert/strict";
import test from "node:test";

import {
  classifyRecoveryIncident,
  executeRecovery,
  type RecoveryActionReceipt,
  type RecoveryHistorySnapshot,
  type RecoveryIncident,
  type RecoveryPlan,
  type RecoveryProofs,
  type RecoveryStore,
} from "../runtime/vnext/recovery-engine";

function emptyProofs(): RecoveryProofs {
  return {
    sameRequestFingerprint: false,
    operationIdempotent: false,
    retryBudgetAvailable: false,
    noCommittedMutation: false,
    expectedStateVersionFresh: false,

    repairTargetUnique: false,
    authoritativeTargetResolved: false,
    targetSealed: false,
    targetAvailable: false,
    targetHashVerified: false,
    targetContractPinsMatch: false,

    expectedHashKnown: false,
    candidateLocatorDurable: false,
    candidateBytesHashVerified: false,
    artifactIdentityExact: false,

    changesAnalyticalMeaning: false,
    rewritesHistoricalArtifact: false,
    rewritesHistoricalEvent: false,
    requiresAnalystJudgment: false,
    identityAmbiguous: false,
    contractDriftUnresolved: false,
    dataCutoffChangeRequired: false,
  };
}

function incident(
  overrides: Partial<RecoveryIncident> = {},
): RecoveryIncident {
  return {
    incidentId: "incident-gate11",
    runId: "run-gate11",
    kind: "UNKNOWN",
    operationId: null,
    requestFingerprint: null,
    attempt: 1,
    maxAttempts: 3,
    proofs: emptyProofs(),
    ...overrides,
  };
}

function asmFixture(): RecoveryIncident {
  return incident({
    incidentId: "fixture-asm",
    kind: "TRANSIENT_EXECUTION_FAILURE",
    operationId: "asm-op-1",
    requestFingerprint: "sha256:asm-same-request",
    attempt: 1,
    maxAttempts: 3,
    proofs: {
      ...emptyProofs(),
      sameRequestFingerprint: true,
      operationIdempotent: true,
      retryBudgetAvailable: true,
      noCommittedMutation: true,
      expectedStateVersionFresh: true,
    },
  });
}

function stMicroFixture(): RecoveryIncident {
  return incident({
    incidentId: "fixture-stmicro",
    kind: "MANIFEST_POINTER_MISMATCH",
    proofs: {
      ...emptyProofs(),
      repairTargetUnique: true,
      authoritativeTargetResolved: true,
      targetSealed: true,
      targetAvailable: true,
      targetHashVerified: true,
      targetContractPinsMatch: true,
      expectedStateVersionFresh: true,
    },
  });
}

function topicusFixture(): RecoveryIncident {
  return incident({
    incidentId: "fixture-topicus",
    kind: "ARTIFACT_LOCATOR_UNRESOLVED",
    proofs: {
      ...emptyProofs(),
      expectedHashKnown: true,
      candidateLocatorDurable: true,
      candidateBytesHashVerified: true,
      artifactIdentityExact: true,
      expectedStateVersionFresh: true,
    },
  });
}

class InMemoryRecoveryStore implements RecoveryStore {
  history: RecoveryHistorySnapshot = {
    runId: "run-gate11",
    entries: [
      {
        sequence: 1,
        eventId: "event-1",
        fingerprint: "sha256:history-1",
      },
      {
        sequence: 2,
        eventId: "event-2",
        fingerprint: "sha256:history-2",
      },
    ],
  };

  appliedPlans: RecoveryPlan[] = [];
  rewriteHistory = false;

  async readHistory(): Promise<RecoveryHistorySnapshot> {
    return structuredClone(this.history);
  }

  async applyRecovery(
    recoveryIncident: RecoveryIncident,
    plan: RecoveryPlan,
  ): Promise<RecoveryActionReceipt> {
    this.appliedPlans.push(structuredClone(plan));

    if (this.rewriteHistory) {
      const entries = [...this.history.entries];
      entries[0] = {
        ...entries[0],
        fingerprint: "sha256:illicit-rewrite",
      };
      this.history = {
        ...this.history,
        entries,
      };
    }

    this.history = {
      ...this.history,
      entries: [
        ...this.history.entries,
        {
          sequence: this.history.entries.length + 1,
          eventId: "recovery-" + recoveryIncident.incidentId,
          fingerprint: "sha256:" + plan.action,
        },
      ],
    };

    return {
      incidentId: recoveryIncident.incidentId,
      action: plan.action,
      applied: true,
    };
  }
}

test("Gate 11 ASM regression fixture becomes AUTO_RETRY", async () => {
  const fixture = asmFixture();
  const plan = classifyRecoveryIncident(fixture);

  assert.equal(plan.classification, "AUTO_RETRY");
  assert.equal(plan.action, "RETRY_SAME_REQUEST");
  assert.equal(plan.mutationScope, "EXECUTION_RETRY_ONLY");
  assert.equal(plan.automatic, true);

  const store = new InMemoryRecoveryStore();
  const result = await executeRecovery(store, fixture);

  assert.equal(result.receipt?.applied, true);
  assert.equal(store.appliedPlans.length, 1);
  assert.equal(result.historyBefore.entries.length, 2);
  assert.equal(result.historyAfter.entries.length, 3);
  assert.deepEqual(
    result.historyAfter.entries.slice(0, 2),
    result.historyBefore.entries,
  );
});

test("Gate 11 STMicro regression fixture becomes deterministic repair only when fully proven", async () => {
  const fixture = stMicroFixture();
  const plan = classifyRecoveryIncident(fixture);

  assert.equal(plan.classification, "DETERMINISTIC_AUTO_REPAIR");
  assert.equal(plan.action, "REBIND_PROVEN_MANIFEST_POINTER");
  assert.equal(plan.mutationScope, "CURRENT_ROUTING_METADATA_ONLY");

  const store = new InMemoryRecoveryStore();
  const result = await executeRecovery(store, fixture);
  assert.equal(result.receipt?.applied, true);

  const unproven = stMicroFixture();
  unproven.proofs.targetHashVerified = false;

  assert.equal(
    classifyRecoveryIncident(unproven).classification,
    "HUMAN_REQUIRED",
  );
});

test("Gate 11 Topicus regression fixture becomes verifiable locator recovery", async () => {
  const fixture = topicusFixture();
  const plan = classifyRecoveryIncident(fixture);

  assert.equal(plan.classification, "VERIFIABLE_RECOVERY");
  assert.equal(plan.action, "RESOLVE_AND_VERIFY_LOCATOR");
  assert.equal(
    plan.mutationScope,
    "VERIFIED_LOCATOR_BINDING_ONLY",
  );

  const store = new InMemoryRecoveryStore();
  const result = await executeRecovery(store, fixture);
  assert.equal(result.receipt?.applied, true);

  const wrongBytes = topicusFixture();
  wrongBytes.proofs.candidateBytesHashVerified = false;

  assert.equal(
    classifyRecoveryIncident(wrongBytes).classification,
    "HUMAN_REQUIRED",
  );
});

test("AUTO_RETRY requires same idempotent request, fresh state and remaining budget", () => {
  const fixture = asmFixture();
  fixture.proofs.sameRequestFingerprint = false;

  assert.equal(
    classifyRecoveryIncident(fixture).classification,
    "HUMAN_REQUIRED",
  );

  const stale = asmFixture();
  stale.proofs.expectedStateVersionFresh = false;
  assert.equal(
    classifyRecoveryIncident(stale).classification,
    "HUMAN_REQUIRED",
  );

  const exhausted = asmFixture();
  exhausted.attempt = exhausted.maxAttempts;
  assert.equal(
    classifyRecoveryIncident(exhausted).classification,
    "HUMAN_REQUIRED",
  );
});

test("semantic, authority and historical ambiguity always escalates", () => {
  const risky = stMicroFixture();
  risky.proofs.requiresAnalystJudgment = true;

  const plan = classifyRecoveryIncident(risky);
  assert.equal(plan.classification, "HUMAN_REQUIRED");
  assert.equal(plan.action, "ESCALATE_HUMAN");
  assert.equal(plan.automatic, false);
  assert.equal(plan.mutationScope, "NONE");

  const historicalRewrite = topicusFixture();
  historicalRewrite.proofs.rewritesHistoricalArtifact = true;

  assert.equal(
    classifyRecoveryIncident(historicalRewrite).classification,
    "HUMAN_REQUIRED",
  );
});

test("unresolved contract drift and data-cutoff changes cannot be auto-repaired", () => {
  const contract = incident({
    kind: "CONTRACT_DRIFT",
    proofs: {
      ...emptyProofs(),
      contractDriftUnresolved: true,
    },
  });

  assert.equal(
    classifyRecoveryIncident(contract).classification,
    "HUMAN_REQUIRED",
  );

  const cutoff = incident({
    kind: "DATA_CUTOFF_CONFLICT",
    proofs: {
      ...emptyProofs(),
      dataCutoffChangeRequired: true,
    },
  });

  assert.equal(
    classifyRecoveryIncident(cutoff).classification,
    "HUMAN_REQUIRED",
  );
});

test("unknown incidents fail closed to HUMAN_REQUIRED", () => {
  const plan = classifyRecoveryIncident(incident());

  assert.equal(plan.classification, "HUMAN_REQUIRED");
  assert.equal(plan.action, "ESCALATE_HUMAN");
});

test("HUMAN_REQUIRED executes no automatic mutation", async () => {
  const store = new InMemoryRecoveryStore();
  const result = await executeRecovery(store, incident());

  assert.equal(result.receipt, null);
  assert.equal(store.appliedPlans.length, 0);
  assert.deepEqual(result.historyAfter, result.historyBefore);
});

test("Gate 11 detects any rewrite of prior history during automatic recovery", async () => {
  const store = new InMemoryRecoveryStore();
  store.rewriteHistory = true;

  await assert.rejects(
    executeRecovery(store, asmFixture()),
    /VNEXT_RECOVERY_HISTORY_REWRITE_DETECTED/,
  );
});

test("all three known regression fixtures remain fully automatic under proven conditions", () => {
  const plans = [
    classifyRecoveryIncident(asmFixture()),
    classifyRecoveryIncident(stMicroFixture()),
    classifyRecoveryIncident(topicusFixture()),
  ];

  assert.deepEqual(
    plans.map((plan) => plan.classification),
    [
      "AUTO_RETRY",
      "DETERMINISTIC_AUTO_REPAIR",
      "VERIFIABLE_RECOVERY",
    ],
  );

  assert.ok(plans.every((plan) => plan.automatic));
  assert.ok(
    plans.every((plan) => plan.requiresFreshReread),
  );
});

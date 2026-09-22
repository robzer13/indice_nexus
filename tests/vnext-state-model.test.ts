import assert from "node:assert/strict";
import test from "node:test";

import {
  ANALYTICAL_STATES,
  AUDIT_STATUSES,
  DECISION_STATES,
  EVIDENCE_STATES,
  PRICE_CONDITIONS,
  RUNTIME_STATES,
  STAGE_STATES,
  VALUATION_RELIABILITY_STATES,
  canAnalyticalTransition,
  canAuditTransition,
  canDecisionTransition,
  canEvidenceTransition,
  canPriceConditionTransition,
  canRuntimeTransition,
  canStageTransition,
  canValuationReliabilityTransition,
  validateStateVector,
  type VNextStateVector,
} from "../runtime/vnext/state-model";

test("Gate 6 exposes all eight orthogonal state domains", () => {
  assert.deepEqual(RUNTIME_STATES, [
    "IDLE",
    "READY",
    "RUNNING",
    "PAUSED",
    "BLOCKED",
    "RECOVERING",
    "FAILED",
    "COMPLETE",
  ]);

  assert.deepEqual(STAGE_STATES, [
    "NOT_STARTED",
    "IN_PROGRESS",
    "PAUSED",
    "BLOCKED",
    "COMPLETE",
  ]);

  assert.deepEqual(ANALYTICAL_STATES, [
    "INSUFFICIENT",
    "IN_PROGRESS",
    "PROVISIONALLY_STABLE",
    "LOCKED",
  ]);

  assert.deepEqual(EVIDENCE_STATES, [
    "UNKNOWN",
    "SUFFICIENT",
    "PARTIAL_BUT_DECISIONABLE",
    "INSUFFICIENT",
    "CONFLICTED",
  ]);

  assert.deepEqual(VALUATION_RELIABILITY_STATES, [
    "UNKNOWN",
    "HIGH",
    "MEDIUM",
    "LOW",
    "NOT_ASSESSABLE",
  ]);

  assert.deepEqual(PRICE_CONDITIONS, [
    "UNKNOWN",
    "NOT_ASSESSABLE",
    "ABOVE_REQUIRED_RETURN_PRICE",
    "AT_OR_BELOW_REQUIRED_RETURN_PRICE",
    "AT_OR_BELOW_STRONG_RETURN_PRICE",
    "AT_OR_BELOW_EXCEPTIONAL_RETURN_PRICE",
  ]);

  assert.deepEqual(DECISION_STATES, [
    "UNKNOWN",
    "INVESTABLE_NOW",
    "WAIT_FOR_PRICE",
    "WAIT_FOR_EVIDENCE",
    "REFRESH_REQUIRED",
    "REJECT",
  ]);

  assert.deepEqual(AUDIT_STATUSES, [
    "NOT_RUN",
    "IN_PROGRESS",
    "PASS",
    "FAIL",
    "STALE",
  ]);
});

test("runtime technical blockage is reversible and independent", () => {
  assert.equal(canRuntimeTransition("READY", "RUNNING"), true);
  assert.equal(canRuntimeTransition("RUNNING", "BLOCKED"), true);
  assert.equal(canRuntimeTransition("BLOCKED", "RECOVERING"), true);
  assert.equal(canRuntimeTransition("RECOVERING", "RUNNING"), true);
  assert.equal(canRuntimeTransition("BLOCKED", "COMPLETE"), false);
});

test("stage transitions preserve frozen Registry lifecycle semantics", () => {
  assert.equal(canStageTransition("NOT_STARTED", "IN_PROGRESS"), true);
  assert.equal(canStageTransition("IN_PROGRESS", "PAUSED"), true);
  assert.equal(canStageTransition("IN_PROGRESS", "BLOCKED"), true);
  assert.equal(canStageTransition("IN_PROGRESS", "COMPLETE"), true);
  assert.equal(canStageTransition("PAUSED", "IN_PROGRESS"), true);
  assert.equal(canStageTransition("BLOCKED", "PAUSED"), true);

  assert.equal(canStageTransition("COMPLETE", "IN_PROGRESS"), false);
  assert.equal(
    canStageTransition("COMPLETE", "IN_PROGRESS", {
      controlledReopen: true,
    }),
    true,
  );
});

test("analytical LOCKED is not Certification and requires controlled reopen", () => {
  assert.equal(
    canAnalyticalTransition("IN_PROGRESS", "PROVISIONALLY_STABLE"),
    true,
  );
  assert.equal(canAnalyticalTransition("PROVISIONALLY_STABLE", "LOCKED"), true);
  assert.equal(canAnalyticalTransition("LOCKED", "IN_PROGRESS"), false);
  assert.equal(
    canAnalyticalTransition("LOCKED", "IN_PROGRESS", {
      controlledReopen: true,
    }),
    true,
  );
  assert.equal(
    canAnalyticalTransition("LOCKED", "INSUFFICIENT", {
      controlledReopen: true,
    }),
    true,
  );
});

test("evidence sufficiency can improve or deteriorate as new evidence arrives", () => {
  assert.equal(canEvidenceTransition("UNKNOWN", "INSUFFICIENT"), true);
  assert.equal(canEvidenceTransition("INSUFFICIENT", "SUFFICIENT"), true);
  assert.equal(canEvidenceTransition("SUFFICIENT", "CONFLICTED"), true);
  assert.equal(
    canEvidenceTransition("CONFLICTED", "PARTIAL_BUT_DECISIONABLE"),
    true,
  );
  assert.equal(canEvidenceTransition("SUFFICIENT", "SUFFICIENT"), false);
});

test("valuation reliability uses UNKNOWN only as a pre-evaluation or invalidated execution sentinel", () => {
  assert.equal(
    canValuationReliabilityTransition("UNKNOWN", "NOT_ASSESSABLE"),
    true,
  );
  assert.equal(canValuationReliabilityTransition("LOW", "MEDIUM"), true);
  assert.equal(canValuationReliabilityTransition("HIGH", "UNKNOWN"), false);
  assert.equal(
    canValuationReliabilityTransition("HIGH", "UNKNOWN", {
      invalidateAssessment: true,
    }),
    true,
  );
});

test("price condition is a factual market-to-ladder state and may move in either direction", () => {
  assert.equal(
    canPriceConditionTransition(
      "ABOVE_REQUIRED_RETURN_PRICE",
      "AT_OR_BELOW_REQUIRED_RETURN_PRICE",
    ),
    true,
  );
  assert.equal(
    canPriceConditionTransition(
      "AT_OR_BELOW_EXCEPTIONAL_RETURN_PRICE",
      "ABOVE_REQUIRED_RETURN_PRICE",
    ),
    true,
  );
  assert.equal(canPriceConditionTransition("UNKNOWN", "UNKNOWN"), false);
});

test("REJECT cannot reopen directly without the frozen causal reopen condition", () => {
  assert.equal(canDecisionTransition("WAIT_FOR_PRICE", "INVESTABLE_NOW"), true);
  assert.equal(canDecisionTransition("REJECT", "INVESTABLE_NOW"), false);
  assert.equal(canDecisionTransition("REJECT", "REFRESH_REQUIRED"), false);
  assert.equal(
    canDecisionTransition("REJECT", "REFRESH_REQUIRED", {
      rejectionReopenAuthorized: true,
    }),
    true,
  );
});

test("audit PASS becomes STALE before rerun and FAIL reruns explicitly", () => {
  assert.equal(canAuditTransition("NOT_RUN", "IN_PROGRESS"), true);
  assert.equal(canAuditTransition("IN_PROGRESS", "PASS"), true);
  assert.equal(canAuditTransition("IN_PROGRESS", "FAIL"), true);
  assert.equal(canAuditTransition("PASS", "FAIL"), false);
  assert.equal(canAuditTransition("PASS", "STALE"), true);
  assert.equal(canAuditTransition("STALE", "IN_PROGRESS"), true);
  assert.equal(canAuditTransition("FAIL", "IN_PROGRESS"), true);
});

test("technical blockage does not imply an analytical or investment rejection", () => {
  const state: VNextStateVector = {
    runtimeState: "BLOCKED",
    stageState: "BLOCKED",
    analyticalState: "PROVISIONALLY_STABLE",
    evidenceState: "SUFFICIENT",
    valuationReliability: "HIGH",
    priceCondition: "ABOVE_REQUIRED_RETURN_PRICE",
    decisionState: "WAIT_FOR_PRICE",
    auditStatus: "STALE",
  };

  assert.deepEqual(validateStateVector(state), {
    valid: true,
    errors: [],
  });
});

test("INVESTABLE_NOW requires decisionable evidence, assessable valuation and price support", () => {
  const invalid: VNextStateVector = {
    runtimeState: "READY",
    stageState: "COMPLETE",
    analyticalState: "LOCKED",
    evidenceState: "INSUFFICIENT",
    valuationReliability: "NOT_ASSESSABLE",
    priceCondition: "ABOVE_REQUIRED_RETURN_PRICE",
    decisionState: "INVESTABLE_NOW",
    auditStatus: "PASS",
  };

  const result = validateStateVector(invalid);
  assert.equal(result.valid, false);
  assert.equal(result.errors.length, 4);

  const valid: VNextStateVector = {
    runtimeState: "COMPLETE",
    stageState: "COMPLETE",
    analyticalState: "LOCKED",
    evidenceState: "PARTIAL_BUT_DECISIONABLE",
    valuationReliability: "MEDIUM",
    priceCondition: "AT_OR_BELOW_REQUIRED_RETURN_PRICE",
    decisionState: "INVESTABLE_NOW",
    auditStatus: "PASS",
  };

  assert.deepEqual(validateStateVector(valid), {
    valid: true,
    errors: [],
  });
});

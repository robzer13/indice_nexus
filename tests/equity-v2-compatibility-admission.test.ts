import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  V2_COMPATIBILITY_SEMANTIC_BASE,
  V2_FROZEN_CONTRACT_SET_SHA256,
  evaluateExistingRunCompatibilityAdmission,
  type ExistingRunCompatibilityAdmissionInput,
} from "../lib/orotitan-equity/v2/existing-run-compatibility-admission";

function base(): ExistingRunCompatibilityAdmissionInput {
  return {
    runStatus: "ACTIVE",
    currentStage: "DEEP_DIVE",
    runContractSetSha256: V2_FROZEN_CONTRACT_SET_SHA256,
    contractPinsUnchanged: true,
    sourceArtifactsHashValid: true,
    authorityHashVerified: true,
    schemaHashVerified: true,
    validatorHashVerified: true,
    regressionsPassed: true,
    integrationHasAdmittedCanonicalSnapshot: false,
    readyToPublish: false,
    publicationAuthorizationExists: false,
    publicationEventCount: 0,
    currentSnapshotStateCompatible: true,
    semanticProjections: [
      {
        targetField: "l2_research_fundamentals.analytical_metrics.roiic",
        sourceValue: "NOT_INTERPRETABLE",
        targetValue: "NOT_INTERPRETABLE",
      },
      {
        targetField: "l3_investment_valuation.valuation.return_horizon",
        sourceValue: 4.7835616438356166,
        targetValue: 4.7835616438356166,
      },
    ],
    semanticLoss: "NONE",
    analyticalArtifactMutationRequired: false,
    contractPinMutationRequired: false,
    analyticalContradiction: false,
    unrelatedBlocker: false,
    deepDiveLifecycleStatus: "COMPLETE",
    deepDiveActiveManifestKind: "FINAL",
    readyForIntegration: true,
    integrationStage: null,
    integrationArtifactCount: 0,
  };
}

function sha256(path: string): string {
  return createHash("sha256").update(readFileSync(new URL(path, import.meta.url))).digest("hex");
}

test("V2.0.8 semantic base uses exact V2.0.8 authority/schema/validator bytes", () => {
  assert.equal(
    sha256("../contracts/orotitan-equity/v2/OROTITAN_CANONICAL_SEMANTIC_COMPATIBILITY_PATCH_V2.0.8.md"),
    V2_COMPATIBILITY_SEMANTIC_BASE.authoritySha256,
  );
  assert.equal(
    sha256("../contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V1_COMPAT_V2.0.8.json"),
    V2_COMPATIBILITY_SEMANTIC_BASE.schemaSha256,
  );
  assert.equal(
    sha256("../lib/orotitan-equity/v2/research-snapshot-schema.ts"),
    V2_COMPATIBILITY_SEMANTIC_BASE.validatorSha256,
  );
});

test("TEST A pre-Integration COMPLETE/FINAL/ready exact compatible run is admitted", () => {
  const result = evaluateExistingRunCompatibilityAdmission(base());
  assert.deepEqual(result, {
    admitted: true,
    route: "PRE_INTEGRATION_EXISTING_V2_RUN",
    failures: [],
  });
});

test("TEST B Deep Dive IN_PROGRESS is rejected", () => {
  const input = base();
  input.deepDiveLifecycleStatus = "IN_PROGRESS";
  const result = evaluateExistingRunCompatibilityAdmission(input);
  assert.equal(result.admitted, false);
  if (!result.admitted) assert.ok(result.failures.includes("DEEP_DIVE_NOT_COMPLETE"));
});

test("TEST C READY_FOR_INTEGRATION NO is rejected", () => {
  const input = base();
  input.readyForIntegration = false;
  const result = evaluateExistingRunCompatibilityAdmission(input);
  assert.equal(result.admitted, false);
  if (!result.admitted) assert.ok(result.failures.includes("READY_FOR_INTEGRATION_NOT_YES"));
});

test("TEST D Integration absent but Integration artifacts exist is rejected", () => {
  const input = base();
  input.integrationArtifactCount = 1;
  const result = evaluateExistingRunCompatibilityAdmission(input);
  assert.equal(result.admitted, false);
  if (!result.admitted) assert.ok(result.failures.includes("PRE_INTEGRATION_ARTIFACTS_ALREADY_EXIST"));
});

test("TEST E valid existing Integration checkpoint routes through V2.0.8-compatible path", () => {
  const input = base();
  input.runStatus = "BLOCKED";
  input.currentStage = "INTEGRATION";
  input.integrationArtifactCount = 3;
  input.integrationStage = {
    lifecycleStatus: "BLOCKED",
    activeManifestKind: "CHECKPOINT",
    activeManifestDurable: true,
    sameStageRevision: true,
  };
  const result = evaluateExistingRunCompatibilityAdmission(input);
  assert.deepEqual(result, {
    admitted: true,
    route: "EXISTING_INTEGRATION_CHECKPOINT",
    failures: [],
  });
});

test("TEST F inconsistent existing Integration lifecycle is rejected", () => {
  const input = base();
  input.currentStage = "INTEGRATION";
  input.integrationStage = {
    lifecycleStatus: "COMPLETE",
    activeManifestKind: "FINAL",
    activeManifestDurable: true,
    sameStageRevision: true,
  };
  const result = evaluateExistingRunCompatibilityAdmission(input);
  assert.equal(result.admitted, false);
  if (!result.admitted) assert.deepEqual(result.failures, ["INTEGRATION_STAGE_INCONSISTENT"]);
});

test("TEST G Contract Set mismatch is rejected", () => {
  const input = base();
  input.runContractSetSha256 = "0".repeat(64);
  const result = evaluateExistingRunCompatibilityAdmission(input);
  assert.equal(result.admitted, false);
  if (!result.admitted) assert.ok(result.failures.includes("CONTRACT_SET_MISMATCH"));
});

test("TEST H ROIIC NOT_INTERPRETABLE and fractional return_horizon are lossless", () => {
  const input = base();
  const result = evaluateExistingRunCompatibilityAdmission(input);
  assert.equal(result.admitted, true);
  for (const projection of input.semanticProjections) {
    assert.equal(projection.sourceValue, projection.targetValue);
  }
  assert.equal(input.semanticLoss, "NONE");
});

test("TEST I NOT_INTERPRETABLE in unauthorized returnValue field is rejected", () => {
  const input = base();
  input.semanticProjections = [{
    targetField: "l2_research_fundamentals.analytical_metrics.all_in_roic",
    sourceValue: "NOT_INTERPRETABLE",
    targetValue: "NOT_INTERPRETABLE",
  }];
  const result = evaluateExistingRunCompatibilityAdmission(input);
  assert.equal(result.admitted, false);
  if (!result.admitted) assert.ok(result.failures.includes("SEMANTIC_RULE_NOT_AUTHORIZED"));
});

test("TEST J coercion of NOT_INTERPRETABLE is rejected", () => {
  const input = base();
  input.semanticProjections = [{
    targetField: "l2_research_fundamentals.analytical_metrics.roiic",
    sourceValue: "NOT_INTERPRETABLE",
    targetValue: "UNKNOWN",
  }];
  const result = evaluateExistingRunCompatibilityAdmission(input);
  assert.equal(result.admitted, false);
  if (!result.admitted) assert.ok(result.failures.includes("SEMANTIC_COERCION_FORBIDDEN"));
});

test("TEST K publication event already exists is rejected", () => {
  const input = base();
  input.publicationEventCount = 1;
  const result = evaluateExistingRunCompatibilityAdmission(input);
  assert.equal(result.admitted, false);
  if (!result.admitted) assert.ok(result.failures.includes("PUBLICATION_EVENT_EXISTS"));
});

test("TEST L compatibility admission requiring analytical mutation is rejected", () => {
  const input = base();
  input.analyticalArtifactMutationRequired = true;
  const result = evaluateExistingRunCompatibilityAdmission(input);
  assert.equal(result.admitted, false);
  if (!result.admitted) assert.ok(result.failures.includes("ANALYTICAL_ARTIFACT_MUTATION_REQUIRED"));
});

test("all V2.0.2-V2.0.8 field-local semantic rules remain admissible without broadening", () => {
  const rules = [
    ["l2_research_fundamentals.fundamental_states.roic_trend", "NOT_APPLICABLE"],
    ["l2_research_fundamentals.analytical_metrics.roiic", "NOT_INTERPRETABLE"],
    ["l2_research_fundamentals.analytical_metrics.standard_roic", "NOT_INTERPRETABLE"],
    ["l2_research_fundamentals.fundamental_states.roic_trend", "UNKNOWN"],
    ["l2_research_fundamentals.analytical_metrics.roic_ex_goodwill", "NOT_INTERPRETABLE"],
  ] as const;

  for (const [targetField, value] of rules) {
    const input = base();
    input.semanticProjections = [{ targetField, sourceValue: value, targetValue: value }];
    const result = evaluateExistingRunCompatibilityAdmission(input);
    assert.equal(result.admitted, true, `${targetField}=${value}`);
  }
});

test("hash/regression/pin/semantic safety controls each fail closed", () => {
  const mutations: Array<[keyof ExistingRunCompatibilityAdmissionInput, unknown, string]> = [
    ["contractPinsUnchanged", false, "CONTRACT_PINS_CHANGED"],
    ["sourceArtifactsHashValid", false, "SOURCE_ARTIFACT_HASH_INVALID"],
    ["authorityHashVerified", false, "COMPATIBILITY_AUTHORITY_HASH_UNVERIFIED"],
    ["schemaHashVerified", false, "COMPATIBILITY_SCHEMA_HASH_UNVERIFIED"],
    ["validatorHashVerified", false, "COMPATIBILITY_VALIDATOR_HASH_UNVERIFIED"],
    ["regressionsPassed", false, "COMPATIBILITY_REGRESSION_NOT_PASS_ALL"],
    ["integrationHasAdmittedCanonicalSnapshot", true, "INTEGRATION_SNAPSHOT_ALREADY_ADMITTED"],
    ["readyToPublish", true, "READY_TO_PUBLISH_ALREADY_YES"],
    ["publicationAuthorizationExists", true, "PUBLICATION_AUTHORIZATION_EXISTS"],
    ["currentSnapshotStateCompatible", false, "CURRENT_SNAPSHOT_STATE_INCOMPATIBLE"],
    ["semanticLoss", "MATERIAL", "SEMANTIC_LOSS_NOT_NONE"],
    ["contractPinMutationRequired", true, "CONTRACT_PIN_MUTATION_REQUIRED"],
    ["analyticalContradiction", true, "ANALYTICAL_CONTRADICTION"],
    ["unrelatedBlocker", true, "UNRELATED_BLOCKER"],
  ];

  for (const [key, value, reason] of mutations) {
    const input = base();
    (input as unknown as Record<string, unknown>)[key] = value;
    const result = evaluateExistingRunCompatibilityAdmission(input);
    assert.equal(result.admitted, false, String(key));
    if (!result.admitted) assert.ok(result.failures.includes(reason), String(key));
  }
});


test("V2.0.8 admits exact positive finite fractional return_horizon semantics", () => {
  for (const value of [0.25, 0.5, 1.5, 4.7835616438356166, 9.999999]) {
    const input = base();
    input.semanticProjections = [{
      targetField: "l3_investment_valuation.valuation.return_horizon",
      sourceValue: value,
      targetValue: value,
    }];
    const result = evaluateExistingRunCompatibilityAdmission(input);
    assert.equal(result.admitted, true, String(value));
  }
});

test("V2.0.8 rejects invalid numeric and string return_horizon compatibility projections", () => {
  const cases: Array<string | number> = [0, -1, Infinity, "4.7835616438356166"];
  for (const value of cases) {
    const input = base();
    input.semanticProjections = [{
      targetField: "l3_investment_valuation.valuation.return_horizon",
      sourceValue: value,
      targetValue: value,
    }];
    const result = evaluateExistingRunCompatibilityAdmission(input);
    assert.equal(result.admitted, false, String(value));
    if (!result.admitted) assert.ok(result.failures.includes("SEMANTIC_RULE_NOT_AUTHORIZED"), String(value));
  }
});

test("V2.0.8 rejects NaN return_horizon without coercing it", () => {
  const input = base();
  input.semanticProjections = [{
    targetField: "l3_investment_valuation.valuation.return_horizon",
    sourceValue: Number.NaN,
    targetValue: Number.NaN,
  }];
  const result = evaluateExistingRunCompatibilityAdmission(input);
  assert.equal(result.admitted, false);
  if (!result.admitted) assert.ok(result.failures.includes("SEMANTIC_COERCION_FORBIDDEN"));
});

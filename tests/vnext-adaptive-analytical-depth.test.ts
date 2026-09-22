import assert from "node:assert/strict";
import test from "node:test";

import fixtures from "./fixtures/vnext/adaptive-analytical-depth.v0.1.json";

import {
  assertValidAnalyticalDepthConsumption,
  deriveMinimumAnalyticalDepth,
  resolveAnalyticalDepth,
  type AnalyticalDepth,
  type AnalyticalDepthContext,
  type AnalyticalDepthConsumptionRecord,
  type DepthTriggerCode,
} from "../runtime/vnext/adaptive-analytical-depth";

for (const fixture of fixtures.cases) {
  test(`Adaptive analytical depth fixture: ${fixture.id}`, () => {
    const decision = resolveAnalyticalDepth(
      fixture.context as AnalyticalDepthContext,
      "requestedDepth" in fixture
        ? (fixture.requestedDepth as AnalyticalDepth)
        : undefined,
    );

    assert.equal(
      decision.minimumDepth,
      fixture.expected.minimumDepth,
    );
    assert.equal(
      decision.selectedDepth,
      fixture.expected.selectedDepth,
    );
    assert.equal(
      decision.secondAnalystEligible,
      fixture.expected.secondAnalystEligible,
    );

    const codes = decision.triggers.map((trigger) => trigger.code);

    if ("triggerCodes" in fixture.expected) {
      assert.deepEqual(codes, fixture.expected.triggerCodes);
    }

    if ("requiredTrigger" in fixture.expected) {
      assert.equal(
        codes.includes(
          fixture.expected.requiredTrigger as DepthTriggerCode,
        ),
        true,
      );
    }
  });
}

test("DEPTH_1 is selected only when no escalation trigger is present", () => {
  const derived = deriveMinimumAnalyticalDepth(
    fixtures.cases[0].context as AnalyticalDepthContext,
  );

  assert.equal(derived.minimumDepth, "DEPTH_1");
  assert.deepEqual(
    derived.triggers.map((trigger) => trigger.code),
    ["BASELINE_SIMPLE"],
  );
});

test("a requested depth cannot downgrade below the deterministic minimum", () => {
  assert.throws(
    () =>
      resolveAnalyticalDepth(
        fixtures.cases[2].context as AnalyticalDepthContext,
        "DEPTH_2",
      ),
    /VNEXT_ANALYTICAL_DEPTH_DOWNGRADE_BELOW_REQUIRED_MINIMUM/,
  );
});

test("manual deeper request is traceable and cannot erase baseline derivation", () => {
  const decision = resolveAnalyticalDepth(
    fixtures.cases[7].context as AnalyticalDepthContext,
    "DEPTH_2",
  );

  assert.equal(decision.minimumDepth, "DEPTH_1");
  assert.equal(decision.selectedDepth, "DEPTH_2");
  assert.equal(
    decision.triggers.some(
      (trigger) => trigger.code === "MANUAL_DEEPER_REQUEST",
    ),
    true,
  );
});

test("DEPTH_3 enables independent second analyst but does not require one to run", () => {
  const decision = resolveAnalyticalDepth(
    fixtures.cases[3].context as AnalyticalDepthContext,
  );

  assert.equal(decision.selectedDepth, "DEPTH_3");
  assert.equal(decision.secondAnalystEligible, true);
  assert.equal(
    decision.reconciliationRequiredIfSecondAnalystRuns,
    true,
  );
});

test("second analyst execution at DEPTH_3 requires a reconciliation artifact", () => {
  const decision = resolveAnalyticalDepth(
    fixtures.cases[3].context as AnalyticalDepthContext,
  );

  const record: AnalyticalDepthConsumptionRecord = {
    runId: "RUN-001",
    stageCode: "DEEP_DIVE",
    moduleId: "RETURN_NORMALIZATION",
    executionId: "EXEC-001",
    selectedDepth: "DEPTH_3",
    triggerCodes: ["MARGINAL_RETURN_UNCERTAIN"],
    secondAnalystExecuted: true,
    reconciliationArtifactId: null,
    providerRequestIds: [],
  };

  assert.throws(
    () => assertValidAnalyticalDepthConsumption(decision, record),
    /VNEXT_DEPTH_TRACE_RECONCILIATION_REQUIRED/,
  );
});

test("valid DEPTH_3 second-analyst execution is fully traceable", () => {
  const decision = resolveAnalyticalDepth(
    fixtures.cases[3].context as AnalyticalDepthContext,
  );

  const record: AnalyticalDepthConsumptionRecord = {
    runId: "RUN-001",
    stageCode: "DEEP_DIVE",
    moduleId: "RETURN_NORMALIZATION",
    executionId: "EXEC-001",
    selectedDepth: "DEPTH_3",
    triggerCodes: ["MARGINAL_RETURN_UNCERTAIN"],
    secondAnalystExecuted: true,
    reconciliationArtifactId: "ART-RECON-001",
    providerRequestIds: [],
  };

  assert.doesNotThrow(() =>
    assertValidAnalyticalDepthConsumption(decision, record),
  );
});

test("DEPTH_1 cannot execute a second analyst through the depth policy", () => {
  const decision = resolveAnalyticalDepth(
    fixtures.cases[0].context as AnalyticalDepthContext,
  );

  const record: AnalyticalDepthConsumptionRecord = {
    runId: "RUN-002",
    stageCode: "DEEP_DIVE",
    moduleId: "WEAK_LINK_TAXONOMY",
    executionId: "EXEC-002",
    selectedDepth: "DEPTH_1",
    triggerCodes: ["BASELINE_SIMPLE"],
    secondAnalystExecuted: true,
    reconciliationArtifactId: "ART-RECON-002",
    providerRequestIds: [],
  };

  assert.throws(
    () => assertValidAnalyticalDepthConsumption(decision, record),
    /VNEXT_DEPTH_TRACE_SECOND_ANALYST_NOT_ELIGIBLE/,
  );
});

test("depth consumption must reference only triggers from the exact decision", () => {
  const decision = resolveAnalyticalDepth(
    fixtures.cases[1].context as AnalyticalDepthContext,
  );

  const record: AnalyticalDepthConsumptionRecord = {
    runId: "RUN-003",
    stageCode: "DEEP_DIVE",
    moduleId: "OWNER_CASH",
    executionId: "EXEC-003",
    selectedDepth: "DEPTH_2",
    triggerCodes: ["MATERIAL_WEAK_LINK"],
    secondAnalystExecuted: false,
    reconciliationArtifactId: null,
    providerRequestIds: [],
  };

  assert.throws(
    () => assertValidAnalyticalDepthConsumption(decision, record),
    /VNEXT_DEPTH_TRACE_UNAUTHORIZED_TRIGGER_CODE/,
  );
});

test("depth consumption must preserve the complete exact trigger set", () => {
  const decision = resolveAnalyticalDepth(
    fixtures.cases[1].context as AnalyticalDepthContext,
  );

  const record: AnalyticalDepthConsumptionRecord = {
    runId: "RUN-004",
    stageCode: "DEEP_DIVE",
    moduleId: "VALUATION_DIAGNOSTIC_INTEGRITY",
    executionId: "EXEC-004",
    selectedDepth: "DEPTH_2",
    triggerCodes: ["SIGNIFICANT_UNCERTAINTY"],
    secondAnalystExecuted: false,
    reconciliationArtifactId: null,
    providerRequestIds: [],
  };

  assert.throws(
    () => assertValidAnalyticalDepthConsumption(decision, record),
    /VNEXT_DEPTH_TRACE_MISSING_DECISION_TRIGGER/,
  );
});

test("depth consumption rejects duplicate trigger provenance", () => {
  const decision = resolveAnalyticalDepth(
    fixtures.cases[3].context as AnalyticalDepthContext,
  );

  const record: AnalyticalDepthConsumptionRecord = {
    runId: "RUN-005",
    stageCode: "DEEP_DIVE",
    moduleId: "RETURN_NORMALIZATION",
    executionId: "EXEC-005",
    selectedDepth: "DEPTH_3",
    triggerCodes: [
      "MARGINAL_RETURN_UNCERTAIN",
      "MARGINAL_RETURN_UNCERTAIN",
    ],
    secondAnalystExecuted: false,
    reconciliationArtifactId: null,
    providerRequestIds: [],
  };

  assert.throws(
    () => assertValidAnalyticalDepthConsumption(decision, record),
    /VNEXT_DEPTH_TRACE_DUPLICATE_TRIGGER_CODE/,
  );
});

test("second-analyst reconciliation artifact id cannot be blank", () => {
  const decision = resolveAnalyticalDepth(
    fixtures.cases[3].context as AnalyticalDepthContext,
  );

  const record: AnalyticalDepthConsumptionRecord = {
    runId: "RUN-006",
    stageCode: "DEEP_DIVE",
    moduleId: "RETURN_NORMALIZATION",
    executionId: "EXEC-006",
    selectedDepth: "DEPTH_3",
    triggerCodes: ["MARGINAL_RETURN_UNCERTAIN"],
    secondAnalystExecuted: true,
    reconciliationArtifactId: "   ",
    providerRequestIds: [],
  };

  assert.throws(
    () => assertValidAnalyticalDepthConsumption(decision, record),
    /VNEXT_DEPTH_TRACE_RECONCILIATION_ARTIFACT_ID_REQUIRED/,
  );
});

test("adaptive depth policy is provider-neutral and does not require Azure", async () => {
  const source = await import("node:fs/promises").then((fs) =>
    fs.readFile(
      "runtime/vnext/adaptive-analytical-depth.ts",
      "utf8",
    ),
  );

  for (const forbidden of [
    /AzureProvider/,
    /AnalyticalModelProvider/,
    /openai\.azure\.com/i,
    /services\.ai\.azure\.com/i,
    /process\.env/,
    /fetch\s*\(/,
  ]) {
    assert.equal(forbidden.test(source), false);
  }
});

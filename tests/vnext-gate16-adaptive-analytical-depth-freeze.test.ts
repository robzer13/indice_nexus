import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  ANALYTICAL_DEPTHS,
  DEPTH_TRIGGER_CODES,
  assertValidAnalyticalDepthConsumption,
  deriveMinimumAnalyticalDepth,
  resolveAnalyticalDepth,
  type AnalyticalDepthContext,
  type AnalyticalDepthConsumptionRecord,
} from "../runtime/vnext/adaptive-analytical-depth";

const BASE_CONTEXT: AnalyticalDepthContext = {
  materialWeakLink: "NO",
  weakLinkUnresolved: false,
  marginalReturnUncertain: false,
  valuationReliability: "HIGH",
  materialSourceConflictCount: 0,
  significantUncertaintyCount: 0,
  criticalUnknownCount: 0,
  executionConfidence: "HIGH",
  decisionBoundarySensitivity: "LOW",
  materialCounterevidenceRisk: false,
};

function context(
  patch: Partial<AnalyticalDepthContext> = {},
): AnalyticalDepthContext {
  return {
    ...BASE_CONTEXT,
    ...patch,
  };
}

test("Gate 16 freezes exactly three analytical depth states", () => {
  assert.deepEqual(ANALYTICAL_DEPTHS, [
    "DEPTH_1",
    "DEPTH_2",
    "DEPTH_3",
  ]);
});

test("Gate 16 freezes the exact trigger taxonomy", () => {
  assert.deepEqual(DEPTH_TRIGGER_CODES, [
    "BASELINE_SIMPLE",
    "SIGNIFICANT_UNCERTAINTY",
    "CRITICAL_UNKNOWN",
    "EXECUTION_CONFIDENCE_MEDIUM",
    "EXECUTION_CONFIDENCE_LOW",
    "MATERIAL_WEAK_LINK",
    "WEAK_LINK_UNRESOLVED",
    "MARGINAL_RETURN_UNCERTAIN",
    "VALUATION_RELIABILITY_LOW",
    "VALUATION_RELIABILITY_NOT_ASSESSABLE",
    "MATERIAL_SOURCE_CONFLICT",
    "DECISION_BOUNDARY_MEDIUM_SENSITIVITY",
    "DECISION_BOUNDARY_HIGH_SENSITIVITY",
    "MATERIAL_COUNTEREVIDENCE_RISK",
    "MANUAL_DEEPER_REQUEST",
  ]);
});

test("Gate 16 simple baseline remains DEPTH_1", () => {
  const decision = deriveMinimumAnalyticalDepth(context());

  assert.equal(decision.minimumDepth, "DEPTH_1");
  assert.deepEqual(
    decision.triggers.map((trigger) => trigger.code),
    ["BASELINE_SIMPLE"],
  );
});

test("every frozen DEPTH_2 trigger deterministically routes to DEPTH_2", () => {
  const cases = [
    [
      "SIGNIFICANT_UNCERTAINTY",
      { significantUncertaintyCount: 1 },
    ],
    [
      "CRITICAL_UNKNOWN",
      { criticalUnknownCount: 1 },
    ],
    [
      "EXECUTION_CONFIDENCE_MEDIUM",
      { executionConfidence: "MEDIUM" },
    ],
    [
      "DECISION_BOUNDARY_MEDIUM_SENSITIVITY",
      { decisionBoundarySensitivity: "MEDIUM" },
    ],
  ] as const;

  for (const [expectedTrigger, patch] of cases) {
    const decision = deriveMinimumAnalyticalDepth(context(patch));

    assert.equal(
      decision.minimumDepth,
      "DEPTH_2",
      expectedTrigger,
    );
    assert.equal(
      decision.triggers.some(
        (trigger) => trigger.code === expectedTrigger,
      ),
      true,
      expectedTrigger,
    );
  }
});

test("every frozen DEPTH_3 trigger deterministically routes to DEPTH_3", () => {
  const cases = [
    [
      "MATERIAL_WEAK_LINK",
      { materialWeakLink: "YES" },
    ],
    [
      "WEAK_LINK_UNRESOLVED",
      { weakLinkUnresolved: true },
    ],
    [
      "MARGINAL_RETURN_UNCERTAIN",
      { marginalReturnUncertain: true },
    ],
    [
      "VALUATION_RELIABILITY_LOW",
      { valuationReliability: "LOW" },
    ],
    [
      "VALUATION_RELIABILITY_NOT_ASSESSABLE",
      { valuationReliability: "NOT_ASSESSABLE" },
    ],
    [
      "MATERIAL_SOURCE_CONFLICT",
      { materialSourceConflictCount: 1 },
    ],
    [
      "EXECUTION_CONFIDENCE_LOW",
      { executionConfidence: "LOW" },
    ],
    [
      "DECISION_BOUNDARY_HIGH_SENSITIVITY",
      { decisionBoundarySensitivity: "HIGH" },
    ],
    [
      "MATERIAL_COUNTEREVIDENCE_RISK",
      { materialCounterevidenceRisk: true },
    ],
  ] as const;

  for (const [expectedTrigger, patch] of cases) {
    const decision = deriveMinimumAnalyticalDepth(context(patch));

    assert.equal(
      decision.minimumDepth,
      "DEPTH_3",
      expectedTrigger,
    );
    assert.equal(
      decision.triggers.some(
        (trigger) => trigger.code === expectedTrigger,
      ),
      true,
      expectedTrigger,
    );
  }
});

test("DEPTH_3 trigger dominates simultaneous DEPTH_2 triggers", () => {
  const decision = deriveMinimumAnalyticalDepth(
    context({
      significantUncertaintyCount: 2,
      criticalUnknownCount: 1,
      executionConfidence: "MEDIUM",
      decisionBoundarySensitivity: "HIGH",
    }),
  );

  assert.equal(decision.minimumDepth, "DEPTH_3");
  assert.equal(
    decision.triggers.some(
      (trigger) =>
        trigger.code ===
        "DECISION_BOUNDARY_HIGH_SENSITIVITY",
    ),
    true,
  );
});

test("deterministic minimum cannot be silently downgraded", () => {
  assert.throws(
    () =>
      resolveAnalyticalDepth(
        context({ materialWeakLink: "YES" }),
        "DEPTH_2",
      ),
    /VNEXT_ANALYTICAL_DEPTH_DOWNGRADE_BELOW_REQUIRED_MINIMUM/,
  );
});

test("manual request may increase depth while retaining exact provenance", () => {
  const decision = resolveAnalyticalDepth(
    context(),
    "DEPTH_3",
  );

  assert.equal(decision.minimumDepth, "DEPTH_1");
  assert.equal(decision.selectedDepth, "DEPTH_3");
  assert.deepEqual(
    decision.triggers.map((trigger) => trigger.code),
    ["BASELINE_SIMPLE", "MANUAL_DEEPER_REQUEST"],
  );
  assert.equal(decision.secondAnalystEligible, true);
});

test("second analyst is rejected below DEPTH_3", () => {
  const decision = resolveAnalyticalDepth(
    context({ significantUncertaintyCount: 1 }),
  );

  const record: AnalyticalDepthConsumptionRecord = {
    runId: "RUN-G16-D2",
    stageCode: "DEEP_DIVE",
    moduleId: "VALUATION_ASSUMPTION_INTEGRITY",
    executionId: "EXEC-G16-D2",
    selectedDepth: "DEPTH_2",
    triggerCodes: ["SIGNIFICANT_UNCERTAINTY"],
    secondAnalystExecuted: true,
    reconciliationArtifactId: "ART-RECON-D2",
    providerRequestIds: [],
  };

  assert.throws(
    () => assertValidAnalyticalDepthConsumption(decision, record),
    /VNEXT_DEPTH_TRACE_SECOND_ANALYST_NOT_ELIGIBLE/,
  );
});

test("DEPTH_3 second analyst requires non-blank reconciliation identity", () => {
  const decision = resolveAnalyticalDepth(
    context({ marginalReturnUncertain: true }),
  );

  const missing: AnalyticalDepthConsumptionRecord = {
    runId: "RUN-G16-D3-A",
    stageCode: "DEEP_DIVE",
    moduleId: "RETURN_NORMALIZATION",
    executionId: "EXEC-G16-D3-A",
    selectedDepth: "DEPTH_3",
    triggerCodes: ["MARGINAL_RETURN_UNCERTAIN"],
    secondAnalystExecuted: true,
    reconciliationArtifactId: null,
    providerRequestIds: [],
  };

  assert.throws(
    () => assertValidAnalyticalDepthConsumption(decision, missing),
    /VNEXT_DEPTH_TRACE_RECONCILIATION_REQUIRED/,
  );

  const blank: AnalyticalDepthConsumptionRecord = {
    ...missing,
    executionId: "EXEC-G16-D3-B",
    reconciliationArtifactId: "   ",
  };

  assert.throws(
    () => assertValidAnalyticalDepthConsumption(decision, blank),
    /VNEXT_DEPTH_TRACE_RECONCILIATION_ARTIFACT_ID_REQUIRED/,
  );

  const valid: AnalyticalDepthConsumptionRecord = {
    ...missing,
    executionId: "EXEC-G16-D3-C",
    reconciliationArtifactId: "ART-RECON-G16",
  };

  assert.doesNotThrow(() =>
    assertValidAnalyticalDepthConsumption(decision, valid),
  );
});

test("consumption record must preserve the complete exact trigger set", () => {
  const decision = resolveAnalyticalDepth(
    context({
      significantUncertaintyCount: 1,
      executionConfidence: "MEDIUM",
      decisionBoundarySensitivity: "MEDIUM",
    }),
  );

  const incomplete: AnalyticalDepthConsumptionRecord = {
    runId: "RUN-G16-TRACE",
    stageCode: "DEEP_DIVE",
    moduleId: "VALUATION_DIAGNOSTIC_INTEGRITY",
    executionId: "EXEC-G16-TRACE",
    selectedDepth: "DEPTH_2",
    triggerCodes: ["SIGNIFICANT_UNCERTAINTY"],
    secondAnalystExecuted: false,
    reconciliationArtifactId: null,
    providerRequestIds: [],
  };

  assert.throws(
    () =>
      assertValidAnalyticalDepthConsumption(
        decision,
        incomplete,
      ),
    /VNEXT_DEPTH_TRACE_MISSING_DECISION_TRIGGER/,
  );

  const complete: AnalyticalDepthConsumptionRecord = {
    ...incomplete,
    triggerCodes: [
      "SIGNIFICANT_UNCERTAINTY",
      "EXECUTION_CONFIDENCE_MEDIUM",
      "DECISION_BOUNDARY_MEDIUM_SENSITIVITY",
    ],
  };

  assert.doesNotThrow(() =>
    assertValidAnalyticalDepthConsumption(decision, complete),
  );
});

test("Gate 16 runtime remains provider-neutral and mutation-free", () => {
  const source = readFileSync(
    "runtime/vnext/adaptive-analytical-depth.ts",
    "utf8",
  );

  const forbidden = [
    /AzureProvider/,
    /OpenAIProvider/,
    /AnthropicProvider/,
    /GeminiProvider/,
    /openai\.azure\.com/i,
    /services\.ai\.azure\.com/i,
    /createClient\s*\(/,
    /supabase/i,
    /process\.env/,
    /fetch\s*\(/,
    /\.(insert|update|upsert|delete)\s*\(/,
  ];

  for (const pattern of forbidden) {
    assert.equal(
      pattern.test(source),
      false,
      `unexpected Gate 16 runtime dependency: ${pattern}`,
    );
  }
});

test("Gate 16 freeze artifact pins the merged implementation boundary", () => {
  const freeze = readFileSync(
    "contracts/orotitan-equity/vnext/OROTITAN_VNEXT_GATE16_ADAPTIVE_ANALYTICAL_DEPTH_FREEZE_V1.0.md",
    "utf8",
  );

  assert.match(
    freeze,
    /IMPLEMENTATION_MERGE_SHA[\s\S]*155a09624633536407c67482fab8d4eeb9f6dd80/,
  );
  assert.match(freeze, /RESULT = PASS \/ FROZEN/);
  assert.match(freeze, /NEXT = GATE 17/);
});

import assert from "node:assert/strict";
import test from "node:test";

import smokeAttempt from "../calibration/vnext/OROTITAN_GATE18_SMOKE_ATTEMPT_001.json";

import {
  aggregateGate18RawObservations,
  assertObservationIntegrity,
  assertValidGate18EngineeringReceipt,
  type Gate18CalibrationObservation,
  type Gate18EngineeringReceipt,
  type Gate18QualityAdjudication,
} from "../runtime/vnext/model-calibration-evidence";

const SHA = "a".repeat(64);

function engineering(
  overrides: Partial<Gate18EngineeringReceipt> = {},
): Gate18EngineeringReceipt {
  return {
    invocation: {
      caseId: "CASE-001",
      displayName: "Example Co",
      sourceRunId: "RUN-001",
      dataCutoff: "2026-09-19",
      moduleId: "MOAT_PROOF",
      modelLabel: "SOL",
      modelId: "openai/gpt-5.6-sol",
      repetition: 1,
      promptTemplateId: "PROMPT-001",
      promptTemplateVersion: "0.1.0",
      promptTemplateSha256: SHA,
      generationSchemaId: "SCHEMA-001",
      generationSchemaVersion: "0.1.0",
      generationSchemaSha256: SHA,
      evidencePacketSha256: SHA,
    },
    executionId: "EXEC-001",
    providerRequestId: "REQ-001",
    schemaValid: true,
    semanticValid: true,
    latencyMs: 1200,
    inputTokens: 1000,
    cachedInputTokens: 0,
    outputTokens: 200,
    reasoningTokens: 300,
    totalTokens: 1500,
    retryCount: 0,
    estimatedCostUsd: 0.25,
    costProvenance: "AI_GATEWAY_RECEIPT",
    finishReason: "stop",
    responseSha256: SHA,
    ...overrides,
  };
}

function adjudication(
  executionId = "EXEC-001",
): Gate18QualityAdjudication {
  return {
    adjudicationId: "ADJ-001",
    executionId,
    adjudicatorType: "HUMAN",
    adjudicatorId: "HUMAN-001",
    factualErrors: {
      critical: 0,
      material: 1,
      minor: 2,
    },
    unsupportedMaterialClaims: 1,
    evidenceCoverage: {
      supportedRequiredFindings: 8,
      requiredFindings: 10,
    },
    contradictionDetection: {
      detectedMaterialConflicts: 3,
      requiredMaterialConflicts: 4,
    },
    counterEvidenceHandling: {
      handledMaterialCounterEvidence: 2,
      requiredMaterialCounterEvidence: 3,
    },
    criticalDecisionError: false,
    evidenceRefs: ["E-001", "C-001"],
    notes: null,
  };
}

test("Gate 18 engineering receipt validates raw traceability without model score", () => {
  const receipt = engineering();

  assert.doesNotThrow(() =>
    assertValidGate18EngineeringReceipt(receipt),
  );

  assert.equal(
    "score" in (receipt as unknown as Record<string, unknown>),
    false,
  );
});

test("Gate 18 cost requires explicit provenance", () => {
  const receipt = engineering({
    estimatedCostUsd: 0.12,
    costProvenance: null,
  });

  assert.throws(
    () => assertValidGate18EngineeringReceipt(receipt),
    /VNEXT_GATE18_COST_PROVENANCE_REQUIRED/,
  );
});

test("Gate 18 adjudication must bind to the exact execution", () => {
  const observation: Gate18CalibrationObservation = {
    engineering: engineering(),
    adjudication: adjudication("EXEC-OTHER"),
  };

  assert.throws(
    () => assertObservationIntegrity(observation),
    /VNEXT_GATE18_ADJUDICATION_EXECUTION_MISMATCH/,
  );
});

test("Gate 18 raw aggregation preserves counts and distributions without ranking", () => {
  const observations: Gate18CalibrationObservation[] = [
    {
      engineering: engineering(),
      adjudication: adjudication(),
    },
    {
      engineering: engineering({
        invocation: {
          ...engineering().invocation,
          modelLabel: "TERRA",
          modelId: "openai/gpt-5.6-terra",
        },
        executionId: "EXEC-002",
        latencyMs: 800,
        estimatedCostUsd: 0.1,
      }),
      adjudication: {
        ...adjudication("EXEC-002"),
        adjudicationId: "ADJ-002",
        factualErrors: {
          critical: 0,
          material: 0,
          minor: 1,
        },
        unsupportedMaterialClaims: 0,
      },
    },
  ];

  const aggregates = aggregateGate18RawObservations(observations);

  assert.equal(aggregates.length, 2);

  const sol = aggregates.find(
    (aggregate) => aggregate.modelLabel === "SOL",
  );
  const terra = aggregates.find(
    (aggregate) => aggregate.modelLabel === "TERRA",
  );

  assert.equal(sol?.observations, 1);
  assert.equal(sol?.factualErrorsMaterial, 1);
  assert.deepEqual(sol?.latenciesMs, [1200]);

  assert.equal(terra?.observations, 1);
  assert.equal(terra?.factualErrorsMaterial, 0);
  assert.deepEqual(terra?.estimatedCostsUsd, [0.1]);

  for (const aggregate of aggregates) {
    assert.equal(
      "rank" in (aggregate as unknown as Record<string, unknown>),
      false,
    );
    assert.equal(
      "score" in (aggregate as unknown as Record<string, unknown>),
      false,
    );
    assert.equal(
      "winner" in (aggregate as unknown as Record<string, unknown>),
      false,
    );
  }
});

test("Gate 18 smoke blocker is recorded as provider admission failure, not model evidence", () => {
  assert.equal(
    smokeAttempt.calibration_status,
    "BLOCKED_EXTERNAL_PROVIDER_ADMISSION",
  );
  assert.equal(smokeAttempt.gateway.reached, true);
  assert.equal(smokeAttempt.gateway.inner_status_code, 403);
  assert.equal(
    smokeAttempt.gateway.error_type,
    "customer_verification_required",
  );

  assert.equal(
    smokeAttempt.result.physical_model_generation_confirmed,
    false,
  );
  assert.equal(smokeAttempt.result.schema_receipts_valid, 0);
  assert.equal(
    smokeAttempt.result.analytical_quality_observations,
    0,
  );
  assert.equal(smokeAttempt.result.winner_selected, false);

  assert.equal(
    smokeAttempt.classification.analytical_methodology_failure,
    false,
  );
  assert.equal(
    smokeAttempt.classification.model_quality_failure,
    false,
  );
});

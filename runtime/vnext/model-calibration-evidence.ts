export const GATE18_ADJUDICATOR_TYPES = [
  "HUMAN",
  "DETERMINISTIC",
] as const;

export type Gate18AdjudicatorType =
  (typeof GATE18_ADJUDICATOR_TYPES)[number];

export interface Gate18InvocationIdentity {
  caseId: string;
  displayName: string;
  sourceRunId: string;
  dataCutoff: string;
  moduleId: string;
  modelLabel: "LUNA" | "TERRA" | "SOL" | "ASTRA";
  modelId: string;
  repetition: number;
  promptTemplateId: string;
  promptTemplateVersion: string;
  promptTemplateSha256: string;
  generationSchemaId: string;
  generationSchemaVersion: string;
  generationSchemaSha256: string;
  evidencePacketSha256: string;
}

export interface Gate18EngineeringReceipt {
  invocation: Gate18InvocationIdentity;
  executionId: string;
  providerRequestId: string | null;
  schemaValid: boolean;
  semanticValid: boolean;
  latencyMs: number;
  inputTokens: number | null;
  cachedInputTokens: number | null;
  outputTokens: number | null;
  reasoningTokens: number | null;
  totalTokens: number | null;
  retryCount: number;
  estimatedCostUsd: number | null;
  costProvenance: string | null;
  finishReason: string | null;
  responseSha256: string;
}

export interface Gate18QualityAdjudication {
  adjudicationId: string;
  executionId: string;
  adjudicatorType: Gate18AdjudicatorType;
  adjudicatorId: string;
  factualErrors: {
    critical: number;
    material: number;
    minor: number;
  };
  unsupportedMaterialClaims: number;
  evidenceCoverage: {
    supportedRequiredFindings: number;
    requiredFindings: number;
  };
  contradictionDetection: {
    detectedMaterialConflicts: number;
    requiredMaterialConflicts: number;
  };
  counterEvidenceHandling: {
    handledMaterialCounterEvidence: number;
    requiredMaterialCounterEvidence: number;
  };
  criticalDecisionError: boolean;
  evidenceRefs: readonly string[];
  notes: string | null;
}

export interface Gate18CalibrationObservation {
  engineering: Gate18EngineeringReceipt;
  adjudication: Gate18QualityAdjudication | null;
}

export interface Gate18ModelRawAggregate {
  modelLabel: Gate18InvocationIdentity["modelLabel"];
  modelId: string;
  observations: number;
  adjudicatedObservations: number;
  schemaPasses: number;
  semanticPasses: number;
  criticalDecisionErrors: number;
  factualErrorsCritical: number;
  factualErrorsMaterial: number;
  factualErrorsMinor: number;
  unsupportedMaterialClaims: number;
  supportedRequiredFindings: number;
  requiredFindings: number;
  detectedMaterialConflicts: number;
  requiredMaterialConflicts: number;
  handledMaterialCounterEvidence: number;
  requiredMaterialCounterEvidence: number;
  latenciesMs: readonly number[];
  estimatedCostsUsd: readonly number[];
  retries: readonly number[];
}

const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

function assertNonBlank(value: string, code: string): void {
  if (value.trim().length === 0) {
    throw new Error(code);
  }
}

function assertNonNegativeInteger(
  value: number,
  code: string,
): void {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(code);
  }
}

function assertSha256(value: string, code: string): void {
  if (!SHA256_PATTERN.test(value)) {
    throw new Error(code);
  }
}

function assertRatioCounts(
  numerator: number,
  denominator: number,
  code: string,
): void {
  assertNonNegativeInteger(numerator, code);
  assertNonNegativeInteger(denominator, code);

  if (numerator > denominator) {
    throw new Error(code);
  }
}

export function assertValidGate18EngineeringReceipt(
  receipt: Gate18EngineeringReceipt,
): void {
  const { invocation } = receipt;

  for (const [value, code] of [
    [invocation.caseId, "VNEXT_GATE18_CASE_ID_REQUIRED"],
    [invocation.displayName, "VNEXT_GATE18_DISPLAY_NAME_REQUIRED"],
    [invocation.sourceRunId, "VNEXT_GATE18_SOURCE_RUN_ID_REQUIRED"],
    [invocation.moduleId, "VNEXT_GATE18_MODULE_ID_REQUIRED"],
    [invocation.modelId, "VNEXT_GATE18_MODEL_ID_REQUIRED"],
    [
      invocation.promptTemplateId,
      "VNEXT_GATE18_PROMPT_TEMPLATE_ID_REQUIRED",
    ],
    [
      invocation.promptTemplateVersion,
      "VNEXT_GATE18_PROMPT_TEMPLATE_VERSION_REQUIRED",
    ],
    [
      invocation.generationSchemaId,
      "VNEXT_GATE18_GENERATION_SCHEMA_ID_REQUIRED",
    ],
    [
      invocation.generationSchemaVersion,
      "VNEXT_GATE18_GENERATION_SCHEMA_VERSION_REQUIRED",
    ],
    [receipt.executionId, "VNEXT_GATE18_EXECUTION_ID_REQUIRED"],
  ] as const) {
    assertNonBlank(value, code);
  }

  if (!DATE_PATTERN.test(invocation.dataCutoff)) {
    throw new Error("VNEXT_GATE18_DATA_CUTOFF_INVALID");
  }

  assertNonNegativeInteger(
    invocation.repetition,
    "VNEXT_GATE18_REPETITION_INVALID",
  );

  if (invocation.repetition < 1) {
    throw new Error("VNEXT_GATE18_REPETITION_INVALID");
  }

  for (const [value, code] of [
    [
      invocation.promptTemplateSha256,
      "VNEXT_GATE18_PROMPT_SHA256_INVALID",
    ],
    [
      invocation.generationSchemaSha256,
      "VNEXT_GATE18_GENERATION_SCHEMA_SHA256_INVALID",
    ],
    [
      invocation.evidencePacketSha256,
      "VNEXT_GATE18_EVIDENCE_PACKET_SHA256_INVALID",
    ],
    [receipt.responseSha256, "VNEXT_GATE18_RESPONSE_SHA256_INVALID"],
  ] as const) {
    assertSha256(value, code);
  }

  if (
    !Number.isFinite(receipt.latencyMs) ||
    receipt.latencyMs < 0
  ) {
    throw new Error("VNEXT_GATE18_LATENCY_INVALID");
  }

  assertNonNegativeInteger(
    receipt.retryCount,
    "VNEXT_GATE18_RETRY_COUNT_INVALID",
  );

  for (const count of [
    receipt.inputTokens,
    receipt.cachedInputTokens,
    receipt.outputTokens,
    receipt.reasoningTokens,
    receipt.totalTokens,
  ]) {
    if (
      count !== null &&
      (!Number.isInteger(count) || count < 0)
    ) {
      throw new Error("VNEXT_GATE18_TOKEN_USAGE_INVALID");
    }
  }

  if (
    receipt.estimatedCostUsd !== null &&
    (!Number.isFinite(receipt.estimatedCostUsd) ||
      receipt.estimatedCostUsd < 0)
  ) {
    throw new Error("VNEXT_GATE18_COST_INVALID");
  }

  if (
    receipt.estimatedCostUsd !== null &&
    (receipt.costProvenance === null ||
      receipt.costProvenance.trim().length === 0)
  ) {
    throw new Error(
      "VNEXT_GATE18_COST_PROVENANCE_REQUIRED",
    );
  }
}

export function assertValidGate18QualityAdjudication(
  adjudication: Gate18QualityAdjudication,
): void {
  for (const [value, code] of [
    [
      adjudication.adjudicationId,
      "VNEXT_GATE18_ADJUDICATION_ID_REQUIRED",
    ],
    [
      adjudication.executionId,
      "VNEXT_GATE18_ADJUDICATION_EXECUTION_ID_REQUIRED",
    ],
    [
      adjudication.adjudicatorId,
      "VNEXT_GATE18_ADJUDICATOR_ID_REQUIRED",
    ],
  ] as const) {
    assertNonBlank(value, code);
  }

  if (
    !GATE18_ADJUDICATOR_TYPES.includes(
      adjudication.adjudicatorType,
    )
  ) {
    throw new Error("VNEXT_GATE18_ADJUDICATOR_TYPE_INVALID");
  }

  for (const value of [
    adjudication.factualErrors.critical,
    adjudication.factualErrors.material,
    adjudication.factualErrors.minor,
    adjudication.unsupportedMaterialClaims,
  ]) {
    assertNonNegativeInteger(
      value,
      "VNEXT_GATE18_QUALITY_COUNT_INVALID",
    );
  }

  assertRatioCounts(
    adjudication.evidenceCoverage.supportedRequiredFindings,
    adjudication.evidenceCoverage.requiredFindings,
    "VNEXT_GATE18_EVIDENCE_COVERAGE_INVALID",
  );
  assertRatioCounts(
    adjudication.contradictionDetection.detectedMaterialConflicts,
    adjudication.contradictionDetection.requiredMaterialConflicts,
    "VNEXT_GATE18_CONTRADICTION_COUNTS_INVALID",
  );
  assertRatioCounts(
    adjudication.counterEvidenceHandling
      .handledMaterialCounterEvidence,
    adjudication.counterEvidenceHandling
      .requiredMaterialCounterEvidence,
    "VNEXT_GATE18_COUNTEREVIDENCE_COUNTS_INVALID",
  );

  if (adjudication.evidenceRefs.length === 0) {
    throw new Error("VNEXT_GATE18_ADJUDICATION_EVIDENCE_REQUIRED");
  }

  for (const ref of adjudication.evidenceRefs) {
    assertNonBlank(
      ref,
      "VNEXT_GATE18_ADJUDICATION_EVIDENCE_REF_REQUIRED",
    );
  }
}

export function assertObservationIntegrity(
  observation: Gate18CalibrationObservation,
): void {
  assertValidGate18EngineeringReceipt(observation.engineering);

  if (observation.adjudication === null) {
    return;
  }

  assertValidGate18QualityAdjudication(
    observation.adjudication,
  );

  if (
    observation.adjudication.executionId !==
    observation.engineering.executionId
  ) {
    throw new Error(
      "VNEXT_GATE18_ADJUDICATION_EXECUTION_MISMATCH",
    );
  }
}

export function aggregateGate18RawObservations(
  observations: readonly Gate18CalibrationObservation[],
): readonly Gate18ModelRawAggregate[] {
  const aggregates = new Map<
    string,
    {
      aggregate: Gate18ModelRawAggregate;
      modelId: string;
    }
  >();

  for (const observation of observations) {
    assertObservationIntegrity(observation);

    const { engineering, adjudication } = observation;
    const key = engineering.invocation.modelLabel;
    const existing = aggregates.get(key);

    if (
      existing &&
      existing.modelId !== engineering.invocation.modelId
    ) {
      throw new Error(
        "VNEXT_GATE18_MODEL_LABEL_PHYSICAL_ID_DRIFT",
      );
    }

    const aggregate =
      existing?.aggregate ??
      {
        modelLabel: engineering.invocation.modelLabel,
        modelId: engineering.invocation.modelId,
        observations: 0,
        adjudicatedObservations: 0,
        schemaPasses: 0,
        semanticPasses: 0,
        criticalDecisionErrors: 0,
        factualErrorsCritical: 0,
        factualErrorsMaterial: 0,
        factualErrorsMinor: 0,
        unsupportedMaterialClaims: 0,
        supportedRequiredFindings: 0,
        requiredFindings: 0,
        detectedMaterialConflicts: 0,
        requiredMaterialConflicts: 0,
        handledMaterialCounterEvidence: 0,
        requiredMaterialCounterEvidence: 0,
        latenciesMs: [],
        estimatedCostsUsd: [],
        retries: [],
      };

    aggregate.observations += 1;
    aggregate.schemaPasses += engineering.schemaValid ? 1 : 0;
    aggregate.semanticPasses += engineering.semanticValid ? 1 : 0;
    (aggregate.latenciesMs as number[]).push(engineering.latencyMs);
    (aggregate.retries as number[]).push(engineering.retryCount);

    if (engineering.estimatedCostUsd !== null) {
      (aggregate.estimatedCostsUsd as number[]).push(
        engineering.estimatedCostUsd,
      );
    }

    if (adjudication !== null) {
      aggregate.adjudicatedObservations += 1;
      aggregate.criticalDecisionErrors +=
        adjudication.criticalDecisionError ? 1 : 0;
      aggregate.factualErrorsCritical +=
        adjudication.factualErrors.critical;
      aggregate.factualErrorsMaterial +=
        adjudication.factualErrors.material;
      aggregate.factualErrorsMinor +=
        adjudication.factualErrors.minor;
      aggregate.unsupportedMaterialClaims +=
        adjudication.unsupportedMaterialClaims;
      aggregate.supportedRequiredFindings +=
        adjudication.evidenceCoverage.supportedRequiredFindings;
      aggregate.requiredFindings +=
        adjudication.evidenceCoverage.requiredFindings;
      aggregate.detectedMaterialConflicts +=
        adjudication.contradictionDetection
          .detectedMaterialConflicts;
      aggregate.requiredMaterialConflicts +=
        adjudication.contradictionDetection
          .requiredMaterialConflicts;
      aggregate.handledMaterialCounterEvidence +=
        adjudication.counterEvidenceHandling
          .handledMaterialCounterEvidence;
      aggregate.requiredMaterialCounterEvidence +=
        adjudication.counterEvidenceHandling
          .requiredMaterialCounterEvidence;
    }

    aggregates.set(key, {
      aggregate,
      modelId: engineering.invocation.modelId,
    });
  }

  return [...aggregates.values()]
    .map(({ aggregate }) => aggregate)
    .sort((left, right) =>
      left.modelLabel.localeCompare(right.modelLabel),
    );
}

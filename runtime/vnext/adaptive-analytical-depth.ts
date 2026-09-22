export const ANALYTICAL_DEPTHS = [
  "DEPTH_1",
  "DEPTH_2",
  "DEPTH_3",
] as const;

export type AnalyticalDepth = (typeof ANALYTICAL_DEPTHS)[number];

export const DEPTH_TRIGGER_CODES = [
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
] as const;

export type DepthTriggerCode =
  (typeof DEPTH_TRIGGER_CODES)[number];

export type TriState = "YES" | "NO" | "UNKNOWN";

export type ExecutionConfidence = "HIGH" | "MEDIUM" | "LOW";

export type ValuationReliability =
  | "HIGH"
  | "MEDIUM"
  | "LOW"
  | "NOT_ASSESSABLE"
  | "UNKNOWN";

export type DecisionBoundarySensitivity =
  | "LOW"
  | "MEDIUM"
  | "HIGH"
  | "UNKNOWN";

export interface AnalyticalDepthContext {
  materialWeakLink: TriState;
  weakLinkUnresolved: boolean;
  marginalReturnUncertain: boolean;
  valuationReliability: ValuationReliability;
  materialSourceConflictCount: number;
  significantUncertaintyCount: number;
  criticalUnknownCount: number;
  executionConfidence: ExecutionConfidence;
  decisionBoundarySensitivity: DecisionBoundarySensitivity;
  materialCounterevidenceRisk: boolean;
}

export interface DepthTriggerRecord {
  code: DepthTriggerCode;
  minimumDepth: AnalyticalDepth;
  rationale: string;
}

export interface AnalyticalDepthDecision {
  minimumDepth: AnalyticalDepth;
  selectedDepth: AnalyticalDepth;
  triggers: readonly DepthTriggerRecord[];
  secondAnalystEligible: boolean;
  reconciliationRequiredIfSecondAnalystRuns: boolean;
  deterministic: true;
}

export interface AnalyticalDepthConsumptionRecord {
  runId: string;
  stageCode: string;
  moduleId: string;
  executionId: string;
  selectedDepth: AnalyticalDepth;
  triggerCodes: readonly DepthTriggerCode[];
  secondAnalystExecuted: boolean;
  reconciliationArtifactId: string | null;
  providerRequestIds: readonly string[];
}

const DEPTH_RANK: Record<AnalyticalDepth, number> = {
  DEPTH_1: 1,
  DEPTH_2: 2,
  DEPTH_3: 3,
};

function assertNonBlank(value: string, code: string): void {
  if (value.trim().length === 0) {
    throw new Error(code);
  }
}

function maxDepth(
  left: AnalyticalDepth,
  right: AnalyticalDepth,
): AnalyticalDepth {
  return DEPTH_RANK[left] >= DEPTH_RANK[right] ? left : right;
}

function addTrigger(
  triggers: DepthTriggerRecord[],
  code: DepthTriggerCode,
  minimumDepth: AnalyticalDepth,
  rationale: string,
): void {
  triggers.push({ code, minimumDepth, rationale });
}

export function deriveMinimumAnalyticalDepth(
  context: AnalyticalDepthContext,
): {
  minimumDepth: AnalyticalDepth;
  triggers: readonly DepthTriggerRecord[];
} {
  if (
    !Number.isInteger(context.materialSourceConflictCount) ||
    context.materialSourceConflictCount < 0
  ) {
    throw new Error(
      "VNEXT_ANALYTICAL_DEPTH_INVALID_MATERIAL_CONFLICT_COUNT",
    );
  }

  if (
    !Number.isInteger(context.significantUncertaintyCount) ||
    context.significantUncertaintyCount < 0
  ) {
    throw new Error(
      "VNEXT_ANALYTICAL_DEPTH_INVALID_SIGNIFICANT_UNCERTAINTY_COUNT",
    );
  }

  if (
    !Number.isInteger(context.criticalUnknownCount) ||
    context.criticalUnknownCount < 0
  ) {
    throw new Error(
      "VNEXT_ANALYTICAL_DEPTH_INVALID_CRITICAL_UNKNOWN_COUNT",
    );
  }

  const triggers: DepthTriggerRecord[] = [];

  if (context.significantUncertaintyCount > 0) {
    addTrigger(
      triggers,
      "SIGNIFICANT_UNCERTAINTY",
      "DEPTH_2",
      "At least one significant uncertainty requires analysis beyond the simple baseline.",
    );
  }

  if (context.criticalUnknownCount > 0) {
    addTrigger(
      triggers,
      "CRITICAL_UNKNOWN",
      "DEPTH_2",
      "A critical UNKNOWN requires explicit additional analytical work.",
    );
  }

  if (context.executionConfidence === "MEDIUM") {
    addTrigger(
      triggers,
      "EXECUTION_CONFIDENCE_MEDIUM",
      "DEPTH_2",
      "Medium execution confidence requires additional analytical depth.",
    );
  }

  if (context.decisionBoundarySensitivity === "MEDIUM") {
    addTrigger(
      triggers,
      "DECISION_BOUNDARY_MEDIUM_SENSITIVITY",
      "DEPTH_2",
      "The decision is moderately sensitive to a boundary condition.",
    );
  }

  if (context.materialWeakLink === "YES") {
    addTrigger(
      triggers,
      "MATERIAL_WEAK_LINK",
      "DEPTH_3",
      "A material weak link is decision-relevant and requires maximum analytical depth.",
    );
  }

  if (context.weakLinkUnresolved) {
    addTrigger(
      triggers,
      "WEAK_LINK_UNRESOLVED",
      "DEPTH_3",
      "An unresolved weak-link question requires maximum analytical depth.",
    );
  }

  if (context.marginalReturnUncertain) {
    addTrigger(
      triggers,
      "MARGINAL_RETURN_UNCERTAIN",
      "DEPTH_3",
      "Uncertain marginal/cohort return is a hard analytical-depth trigger.",
    );
  }

  if (context.valuationReliability === "LOW") {
    addTrigger(
      triggers,
      "VALUATION_RELIABILITY_LOW",
      "DEPTH_3",
      "Low valuation reliability requires maximum analytical depth.",
    );
  }

  if (context.valuationReliability === "NOT_ASSESSABLE") {
    addTrigger(
      triggers,
      "VALUATION_RELIABILITY_NOT_ASSESSABLE",
      "DEPTH_3",
      "A non-assessable valuation reliability state requires maximum analytical depth before a decision can rely on valuation.",
    );
  }

  if (context.materialSourceConflictCount > 0) {
    addTrigger(
      triggers,
      "MATERIAL_SOURCE_CONFLICT",
      "DEPTH_3",
      "At least one material source conflict remains active.",
    );
  }

  if (context.executionConfidence === "LOW") {
    addTrigger(
      triggers,
      "EXECUTION_CONFIDENCE_LOW",
      "DEPTH_3",
      "Low execution confidence is a hard analytical-depth trigger.",
    );
  }

  if (context.decisionBoundarySensitivity === "HIGH") {
    addTrigger(
      triggers,
      "DECISION_BOUNDARY_HIGH_SENSITIVITY",
      "DEPTH_3",
      "The decision is highly sensitive to a boundary condition.",
    );
  }

  if (context.materialCounterevidenceRisk) {
    addTrigger(
      triggers,
      "MATERIAL_COUNTEREVIDENCE_RISK",
      "DEPTH_3",
      "Material counter-evidence risk requires deeper adversarial analysis.",
    );
  }

  if (triggers.length === 0) {
    return {
      minimumDepth: "DEPTH_1",
      triggers: [
        {
          code: "BASELINE_SIMPLE",
          minimumDepth: "DEPTH_1",
          rationale:
            "No material uncertainty, conflict, weak-link, valuation-reliability or decision-boundary trigger requires deeper analysis.",
        },
      ],
    };
  }

  let minimumDepth: AnalyticalDepth = "DEPTH_1";
  for (const trigger of triggers) {
    minimumDepth = maxDepth(minimumDepth, trigger.minimumDepth);
  }

  return {
    minimumDepth,
    triggers,
  };
}

export function resolveAnalyticalDepth(
  context: AnalyticalDepthContext,
  requestedDepth?: AnalyticalDepth,
): AnalyticalDepthDecision {
  const derived = deriveMinimumAnalyticalDepth(context);
  const triggers = [...derived.triggers];
  let selectedDepth = derived.minimumDepth;

  if (requestedDepth !== undefined) {
    if (!ANALYTICAL_DEPTHS.includes(requestedDepth)) {
      throw new Error(
        "VNEXT_ANALYTICAL_DEPTH_INVALID_REQUESTED_DEPTH",
      );
    }

    if (
      DEPTH_RANK[requestedDepth] < DEPTH_RANK[derived.minimumDepth]
    ) {
      throw new Error(
        "VNEXT_ANALYTICAL_DEPTH_DOWNGRADE_BELOW_REQUIRED_MINIMUM",
      );
    }

    if (
      DEPTH_RANK[requestedDepth] > DEPTH_RANK[derived.minimumDepth]
    ) {
      selectedDepth = requestedDepth;
      triggers.push({
        code: "MANUAL_DEEPER_REQUEST",
        minimumDepth: requestedDepth,
        rationale:
          "A deeper-than-minimum analytical pass was explicitly requested; the deterministic minimum was not reduced.",
      });
    }
  }

  return {
    minimumDepth: derived.minimumDepth,
    selectedDepth,
    triggers,
    secondAnalystEligible: selectedDepth === "DEPTH_3",
    reconciliationRequiredIfSecondAnalystRuns:
      selectedDepth === "DEPTH_3",
    deterministic: true,
  };
}

export function assertValidAnalyticalDepthConsumption(
  decision: AnalyticalDepthDecision,
  record: AnalyticalDepthConsumptionRecord,
): void {
  assertNonBlank(record.runId, "VNEXT_DEPTH_TRACE_RUN_ID_REQUIRED");
  assertNonBlank(
    record.stageCode,
    "VNEXT_DEPTH_TRACE_STAGE_CODE_REQUIRED",
  );
  assertNonBlank(
    record.moduleId,
    "VNEXT_DEPTH_TRACE_MODULE_ID_REQUIRED",
  );
  assertNonBlank(
    record.executionId,
    "VNEXT_DEPTH_TRACE_EXECUTION_ID_REQUIRED",
  );

  if (record.selectedDepth !== decision.selectedDepth) {
    throw new Error(
      "VNEXT_DEPTH_TRACE_SELECTED_DEPTH_MISMATCH",
    );
  }

  const decisionTriggerCodes = new Set(
    decision.triggers.map((trigger) => trigger.code),
  );

  if (record.triggerCodes.length === 0) {
    throw new Error(
      "VNEXT_DEPTH_TRACE_TRIGGER_REQUIRED",
    );
  }

  const recordedTriggerCodes = new Set(record.triggerCodes);

  if (recordedTriggerCodes.size !== record.triggerCodes.length) {
    throw new Error(
      "VNEXT_DEPTH_TRACE_DUPLICATE_TRIGGER_CODE",
    );
  }

  for (const code of record.triggerCodes) {
    if (!decisionTriggerCodes.has(code)) {
      throw new Error(
        "VNEXT_DEPTH_TRACE_UNAUTHORIZED_TRIGGER_CODE",
      );
    }
  }

  for (const code of decisionTriggerCodes) {
    if (!recordedTriggerCodes.has(code)) {
      throw new Error(
        "VNEXT_DEPTH_TRACE_MISSING_DECISION_TRIGGER",
      );
    }
  }

  if (
    record.secondAnalystExecuted &&
    !decision.secondAnalystEligible
  ) {
    throw new Error(
      "VNEXT_DEPTH_TRACE_SECOND_ANALYST_NOT_ELIGIBLE",
    );
  }

  if (
    record.secondAnalystExecuted &&
    record.reconciliationArtifactId === null
  ) {
    throw new Error(
      "VNEXT_DEPTH_TRACE_RECONCILIATION_REQUIRED",
    );
  }

  if (
    record.secondAnalystExecuted &&
    record.reconciliationArtifactId !== null
  ) {
    assertNonBlank(
      record.reconciliationArtifactId,
      "VNEXT_DEPTH_TRACE_RECONCILIATION_ARTIFACT_ID_REQUIRED",
    );
  }

  if (
    !record.secondAnalystExecuted &&
    record.reconciliationArtifactId !== null
  ) {
    throw new Error(
      "VNEXT_DEPTH_TRACE_RECONCILIATION_WITHOUT_SECOND_ANALYST",
    );
  }
}

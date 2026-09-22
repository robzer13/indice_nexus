export const RUNTIME_STATES = [
  "IDLE",
  "READY",
  "RUNNING",
  "PAUSED",
  "BLOCKED",
  "RECOVERING",
  "FAILED",
  "COMPLETE",
] as const;

export type RuntimeState = (typeof RUNTIME_STATES)[number];

export const STAGE_STATES = [
  "NOT_STARTED",
  "IN_PROGRESS",
  "PAUSED",
  "BLOCKED",
  "COMPLETE",
] as const;

export type StageState = (typeof STAGE_STATES)[number];

export const ANALYTICAL_STATES = [
  "INSUFFICIENT",
  "IN_PROGRESS",
  "PROVISIONALLY_STABLE",
  "LOCKED",
] as const;

export type AnalyticalState = (typeof ANALYTICAL_STATES)[number];

export const EVIDENCE_STATES = [
  "UNKNOWN",
  "SUFFICIENT",
  "PARTIAL_BUT_DECISIONABLE",
  "INSUFFICIENT",
  "CONFLICTED",
] as const;

export type EvidenceState = (typeof EVIDENCE_STATES)[number];

export const VALUATION_RELIABILITY_STATES = [
  "UNKNOWN",
  "HIGH",
  "MEDIUM",
  "LOW",
  "NOT_ASSESSABLE",
] as const;

export type ValuationReliability =
  (typeof VALUATION_RELIABILITY_STATES)[number];

export const PRICE_CONDITIONS = [
  "UNKNOWN",
  "NOT_ASSESSABLE",
  "ABOVE_REQUIRED_RETURN_PRICE",
  "AT_OR_BELOW_REQUIRED_RETURN_PRICE",
  "AT_OR_BELOW_STRONG_RETURN_PRICE",
  "AT_OR_BELOW_EXCEPTIONAL_RETURN_PRICE",
] as const;

export type PriceCondition = (typeof PRICE_CONDITIONS)[number];

export const DECISION_STATES = [
  "UNKNOWN",
  "INVESTABLE_NOW",
  "WAIT_FOR_PRICE",
  "WAIT_FOR_EVIDENCE",
  "REFRESH_REQUIRED",
  "REJECT",
] as const;

export type DecisionState = (typeof DECISION_STATES)[number];

export const AUDIT_STATUSES = [
  "NOT_RUN",
  "IN_PROGRESS",
  "PASS",
  "FAIL",
  "STALE",
] as const;

export type AuditStatus = (typeof AUDIT_STATUSES)[number];

export interface StateTransitionContext {
  controlledReopen?: boolean;
  invalidateAssessment?: boolean;
  rejectionReopenAuthorized?: boolean;
}

function contains<T extends string>(values: readonly T[], value: T): boolean {
  return values.includes(value);
}

export const RUNTIME_TRANSITIONS: Readonly<
  Record<RuntimeState, readonly RuntimeState[]>
> = {
  IDLE: ["READY"],
  READY: ["RUNNING", "BLOCKED"],
  RUNNING: ["PAUSED", "BLOCKED", "RECOVERING", "FAILED", "COMPLETE"],
  PAUSED: ["READY", "RUNNING", "BLOCKED"],
  BLOCKED: ["READY", "RECOVERING", "FAILED"],
  RECOVERING: ["READY", "RUNNING", "BLOCKED", "FAILED"],
  FAILED: ["RECOVERING"],
  COMPLETE: ["READY"],
};

export const STAGE_TRANSITIONS: Readonly<
  Record<StageState, readonly StageState[]>
> = {
  NOT_STARTED: ["IN_PROGRESS"],
  IN_PROGRESS: ["PAUSED", "BLOCKED", "COMPLETE"],
  PAUSED: ["IN_PROGRESS", "BLOCKED"],
  BLOCKED: ["IN_PROGRESS", "PAUSED"],
  COMPLETE: [],
};

export const ANALYTICAL_TRANSITIONS: Readonly<
  Record<AnalyticalState, readonly AnalyticalState[]>
> = {
  INSUFFICIENT: ["IN_PROGRESS"],
  IN_PROGRESS: ["INSUFFICIENT", "PROVISIONALLY_STABLE", "LOCKED"],
  PROVISIONALLY_STABLE: ["INSUFFICIENT", "IN_PROGRESS", "LOCKED"],
  LOCKED: [],
};

export const EVIDENCE_TRANSITIONS: Readonly<
  Record<EvidenceState, readonly EvidenceState[]>
> = Object.fromEntries(
  EVIDENCE_STATES.map((from) => [
    from,
    EVIDENCE_STATES.filter((to) => to !== from),
  ]),
) as Record<EvidenceState, readonly EvidenceState[]>;

export const PRICE_CONDITION_TRANSITIONS: Readonly<
  Record<PriceCondition, readonly PriceCondition[]>
> = Object.fromEntries(
  PRICE_CONDITIONS.map((from) => [
    from,
    PRICE_CONDITIONS.filter((to) => to !== from),
  ]),
) as Record<PriceCondition, readonly PriceCondition[]>;

export const DECISION_TRANSITIONS: Readonly<
  Record<DecisionState, readonly DecisionState[]>
> = {
  UNKNOWN: [
    "INVESTABLE_NOW",
    "WAIT_FOR_PRICE",
    "WAIT_FOR_EVIDENCE",
    "REFRESH_REQUIRED",
    "REJECT",
  ],
  INVESTABLE_NOW: [
    "WAIT_FOR_PRICE",
    "WAIT_FOR_EVIDENCE",
    "REFRESH_REQUIRED",
    "REJECT",
  ],
  WAIT_FOR_PRICE: [
    "INVESTABLE_NOW",
    "WAIT_FOR_EVIDENCE",
    "REFRESH_REQUIRED",
    "REJECT",
  ],
  WAIT_FOR_EVIDENCE: [
    "INVESTABLE_NOW",
    "WAIT_FOR_PRICE",
    "REFRESH_REQUIRED",
    "REJECT",
  ],
  REFRESH_REQUIRED: [
    "INVESTABLE_NOW",
    "WAIT_FOR_PRICE",
    "WAIT_FOR_EVIDENCE",
    "REJECT",
  ],
  REJECT: [],
};

export const AUDIT_TRANSITIONS: Readonly<
  Record<AuditStatus, readonly AuditStatus[]>
> = {
  NOT_RUN: ["IN_PROGRESS"],
  IN_PROGRESS: ["PASS", "FAIL"],
  PASS: ["STALE"],
  FAIL: ["IN_PROGRESS"],
  STALE: ["IN_PROGRESS"],
};

export function canRuntimeTransition(
  from: RuntimeState,
  to: RuntimeState,
): boolean {
  return contains(RUNTIME_TRANSITIONS[from], to);
}

export function canStageTransition(
  from: StageState,
  to: StageState,
  context: StateTransitionContext = {},
): boolean {
  if (from === "COMPLETE" && to === "IN_PROGRESS") {
    return context.controlledReopen === true;
  }
  return contains(STAGE_TRANSITIONS[from], to);
}

export function canAnalyticalTransition(
  from: AnalyticalState,
  to: AnalyticalState,
  context: StateTransitionContext = {},
): boolean {
  if (from === "LOCKED" && (to === "IN_PROGRESS" || to === "INSUFFICIENT")) {
    return context.controlledReopen === true;
  }
  return contains(ANALYTICAL_TRANSITIONS[from], to);
}

export function canEvidenceTransition(
  from: EvidenceState,
  to: EvidenceState,
): boolean {
  return contains(EVIDENCE_TRANSITIONS[from], to);
}

export function canValuationReliabilityTransition(
  from: ValuationReliability,
  to: ValuationReliability,
  context: StateTransitionContext = {},
): boolean {
  if (from === to) {
    return false;
  }

  if (to === "UNKNOWN") {
    return from !== "UNKNOWN" && context.invalidateAssessment === true;
  }

  return true;
}

export function canPriceConditionTransition(
  from: PriceCondition,
  to: PriceCondition,
): boolean {
  return contains(PRICE_CONDITION_TRANSITIONS[from], to);
}

export function canDecisionTransition(
  from: DecisionState,
  to: DecisionState,
  context: StateTransitionContext = {},
): boolean {
  if (from === "REJECT" && to === "REFRESH_REQUIRED") {
    return context.rejectionReopenAuthorized === true;
  }
  return contains(DECISION_TRANSITIONS[from], to);
}

export function canAuditTransition(
  from: AuditStatus,
  to: AuditStatus,
): boolean {
  return contains(AUDIT_TRANSITIONS[from], to);
}

export interface VNextStateVector {
  runtimeState: RuntimeState;
  stageState: StageState;
  analyticalState: AnalyticalState;
  evidenceState: EvidenceState;
  valuationReliability: ValuationReliability;
  priceCondition: PriceCondition;
  decisionState: DecisionState;
  auditStatus: AuditStatus;
}

export interface StateVectorValidationResult {
  valid: boolean;
  errors: readonly string[];
}

export function validateStateVector(
  state: VNextStateVector,
): StateVectorValidationResult {
  const errors: string[] = [];

  if (
    state.valuationReliability === "NOT_ASSESSABLE" &&
    !["UNKNOWN", "NOT_ASSESSABLE"].includes(state.priceCondition)
  ) {
    errors.push(
      "PRICE_CONDITION_REQUIRES_ASSESSABLE_VALUATION: a priced return-zone condition cannot coexist with VALUATION_RELIABILITY=NOT_ASSESSABLE",
    );
  }

  if (state.decisionState === "INVESTABLE_NOW") {
    if (
      ["UNKNOWN", "INSUFFICIENT", "CONFLICTED"].includes(state.evidenceState)
    ) {
      errors.push(
        "INVESTABLE_NOW_REQUIRES_DECISIONABLE_EVIDENCE: evidence must be SUFFICIENT or PARTIAL_BUT_DECISIONABLE",
      );
    }

    if (
      ["UNKNOWN", "NOT_ASSESSABLE"].includes(state.valuationReliability)
    ) {
      errors.push(
        "INVESTABLE_NOW_REQUIRES_ASSESSABLE_VALUATION: valuation reliability cannot be UNKNOWN or NOT_ASSESSABLE",
      );
    }

    if (
      ["UNKNOWN", "NOT_ASSESSABLE", "ABOVE_REQUIRED_RETURN_PRICE"].includes(
        state.priceCondition,
      )
    ) {
      errors.push(
        "INVESTABLE_NOW_REQUIRES_PRICE_SUPPORT: price condition must be at or below an applicable return threshold",
      );
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

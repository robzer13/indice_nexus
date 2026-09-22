export const DECISION_STATES = [
  "UNKNOWN",
  "INVESTABLE_NOW",
  "WAIT_FOR_PRICE",
  "WAIT_FOR_EVIDENCE",
  "REFRESH_REQUIRED",
  "REJECT",
] as const;

export type DecisionState = (typeof DECISION_STATES)[number];

export const DECISION_BASIS_STATES = [
  "YES",
  "NO",
  "UNKNOWN",
] as const;

export type DecisionBasisState =
  (typeof DECISION_BASIS_STATES)[number];

export interface DecisionBasisAssessment {
  state: DecisionBasisState;
  rationale: string;
  evidenceIds: readonly string[];
  assumptionIds: readonly string[];
}

export interface DecisionStateInput {
  proposedDecisionState: DecisionState;
  investmentPolicySatisfied: DecisionBasisAssessment;
  certificationRequirementsSatisfied: DecisionBasisAssessment;
  businessPrepared: DecisionBasisAssessment;
  valuationAdequate: DecisionBasisAssessment;
  materialEconomicUncertainty: DecisionBasisAssessment;
  currentDecisionSupport: DecisionBasisAssessment;
  structuralEconomicsFail: DecisionBasisAssessment;
  thesisBreakingEvidence: DecisionBasisAssessment;
  rationale: string;
}

export interface DecisionStateOutput {
  proposedDecisionState: DecisionState;
  eligibleCanonicalStates: readonly Exclude<DecisionState, "UNKNOWN">[];
  basisIncomplete: boolean;
  selectionAmbiguous: boolean;
  finalizable: boolean;
  validationCodes: readonly string[];
  supportingEvidenceIds: readonly string[];
  assumptionIds: readonly string[];
  rationale: string;
}

const CANONICAL_FINAL_STATES: readonly Exclude<
  DecisionState,
  "UNKNOWN"
>[] = [
  "INVESTABLE_NOW",
  "WAIT_FOR_PRICE",
  "WAIT_FOR_EVIDENCE",
  "REFRESH_REQUIRED",
  "REJECT",
];

function assertNonBlank(value: string, code: string): void {
  if (value.trim().length === 0) throw new Error(code);
}

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values)].sort();
}

function assertAssessment(
  label: string,
  assessment: DecisionBasisAssessment,
): void {
  if (!DECISION_BASIS_STATES.includes(assessment.state)) {
    throw new Error(`VNEXT_DECISION_INVALID_${label}_STATE`);
  }

  assertNonBlank(
    assessment.rationale,
    `VNEXT_DECISION_${label}_RATIONALE_REQUIRED`,
  );

  if (
    assessment.state === "YES" &&
    assessment.evidenceIds.length === 0
  ) {
    throw new Error(
      `VNEXT_DECISION_${label}_YES_REQUIRES_EVIDENCE`,
    );
  }

  if (
    assessment.state === "UNKNOWN" &&
    assessment.evidenceIds.length === 0 &&
    assessment.assumptionIds.length === 0
  ) {
    throw new Error(
      `VNEXT_DECISION_${label}_UNKNOWN_REQUIRES_TRACEABLE_BASIS`,
    );
  }
}

function isYes(value: DecisionBasisAssessment): boolean {
  return value.state === "YES";
}

function isNo(value: DecisionBasisAssessment): boolean {
  return value.state === "NO";
}

function hasUnknown(input: DecisionStateInput): boolean {
  return [
    input.investmentPolicySatisfied,
    input.certificationRequirementsSatisfied,
    input.businessPrepared,
    input.valuationAdequate,
    input.materialEconomicUncertainty,
    input.currentDecisionSupport,
    input.structuralEconomicsFail,
    input.thesisBreakingEvidence,
  ].some((assessment) => assessment.state === "UNKNOWN");
}

/**
 * These are minimum compatibility rules from the frozen NEXT_ACTION
 * definitions. They are not a priority engine and do not replace analyst
 * judgment when more than one state remains plausible.
 */
export function deriveEligibleDecisionStates(
  input: DecisionStateInput,
): Exclude<DecisionState, "UNKNOWN">[] {
  const states: Exclude<DecisionState, "UNKNOWN">[] = [];

  if (
    isYes(input.investmentPolicySatisfied) &&
    isYes(input.certificationRequirementsSatisfied) &&
    isYes(input.currentDecisionSupport) &&
    isNo(input.materialEconomicUncertainty) &&
    isNo(input.structuralEconomicsFail) &&
    isNo(input.thesisBreakingEvidence)
  ) {
    states.push("INVESTABLE_NOW");
  }

  if (
    isYes(input.businessPrepared) &&
    isNo(input.valuationAdequate) &&
    isYes(input.currentDecisionSupport) &&
    isNo(input.materialEconomicUncertainty) &&
    isNo(input.structuralEconomicsFail) &&
    isNo(input.thesisBreakingEvidence)
  ) {
    states.push("WAIT_FOR_PRICE");
  }

  if (isYes(input.materialEconomicUncertainty)) {
    states.push("WAIT_FOR_EVIDENCE");
  }

  if (isNo(input.currentDecisionSupport)) {
    states.push("REFRESH_REQUIRED");
  }

  if (
    isYes(input.structuralEconomicsFail) ||
    isYes(input.thesisBreakingEvidence)
  ) {
    states.push("REJECT");
  }

  return states;
}

export function evaluateDecisionStateArchitecture(
  input: DecisionStateInput,
): DecisionStateOutput {
  if (!DECISION_STATES.includes(input.proposedDecisionState)) {
    throw new Error("VNEXT_DECISION_STATE_INVALID");
  }

  assertNonBlank(
    input.rationale,
    "VNEXT_DECISION_OVERALL_RATIONALE_REQUIRED",
  );

  const assessments: readonly [
    string,
    DecisionBasisAssessment,
  ][] = [
    ["INVESTMENT_POLICY", input.investmentPolicySatisfied],
    [
      "CERTIFICATION_REQUIREMENTS",
      input.certificationRequirementsSatisfied,
    ],
    ["BUSINESS_PREPARED", input.businessPrepared],
    ["VALUATION_ADEQUATE", input.valuationAdequate],
    [
      "MATERIAL_ECONOMIC_UNCERTAINTY",
      input.materialEconomicUncertainty,
    ],
    ["CURRENT_DECISION_SUPPORT", input.currentDecisionSupport],
    ["STRUCTURAL_ECONOMICS_FAIL", input.structuralEconomicsFail],
    ["THESIS_BREAKING_EVIDENCE", input.thesisBreakingEvidence],
  ];

  for (const [label, assessment] of assessments) {
    assertAssessment(label, assessment);
  }

  const eligibleCanonicalStates =
    deriveEligibleDecisionStates(input);

  const basisIncomplete = hasUnknown(input);
  const selectionAmbiguous =
    eligibleCanonicalStates.length > 1;

  const validationCodes: string[] = [];

  if (input.proposedDecisionState === "UNKNOWN") {
    validationCodes.push("DECISION_NOT_YET_FINAL");
  } else if (
    !eligibleCanonicalStates.includes(
      input.proposedDecisionState,
    )
  ) {
    validationCodes.push(
      "PROPOSED_DECISION_INCOMPATIBLE_WITH_BASIS",
    );
  }

  if (selectionAmbiguous) {
    validationCodes.push(
      "MULTIPLE_CANONICAL_STATES_REQUIRE_ANALYST_RESOLUTION",
    );
  }

  if (
    input.proposedDecisionState === "INVESTABLE_NOW" &&
    (
      isYes(input.materialEconomicUncertainty) ||
      isNo(input.currentDecisionSupport) ||
      isYes(input.structuralEconomicsFail) ||
      isYes(input.thesisBreakingEvidence)
    )
  ) {
    validationCodes.push(
      "INVESTABLE_NOW_BLOCKED_BY_MATERIAL_CONTRADICTION",
    );
  }

  if (
    input.proposedDecisionState === "WAIT_FOR_PRICE" &&
    isYes(input.materialEconomicUncertainty)
  ) {
    validationCodes.push(
      "WAIT_FOR_PRICE_BLOCKED_BY_MATERIAL_UNCERTAINTY",
    );
  }

  if (
    input.proposedDecisionState === "REFRESH_REQUIRED" &&
    isYes(input.currentDecisionSupport)
  ) {
    validationCodes.push(
      "REFRESH_REQUIRED_CONTRADICTS_CURRENT_SUPPORT",
    );
  }

  if (
    input.proposedDecisionState === "REJECT" &&
    isNo(input.structuralEconomicsFail) &&
    isNo(input.thesisBreakingEvidence)
  ) {
    validationCodes.push(
      "REJECT_REQUIRES_STRUCTURAL_OR_THESIS_BREAKING_BASIS",
    );
  }

  const finalizable =
    input.proposedDecisionState !== "UNKNOWN" &&
    validationCodes.length === 0;

  const supportingEvidenceIds = uniqueSorted(
    assessments.flatMap(([, assessment]) =>
      [...assessment.evidenceIds]
    ),
  );

  const assumptionIds = uniqueSorted(
    assessments.flatMap(([, assessment]) =>
      [...assessment.assumptionIds]
    ),
  );

  return {
    proposedDecisionState: input.proposedDecisionState,
    eligibleCanonicalStates,
    basisIncomplete,
    selectionAmbiguous,
    finalizable,
    validationCodes,
    supportingEvidenceIds,
    assumptionIds,
    rationale: input.rationale,
  };
}

export function assertDecisionStateFinalizable(
  output: DecisionStateOutput,
): void {
  if (!output.finalizable) {
    const details =
      output.validationCodes.length > 0
        ? output.validationCodes.join(",")
        : "DECISION_NOT_FINALIZABLE";

    throw new Error(
      `VNEXT_DECISION_STATE_NOT_FINALIZABLE: ${details}`,
    );
  }
}

export const TRI_STATES = ["YES", "NO", "UNKNOWN"] as const;
export type TriState = (typeof TRI_STATES)[number];

export const DIAGNOSTIC_STATES = [
  "NUMERIC_VALID",
  "NOT_ASSESSABLE",
  "NOT_AVAILABLE",
  "INVALID",
] as const;
export type DiagnosticState = (typeof DIAGNOSTIC_STATES)[number];

export const N_SELECTION_STATES = [
  "MATURE_NORMALIZATION_RETURN",
  "NO_MULTIPLE_EXPANSION_RETURN",
  "NOT_ASSESSABLE",
  "NOT_AVAILABLE",
  "INVALID",
] as const;
export type NSelectionState = (typeof N_SELECTION_STATES)[number];

export const REVERSE_DCF_STATES = [
  "COMPLETE",
  "NOT_ASSESSABLE",
  "INVALID",
] as const;
export type ReverseDcfState = (typeof REVERSE_DCF_STATES)[number];

export const VALUATION_RELIABILITY_STATES = [
  "HIGH",
  "MEDIUM",
  "LOW",
  "NOT_ASSESSABLE",
] as const;
export type ValuationReliability =
  (typeof VALUATION_RELIABILITY_STATES)[number];

export const MATH_CHECKS = [
  "MARKET_CAP_IDENTITY",
  "EV_BRIDGE",
  "FCF_YIELD",
  "PE",
  "CAGR",
  "TERMINAL_REVENUE_BRIDGE",
  "TERMINAL_METRIC_BRIDGE",
  "PV_BRIDGE",
  "EQUITY_BRIDGE",
  "VALUE_PER_SHARE",
  "IRR_EXACT",
  "MONOTONICITY_DISCOUNT_RATE",
  "MONOTONICITY_TERMINAL_GROWTH",
  "MONOTONICITY_CASH",
  "MONOTONICITY_DEBT",
] as const;
export type MathCheckName = (typeof MATH_CHECKS)[number];

export const MATH_CHECK_STATES = [
  "PASS",
  "FAIL",
  "NOT_APPLICABLE",
  "UNKNOWN",
] as const;
export type MathCheckState = (typeof MATH_CHECK_STATES)[number];

export interface TraceableAssessment {
  state: TriState;
  rationale: string;
  evidenceIds: readonly string[];
  assumptionIds: readonly string[];
}

export interface ReturnDiagnostic {
  state: DiagnosticState;
  validityReconciled: TraceableAssessment;
  calculationIds: readonly string[];
  rationale: string;
}

export interface ReverseDcfDiagnostic {
  state: ReverseDcfState;
  solvedVariables: readonly string[];
  translatedToBusinessEconomics: TraceableAssessment;
  comparedWithFrozenFundamentals: TraceableAssessment;
  calculationIds: readonly string[];
  rationale: string;
}

export interface ExpectedReturnDiagnostic {
  state: DiagnosticState;
  calculationIds: readonly string[];
  interimCashFlowsModeled: boolean;
  exactIrrUsed: TraceableAssessment;
  returnDecompositionReconciled: TraceableAssessment;
  naiveArithmeticAdditionUsed: boolean;
  fiveYearDefensible: boolean;
  fiveYearProduced: boolean;
  tenYearProduced: boolean;
  tenYearHorizonSupported: TraceableAssessment;
  rationale: string;
}

export interface ValuationMathCheck {
  check: MathCheckName;
  state: MathCheckState;
  calculationIds: readonly string[];
  rationale: string;
}

export interface ValuationDiagnosticIntegrityInput {
  valuationAssumptionIntegrityPassed: boolean;

  matureNormalization: ReturnDiagnostic;
  sameMultiple: ReturnDiagnostic;

  reverseDcf: ReverseDcfDiagnostic;
  expectedReturn: ExpectedReturnDiagnostic;

  normalizedMultipleCrossCheckUsed: boolean;
  normalizedMultipleComparabilityEstablished: TraceableAssessment;

  cyclicalNormalizationRequired: boolean;
  midCycleNormalizationEstablished: TraceableAssessment;

  marginOfSafetyInputsComplete: TraceableAssessment;
  priceLadderUsesSameFundamentals: TraceableAssessment;
  priceLadderUsesConfiguredReturnLevels: TraceableAssessment;

  valueMostSensitiveTo: readonly string[];
  valuationReliability: ValuationReliability;

  terminalValueUsed: boolean;
  mathChecks: readonly ValuationMathCheck[];

  counterEvidenceIds: readonly string[];
  rationale: string;
}

export interface ValuationDiagnosticIntegrityOutput {
  selectedNBasis: NSelectionState;
  numericOvsPermittedByNSelection: boolean;
  valuationReliability: ValuationReliability;
  validationCodes: readonly string[];
  limitations: readonly string[];
  finalizable: boolean;
  calculationIds: readonly string[];
  evidenceIds: readonly string[];
  assumptionIds: readonly string[];
  counterEvidenceIds: readonly string[];
  rationale: string;
}

function assertNonBlank(value: string, code: string): void {
  if (value.trim().length === 0) throw new Error(code);
}

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values)].sort();
}

function assertTraceableAssessment(
  label: string,
  assessment: TraceableAssessment,
): void {
  if (!TRI_STATES.includes(assessment.state)) {
    throw new Error(
      `VNEXT_VALUATION_DIAGNOSTIC_INVALID_${label}_STATE`,
    );
  }
  assertNonBlank(
    assessment.rationale,
    `VNEXT_VALUATION_DIAGNOSTIC_${label}_RATIONALE_REQUIRED`,
  );
  if (
    assessment.state === "YES" &&
    assessment.evidenceIds.length === 0
  ) {
    throw new Error(
      `VNEXT_VALUATION_DIAGNOSTIC_${label}_YES_REQUIRES_EVIDENCE`,
    );
  }
  if (
    assessment.state === "UNKNOWN" &&
    assessment.evidenceIds.length === 0 &&
    assessment.assumptionIds.length === 0
  ) {
    throw new Error(
      `VNEXT_VALUATION_DIAGNOSTIC_${label}_UNKNOWN_REQUIRES_TRACEABLE_BASIS`,
    );
  }
}

function assertReturnDiagnostic(
  label: string,
  diagnostic: ReturnDiagnostic,
): void {
  if (!DIAGNOSTIC_STATES.includes(diagnostic.state)) {
    throw new Error(
      `VNEXT_VALUATION_DIAGNOSTIC_INVALID_${label}_DIAGNOSTIC_STATE`,
    );
  }
  assertNonBlank(
    diagnostic.rationale,
    `VNEXT_VALUATION_DIAGNOSTIC_${label}_RATIONALE_REQUIRED`,
  );
  assertTraceableAssessment(
    `${label}_VALIDITY`,
    diagnostic.validityReconciled,
  );
  if (
    diagnostic.state === "NUMERIC_VALID" &&
    diagnostic.calculationIds.length === 0
  ) {
    throw new Error(
      `VNEXT_VALUATION_DIAGNOSTIC_${label}_NUMERIC_REQUIRES_CALCULATION`,
    );
  }
}

export function selectNDiagnostic(
  mature: ReturnDiagnostic,
  sameMultiple: ReturnDiagnostic,
): NSelectionState {
  if (mature.state === "INVALID") return "INVALID";

  if (
    mature.state === "NUMERIC_VALID" &&
    mature.validityReconciled.state === "YES"
  ) {
    return "MATURE_NORMALIZATION_RETURN";
  }

  if (
    mature.state === "NOT_ASSESSABLE" ||
    mature.state === "NOT_AVAILABLE"
  ) {
    if (sameMultiple.state === "INVALID") return "INVALID";
    if (
      sameMultiple.state === "NUMERIC_VALID" &&
      sameMultiple.validityReconciled.state === "YES"
    ) {
      return "NO_MULTIPLE_EXPANSION_RETURN";
    }

    if (
      mature.state === "NOT_ASSESSABLE" ||
      sameMultiple.state === "NOT_ASSESSABLE"
    ) {
      return "NOT_ASSESSABLE";
    }
    return "NOT_AVAILABLE";
  }

  return "INVALID";
}

function requiredMathChecks(
  input: ValuationDiagnosticIntegrityInput,
): MathCheckName[] {
  const required: MathCheckName[] = [
    "MARKET_CAP_IDENTITY",
    "EV_BRIDGE",
    "PV_BRIDGE",
    "EQUITY_BRIDGE",
    "VALUE_PER_SHARE",
    "MONOTONICITY_DISCOUNT_RATE",
    "MONOTONICITY_TERMINAL_GROWTH",
    "MONOTONICITY_CASH",
    "MONOTONICITY_DEBT",
  ];

  if (input.terminalValueUsed) {
    required.push(
      "TERMINAL_REVENUE_BRIDGE",
      "TERMINAL_METRIC_BRIDGE",
    );
  }

  if (
    input.expectedReturn.state === "NUMERIC_VALID" &&
    input.expectedReturn.interimCashFlowsModeled
  ) {
    required.push("IRR_EXACT");
  }

  return required;
}

function validateMathChecks(
  input: ValuationDiagnosticIntegrityInput,
): string[] {
  const validationCodes: string[] = [];
  const seen = new Map<MathCheckName, ValuationMathCheck>();

  for (const item of input.mathChecks) {
    if (!MATH_CHECKS.includes(item.check)) {
      throw new Error(
        "VNEXT_VALUATION_DIAGNOSTIC_INVALID_MATH_CHECK",
      );
    }
    if (!MATH_CHECK_STATES.includes(item.state)) {
      throw new Error(
        "VNEXT_VALUATION_DIAGNOSTIC_INVALID_MATH_CHECK_STATE",
      );
    }
    assertNonBlank(
      item.rationale,
      "VNEXT_VALUATION_DIAGNOSTIC_MATH_CHECK_RATIONALE_REQUIRED",
    );
    if (seen.has(item.check)) {
      throw new Error(
        "VNEXT_VALUATION_DIAGNOSTIC_DUPLICATE_MATH_CHECK",
      );
    }
    seen.set(item.check, item);

    if (item.state === "PASS" && item.calculationIds.length === 0) {
      throw new Error(
        "VNEXT_VALUATION_DIAGNOSTIC_PASS_MATH_CHECK_REQUIRES_CALCULATION",
      );
    }
  }

  for (const check of requiredMathChecks(input)) {
    const result = seen.get(check);
    if (!result) {
      validationCodes.push(
        `REQUIRED_MATH_CHECK_MISSING:${check}`,
      );
      continue;
    }
    if (result.state !== "PASS") {
      validationCodes.push(
        `REQUIRED_MATH_CHECK_NOT_PASS:${check}`,
      );
    }
  }

  return validationCodes;
}

export function evaluateValuationDiagnosticIntegrity(
  input: ValuationDiagnosticIntegrityInput,
): ValuationDiagnosticIntegrityOutput {
  assertNonBlank(
    input.rationale,
    "VNEXT_VALUATION_DIAGNOSTIC_INTEGRITY_RATIONALE_REQUIRED",
  );

  if (
    !VALUATION_RELIABILITY_STATES.includes(
      input.valuationReliability,
    )
  ) {
    throw new Error(
      "VNEXT_VALUATION_DIAGNOSTIC_INVALID_RELIABILITY_STATE",
    );
  }

  assertReturnDiagnostic(
    "MATURE_NORMALIZATION",
    input.matureNormalization,
  );
  assertReturnDiagnostic("SAME_MULTIPLE", input.sameMultiple);

  assertNonBlank(
    input.reverseDcf.rationale,
    "VNEXT_VALUATION_DIAGNOSTIC_REVERSE_DCF_RATIONALE_REQUIRED",
  );
  if (!REVERSE_DCF_STATES.includes(input.reverseDcf.state)) {
    throw new Error(
      "VNEXT_VALUATION_DIAGNOSTIC_INVALID_REVERSE_DCF_STATE",
    );
  }
  assertTraceableAssessment(
    "REVERSE_DCF_BUSINESS_TRANSLATION",
    input.reverseDcf.translatedToBusinessEconomics,
  );
  assertTraceableAssessment(
    "REVERSE_DCF_FROZEN_FUNDAMENTALS_COMPARISON",
    input.reverseDcf.comparedWithFrozenFundamentals,
  );

  assertNonBlank(
    input.expectedReturn.rationale,
    "VNEXT_VALUATION_DIAGNOSTIC_EXPECTED_RETURN_RATIONALE_REQUIRED",
  );
  if (!DIAGNOSTIC_STATES.includes(input.expectedReturn.state)) {
    throw new Error(
      "VNEXT_VALUATION_DIAGNOSTIC_INVALID_EXPECTED_RETURN_STATE",
    );
  }
  assertTraceableAssessment(
    "EXPECTED_RETURN_EXACT_IRR",
    input.expectedReturn.exactIrrUsed,
  );
  assertTraceableAssessment(
    "EXPECTED_RETURN_DECOMPOSITION",
    input.expectedReturn.returnDecompositionReconciled,
  );
  assertTraceableAssessment(
    "EXPECTED_RETURN_10Y_HORIZON",
    input.expectedReturn.tenYearHorizonSupported,
  );

  assertTraceableAssessment(
    "NORMALIZED_MULTIPLE_COMPARABILITY",
    input.normalizedMultipleComparabilityEstablished,
  );
  assertTraceableAssessment(
    "MID_CYCLE_NORMALIZATION",
    input.midCycleNormalizationEstablished,
  );
  assertTraceableAssessment(
    "MARGIN_OF_SAFETY_INPUTS",
    input.marginOfSafetyInputsComplete,
  );
  assertTraceableAssessment(
    "PRICE_LADDER_SAME_FUNDAMENTALS",
    input.priceLadderUsesSameFundamentals,
  );
  assertTraceableAssessment(
    "PRICE_LADDER_CONFIGURED_RETURN_LEVELS",
    input.priceLadderUsesConfiguredReturnLevels,
  );

  const validationCodes = validateMathChecks(input);
  const limitations: string[] = [];

  if (!input.valuationAssumptionIntegrityPassed) {
    validationCodes.push(
      "VALUATION_ASSUMPTION_INTEGRITY_REQUIRED",
    );
  }

  if (
    input.matureNormalization.state === "NUMERIC_VALID" &&
    input.matureNormalization.validityReconciled.state !== "YES"
  ) {
    validationCodes.push(
      "MATURE_NORMALIZATION_NUMERIC_REQUIRES_VALID_RECONCILIATION",
    );
  }

  if (input.matureNormalization.state === "INVALID") {
    validationCodes.push(
      "MATURE_NORMALIZATION_INVALID_FAIL_CLOSED",
    );
  }

  if (
    input.sameMultiple.state === "NUMERIC_VALID" &&
    input.sameMultiple.validityReconciled.state !== "YES"
  ) {
    validationCodes.push(
      "SAME_MULTIPLE_NUMERIC_REQUIRES_VALID_RECONCILIATION",
    );
  }

  if (input.sameMultiple.state === "INVALID") {
    validationCodes.push("SAME_MULTIPLE_DIAGNOSTIC_INVALID");
  }

  const selectedNBasis = selectNDiagnostic(
    input.matureNormalization,
    input.sameMultiple,
  );

  if (selectedNBasis === "INVALID") {
    validationCodes.push("N_SELECTION_INVALID_FAIL_CLOSED");
  } else if (
    selectedNBasis === "NOT_ASSESSABLE" ||
    selectedNBasis === "NOT_AVAILABLE"
  ) {
    limitations.push(`N_SELECTION_${selectedNBasis}`);
  }

  if (
    input.matureNormalization.state === "NUMERIC_VALID" &&
    selectedNBasis !== "MATURE_NORMALIZATION_RETURN"
  ) {
    validationCodes.push(
      "MATURE_NORMALIZATION_PRECEDENCE_VIOLATED",
    );
  }

  if (input.reverseDcf.state === "INVALID") {
    validationCodes.push("REVERSE_DCF_INVALID");
  } else if (input.reverseDcf.state === "NOT_ASSESSABLE") {
    limitations.push("REVERSE_DCF_NOT_ASSESSABLE");
  } else {
    if (input.reverseDcf.solvedVariables.length !== 1) {
      validationCodes.push(
        "REVERSE_DCF_MUST_SOLVE_ONE_MATERIAL_VARIABLE",
      );
    }
    if (
      input.reverseDcf.translatedToBusinessEconomics.state !== "YES"
    ) {
      validationCodes.push(
        "REVERSE_DCF_REQUIRES_BUSINESS_ECONOMICS_TRANSLATION",
      );
    }
    if (
      input.reverseDcf.comparedWithFrozenFundamentals.state !== "YES"
    ) {
      validationCodes.push(
        "REVERSE_DCF_REQUIRES_FROZEN_FUNDAMENTALS_COMPARISON",
      );
    }
    if (input.reverseDcf.calculationIds.length === 0) {
      validationCodes.push(
        "REVERSE_DCF_COMPLETE_REQUIRES_CALCULATION",
      );
    }
  }

  if (input.expectedReturn.state === "INVALID") {
    validationCodes.push("EXPECTED_RETURN_INVALID");
  } else if (
    input.expectedReturn.state === "NOT_ASSESSABLE" ||
    input.expectedReturn.state === "NOT_AVAILABLE"
  ) {
    limitations.push(
      `EXPECTED_RETURN_${input.expectedReturn.state}`,
    );
  } else {
    if (input.expectedReturn.calculationIds.length === 0) {
      validationCodes.push(
        "NUMERIC_EXPECTED_RETURN_REQUIRES_CALCULATION",
      );
    }
    if (
      input.expectedReturn.interimCashFlowsModeled &&
      input.expectedReturn.exactIrrUsed.state !== "YES"
    ) {
      validationCodes.push(
        "INTERIM_CASH_FLOWS_REQUIRE_EXACT_IRR",
      );
    }
    if (
      input.expectedReturn.returnDecompositionReconciled.state !==
      "YES"
    ) {
      validationCodes.push(
        "EXPECTED_RETURN_DECOMPOSITION_MUST_RECONCILE",
      );
    }
    if (input.expectedReturn.naiveArithmeticAdditionUsed) {
      validationCodes.push(
        "NAIVE_EXPECTED_RETURN_ADDITION_FORBIDDEN",
      );
    }
  }

  if (
    input.expectedReturn.fiveYearDefensible &&
    !input.expectedReturn.fiveYearProduced
  ) {
    validationCodes.push(
      "FIVE_YEAR_EXPECTED_RETURN_REQUIRED_WHEN_DEFENSIBLE",
    );
  }

  if (
    input.expectedReturn.tenYearProduced &&
    input.expectedReturn.tenYearHorizonSupported.state !== "YES"
  ) {
    validationCodes.push(
      "TEN_YEAR_EXPECTED_RETURN_REQUIRES_ECONOMIC_HORIZON_SUPPORT",
    );
  }

  if (
    input.normalizedMultipleCrossCheckUsed &&
    input.normalizedMultipleComparabilityEstablished.state !== "YES"
  ) {
    validationCodes.push(
      "NORMALIZED_MULTIPLE_REQUIRES_COMPARABILITY",
    );
  }

  if (
    input.cyclicalNormalizationRequired &&
    input.midCycleNormalizationEstablished.state !== "YES"
  ) {
    validationCodes.push(
      "CYCLICAL_VALUATION_REQUIRES_MID_CYCLE_NORMALIZATION",
    );
  }

  if (input.marginOfSafetyInputsComplete.state !== "YES") {
    validationCodes.push(
      "MARGIN_OF_SAFETY_INPUT_SET_INCOMPLETE",
    );
  }

  if (input.priceLadderUsesSameFundamentals.state !== "YES") {
    validationCodes.push(
      "PRICE_LADDER_MUST_USE_SAME_FUNDAMENTALS",
    );
  }

  if (
    input.priceLadderUsesConfiguredReturnLevels.state !== "YES"
  ) {
    validationCodes.push(
      "PRICE_LADDER_MUST_USE_CONFIGURED_RETURN_LEVELS",
    );
  }

  if (input.valuationReliability === "NOT_ASSESSABLE") {
    limitations.push("VALUATION_RELIABILITY_NOT_ASSESSABLE");
  } else if (
    input.valueMostSensitiveTo.length < 2 ||
    input.valueMostSensitiveTo.length > 4
  ) {
    validationCodes.push(
      "VALUE_MOST_SENSITIVE_TO_REQUIRES_TWO_TO_FOUR_VARIABLES",
    );
  }

  const calculationIds = uniqueSorted([
    ...input.matureNormalization.calculationIds,
    ...input.sameMultiple.calculationIds,
    ...input.reverseDcf.calculationIds,
    ...input.expectedReturn.calculationIds,
    ...input.mathChecks.flatMap((check) => check.calculationIds),
  ]);

  const assessments: readonly TraceableAssessment[] = [
    input.matureNormalization.validityReconciled,
    input.sameMultiple.validityReconciled,
    input.reverseDcf.translatedToBusinessEconomics,
    input.reverseDcf.comparedWithFrozenFundamentals,
    input.expectedReturn.exactIrrUsed,
    input.expectedReturn.returnDecompositionReconciled,
    input.expectedReturn.tenYearHorizonSupported,
    input.normalizedMultipleComparabilityEstablished,
    input.midCycleNormalizationEstablished,
    input.marginOfSafetyInputsComplete,
    input.priceLadderUsesSameFundamentals,
    input.priceLadderUsesConfiguredReturnLevels,
  ];

  const numericOvsPermittedByNSelection =
    (selectedNBasis === "MATURE_NORMALIZATION_RETURN" ||
      selectedNBasis === "NO_MULTIPLE_EXPANSION_RETURN") &&
    input.valuationReliability !== "NOT_ASSESSABLE" &&
    validationCodes.length === 0;

  return {
    selectedNBasis,
    numericOvsPermittedByNSelection,
    valuationReliability: input.valuationReliability,
    validationCodes: uniqueSorted(validationCodes),
    limitations: uniqueSorted(limitations),
    finalizable: validationCodes.length === 0,
    calculationIds,
    evidenceIds: uniqueSorted(
      assessments.flatMap((assessment) => [
        ...assessment.evidenceIds,
      ]),
    ),
    assumptionIds: uniqueSorted(
      assessments.flatMap((assessment) => [
        ...assessment.assumptionIds,
      ]),
    ),
    counterEvidenceIds: uniqueSorted(input.counterEvidenceIds),
    rationale: input.rationale,
  };
}

export function assertValuationDiagnosticIntegrityFinalizable(
  output: ValuationDiagnosticIntegrityOutput,
): void {
  if (!output.finalizable) {
    throw new Error(
      `VNEXT_VALUATION_DIAGNOSTIC_INTEGRITY_NOT_FINALIZABLE: ${output.validationCodes.join(",")}`,
    );
  }
}

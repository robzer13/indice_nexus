export const TRI_STATES = ["YES", "NO", "UNKNOWN"] as const;
export type TriState = (typeof TRI_STATES)[number];

export const VALUATION_CASH_FLOW_BASES = [
  "FCFF",
  "FCFE",
  "OWNER_EARNINGS",
  "SECTOR_VALID",
] as const;

export type ValuationCashFlowBasis =
  (typeof VALUATION_CASH_FLOW_BASES)[number];

export const DISCOUNT_RATE_BASES = [
  "WACC",
  "COST_OF_EQUITY",
  "SECTOR_VALID",
] as const;

export type DiscountRateBasis =
  (typeof DISCOUNT_RATE_BASES)[number];

export const OWNER_EARNINGS_STATES = [
  "POINT",
  "RANGE",
  "UNKNOWN",
  "NOT_APPLICABLE",
] as const;

export type OwnerEarningsState =
  (typeof OWNER_EARNINGS_STATES)[number];

export const ASSUMPTION_EPISTEMIC_TYPES = [
  "FACT",
  "CONSENSUS",
  "ESTIMATE",
  "ASSUMPTION",
  "UNKNOWN",
] as const;

export type AssumptionEpistemicType =
  (typeof ASSUMPTION_EPISTEMIC_TYPES)[number];

export const ASSUMPTION_SENSITIVITY = [
  "HIGH",
  "MEDIUM",
  "LOW",
  "UNKNOWN",
] as const;

export type AssumptionSensitivity =
  (typeof ASSUMPTION_SENSITIVITY)[number];

export interface TraceableAssessment {
  state: TriState;
  rationale: string;
  evidenceIds: readonly string[];
  assumptionIds: readonly string[];
}

export interface MaterialValuationAssumption {
  assumptionId: string;
  variable: string;
  valueOrRange: string;
  epistemicType: AssumptionEpistemicType;
  sourceOrRationale: string;
  sensitivity: AssumptionSensitivity;
  usedIn: readonly string[];
  evidenceIds: readonly string[];
  isCritical: boolean;
}

export interface ValuationAssumptionIntegrityInput {
  cashFlowBasis: ValuationCashFlowBasis;
  discountRateBasis: DiscountRateBasis;
  ownerEarningsState: OwnerEarningsState;

  fundamentalDriverChainEstablished: TraceableAssessment;
  freeStandingRevenueCagrUsedWithoutDriverSupport: boolean;

  runwayHorizonMechanicallyMappedToForecastYears: boolean;
  fadeCauseEstablished: TraceableAssessment;

  reinvestmentConsistencyEstablished: TraceableAssessment;
  highGrowthHighDistributionLowCapitalNeedCombinationPresent: boolean;
  highGrowthHighDistributionLowCapitalNeedSupported: TraceableAssessment;

  materialMarginChangeAssumed: boolean;
  marginDriversEstablished: TraceableAssessment;
  marginExpansionDoubleCountedInTerminalValue: boolean;

  shareCountBridgeEstablished: TraceableAssessment;
  buybacksAssumedOrExecuted: boolean;
  buybacksSupportedByFinancingBridge: TraceableAssessment;

  terminalGrowthReinvestmentReturnReconciled: TraceableAssessment;
  terminalEconomicsConsistentWithMaturity: TraceableAssessment;
  erodingMoatWithPermanentExcessReturnsWithoutFade: boolean;

  discountRateReproducibleAndCurrencyConsistent: TraceableAssessment;
  economicDiscountRateSeparatedFromInvestorRequiredReturn:
    TraceableAssessment;

  futureMnaMaterial: TraceableAssessment;
  futureMnaSupportedByTargetPoolCapitalCapacityAndReturns:
    TraceableAssessment;

  optionalityIncludedInBase: boolean;
  optionalityDoubleCountedInTerminalValue: boolean;

  scenarioAssumptionsCausal: TraceableAssessment;
  defaultScenarioProbabilitiesUsedWithoutSupport: boolean;

  materialAssumptions: readonly MaterialValuationAssumption[];
  assumptionIdsLeakedIntoFactualNarrative: readonly string[];
  unresolvedCriticalPlaceholders: readonly string[];
  unresolvedNonCriticalPlaceholders: readonly string[];

  counterEvidenceIds: readonly string[];
  rationale: string;
}

export interface ValuationAssumptionIntegrityOutput {
  validationCodes: readonly string[];
  limitations: readonly string[];
  finalizable: boolean;
  materialAssumptionIds: readonly string[];
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
      `VNEXT_VALUATION_ASSUMPTION_INVALID_${label}_STATE`,
    );
  }

  assertNonBlank(
    assessment.rationale,
    `VNEXT_VALUATION_ASSUMPTION_${label}_RATIONALE_REQUIRED`,
  );

  if (
    assessment.state === "YES" &&
    assessment.evidenceIds.length === 0
  ) {
    throw new Error(
      `VNEXT_VALUATION_ASSUMPTION_${label}_YES_REQUIRES_EVIDENCE`,
    );
  }

  if (
    assessment.state === "UNKNOWN" &&
    assessment.evidenceIds.length === 0 &&
    assessment.assumptionIds.length === 0
  ) {
    throw new Error(
      `VNEXT_VALUATION_ASSUMPTION_${label}_UNKNOWN_REQUIRES_TRACEABLE_BASIS`,
    );
  }
}

function containsPlaceholder(value: string): boolean {
  const normalized = value.trim().toUpperCase();
  return (
    normalized === "TBD" ||
    normalized === "XXX" ||
    normalized === "[SOURCE]" ||
    normalized === "N/A?" ||
    normalized.includes("€?M") ||
    normalized.includes("PLACEHOLDER")
  );
}

function validateMaterialAssumptions(
  assumptions: readonly MaterialValuationAssumption[],
): string[] {
  const validationCodes: string[] = [];
  const ids = new Set<string>();

  for (const assumption of assumptions) {
    assertNonBlank(
      assumption.assumptionId,
      "VNEXT_VALUATION_ASSUMPTION_ID_REQUIRED",
    );
    assertNonBlank(
      assumption.variable,
      "VNEXT_VALUATION_ASSUMPTION_VARIABLE_REQUIRED",
    );
    assertNonBlank(
      assumption.valueOrRange,
      "VNEXT_VALUATION_ASSUMPTION_VALUE_REQUIRED",
    );
    assertNonBlank(
      assumption.sourceOrRationale,
      "VNEXT_VALUATION_ASSUMPTION_SOURCE_OR_RATIONALE_REQUIRED",
    );

    if (ids.has(assumption.assumptionId)) {
      throw new Error(
        "VNEXT_VALUATION_ASSUMPTION_DUPLICATE_ASSUMPTION_ID",
      );
    }
    ids.add(assumption.assumptionId);

    if (
      !ASSUMPTION_EPISTEMIC_TYPES.includes(
        assumption.epistemicType,
      )
    ) {
      throw new Error(
        "VNEXT_VALUATION_ASSUMPTION_INVALID_EPISTEMIC_TYPE",
      );
    }

    if (
      !ASSUMPTION_SENSITIVITY.includes(assumption.sensitivity)
    ) {
      throw new Error(
        "VNEXT_VALUATION_ASSUMPTION_INVALID_SENSITIVITY",
      );
    }

    if (assumption.usedIn.length === 0) {
      validationCodes.push(
        "MATERIAL_ASSUMPTION_USED_IN_REQUIRED",
      );
    }

    if (
      assumption.epistemicType === "FACT" &&
      assumption.evidenceIds.length === 0
    ) {
      validationCodes.push(
        "FACT_ASSUMPTION_REQUIRES_EVIDENCE",
      );
    }

    if (
      assumption.epistemicType === "UNKNOWN" &&
      assumption.isCritical
    ) {
      validationCodes.push(
        "CRITICAL_MATERIAL_ASSUMPTION_CANNOT_REMAIN_UNKNOWN",
      );
    }

    if (
      containsPlaceholder(assumption.valueOrRange) ||
      containsPlaceholder(assumption.sourceOrRationale)
    ) {
      validationCodes.push(
        "PLACEHOLDER_IN_MATERIAL_ASSUMPTION",
      );
    }
  }

  return validationCodes;
}

export function evaluateValuationAssumptionIntegrity(
  input: ValuationAssumptionIntegrityInput,
): ValuationAssumptionIntegrityOutput {
  assertNonBlank(
    input.rationale,
    "VNEXT_VALUATION_ASSUMPTION_INTEGRITY_RATIONALE_REQUIRED",
  );

  const traceables: readonly [string, TraceableAssessment][] = [
    [
      "FUNDAMENTAL_DRIVER_CHAIN",
      input.fundamentalDriverChainEstablished,
    ],
    ["FADE_CAUSE", input.fadeCauseEstablished],
    [
      "REINVESTMENT_CONSISTENCY",
      input.reinvestmentConsistencyEstablished,
    ],
    [
      "HIGH_GROWTH_DISTRIBUTION_CAPITAL_SUPPORT",
      input.highGrowthHighDistributionLowCapitalNeedSupported,
    ],
    ["MARGIN_DRIVERS", input.marginDriversEstablished],
    ["SHARE_COUNT_BRIDGE", input.shareCountBridgeEstablished],
    [
      "BUYBACK_FINANCING_BRIDGE",
      input.buybacksSupportedByFinancingBridge,
    ],
    [
      "TERMINAL_GROWTH_REINVESTMENT_RETURN",
      input.terminalGrowthReinvestmentReturnReconciled,
    ],
    [
      "TERMINAL_MATURITY_CONSISTENCY",
      input.terminalEconomicsConsistentWithMaturity,
    ],
    [
      "DISCOUNT_RATE_REPRODUCIBILITY",
      input.discountRateReproducibleAndCurrencyConsistent,
    ],
    [
      "DISCOUNT_RATE_POLICY_SEPARATION",
      input.economicDiscountRateSeparatedFromInvestorRequiredReturn,
    ],
    ["FUTURE_MNA_MATERIAL", input.futureMnaMaterial],
    [
      "FUTURE_MNA_SUPPORT",
      input.futureMnaSupportedByTargetPoolCapitalCapacityAndReturns,
    ],
    ["SCENARIO_CAUSALITY", input.scenarioAssumptionsCausal],
  ];

  for (const [label, assessment] of traceables) {
    assertTraceableAssessment(label, assessment);
  }

  const validationCodes = validateMaterialAssumptions(
    input.materialAssumptions,
  );
  const limitations: string[] = [];

  if (
    input.cashFlowBasis === "FCFF" &&
    input.discountRateBasis !== "WACC"
  ) {
    validationCodes.push(
      "FCFF_MUST_BE_DISCOUNTED_AT_WACC",
    );
  }

  if (
    (input.cashFlowBasis === "FCFE" ||
      input.cashFlowBasis === "OWNER_EARNINGS") &&
    input.discountRateBasis !== "COST_OF_EQUITY"
  ) {
    validationCodes.push(
      "EQUITY_CASH_FLOW_MUST_BE_DISCOUNTED_AT_COST_OF_EQUITY",
    );
  }

  if (
    input.cashFlowBasis === "SECTOR_VALID" &&
    input.discountRateBasis !== "SECTOR_VALID"
  ) {
    validationCodes.push(
      "SECTOR_VALID_CASH_FLOW_REQUIRES_SECTOR_VALID_DISCOUNT_BASIS",
    );
  }

  if (
    input.cashFlowBasis === "OWNER_EARNINGS" &&
    input.ownerEarningsState === "UNKNOWN"
  ) {
    validationCodes.push(
      "OWNER_EARNINGS_UNKNOWN_CANNOT_SUPPORT_PRECISE_OWNER_EARNINGS_DCF",
    );
  }

  if (input.fundamentalDriverChainEstablished.state !== "YES") {
    validationCodes.push(
      "FUNDAMENTAL_DRIVER_CHAIN_REQUIRED",
    );
  }

  if (input.freeStandingRevenueCagrUsedWithoutDriverSupport) {
    validationCodes.push(
      "FREE_STANDING_REVENUE_CAGR_WITHOUT_DRIVER_SUPPORT_FORBIDDEN",
    );
  }

  if (input.runwayHorizonMechanicallyMappedToForecastYears) {
    validationCodes.push(
      "RUNWAY_HORIZON_CANNOT_MECHANICALLY_SET_FORECAST_YEARS",
    );
  }

  if (input.fadeCauseEstablished.state !== "YES") {
    validationCodes.push("FADE_CAUSE_REQUIRED");
  }

  if (input.reinvestmentConsistencyEstablished.state !== "YES") {
    validationCodes.push(
      "GROWTH_REINVESTMENT_MARGINAL_RETURN_MUST_RECONCILE",
    );
  }

  if (
    input.highGrowthHighDistributionLowCapitalNeedCombinationPresent &&
    input.highGrowthHighDistributionLowCapitalNeedSupported.state !==
      "YES"
  ) {
    validationCodes.push(
      "HIGH_GROWTH_HIGH_DISTRIBUTION_LOW_CAPITAL_NEEDS_SUPPORT",
    );
  }

  if (
    input.materialMarginChangeAssumed &&
    input.marginDriversEstablished.state !== "YES"
  ) {
    validationCodes.push("MARGIN_DRIVER_DECOMPOSITION_REQUIRED");
  }

  if (input.marginExpansionDoubleCountedInTerminalValue) {
    validationCodes.push(
      "MARGIN_EXPANSION_DOUBLE_COUNTED_IN_TERMINAL_VALUE",
    );
  }

  if (input.shareCountBridgeEstablished.state !== "YES") {
    validationCodes.push("FUTURE_SHARE_COUNT_BRIDGE_REQUIRED");
  }

  if (
    input.buybacksAssumedOrExecuted &&
    input.buybacksSupportedByFinancingBridge.state !== "YES"
  ) {
    validationCodes.push(
      "BUYBACK_ASSUMPTIONS_REQUIRE_FINANCING_SUPPORT",
    );
  }

  if (
    input.terminalGrowthReinvestmentReturnReconciled.state !== "YES"
  ) {
    validationCodes.push(
      "TERMINAL_GROWTH_REINVESTMENT_RETURN_MUST_RECONCILE",
    );
  }

  if (
    input.terminalEconomicsConsistentWithMaturity.state !== "YES"
  ) {
    validationCodes.push(
      "TERMINAL_ECONOMICS_MUST_BE_CONSISTENT_WITH_MATURITY",
    );
  }

  if (input.erodingMoatWithPermanentExcessReturnsWithoutFade) {
    validationCodes.push(
      "ERODING_MOAT_CANNOT_SUPPORT_PERMANENT_EXCESS_RETURNS_WITHOUT_FADE",
    );
  }

  if (
    input.discountRateReproducibleAndCurrencyConsistent.state !==
      "YES"
  ) {
    validationCodes.push(
      "DISCOUNT_RATE_MUST_BE_REPRODUCIBLE_AND_CURRENCY_CONSISTENT",
    );
  }

  if (
    input.economicDiscountRateSeparatedFromInvestorRequiredReturn
      .state !== "YES"
  ) {
    validationCodes.push(
      "ECONOMIC_DISCOUNT_RATE_MUST_BE_SEPARATE_FROM_INVESTOR_REQUIRED_RETURN",
    );
  }

  if (
    input.futureMnaMaterial.state === "YES" &&
    input.futureMnaSupportedByTargetPoolCapitalCapacityAndReturns
      .state !== "YES"
  ) {
    validationCodes.push(
      "MATERIAL_FUTURE_MNA_REQUIRES_TARGET_POOL_CAPITAL_CAPACITY_AND_RETURN_SUPPORT",
    );
  }

  if (
    input.futureMnaMaterial.state === "UNKNOWN"
  ) {
    limitations.push("FUTURE_MNA_MATERIALITY_UNKNOWN");
  }

  if (
    input.optionalityIncludedInBase &&
    input.optionalityDoubleCountedInTerminalValue
  ) {
    validationCodes.push(
      "OPTIONALITY_DOUBLE_COUNTED_BETWEEN_BASE_AND_TERMINAL",
    );
  }

  if (input.scenarioAssumptionsCausal.state !== "YES") {
    validationCodes.push(
      "SCENARIO_ASSUMPTIONS_MUST_BE_CAUSAL",
    );
  }

  if (input.defaultScenarioProbabilitiesUsedWithoutSupport) {
    validationCodes.push(
      "UNSUPPORTED_DEFAULT_SCENARIO_PROBABILITIES_FORBIDDEN",
    );
  }

  if (input.assumptionIdsLeakedIntoFactualNarrative.length > 0) {
    validationCodes.push(
      "ASSUMPTION_LEAKAGE_INTO_FACTUAL_NARRATIVE",
    );
  }

  if (input.unresolvedCriticalPlaceholders.length > 0) {
    validationCodes.push(
      "CRITICAL_PLACEHOLDER_BLOCKS_VALUATION_ASSUMPTION_INTEGRITY",
    );
  }

  if (input.unresolvedNonCriticalPlaceholders.length > 0) {
    limitations.push(
      "NON_CRITICAL_PLACEHOLDER_MUST_REMAIN_UNKNOWN_LIMITATION",
    );
  }

  const materialAssumptionIds = uniqueSorted(
    input.materialAssumptions.map(
      (assumption) => assumption.assumptionId,
    ),
  );

  const evidenceIds = uniqueSorted([
    ...traceables.flatMap(([, assessment]) => [
      ...assessment.evidenceIds,
    ]),
    ...input.materialAssumptions.flatMap(
      (assumption) => assumption.evidenceIds,
    ),
  ]);

  const assumptionIds = uniqueSorted([
    ...traceables.flatMap(([, assessment]) => [
      ...assessment.assumptionIds,
    ]),
    ...materialAssumptionIds,
  ]);

  return {
    validationCodes: uniqueSorted(validationCodes),
    limitations: uniqueSorted(limitations),
    finalizable: validationCodes.length === 0,
    materialAssumptionIds,
    evidenceIds,
    assumptionIds,
    counterEvidenceIds: uniqueSorted(input.counterEvidenceIds),
    rationale: input.rationale,
  };
}

export function assertValuationAssumptionIntegrityFinalizable(
  output: ValuationAssumptionIntegrityOutput,
): void {
  if (!output.finalizable) {
    throw new Error(
      `VNEXT_VALUATION_ASSUMPTION_INTEGRITY_NOT_FINALIZABLE: ${output.validationCodes.join(",")}`,
    );
  }
}

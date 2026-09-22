export const RETURN_MEASURES = [
  "STANDARD_ROIC",
  "ALL_IN_ROIC",
  "ROIIC",
  "ROIC_EX_GOODWILL",
  "R_AND_D_ADJUSTED_ROIC",
  "ACQUISITION_COHORT_RETURN",
  "ALTERNATIVE_ECONOMICS",
  "SECTOR_VALID_RETURN_FRAMEWORK",
] as const;

export type ReturnMeasure = (typeof RETURN_MEASURES)[number];

export const TRI_STATES = ["YES", "NO", "UNKNOWN"] as const;
export type TriState = (typeof TRI_STATES)[number];

export type ReturnInterpretability =
  | "INTERPRETABLE"
  | "INTERPRETABLE_WITH_LIMITATIONS"
  | "NOT_INTERPRETABLE"
  | "UNKNOWN";

export type ReturnMetadataState =
  | "HIGH"
  | "MEDIUM"
  | "LOW"
  | "UNKNOWN";

export type InvestedCapitalCondition =
  | "NORMAL_POSITIVE"
  | "NEAR_ZERO"
  | "NEGATIVE"
  | "UNKNOWN";

export type DeltaIcState =
  | "POSITIVE"
  | "NEAR_ZERO"
  | "NEGATIVE"
  | "UNKNOWN";

export interface TraceableAssessment {
  state: TriState;
  rationale: string;
  evidenceIds: readonly string[];
  assumptionIds: readonly string[];
}

export interface RoiicNormalizationInput {
  attempted: boolean;
  deltaIcState: DeltaIcState;
  deltaIcMaterial: TraceableAssessment;
  samePerimeter: TraceableAssessment;
  noUnexplainedMajorMna: TraceableAssessment;
  noMajorDivestiture: TraceableAssessment;
  noMajorAccountingReclassification: TraceableAssessment;
  investmentAndProfitPeriodsLinked: TraceableAssessment;
  nopatNormalized: TraceableAssessment;
  primaryWindowYears: 1 | 3 | 5 | null;
}

export interface RAndDNormalizationInput {
  adjustmentUsed: boolean;
  materialRecurringMultiPeriod: TraceableAssessment;
  expensingDistortsComparison: TraceableAssessment;
  currentRndAddedBack: boolean;
  denominatorRndAssetIncluded: boolean;
  amortizationIncluded: boolean;
}

export interface ReturnNormalizationInput {
  frameworkApplicability:
    | "INDUSTRIAL_APPLICABLE"
    | "INDUSTRIAL_NOT_APPLICABLE";
  acquisitionCapitalMaterial: TraceableAssessment;
  investedCapitalCondition: InvestedCapitalCondition;
  headlineMeasures: readonly ReturnMeasure[];
  diagnosticMeasures: readonly ReturnMeasure[];
  specialMeasures: readonly ReturnMeasure[];
  alternativeEconomicsEvidenceIds: readonly string[];
  roiic: RoiicNormalizationInput;
  rAndD: RAndDNormalizationInput;
  dataQuality: ReturnMetadataState;
  attributability: ReturnMetadataState;
  statedInterpretability: ReturnInterpretability;
  rationale: string;
}

export interface ReturnNormalizationOutput {
  frameworkApplicability:
    | "INDUSTRIAL_APPLICABLE"
    | "INDUSTRIAL_NOT_APPLICABLE";
  headlineMeasures: readonly ReturnMeasure[];
  diagnosticMeasures: readonly ReturnMeasure[];
  specialMeasures: readonly ReturnMeasure[];
  requiredMeasures: readonly ReturnMeasure[];
  roiicInterpretability: ReturnInterpretability;
  statedInterpretability: ReturnInterpretability;
  dataQuality: ReturnMetadataState;
  attributability: ReturnMetadataState;
  validationCodes: readonly string[];
  limitations: readonly string[];
  finalizable: boolean;
  supportingEvidenceIds: readonly string[];
  assumptionIds: readonly string[];
  rationale: string;
}

function assertNonBlank(value: string, code: string): void {
  if (value.trim().length === 0) throw new Error(code);
}

function uniqueSorted<T extends string>(values: readonly T[]): T[] {
  return [...new Set(values)].sort() as T[];
}

function assertTraceableAssessment(
  label: string,
  assessment: TraceableAssessment,
): void {
  if (!TRI_STATES.includes(assessment.state)) {
    throw new Error(`VNEXT_RETURN_INVALID_${label}_STATE`);
  }

  assertNonBlank(
    assessment.rationale,
    `VNEXT_RETURN_${label}_RATIONALE_REQUIRED`,
  );

  if (
    assessment.state === "YES" &&
    assessment.evidenceIds.length === 0
  ) {
    throw new Error(
      `VNEXT_RETURN_${label}_YES_REQUIRES_EVIDENCE`,
    );
  }

  if (
    assessment.state === "UNKNOWN" &&
    assessment.evidenceIds.length === 0 &&
    assessment.assumptionIds.length === 0
  ) {
    throw new Error(
      `VNEXT_RETURN_${label}_UNKNOWN_REQUIRES_TRACEABLE_BASIS`,
    );
  }
}

function assertMeasureList(
  label: string,
  values: readonly ReturnMeasure[],
): void {
  for (const value of values) {
    if (!RETURN_MEASURES.includes(value)) {
      throw new Error(`VNEXT_RETURN_INVALID_${label}_MEASURE`);
    }
  }

  if (new Set(values).size !== values.length) {
    throw new Error(`VNEXT_RETURN_DUPLICATE_${label}_MEASURE`);
  }
}

function includes(
  values: readonly ReturnMeasure[],
  measure: ReturnMeasure,
): boolean {
  return values.includes(measure);
}

function triGateState(
  assessments: readonly TraceableAssessment[],
): "PASS" | "FAIL" | "UNKNOWN" {
  if (assessments.some((assessment) => assessment.state === "NO")) {
    return "FAIL";
  }

  if (
    assessments.some((assessment) => assessment.state === "UNKNOWN")
  ) {
    return "UNKNOWN";
  }

  return "PASS";
}

/**
 * Frozen ROIIC interpretability gate.
 *
 * This deliberately does not infer "quality" from a numeric ROIIC. It only
 * checks whether the denominator/perimeter/normalization conditions permit
 * an economically interpretable marginal-return ratio.
 */
export function deriveRoiicInterpretability(
  input: RoiicNormalizationInput,
): ReturnInterpretability {
  if (!input.attempted) return "UNKNOWN";

  if (
    input.deltaIcState === "NEAR_ZERO" ||
    input.deltaIcState === "NEGATIVE"
  ) {
    return "NOT_INTERPRETABLE";
  }

  if (input.deltaIcState === "UNKNOWN") {
    return "UNKNOWN";
  }

  const gate = triGateState([
    input.deltaIcMaterial,
    input.samePerimeter,
    input.noUnexplainedMajorMna,
    input.noMajorDivestiture,
    input.noMajorAccountingReclassification,
    input.investmentAndProfitPeriodsLinked,
    input.nopatNormalized,
  ]);

  if (gate === "FAIL") return "NOT_INTERPRETABLE";
  if (gate === "UNKNOWN") return "UNKNOWN";
  return "INTERPRETABLE";
}

export function evaluateReturnNormalization(
  input: ReturnNormalizationInput,
): ReturnNormalizationOutput {
  assertNonBlank(
    input.rationale,
    "VNEXT_RETURN_NORMALIZATION_RATIONALE_REQUIRED",
  );

  assertMeasureList("HEADLINE", input.headlineMeasures);
  assertMeasureList("DIAGNOSTIC", input.diagnosticMeasures);
  assertMeasureList("SPECIAL", input.specialMeasures);

  const traceables: readonly [string, TraceableAssessment][] = [
    ["ACQUISITION_CAPITAL_MATERIAL", input.acquisitionCapitalMaterial],
    ["ROIIC_DELTA_IC_MATERIAL", input.roiic.deltaIcMaterial],
    ["ROIIC_SAME_PERIMETER", input.roiic.samePerimeter],
    ["ROIIC_NO_UNEXPLAINED_MAJOR_MNA", input.roiic.noUnexplainedMajorMna],
    ["ROIIC_NO_MAJOR_DIVESTITURE", input.roiic.noMajorDivestiture],
    [
      "ROIIC_NO_MAJOR_ACCOUNTING_RECLASSIFICATION",
      input.roiic.noMajorAccountingReclassification,
    ],
    [
      "ROIIC_INVESTMENT_AND_PROFIT_PERIODS_LINKED",
      input.roiic.investmentAndProfitPeriodsLinked,
    ],
    ["ROIIC_NOPAT_NORMALIZED", input.roiic.nopatNormalized],
    [
      "RND_MATERIAL_RECURRING_MULTI_PERIOD",
      input.rAndD.materialRecurringMultiPeriod,
    ],
    [
      "RND_EXPENSING_DISTORTS_COMPARISON",
      input.rAndD.expensingDistortsComparison,
    ],
  ];

  for (const [label, assessment] of traceables) {
    assertTraceableAssessment(label, assessment);
  }

  const validationCodes: string[] = [];
  const limitations: string[] = [];
  const requiredMeasures: ReturnMeasure[] = [];

  const headlineSet = new Set(input.headlineMeasures);
  const diagnosticSet = new Set(input.diagnosticMeasures);
  const specialSet = new Set(input.specialMeasures);

  for (const measure of headlineSet) {
    if (diagnosticSet.has(measure) || specialSet.has(measure)) {
      validationCodes.push(
        "RETURN_MEASURE_ROLE_OVERLAP_FORBIDDEN",
      );
      break;
    }
  }

  for (const measure of diagnosticSet) {
    if (specialSet.has(measure)) {
      validationCodes.push(
        "RETURN_MEASURE_ROLE_OVERLAP_FORBIDDEN",
      );
      break;
    }
  }

  if (
    includes(input.headlineMeasures, "ROIC_EX_GOODWILL")
  ) {
    validationCodes.push(
      "ROIC_EX_GOODWILL_DIAGNOSTIC_ONLY",
    );
  }

  if (
    includes(input.headlineMeasures, "R_AND_D_ADJUSTED_ROIC")
  ) {
    validationCodes.push(
      "R_AND_D_ADJUSTED_ROIC_DIAGNOSTIC_ONLY",
    );
  }

  if (
    input.frameworkApplicability === "INDUSTRIAL_NOT_APPLICABLE"
  ) {
    requiredMeasures.push("SECTOR_VALID_RETURN_FRAMEWORK");

    if (
      !includes(
        input.headlineMeasures,
        "SECTOR_VALID_RETURN_FRAMEWORK",
      )
    ) {
      validationCodes.push(
        "SECTOR_VALID_RETURN_FRAMEWORK_REQUIRED",
      );
    }

    if (
      includes(input.headlineMeasures, "STANDARD_ROIC") ||
      includes(input.headlineMeasures, "ALL_IN_ROIC")
    ) {
      validationCodes.push(
        "INDUSTRIAL_HEADLINE_RETURN_NOT_APPLICABLE",
      );
    }
  } else if (
    input.investedCapitalCondition === "NORMAL_POSITIVE"
  ) {
    requiredMeasures.push("STANDARD_ROIC");

    if (!includes(input.headlineMeasures, "STANDARD_ROIC")) {
      validationCodes.push("STANDARD_ROIC_HEADLINE_REQUIRED");
    }
  }

  if (input.acquisitionCapitalMaterial.state === "YES") {
    requiredMeasures.push(
      "ALL_IN_ROIC",
      "ACQUISITION_COHORT_RETURN",
    );

    if (!includes(input.headlineMeasures, "ALL_IN_ROIC")) {
      validationCodes.push(
        "ALL_IN_ROIC_REQUIRED_FOR_MATERIAL_ACQUISITION_CAPITAL",
      );
    }

    if (
      !includes(
        input.specialMeasures,
        "ACQUISITION_COHORT_RETURN",
      )
    ) {
      validationCodes.push(
        "ACQUISITION_COHORT_RETURN_REQUIRED",
      );
    }

    if (
      input.headlineMeasures.length === 1 &&
      includes(input.headlineMeasures, "ROIC_EX_GOODWILL")
    ) {
      validationCodes.push(
        "EX_GOODWILL_CANNOT_BE_SOLE_HEADLINE_FOR_ACQUIRER",
      );
    }
  }

  if (
    input.investedCapitalCondition === "NEAR_ZERO" ||
    input.investedCapitalCondition === "NEGATIVE"
  ) {
    if (
      includes(input.headlineMeasures, "STANDARD_ROIC") ||
      includes(input.headlineMeasures, "ALL_IN_ROIC")
    ) {
      validationCodes.push(
        "EXPLOSIVE_DENOMINATOR_ROIC_NOT_HEADLINE_QUALITY_EVIDENCE",
      );
    }

    if (input.alternativeEconomicsEvidenceIds.length === 0) {
      validationCodes.push(
        "ALTERNATIVE_ECONOMICS_REQUIRED_FOR_UNINTERPRETABLE_ROIC",
      );
    }

    limitations.push("ROIC_NOT_ECONOMICALLY_INTERPRETABLE");
  }

  const roiicInterpretability =
    deriveRoiicInterpretability(input.roiic);

  if (input.roiic.attempted) {
    requiredMeasures.push("ROIIC");

    if (input.roiic.primaryWindowYears === 1) {
      validationCodes.push("ROIIC_1Y_DIAGNOSTIC_ONLY");
    }

    if (input.roiic.primaryWindowYears === null) {
      validationCodes.push("ROIIC_WINDOW_REQUIRED_WHEN_ATTEMPTED");
    }

    if (
      roiicInterpretability !== "INTERPRETABLE" &&
      includes(input.headlineMeasures, "ROIIC")
    ) {
      validationCodes.push(
        "ROIIC_NOT_INTERPRETABLE_CANNOT_BE_HEADLINE",
      );
    }

    if (roiicInterpretability === "UNKNOWN") {
      limitations.push("ROIIC_INTERPRETABILITY_UNKNOWN");
    }

    if (roiicInterpretability === "NOT_INTERPRETABLE") {
      limitations.push("ROIIC_NOT_ECONOMICALLY_INTERPRETABLE");
    }
  }

  if (input.rAndD.adjustmentUsed) {
    requiredMeasures.push("R_AND_D_ADJUSTED_ROIC");

    if (
      input.rAndD.materialRecurringMultiPeriod.state !== "YES" ||
      input.rAndD.expensingDistortsComparison.state !== "YES"
    ) {
      validationCodes.push(
        "RND_ADJUSTMENT_REQUIRES_MATERIAL_RECURRING_MULTI_PERIOD_DISTORTION",
      );
    }

    if (
      !includes(
        input.diagnosticMeasures,
        "R_AND_D_ADJUSTED_ROIC",
      )
    ) {
      validationCodes.push(
        "RND_ADJUSTED_ROIC_MUST_BE_DIAGNOSTIC",
      );
    }

    if (
      input.rAndD.currentRndAddedBack &&
      (
        !input.rAndD.denominatorRndAssetIncluded ||
        !input.rAndD.amortizationIncluded
      )
    ) {
      validationCodes.push(
        "RND_ADD_BACK_REQUIRES_ASSET_AND_AMORTIZATION_SYMMETRY",
      );
    }
  }

  if (input.dataQuality === "UNKNOWN") {
    limitations.push("DATA_QUALITY_UNKNOWN");
  }

  if (input.attributability === "UNKNOWN") {
    limitations.push("ATTRIBUTABILITY_UNKNOWN");
  }

  if (input.statedInterpretability === "UNKNOWN") {
    limitations.push("RETURN_INTERPRETABILITY_UNKNOWN");
  }

  const allEvidenceIds = uniqueSorted([
    ...traceables.flatMap(([, assessment]) =>
      [...assessment.evidenceIds]
    ),
    ...input.alternativeEconomicsEvidenceIds,
  ]);

  const assumptionIds = uniqueSorted(
    traceables.flatMap(([, assessment]) =>
      [...assessment.assumptionIds]
    ),
  );

  return {
    frameworkApplicability: input.frameworkApplicability,
    headlineMeasures: uniqueSorted(input.headlineMeasures),
    diagnosticMeasures: uniqueSorted(input.diagnosticMeasures),
    specialMeasures: uniqueSorted(input.specialMeasures),
    requiredMeasures: uniqueSorted(requiredMeasures),
    roiicInterpretability,
    statedInterpretability: input.statedInterpretability,
    dataQuality: input.dataQuality,
    attributability: input.attributability,
    validationCodes: uniqueSorted(validationCodes),
    limitations: uniqueSorted(limitations),
    finalizable: validationCodes.length === 0,
    supportingEvidenceIds: allEvidenceIds,
    assumptionIds,
    rationale: input.rationale,
  };
}

export function assertReturnNormalizationFinalizable(
  output: ReturnNormalizationOutput,
): void {
  if (!output.finalizable) {
    throw new Error(
      `VNEXT_RETURN_NORMALIZATION_NOT_FINALIZABLE: ${output.validationCodes.join(",")}`,
    );
  }
}

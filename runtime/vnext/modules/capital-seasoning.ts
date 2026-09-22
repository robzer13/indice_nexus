export const CAPITAL_SEASONING_STATES = [
  "COMMITTED",
  "DEPLOYED",
  "IN_SERVICE",
  "RAMPING",
  "STABILIZING",
  "SEASONED",
  "UNKNOWN",
] as const;

export type CapitalSeasoningState =
  (typeof CAPITAL_SEASONING_STATES)[number];

export const CAPITAL_TYPES = [
  "CAPEX",
  "WORKING_CAPITAL",
  "R_AND_D",
  "SALESFORCE",
  "IMPLEMENTATION",
  "REGULATORY_CAPITAL",
  "ACQUISITION_CAPITAL",
  "OTHER",
] as const;

export type CapitalType = (typeof CAPITAL_TYPES)[number];

export const TRI_STATES = ["YES", "NO", "UNKNOWN"] as const;
export type TriState = (typeof TRI_STATES)[number];

export type InvestmentLag =
  | "LOW"
  | "MEDIUM"
  | "HIGH"
  | "UNKNOWN";

export type ReturnEvidenceUse =
  | "MATURE_RETURN_EVIDENCE_ALLOWED"
  | "UNSEASONED_DO_NOT_JUDGE"
  | "UNKNOWN";

export type PortfolioSeasoningState =
  | "FULLY_SEASONED"
  | "MIXED_SEASONING"
  | "UNSEASONED"
  | "UNKNOWN"
  | "NOT_APPLICABLE";

export interface TraceableAssessment {
  state: TriState;
  rationale: string;
  evidenceIds: readonly string[];
  assumptionIds: readonly string[];
}

export interface CapitalCohortInput {
  cohortId: string;
  capitalType: CapitalType;
  materiality: TraceableAssessment;
  committed: TraceableAssessment;
  deployed: TraceableAssessment;
  inService: TraceableAssessment;
  utilizationEstablished: TraceableAssessment;
  stabilizationEstablished: TraceableAssessment;
  matureReturnEvidenceEstablished: TraceableAssessment;
  investmentLag: InvestmentLag;
  proposedSeasoningState: CapitalSeasoningState;
  returnEvidenceIds: readonly string[];
  counterEvidenceIds: readonly string[];
  rationale: string;
}

export interface CapitalCohortOutput {
  cohortId: string;
  capitalType: CapitalType;
  materiality: TriState;
  investmentLag: InvestmentLag;
  seasoningState: CapitalSeasoningState;
  returnEvidenceUse: ReturnEvidenceUse;
  validationCodes: readonly string[];
  evidenceIds: readonly string[];
  counterEvidenceIds: readonly string[];
  assumptionIds: readonly string[];
  rationale: string;
}

export interface CapitalSeasoningInput {
  applicable: boolean;
  cohorts: readonly CapitalCohortInput[];
  rationale: string;
}

export interface CapitalSeasoningOutput {
  applicable: boolean;
  cohorts: readonly CapitalCohortOutput[];
  portfolioSeasoningState: PortfolioSeasoningState;
  materialSeasonedCohortIds: readonly string[];
  materialUnseasonedCohortIds: readonly string[];
  materialUnknownCohortIds: readonly string[];
  validationCodes: readonly string[];
  finalizable: boolean;
  evidenceIds: readonly string[];
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
    throw new Error(`VNEXT_CAPITAL_SEASONING_INVALID_${label}_STATE`);
  }

  assertNonBlank(
    assessment.rationale,
    `VNEXT_CAPITAL_SEASONING_${label}_RATIONALE_REQUIRED`,
  );

  if (
    assessment.state === "YES" &&
    assessment.evidenceIds.length === 0
  ) {
    throw new Error(
      `VNEXT_CAPITAL_SEASONING_${label}_YES_REQUIRES_EVIDENCE`,
    );
  }

  if (
    assessment.state === "UNKNOWN" &&
    assessment.evidenceIds.length === 0 &&
    assessment.assumptionIds.length === 0
  ) {
    throw new Error(
      `VNEXT_CAPITAL_SEASONING_${label}_UNKNOWN_REQUIRES_TRACEABLE_BASIS`,
    );
  }
}

function allYes(
  assessments: readonly TraceableAssessment[],
): boolean {
  return assessments.every((assessment) => assessment.state === "YES");
}

function anyUnknown(
  assessments: readonly TraceableAssessment[],
): boolean {
  return assessments.some(
    (assessment) => assessment.state === "UNKNOWN",
  );
}

function anyNo(
  assessments: readonly TraceableAssessment[],
): boolean {
  return assessments.some((assessment) => assessment.state === "NO");
}

/**
 * Capital seasoning is evidence/event based. No elapsed-time threshold is
 * encoded here. The function only identifies the most advanced state that
 * is supported by the supplied traceable operating evidence.
 */
export function deriveSupportedSeasoningState(
  cohort: CapitalCohortInput,
): CapitalSeasoningState {
  if (
    allYes([
      cohort.committed,
      cohort.deployed,
      cohort.inService,
      cohort.utilizationEstablished,
      cohort.stabilizationEstablished,
      cohort.matureReturnEvidenceEstablished,
    ])
  ) {
    return "SEASONED";
  }

  if (
    allYes([
      cohort.committed,
      cohort.deployed,
      cohort.inService,
      cohort.utilizationEstablished,
      cohort.stabilizationEstablished,
    ]) &&
    cohort.matureReturnEvidenceEstablished.state === "NO"
  ) {
    return "STABILIZING";
  }

  if (
    allYes([
      cohort.committed,
      cohort.deployed,
      cohort.inService,
      cohort.utilizationEstablished,
    ]) &&
    cohort.stabilizationEstablished.state === "NO"
  ) {
    return "RAMPING";
  }

  if (
    allYes([
      cohort.committed,
      cohort.deployed,
      cohort.inService,
    ]) &&
    cohort.utilizationEstablished.state === "NO"
  ) {
    return "IN_SERVICE";
  }

  if (
    allYes([cohort.committed, cohort.deployed]) &&
    cohort.inService.state === "NO"
  ) {
    return "DEPLOYED";
  }

  if (
    cohort.committed.state === "YES" &&
    cohort.deployed.state === "NO"
  ) {
    return "COMMITTED";
  }

  return "UNKNOWN";
}

export function deriveReturnEvidenceUse(
  seasoningState: CapitalSeasoningState,
): ReturnEvidenceUse {
  if (seasoningState === "SEASONED") {
    return "MATURE_RETURN_EVIDENCE_ALLOWED";
  }

  if (seasoningState === "UNKNOWN") {
    return "UNKNOWN";
  }

  return "UNSEASONED_DO_NOT_JUDGE";
}

function evaluateCohort(
  cohort: CapitalCohortInput,
): CapitalCohortOutput {
  assertNonBlank(
    cohort.cohortId,
    "VNEXT_CAPITAL_SEASONING_COHORT_ID_REQUIRED",
  );
  assertNonBlank(
    cohort.rationale,
    "VNEXT_CAPITAL_SEASONING_COHORT_RATIONALE_REQUIRED",
  );

  if (!CAPITAL_TYPES.includes(cohort.capitalType)) {
    throw new Error(
      "VNEXT_CAPITAL_SEASONING_INVALID_CAPITAL_TYPE",
    );
  }

  if (
    !CAPITAL_SEASONING_STATES.includes(
      cohort.proposedSeasoningState,
    )
  ) {
    throw new Error(
      "VNEXT_CAPITAL_SEASONING_INVALID_PROPOSED_STATE",
    );
  }

  const assessments: readonly [string, TraceableAssessment][] = [
    ["MATERIALITY", cohort.materiality],
    ["COMMITTED", cohort.committed],
    ["DEPLOYED", cohort.deployed],
    ["IN_SERVICE", cohort.inService],
    ["UTILIZATION", cohort.utilizationEstablished],
    ["STABILIZATION", cohort.stabilizationEstablished],
    [
      "MATURE_RETURN_EVIDENCE",
      cohort.matureReturnEvidenceEstablished,
    ],
  ];

  for (const [label, assessment] of assessments) {
    assertTraceableAssessment(label, assessment);
  }

  const supportedState = deriveSupportedSeasoningState(cohort);
  const validationCodes: string[] = [];

  if (
    cohort.proposedSeasoningState !== supportedState &&
    cohort.proposedSeasoningState !== "UNKNOWN"
  ) {
    validationCodes.push(
      "PROPOSED_SEASONING_STATE_NOT_SUPPORTED_BY_EVIDENCE",
    );
  }

  if (
    cohort.proposedSeasoningState === "SEASONED" &&
    cohort.matureReturnEvidenceEstablished.state !== "YES"
  ) {
    validationCodes.push(
      "SEASONED_REQUIRES_MATURE_RETURN_EVIDENCE",
    );
  }

  const sequence = [
    cohort.committed,
    cohort.deployed,
    cohort.inService,
    cohort.utilizationEstablished,
    cohort.stabilizationEstablished,
    cohort.matureReturnEvidenceEstablished,
  ];

  let seenNo = false;
  for (const assessment of sequence) {
    if (assessment.state === "NO") seenNo = true;
    if (seenNo && assessment.state === "YES") {
      validationCodes.push(
        "CAPITAL_SEASONING_SEQUENCE_CONFLICT",
      );
      break;
    }
  }

  if (
    anyUnknown(sequence) &&
    cohort.proposedSeasoningState === "SEASONED"
  ) {
    validationCodes.push(
      "UNKNOWN_CANNOT_BE_COERCED_TO_SEASONED",
    );
  }

  if (
    anyNo(sequence) &&
    cohort.proposedSeasoningState === "SEASONED"
  ) {
    validationCodes.push(
      "NEGATIVE_STAGE_EVIDENCE_BLOCKS_SEASONED",
    );
  }

  const evidenceIds = uniqueSorted([
    ...assessments.flatMap(([, assessment]) => [
      ...assessment.evidenceIds,
    ]),
    ...cohort.returnEvidenceIds,
  ]);

  const assumptionIds = uniqueSorted(
    assessments.flatMap(([, assessment]) => [
      ...assessment.assumptionIds,
    ]),
  );

  return {
    cohortId: cohort.cohortId,
    capitalType: cohort.capitalType,
    materiality: cohort.materiality.state,
    investmentLag: cohort.investmentLag,
    seasoningState:
      cohort.proposedSeasoningState === "UNKNOWN"
        ? supportedState
        : cohort.proposedSeasoningState,
    returnEvidenceUse: deriveReturnEvidenceUse(
      cohort.proposedSeasoningState === "UNKNOWN"
        ? supportedState
        : cohort.proposedSeasoningState,
    ),
    validationCodes: uniqueSorted(validationCodes),
    evidenceIds,
    counterEvidenceIds: uniqueSorted(
      cohort.counterEvidenceIds,
    ),
    assumptionIds,
    rationale: cohort.rationale,
  };
}

export function evaluateCapitalSeasoning(
  input: CapitalSeasoningInput,
): CapitalSeasoningOutput {
  assertNonBlank(
    input.rationale,
    "VNEXT_CAPITAL_SEASONING_RATIONALE_REQUIRED",
  );

  if (!input.applicable) {
    if (input.cohorts.length > 0) {
      throw new Error(
        "VNEXT_CAPITAL_SEASONING_NOT_APPLICABLE_WITH_COHORTS",
      );
    }

    return {
      applicable: false,
      cohorts: [],
      portfolioSeasoningState: "NOT_APPLICABLE",
      materialSeasonedCohortIds: [],
      materialUnseasonedCohortIds: [],
      materialUnknownCohortIds: [],
      validationCodes: [],
      finalizable: true,
      evidenceIds: [],
      assumptionIds: [],
      rationale: input.rationale,
    };
  }

  if (input.cohorts.length === 0) {
    throw new Error(
      "VNEXT_CAPITAL_SEASONING_APPLICABLE_REQUIRES_COHORTS",
    );
  }

  const ids = input.cohorts.map((cohort) => cohort.cohortId);
  if (new Set(ids).size !== ids.length) {
    throw new Error(
      "VNEXT_CAPITAL_SEASONING_DUPLICATE_COHORT_ID",
    );
  }

  const cohorts = input.cohorts.map(evaluateCohort);

  const material = cohorts.filter(
    (cohort) => cohort.materiality === "YES",
  );
  const materialUnknownMateriality = cohorts.filter(
    (cohort) => cohort.materiality === "UNKNOWN",
  );

  const materialSeasonedCohortIds = material
    .filter((cohort) => cohort.seasoningState === "SEASONED")
    .map((cohort) => cohort.cohortId);

  const materialUnseasonedCohortIds = material
    .filter(
      (cohort) =>
        cohort.seasoningState !== "SEASONED" &&
        cohort.seasoningState !== "UNKNOWN",
    )
    .map((cohort) => cohort.cohortId);

  const materialUnknownCohortIds = [
    ...material
      .filter((cohort) => cohort.seasoningState === "UNKNOWN")
      .map((cohort) => cohort.cohortId),
    ...materialUnknownMateriality.map((cohort) => cohort.cohortId),
  ];

  let portfolioSeasoningState: PortfolioSeasoningState;

  if (materialUnknownCohortIds.length > 0) {
    portfolioSeasoningState = "UNKNOWN";
  } else if (material.length === 0) {
    portfolioSeasoningState = "NOT_APPLICABLE";
  } else if (
    materialSeasonedCohortIds.length === material.length
  ) {
    portfolioSeasoningState = "FULLY_SEASONED";
  } else if (materialSeasonedCohortIds.length === 0) {
    portfolioSeasoningState = "UNSEASONED";
  } else {
    portfolioSeasoningState = "MIXED_SEASONING";
  }

  const validationCodes = uniqueSorted(
    cohorts.flatMap((cohort) => [
      ...cohort.validationCodes,
    ]),
  );

  return {
    applicable: true,
    cohorts,
    portfolioSeasoningState,
    materialSeasonedCohortIds:
      uniqueSorted(materialSeasonedCohortIds),
    materialUnseasonedCohortIds:
      uniqueSorted(materialUnseasonedCohortIds),
    materialUnknownCohortIds:
      uniqueSorted(materialUnknownCohortIds),
    validationCodes,
    finalizable: validationCodes.length === 0,
    evidenceIds: uniqueSorted(
      cohorts.flatMap((cohort) => [...cohort.evidenceIds]),
    ),
    assumptionIds: uniqueSorted(
      cohorts.flatMap((cohort) => [...cohort.assumptionIds]),
    ),
    rationale: input.rationale,
  };
}

export function assertCapitalSeasoningFinalizable(
  output: CapitalSeasoningOutput,
): void {
  if (!output.finalizable) {
    throw new Error(
      `VNEXT_CAPITAL_SEASONING_NOT_FINALIZABLE: ${output.validationCodes.join(",")}`,
    );
  }
}

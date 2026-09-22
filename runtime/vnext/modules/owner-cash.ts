export const TRI_STATES = ["YES", "NO", "UNKNOWN"] as const;
export type TriState = (typeof TRI_STATES)[number];

export type OwnerEarningsForm =
  | "POINT"
  | "RANGE"
  | "UNKNOWN"
  | "NOT_APPLICABLE";

export type CashFramework =
  | "INDUSTRIAL_FCF_OWNER_EARNINGS"
  | "SECTOR_DISTRIBUTABLE_CAPITAL";

export const MAINTENANCE_COMPONENTS = [
  "PHYSICAL_CAPEX",
  "INTANGIBLE_INVESTMENT",
  "WORKING_CAPITAL",
  "LEASE_CASH",
  "RECURRING_RESTRUCTURING",
  "ECONOMIC_CASH_TAXES",
  "OTHER",
] as const;

export type MaintenanceComponent =
  (typeof MAINTENANCE_COMPONENTS)[number];

export interface TraceableAssessment {
  state: TriState;
  rationale: string;
  evidenceIds: readonly string[];
  assumptionIds: readonly string[];
}

export interface MaintenanceComponentAssessment {
  component: MaintenanceComponent;
  materiality: TraceableAssessment;
  treatmentResolved: TraceableAssessment;
}

export interface OwnerCashInput {
  cashFramework: CashFramework;
  reportedFcfPresent: boolean;
  reportedFcfSeparatedFromStandardizedFcf: TraceableAssessment;
  standardizedFcfPresent: boolean;
  standardizedFcfUsesCfoMinusCashCapex: TraceableAssessment;
  ownerEarningsSeparatedFromStandardizedFcf: TraceableAssessment;
  ownerEarningsForm: OwnerEarningsForm;
  maintenanceComponents: readonly MaintenanceComponentAssessment[];
  maintenanceDoubleCountedAgainstTotalCapexFcf: boolean;
  workingCapitalDoubleCountedOutsideCfo: boolean;
  temporaryWorkingCapitalBenefit: TraceableAssessment;
  temporaryWorkingCapitalTreatedAsStructuralOwnerCash: boolean;
  deferredMaintenanceMaterial: TraceableAssessment;
  deferredMaintenanceReflectedInOwnerCash: TraceableAssessment;
  recurringAdjustmentsMaterial: TraceableAssessment;
  recurringAdjustmentsRetainedAsEconomicCost: TraceableAssessment;
  sbcMaterial: TraceableAssessment;
  sbcEconomicCostRecognized: TraceableAssessment;
  sbcPenaltyCount: 0 | 1 | 2;
  acquisitionCapitalMaterial: TraceableAssessment;
  acquisitionsSubtractedFromStandardizedFcf: boolean;
  organicCashSeparatedFromAcquisitionDeployment: TraceableAssessment;
  perShareCashProduced: boolean;
  perShareUsesEconomicDilutedShares: TraceableAssessment;
  sectorDistributableCapitalFrameworkEstablished: TraceableAssessment;
  counterEvidenceIds: readonly string[];
  rationale: string;
}

export interface OwnerCashOutput {
  cashFramework: CashFramework;
  ownerEarningsForm: OwnerEarningsForm;
  materialMaintenanceComponents: readonly MaintenanceComponent[];
  unresolvedMaterialMaintenanceComponents: readonly MaintenanceComponent[];
  unknownMaintenanceComponents: readonly MaintenanceComponent[];
  validationCodes: readonly string[];
  limitations: readonly string[];
  finalizable: boolean;
  evidenceIds: readonly string[];
  counterEvidenceIds: readonly string[];
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
    throw new Error(`VNEXT_OWNER_CASH_INVALID_${label}_STATE`);
  }

  assertNonBlank(
    assessment.rationale,
    `VNEXT_OWNER_CASH_${label}_RATIONALE_REQUIRED`,
  );

  if (
    assessment.state === "YES" &&
    assessment.evidenceIds.length === 0
  ) {
    throw new Error(
      `VNEXT_OWNER_CASH_${label}_YES_REQUIRES_EVIDENCE`,
    );
  }

  if (
    assessment.state === "UNKNOWN" &&
    assessment.evidenceIds.length === 0 &&
    assessment.assumptionIds.length === 0
  ) {
    throw new Error(
      `VNEXT_OWNER_CASH_${label}_UNKNOWN_REQUIRES_TRACEABLE_BASIS`,
    );
  }
}

function validateMaintenanceComponents(
  components: readonly MaintenanceComponentAssessment[],
): void {
  const seen = new Set<string>();

  for (const item of components) {
    if (!MAINTENANCE_COMPONENTS.includes(item.component)) {
      throw new Error(
        "VNEXT_OWNER_CASH_INVALID_MAINTENANCE_COMPONENT",
      );
    }

    if (seen.has(item.component)) {
      throw new Error(
        "VNEXT_OWNER_CASH_DUPLICATE_MAINTENANCE_COMPONENT",
      );
    }
    seen.add(item.component);

    assertTraceableAssessment(
      `MAINTENANCE_${item.component}_MATERIALITY`,
      item.materiality,
    );
    assertTraceableAssessment(
      `MAINTENANCE_${item.component}_TREATMENT`,
      item.treatmentResolved,
    );
  }
}

export function evaluateOwnerCash(
  input: OwnerCashInput,
): OwnerCashOutput {
  assertNonBlank(
    input.rationale,
    "VNEXT_OWNER_CASH_RATIONALE_REQUIRED",
  );

  const traceables: readonly [string, TraceableAssessment][] = [
    [
      "REPORTED_FCF_SEPARATION",
      input.reportedFcfSeparatedFromStandardizedFcf,
    ],
    [
      "STANDARDIZED_FCF_FORMULA",
      input.standardizedFcfUsesCfoMinusCashCapex,
    ],
    [
      "OWNER_EARNINGS_SEPARATION",
      input.ownerEarningsSeparatedFromStandardizedFcf,
    ],
    [
      "TEMPORARY_WORKING_CAPITAL_BENEFIT",
      input.temporaryWorkingCapitalBenefit,
    ],
    [
      "DEFERRED_MAINTENANCE_MATERIAL",
      input.deferredMaintenanceMaterial,
    ],
    [
      "DEFERRED_MAINTENANCE_REFLECTED",
      input.deferredMaintenanceReflectedInOwnerCash,
    ],
    [
      "RECURRING_ADJUSTMENTS_MATERIAL",
      input.recurringAdjustmentsMaterial,
    ],
    [
      "RECURRING_ADJUSTMENTS_ECONOMIC_COST",
      input.recurringAdjustmentsRetainedAsEconomicCost,
    ],
    ["SBC_MATERIAL", input.sbcMaterial],
    ["SBC_ECONOMIC_COST_RECOGNIZED", input.sbcEconomicCostRecognized],
    [
      "ACQUISITION_CAPITAL_MATERIAL",
      input.acquisitionCapitalMaterial,
    ],
    [
      "ORGANIC_ACQUISITION_CASH_SEPARATION",
      input.organicCashSeparatedFromAcquisitionDeployment,
    ],
    [
      "PER_SHARE_ECONOMIC_DILUTED_SHARES",
      input.perShareUsesEconomicDilutedShares,
    ],
    [
      "SECTOR_DISTRIBUTABLE_CAPITAL_FRAMEWORK",
      input.sectorDistributableCapitalFrameworkEstablished,
    ],
  ];

  for (const [label, assessment] of traceables) {
    assertTraceableAssessment(label, assessment);
  }
  validateMaintenanceComponents(input.maintenanceComponents);

  const validationCodes: string[] = [];
  const limitations: string[] = [];

  const materialMaintenanceComponents = input.maintenanceComponents
    .filter((item) => item.materiality.state === "YES")
    .map((item) => item.component);

  const unresolvedMaterialMaintenanceComponents =
    input.maintenanceComponents
      .filter(
        (item) =>
          item.materiality.state === "YES" &&
          item.treatmentResolved.state !== "YES",
      )
      .map((item) => item.component);

  const unknownMaintenanceComponents = input.maintenanceComponents
    .filter(
      (item) =>
        item.materiality.state === "UNKNOWN" ||
        item.treatmentResolved.state === "UNKNOWN",
    )
    .map((item) => item.component);

  if (input.cashFramework === "INDUSTRIAL_FCF_OWNER_EARNINGS") {
    if (!input.standardizedFcfPresent) {
      validationCodes.push(
        "STANDARDIZED_FCF_REQUIRED_FOR_INDUSTRIAL_FRAMEWORK",
      );
    }

    if (
      input.standardizedFcfUsesCfoMinusCashCapex.state !== "YES"
    ) {
      validationCodes.push(
        "STANDARDIZED_FCF_FORMULA_NOT_ESTABLISHED",
      );
    }

    if (input.ownerEarningsForm === "NOT_APPLICABLE") {
      validationCodes.push(
        "OWNER_EARNINGS_NOT_APPLICABLE_INVALID_FOR_INDUSTRIAL_FRAMEWORK",
      );
    }
  } else {
    if (
      input.sectorDistributableCapitalFrameworkEstablished.state !==
      "YES"
    ) {
      validationCodes.push(
        "SECTOR_DISTRIBUTABLE_CAPITAL_FRAMEWORK_REQUIRED",
      );
    }

    if (input.standardizedFcfPresent) {
      validationCodes.push(
        "INDUSTRIAL_STANDARDIZED_FCF_NOT_HEADLINE_FOR_SECTOR_FRAMEWORK",
      );
    }

    if (input.ownerEarningsForm !== "NOT_APPLICABLE") {
      limitations.push(
        "INDUSTRIAL_OWNER_EARNINGS_NOT_HEADLINE_FOR_SECTOR_FRAMEWORK",
      );
    }
  }

  if (
    input.reportedFcfPresent &&
    input.reportedFcfSeparatedFromStandardizedFcf.state !== "YES"
  ) {
    validationCodes.push(
      "REPORTED_FCF_MUST_REMAIN_SEPARATE_FROM_STANDARDIZED_FCF",
    );
  }

  if (
    input.cashFramework === "INDUSTRIAL_FCF_OWNER_EARNINGS" &&
    input.ownerEarningsSeparatedFromStandardizedFcf.state !== "YES"
  ) {
    validationCodes.push(
      "OWNER_EARNINGS_MUST_REMAIN_SEPARATE_FROM_STANDARDIZED_FCF",
    );
  }

  if (
    input.ownerEarningsForm === "POINT" &&
    (
      unresolvedMaterialMaintenanceComponents.length > 0 ||
      unknownMaintenanceComponents.length > 0
    )
  ) {
    validationCodes.push(
      "OWNER_EARNINGS_POINT_REQUIRES_RESOLVED_MAINTENANCE",
    );
  }

  if (
    input.ownerEarningsForm === "RANGE" &&
    (
      unresolvedMaterialMaintenanceComponents.length > 0 ||
      unknownMaintenanceComponents.length > 0
    )
  ) {
    limitations.push(
      "OWNER_EARNINGS_RANGE_REFLECTS_MAINTENANCE_UNCERTAINTY",
    );
  }

  if (input.ownerEarningsForm === "UNKNOWN") {
    limitations.push("OWNER_EARNINGS_UNKNOWN");
  }

  if (input.maintenanceDoubleCountedAgainstTotalCapexFcf) {
    validationCodes.push("MAINTENANCE_REINVESTMENT_DOUBLE_COUNTED");
  }

  if (input.workingCapitalDoubleCountedOutsideCfo) {
    validationCodes.push("WORKING_CAPITAL_DOUBLE_COUNTED");
  }

  if (
    input.temporaryWorkingCapitalBenefit.state === "YES" &&
    input.temporaryWorkingCapitalTreatedAsStructuralOwnerCash
  ) {
    validationCodes.push(
      "TEMPORARY_WORKING_CAPITAL_CANNOT_BE_STRUCTURAL_OWNER_CASH",
    );
  }

  if (
    input.deferredMaintenanceMaterial.state === "YES" &&
    input.deferredMaintenanceReflectedInOwnerCash.state !== "YES"
  ) {
    validationCodes.push(
      "MATERIAL_DEFERRED_MAINTENANCE_MUST_BE_REFLECTED",
    );
  }

  if (
    input.recurringAdjustmentsMaterial.state === "YES" &&
    input.recurringAdjustmentsRetainedAsEconomicCost.state !== "YES"
  ) {
    validationCodes.push(
      "MATERIAL_RECURRING_ADJUSTMENTS_MUST_REMAIN_ECONOMIC_COST",
    );
  }

  if (input.sbcPenaltyCount === 2) {
    validationCodes.push("SBC_DOUBLE_PENALTY_FORBIDDEN");
  }

  if (input.sbcMaterial.state === "YES") {
    if (input.sbcEconomicCostRecognized.state !== "YES") {
      validationCodes.push("MATERIAL_SBC_ECONOMIC_COST_REQUIRED");
    }

    if (input.sbcPenaltyCount !== 1) {
      validationCodes.push("MATERIAL_SBC_REQUIRES_ONE_PENALTY");
    }
  }

  if (input.acquisitionsSubtractedFromStandardizedFcf) {
    validationCodes.push(
      "ACQUISITIONS_NOT_SUBTRACTED_FROM_STANDARDIZED_FCF_BY_DEFAULT",
    );
  }

  if (
    input.acquisitionCapitalMaterial.state === "YES" &&
    input.organicCashSeparatedFromAcquisitionDeployment.state !==
      "YES"
  ) {
    validationCodes.push(
      "MATERIAL_ACQUISITION_CAPITAL_REQUIRES_ORGANIC_CASH_SEPARATION",
    );
  }

  if (
    input.perShareCashProduced &&
    input.perShareUsesEconomicDilutedShares.state !== "YES"
  ) {
    validationCodes.push(
      "PER_SHARE_CASH_REQUIRES_ECONOMIC_DILUTED_SHARES",
    );
  }

  if (
    input.deferredMaintenanceMaterial.state === "UNKNOWN"
  ) {
    limitations.push("DEFERRED_MAINTENANCE_UNKNOWN");
  }

  if (input.acquisitionCapitalMaterial.state === "UNKNOWN") {
    limitations.push("ACQUISITION_CAPITAL_MATERIALITY_UNKNOWN");
  }

  const maintenanceEvidence = input.maintenanceComponents.flatMap(
    (item) => [
      ...item.materiality.evidenceIds,
      ...item.treatmentResolved.evidenceIds,
    ],
  );

  const maintenanceAssumptions = input.maintenanceComponents.flatMap(
    (item) => [
      ...item.materiality.assumptionIds,
      ...item.treatmentResolved.assumptionIds,
    ],
  );

  return {
    cashFramework: input.cashFramework,
    ownerEarningsForm: input.ownerEarningsForm,
    materialMaintenanceComponents:
      uniqueSorted(materialMaintenanceComponents),
    unresolvedMaterialMaintenanceComponents:
      uniqueSorted(unresolvedMaterialMaintenanceComponents),
    unknownMaintenanceComponents:
      uniqueSorted(unknownMaintenanceComponents),
    validationCodes: uniqueSorted(validationCodes),
    limitations: uniqueSorted(limitations),
    finalizable: validationCodes.length === 0,
    evidenceIds: uniqueSorted([
      ...traceables.flatMap(([, assessment]) => [
        ...assessment.evidenceIds,
      ]),
      ...maintenanceEvidence,
    ]),
    counterEvidenceIds: uniqueSorted(input.counterEvidenceIds),
    assumptionIds: uniqueSorted([
      ...traceables.flatMap(([, assessment]) => [
        ...assessment.assumptionIds,
      ]),
      ...maintenanceAssumptions,
    ]),
    rationale: input.rationale,
  };
}

export function assertOwnerCashFinalizable(
  output: OwnerCashOutput,
): void {
  if (!output.finalizable) {
    throw new Error(
      `VNEXT_OWNER_CASH_NOT_FINALIZABLE: ${output.validationCodes.join(",")}`,
    );
  }
}

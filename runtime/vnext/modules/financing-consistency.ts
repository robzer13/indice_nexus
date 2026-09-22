export const TRI_STATES = ["YES", "NO", "UNKNOWN"] as const;
export type TriState = (typeof TRI_STATES)[number];

export interface TraceableAssessment {
  state: TriState;
  rationale: string;
  evidenceIds: readonly string[];
  assumptionIds: readonly string[];
}

export interface FinancingConsistencyInput {
  capitalAllocationCashBridgeReconciled: TraceableAssessment;
  netCashDebtBridgeReconciled: TraceableAssessment;

  debtFundingMaterial: TraceableAssessment;
  debtCashInterestEvTreatmentReconciled: TraceableAssessment;

  equityIssuanceMaterial: TraceableAssessment;
  equityIssuanceReflectedInEconomicShares: TraceableAssessment;

  supplierFinanceMaterial: TraceableAssessment;
  supplierFinanceDebtLikeTreatmentReconciled: TraceableAssessment;

  factoringMaterial: TraceableAssessment;
  factoringFinancingTransferReconciled: TraceableAssessment;

  leasesMaterial: TraceableAssessment;
  leaseCashRoicEvTreatmentConsistent: TraceableAssessment;

  acquisitionFundingMaterial: TraceableAssessment;
  acquisitionFundingBridgeEstablished: TraceableAssessment;

  buybacksAssumedOrExecuted: boolean;
  buybackFundingBridgeEstablished: TraceableAssessment;
  buybackShareCountEffectReconciled: TraceableAssessment;

  dividendsAssumedOrExecuted: boolean;
  dividendsConsistentWithRetainedCapitalNeeds: TraceableAssessment;

  assetSalesMaterial: TraceableAssessment;
  assetSalesKeptSeparateFromOperatingCash: TraceableAssessment;

  counterEvidenceIds: readonly string[];
  rationale: string;
}

export interface FinancingConsistencyOutput {
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

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values)].sort();
}

function assertTraceableAssessment(
  label: string,
  assessment: TraceableAssessment,
): void {
  if (!TRI_STATES.includes(assessment.state)) {
    throw new Error(
      `VNEXT_FINANCING_CONSISTENCY_INVALID_${label}_STATE`,
    );
  }

  assertNonBlank(
    assessment.rationale,
    `VNEXT_FINANCING_CONSISTENCY_${label}_RATIONALE_REQUIRED`,
  );

  if (
    assessment.state === "YES" &&
    assessment.evidenceIds.length === 0
  ) {
    throw new Error(
      `VNEXT_FINANCING_CONSISTENCY_${label}_YES_REQUIRES_EVIDENCE`,
    );
  }

  if (
    assessment.state === "UNKNOWN" &&
    assessment.evidenceIds.length === 0 &&
    assessment.assumptionIds.length === 0
  ) {
    throw new Error(
      `VNEXT_FINANCING_CONSISTENCY_${label}_UNKNOWN_REQUIRES_TRACEABLE_BASIS`,
    );
  }
}

function requireYesWhenMaterial(
  material: TraceableAssessment,
  resolved: TraceableAssessment,
  code: string,
  limitations: string[],
  unknownCode: string,
  validationCodes: string[],
): void {
  if (material.state === "YES" && resolved.state !== "YES") {
    validationCodes.push(code);
  }

  if (material.state === "UNKNOWN") {
    limitations.push(unknownCode);
  }
}

export function evaluateFinancingConsistency(
  input: FinancingConsistencyInput,
): FinancingConsistencyOutput {
  assertNonBlank(
    input.rationale,
    "VNEXT_FINANCING_CONSISTENCY_RATIONALE_REQUIRED",
  );

  const traceables: readonly [string, TraceableAssessment][] = [
    [
      "CAPITAL_ALLOCATION_CASH_BRIDGE",
      input.capitalAllocationCashBridgeReconciled,
    ],
    ["NET_CASH_DEBT_BRIDGE", input.netCashDebtBridgeReconciled],
    ["DEBT_FUNDING_MATERIAL", input.debtFundingMaterial],
    [
      "DEBT_CASH_INTEREST_EV_TREATMENT",
      input.debtCashInterestEvTreatmentReconciled,
    ],
    ["EQUITY_ISSUANCE_MATERIAL", input.equityIssuanceMaterial],
    [
      "EQUITY_ISSUANCE_ECONOMIC_SHARES",
      input.equityIssuanceReflectedInEconomicShares,
    ],
    ["SUPPLIER_FINANCE_MATERIAL", input.supplierFinanceMaterial],
    [
      "SUPPLIER_FINANCE_DEBT_LIKE_TREATMENT",
      input.supplierFinanceDebtLikeTreatmentReconciled,
    ],
    ["FACTORING_MATERIAL", input.factoringMaterial],
    [
      "FACTORING_FINANCING_TRANSFER",
      input.factoringFinancingTransferReconciled,
    ],
    ["LEASES_MATERIAL", input.leasesMaterial],
    [
      "LEASE_CASH_ROIC_EV_TREATMENT",
      input.leaseCashRoicEvTreatmentConsistent,
    ],
    [
      "ACQUISITION_FUNDING_MATERIAL",
      input.acquisitionFundingMaterial,
    ],
    [
      "ACQUISITION_FUNDING_BRIDGE",
      input.acquisitionFundingBridgeEstablished,
    ],
    ["BUYBACK_FUNDING_BRIDGE", input.buybackFundingBridgeEstablished],
    [
      "BUYBACK_SHARE_COUNT_EFFECT",
      input.buybackShareCountEffectReconciled,
    ],
    [
      "DIVIDEND_RETAINED_CAPITAL_NEEDS",
      input.dividendsConsistentWithRetainedCapitalNeeds,
    ],
    ["ASSET_SALES_MATERIAL", input.assetSalesMaterial],
    [
      "ASSET_SALES_OPERATING_CASH_SEPARATION",
      input.assetSalesKeptSeparateFromOperatingCash,
    ],
  ];

  for (const [label, assessment] of traceables) {
    assertTraceableAssessment(label, assessment);
  }

  const validationCodes: string[] = [];
  const limitations: string[] = [];

  if (
    input.capitalAllocationCashBridgeReconciled.state !== "YES"
  ) {
    validationCodes.push(
      "CAPITAL_ALLOCATION_CASH_BRIDGE_MUST_RECONCILE",
    );
  }

  if (input.netCashDebtBridgeReconciled.state !== "YES") {
    validationCodes.push("NET_CASH_DEBT_BRIDGE_MUST_RECONCILE");
  }

  requireYesWhenMaterial(
    input.debtFundingMaterial,
    input.debtCashInterestEvTreatmentReconciled,
    "MATERIAL_DEBT_REQUIRES_CASH_INTEREST_EV_RECONCILIATION",
    limitations,
    "DEBT_FUNDING_MATERIALITY_UNKNOWN",
    validationCodes,
  );

  requireYesWhenMaterial(
    input.equityIssuanceMaterial,
    input.equityIssuanceReflectedInEconomicShares,
    "MATERIAL_EQUITY_ISSUANCE_REQUIRES_ECONOMIC_SHARE_RECONCILIATION",
    limitations,
    "EQUITY_ISSUANCE_MATERIALITY_UNKNOWN",
    validationCodes,
  );

  requireYesWhenMaterial(
    input.supplierFinanceMaterial,
    input.supplierFinanceDebtLikeTreatmentReconciled,
    "MATERIAL_SUPPLIER_FINANCE_REQUIRES_DEBT_LIKE_RECONCILIATION",
    limitations,
    "SUPPLIER_FINANCE_MATERIALITY_UNKNOWN",
    validationCodes,
  );

  requireYesWhenMaterial(
    input.factoringMaterial,
    input.factoringFinancingTransferReconciled,
    "MATERIAL_FACTORING_REQUIRES_FINANCING_TRANSFER_RECONCILIATION",
    limitations,
    "FACTORING_MATERIALITY_UNKNOWN",
    validationCodes,
  );

  requireYesWhenMaterial(
    input.leasesMaterial,
    input.leaseCashRoicEvTreatmentConsistent,
    "MATERIAL_LEASES_REQUIRE_CASH_ROIC_EV_CONSISTENCY",
    limitations,
    "LEASE_MATERIALITY_UNKNOWN",
    validationCodes,
  );

  requireYesWhenMaterial(
    input.acquisitionFundingMaterial,
    input.acquisitionFundingBridgeEstablished,
    "MATERIAL_ACQUISITION_FUNDING_REQUIRES_BRIDGE",
    limitations,
    "ACQUISITION_FUNDING_MATERIALITY_UNKNOWN",
    validationCodes,
  );

  if (input.buybacksAssumedOrExecuted) {
    if (input.buybackFundingBridgeEstablished.state !== "YES") {
      validationCodes.push("BUYBACKS_REQUIRE_FINANCING_BRIDGE");
    }

    if (
      input.buybackShareCountEffectReconciled.state !== "YES"
    ) {
      validationCodes.push(
        "BUYBACKS_REQUIRE_SHARE_COUNT_RECONCILIATION",
      );
    }
  }

  if (
    input.dividendsAssumedOrExecuted &&
    input.dividendsConsistentWithRetainedCapitalNeeds.state !==
      "YES"
  ) {
    validationCodes.push(
      "DIVIDENDS_MUST_RECONCILE_WITH_RETAINED_CAPITAL_NEEDS",
    );
  }

  requireYesWhenMaterial(
    input.assetSalesMaterial,
    input.assetSalesKeptSeparateFromOperatingCash,
    "MATERIAL_ASSET_SALES_MUST_REMAIN_SEPARATE_FROM_OPERATING_CASH",
    limitations,
    "ASSET_SALES_MATERIALITY_UNKNOWN",
    validationCodes,
  );

  const evidenceIds = uniqueSorted(
    traceables.flatMap(([, assessment]) => [
      ...assessment.evidenceIds,
    ]),
  );

  const assumptionIds = uniqueSorted(
    traceables.flatMap(([, assessment]) => [
      ...assessment.assumptionIds,
    ]),
  );

  return {
    validationCodes: uniqueSorted(validationCodes),
    limitations: uniqueSorted(limitations),
    finalizable: validationCodes.length === 0,
    evidenceIds,
    counterEvidenceIds: uniqueSorted(input.counterEvidenceIds),
    assumptionIds,
    rationale: input.rationale,
  };
}

export function assertFinancingConsistencyFinalizable(
  output: FinancingConsistencyOutput,
): void {
  if (!output.finalizable) {
    throw new Error(
      `VNEXT_FINANCING_CONSISTENCY_NOT_FINALIZABLE: ${output.validationCodes.join(",")}`,
    );
  }
}

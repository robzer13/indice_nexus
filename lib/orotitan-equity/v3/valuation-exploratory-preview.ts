/**
 * OroTitan — Valuation exploratory preview, G2 proposal only.
 * Pure computation: no imports, DB, networking, registry, scoring or orders.
 * Does not modify or weaken any canonical V3.1 validator.
 * SECURITY/INTEGRITY: a caller-supplied provenance claim is not attestation;
 * verify any immutable GitHub/Registry upstream refs in a separate trust layer.
 */

export type PreviewBasis = "FCFF" | "FCFE" | "OWNER_EARNINGS";
export type PreviewRateBasis = "WACC" | "COST_OF_EQUITY";
export type PreviewOutputType =
  | "ENTERPRISE_DCF_SCENARIO" | "EQUITY_VALUE_SCENARIO"
  | "PER_SHARE_SCENARIO" | "PRICE_COMPARISON_SCENARIO";

export interface PublicSourceRef {
  sourceId: string;
  publicationDate: string;
  economicEffectiveDate: string;
  locator: string;
  access: "PUBLIC";
}
export interface PreviewCashFlow {
  periodStartDate: string;
  paymentDate: string;
  amount: number;
  construction: "POST_ORIGIN_ONLY";
}
export interface PreviewForecastCase {
  id: string;
  assumptionNote: string;
  sourceIds: readonly string[];
  discountRate: number;
  discountRateNote: string;
  flows: readonly PreviewCashFlow[];
  firstFiscalPeriodStartDate: string;
  fiscalCalendarResolved: boolean;
  fiscalYearMismatchResolved: boolean;
  terminal: {
    date: string;
    growthRate: number;
    nextPeriodCashFlow: number;
    stableEconomicsResolved: boolean;
    assumptionNote: string;
  };
}
export interface PreviewBridgeScenario {
  netEquityBridgeAdjustment: number;
  assumptionNote: string;
  economicDate: string;
}
export interface PreviewShareScenario {
  id: string;
  count: number;
  assumptionNote: string;
  economicDate: string;
}
export interface PreviewRequest {
  mode: "EXPLORATORY_PREVIEW";
  analysisCutoff: string;
  valuationDate: string;
  upstreamFundamentalsCutoff: string;
  upstreamRef: { runId: string; manifestId: string; lockSha256: string };
  upstreamFreshness: "FRESH_VERIFIED" | "HISTORICAL_DISCLOSED";
  sourceRefs: readonly PublicSourceRef[];
  cashFlowBasis: PreviewBasis;
  discountRateBasis: PreviewRateBasis;
  ownerEarningsBasisSupported?: boolean;
  historicalOutputCalibration: false;
  forecastCases: readonly PreviewForecastCase[];
  bridgeScenario?: PreviewBridgeScenario;
  shareScenarios?: readonly PreviewShareScenario[];
  priceReference?: {
    date: string;
    price: number;
    sourceId: string;
    freshnessResolved: boolean;
  };
}
export interface PreviewCaseResult {
  caseId: string;
  method: "ACT/365F_END_PERIOD_NO_AUTO_PRORATION";
  sourceIds: readonly string[];
  forecastAssumption: string;
  rateAssumption: string;
  terminalAssumption: string;
  discountRate: number;
  terminalGrowthRate: number;
  terminalValue: number;
  cashFlows: readonly PreviewCashFlow[];
  discountedFlows: readonly number[];
  pvTerminalValue: number;
  enterpriseValueScenario: number | null;
  equityValueScenario: number | null;
  perShareScenarios: readonly { shareScenarioId: string; assumedShares: number; valuePerShare: number; assumptionNote: string }[];
  priceComparisons: readonly { shareScenarioId: string; hypotheticalPremiumPct: number }[];
}
export interface PreviewResult {
  mode: "EXPLORATORY_PREVIEW";
  authority: "NON_AUTHORITATIVE";
  contractStatus: "G2_PROPOSED_UNAPPROVED";
  canonicalValuationAdmission: "NOT_EVALUATED_IN_PREVIEW";
  valuationLockPermitted: false;
  readyForCertification: false;
  scoringPermitted: false;
  publicationPermitted: false;
  orderPermitted: false;
  registryMutationPermitted: false;
  applicableCutoff: string | null;
  methodVersion: "G2_EXPLORATORY_DRAFT_V0_1";
  evidenceAsOfAndPublicationDates: readonly PublicSourceRef[];
  upstreamRef: PreviewRequest["upstreamRef"] | null;
  admissibleAsPreview: boolean;
  refusalReasons: readonly string[];
  warnings: readonly string[];
  outputTypes: readonly PreviewOutputType[];
  cases: readonly PreviewCaseResult[];
}

const ISO = /^\d{4}-\d{2}-\d{2}$/;
const SHA = /^[0-9a-f]{64}$/;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
function day(value: string): number | null {
  if (!ISO.test(value)) return null;
  const [year, month, date] = value.split("-").map(Number);
  const epoch = Date.UTC(year, month - 1, date);
  const d = new Date(epoch);
  return d.getUTCFullYear() === year && d.getUTCMonth() === month - 1 && d.getUTCDate() === date
    ? Math.round(epoch / 86_400_000) : null;
}
function okText(x: unknown): x is string {
  return typeof x === "string" && x.trim().length > 0;
}
function finite(x: unknown): x is number { return typeof x === "number" && Number.isFinite(x); }
function createRefusal(cutoff: string | null, refusalReasons: string[], warnings: string[], input?: PreviewRequest): PreviewResult {
  return Object.freeze({
    mode: "EXPLORATORY_PREVIEW" as const,
    authority: "NON_AUTHORITATIVE" as const,
    contractStatus: "G2_PROPOSED_UNAPPROVED" as const,
    canonicalValuationAdmission: "NOT_EVALUATED_IN_PREVIEW" as const,
    valuationLockPermitted: false as const,
    readyForCertification: false as const,
    scoringPermitted: false as const,
    publicationPermitted: false as const,
    orderPermitted: false as const,
    registryMutationPermitted: false as const,
    applicableCutoff: cutoff,
    methodVersion: "G2_EXPLORATORY_DRAFT_V0_1" as const,
    evidenceAsOfAndPublicationDates: Object.freeze(Array.isArray(input?.sourceRefs) ? [...input.sourceRefs] : []),
    upstreamRef: input?.upstreamRef ?? null,
    admissibleAsPreview: false,
    refusalReasons: Object.freeze([...new Set(refusalReasons)]),
    warnings: Object.freeze([...new Set(warnings)]),
    outputTypes: Object.freeze([] as PreviewOutputType[]),
    cases: Object.freeze([] as PreviewCaseResult[]),
  });
}

/** Computes only *hypothetical* scenarios; never returns canonical Valuation fields. */
export function evaluateValuationExploratoryPreview(input: PreviewRequest): PreviewResult {
  const reasons: string[] = [], warnings: string[] = [];
  if (input?.mode !== "EXPLORATORY_PREVIEW") reasons.push("UNSUPPORTED_MODE");
  const cutoff = day(input?.analysisCutoff);
  if (cutoff === null || day(input?.valuationDate) === null) reasons.push("INVALID_DATE");
  else if (input.analysisCutoff !== input.valuationDate) reasons.push("VALUATION_DATE_DATA_CUTOFF_MISMATCH");
  const upstream = day(input?.upstreamFundamentalsCutoff);
  if (upstream === null) reasons.push("UPSTREAM_CUTOFF_INVALID");
  else if (cutoff !== null && upstream > cutoff) reasons.push("UPSTREAM_FUNDAMENTALS_AFTER_CUTOFF");
  else if (cutoff !== null && upstream < cutoff) {
    if (input?.upstreamFreshness !== "HISTORICAL_DISCLOSED") reasons.push("HISTORICAL_FUNDAMENTALS_MUST_BE_DISCLOSED");
    warnings.push("HISTORICAL_FUNDAMENTALS_NOT_REVALIDATED_FOR_NEW_CUTOFF");
  }
  if (!input?.upstreamRef || !UUID.test(input.upstreamRef.runId) ||
      !UUID.test(input.upstreamRef.manifestId) || !SHA.test(input.upstreamRef.lockSha256)) {
    reasons.push("UPSTREAM_REF_INVALID");
  }
  if (!["FRESH_VERIFIED", "HISTORICAL_DISCLOSED"].includes(input?.upstreamFreshness)) reasons.push("UPSTREAM_FRESHNESS_INVALID");
  warnings.push("UPSTREAM_REF_MUST_BE_VERIFIED_BY_TRUSTED_LOADER");
  const sourceIds = new Set<string>();
  if (!Array.isArray(input?.sourceRefs) || input.sourceRefs.length === 0) reasons.push("NO_PUBLIC_EVIDENCE");
  else for (const source of input.sourceRefs) {
    if (!okText(source?.sourceId) || !okText(source?.locator) ||
        source.access !== "PUBLIC" || day(source.publicationDate) === null ||
        day(source.economicEffectiveDate) === null) {
      reasons.push("SOURCE_PROVENANCE_INCOMPLETE"); continue;
    }
    if (sourceIds.has(source.sourceId)) reasons.push("DUPLICATE_SOURCE_ID");
    sourceIds.add(source.sourceId);
    if (cutoff !== null && day(source.publicationDate)! > cutoff) reasons.push("SOURCE_PUBLISHED_AFTER_CUTOFF");
    if (cutoff !== null && day(source.economicEffectiveDate)! > cutoff) reasons.push("SOURCE_EFFECTIVE_AFTER_CUTOFF");
  }
  if (!["FCFF", "FCFE", "OWNER_EARNINGS"].includes(input?.cashFlowBasis)) reasons.push("INVALID_CASH_FLOW_BASIS");
  if (input?.cashFlowBasis === "FCFF" && input?.discountRateBasis !== "WACC") reasons.push("FCFF_REQUIRES_WACC");
  if (["FCFE", "OWNER_EARNINGS"].includes(input?.cashFlowBasis) && input?.discountRateBasis !== "COST_OF_EQUITY") reasons.push("EQUITY_FLOWS_REQUIRE_COST_OF_EQUITY");
  if (input?.cashFlowBasis === "OWNER_EARNINGS" && input.ownerEarningsBasisSupported !== true) reasons.push("OWNER_EARNINGS_UNSUPPORTED");
  if (input?.historicalOutputCalibration !== false) reasons.push("HISTORICAL_CALIBRATION_FORBIDDEN");
  if (!Array.isArray(input?.forecastCases) || input.forecastCases.length === 0) reasons.push("FORECAST_MISSING");
  else {
    const ids = new Set<string>();
    for (const c of input.forecastCases) {
      if (!okText(c?.id) || !okText(c?.assumptionNote) || !okText(c?.discountRateNote)) reasons.push("FORECAST_ASSUMPTIONS_UNDOCUMENTED");
      if (!Array.isArray(c.sourceIds) || c.sourceIds.length === 0 || c.sourceIds.some((id: string) => !sourceIds.has(id))) reasons.push("FORECAST_SOURCE_LINEAGE_MISSING");
      if (ids.has(c.id)) reasons.push("DUPLICATE_FORECAST_ID");
      ids.add(c.id);
      if (!finite(c?.discountRate) || c.discountRate <= -1) reasons.push("INVALID_DISCOUNT_RATE");
      if (c?.fiscalCalendarResolved !== true || c?.fiscalYearMismatchResolved !== true) reasons.push("FISCAL_TIMING_UNRESOLVED");
      const firstFiscalDay = day(c?.firstFiscalPeriodStartDate);
      if (firstFiscalDay === null || (cutoff !== null && firstFiscalDay > cutoff)) reasons.push("FIRST_FISCAL_PERIOD_INVALID");
      if (!Array.isArray(c?.flows) || c.flows.length === 0) reasons.push("CASH_FLOW_SCHEDULE_EMPTY");
      else {
        let prev = cutoff;
        for (const f of c.flows) {
          const start = day(f?.periodStartDate),end = day(f?.paymentDate);
          if (start === null || end === null || prev === null || start !== prev || end <= prev ||
              f.construction !== "POST_ORIGIN_ONLY") reasons.push("FORECAST_DATE_CHAIN_OR_STUB_INVALID");
          if (!finite(f?.amount)) reasons.push("NONFINITE_CASH_FLOW");
          prev = end;
        }
        if (day(c?.terminal?.date) !== prev) reasons.push("TERMINAL_DATE_MISMATCH");
      }
      if (!okText(c?.terminal?.assumptionNote) || c.terminal.stableEconomicsResolved !== true ||
          !finite(c.terminal.growthRate) || !finite(c.terminal.nextPeriodCashFlow)) reasons.push("TERMINAL_ASSUMPTIONS_UNSUPPORTED");
      else if (finite(c.discountRate) && c.terminal.growthRate >= c.discountRate) reasons.push("TERMINAL_G_NOT_LESS_THAN_R");
    }
  }
  if (input?.cashFlowBasis !== "FCFF" && input?.bridgeScenario) reasons.push("EQUITY_DCF_EV_BRIDGE_FORBIDDEN");
  if (input?.bridgeScenario) {
    const b = input.bridgeScenario;
    if (!finite(b.netEquityBridgeAdjustment) || !okText(b.assumptionNote) ||
        day(b.economicDate) === null || day(b.economicDate)! > (cutoff ?? -Infinity)) reasons.push("BRIDGE_SCENARIO_INVALID");
    else if (b.economicDate !== input.analysisCutoff) warnings.push("BRIDGE_SCENARIO_HISTORICAL_NOT_CURRENT");
    warnings.push("BRIDGE_IS_ASSUMPTION_NOT_RECONCILED_ACTUAL_NET_DEBT");
  }
  if (input?.shareScenarios !== undefined) {
    if (!Array.isArray(input.shareScenarios)) reasons.push("SHARE_SCENARIOS_INVALID");
    else {
      const ids = new Set<string>();
      for (const share of input.shareScenarios) {
        if (!okText(share?.id) || !okText(share?.assumptionNote) || !finite(share?.count) || share.count <= 0 ||
            !Number.isSafeInteger(share.count) || day(share.economicDate) === null ||
            day(share.economicDate)! > (cutoff ?? -Infinity)) reasons.push("SHARE_SCENARIO_INVALID");
        else if (share.economicDate !== input.analysisCutoff) warnings.push("SHARE_SCENARIO_ANCHOR_HISTORICAL");
        if (ids.has(share?.id)) reasons.push("DUPLICATE_SHARE_SCENARIO_ID");
        ids.add(share?.id);
      }
      if (input.shareScenarios.length === 1) warnings.push("SINGLE_SHARE_SCENARIO_NO_SHARE_SENSITIVITY");
    }
  }
  let canComparePrice = false;
  if (input?.priceReference) {
    const p = input.priceReference;
    if (day(p.date) === null || !finite(p.price) || p.price <= 0 || !sourceIds.has(p.sourceId)) {
      reasons.push("PRICE_PROVENANCE_OR_VALUE_INVALID");
    } else if (cutoff !== null && day(p.date)! > cutoff) {
      warnings.push("PRICE_REFERENCE_AFTER_CUTOFF_OMITTED");
    } else if (p.freshnessResolved !== true) {
      warnings.push("PRICE_STALENESS_NOT_RESOLVED_OMITTED");
    } else canComparePrice = true;
  }
  if (reasons.length > 0) return createRefusal(cutoff === null ? null : input.analysisCutoff, reasons, warnings, input);

  const types = new Set<PreviewOutputType>();
  const computed: PreviewCaseResult[] = [];
  for (const c of input.forecastCases) {
    const flows = c.flows.map(f=> f.amount / Math.pow(1 + c.discountRate, (day(f.paymentDate)! - cutoff!) / 365));
    const terminalValue = c.terminal.nextPeriodCashFlow / (c.discountRate - c.terminal.growthRate);
    const terminalPresent = terminalValue / Math.pow(1 + c.discountRate, (day(c.terminal.date)! - cutoff!) / 365);
    const value = flows.reduce((a,b)=>a+b,0) + terminalPresent;
    if (![...flows,terminalValue,terminalPresent,value].every(Number.isFinite)) {
      return createRefusal(input.analysisCutoff,["NONFINITE_DCF_RESULT"],warnings,input);
    }
    const enterprise = input.cashFlowBasis === "FCFF" ? value : null;
    const equity = input.cashFlowBasis === "FCFF"
      ? (input.bridgeScenario ? value + input.bridgeScenario.netEquityBridgeAdjustment : null)
      : value;
    types.add(input.cashFlowBasis === "FCFF" ? "ENTERPRISE_DCF_SCENARIO" : "EQUITY_VALUE_SCENARIO");
    if (equity !== null) types.add("EQUITY_VALUE_SCENARIO");
    const perShareScenarios: PreviewCaseResult["perShareScenarios"][number][] = [];
    const priceComparisons: PreviewCaseResult["priceComparisons"][number][] = [];
    if (equity !== null && Number.isFinite(equity) && input.shareScenarios?.length) {
      for (const share of input.shareScenarios) {
        const perShare = equity / share.count;
        if (!Number.isFinite(perShare)) return createRefusal(input.analysisCutoff,["NONFINITE_PER_SHARE_RESULT"],warnings,input);
        perShareScenarios.push({shareScenarioId:share.id,assumedShares:share.count,valuePerShare:perShare,assumptionNote:share.assumptionNote});
        types.add("PER_SHARE_SCENARIO");
        if (canComparePrice && input.priceReference) {
          priceComparisons.push({shareScenarioId:share.id,hypotheticalPremiumPct:(perShare / input.priceReference.price - 1) * 100});
          types.add("PRICE_COMPARISON_SCENARIO");
        }
      }
    } else if (input.shareScenarios?.length) warnings.push("PER_SHARE_OMITTED_EQUITY_BRIDGE_MISSING");
    computed.push(Object.freeze({caseId:c.id,method:"ACT/365F_END_PERIOD_NO_AUTO_PRORATION" as const,sourceIds:Object.freeze([...c.sourceIds]),
      forecastAssumption:c.assumptionNote,rateAssumption:c.discountRateNote,terminalAssumption:c.terminal.assumptionNote,
      discountRate:c.discountRate,terminalGrowthRate:c.terminal.growthRate,terminalValue,cashFlows:Object.freeze([...c.flows]),
      discountedFlows:Object.freeze(flows),pvTerminalValue:terminalPresent,
      enterpriseValueScenario:enterprise,equityValueScenario:equity,
      perShareScenarios:Object.freeze(perShareScenarios),priceComparisons:Object.freeze(priceComparisons)}));
  }
  warnings.push("SCENARIOS_ARE_NOT_CANONICAL_ECONOMIC_VALUE");
  if (canComparePrice) warnings.push("PRICE_COMPARISON_IS_NOT_MOS_OR_EXPECTED_RETURN");
  if (input.forecastCases.length === 1) warnings.push("SINGLE_FORECAST_CASE_NO_RATE_SENSITIVITY");
  return Object.freeze({
    ...createRefusal(input.analysisCutoff,[],warnings,input),
    admissibleAsPreview:true,
    outputTypes:Object.freeze([...types]),
    cases:Object.freeze(computed),
  });
}

export type ValuationCashFlowBasis = "FCFF" | "FCFE" | "OWNER_EARNINGS";
export type EconomicShareCountRepresentation = "EXACT" | "BOUNDED" | "UNKNOWN";

export type ValuationDateAlignmentInput = {
  dataCutoff: string;
  valuationDate: string;
  timeOrigin: string;
  hasNumericReferencePrice: boolean;
  referencePriceDate: string | null;
  referencePriceStalenessResolved: boolean;
  economicShareCountDate: string;
  shareCountAsOfDate: string;
  economicShareCountRepresentation: EconomicShareCountRepresentation;
  denominatorBridgeResolved: boolean;
  cashFlowBasis: ValuationCashFlowBasis;
  evToEquityBridgeDate: string | null;
  evBridgeRollforwardResolved: boolean;
  perShareOutputDate: string;
  marketCapShareCountDate: string | null;
  marketCapShareCountResolved: boolean;
};

export type ValuationDateAlignmentResult = {
  valuationDate: string;
  timeOrigin: string;
  referencePriceDate: string | null;
  economicShareCountDate: string;
  shareCountAsOfDate: string;
  evToEquityBridgeDate: string | "NOT_APPLICABLE";
  marketCapDate: string | "NOT_ASSESSABLE";
  perShareOutputDate: string;
  denominatorAdmitted: boolean;
  evBridgeAdmitted: boolean;
  priceDependentOutputsAdmitted: boolean;
  marketCapAdmitted: boolean;
  valuationAdmission: boolean;
  blockers: string[];
};

export class ValuationDateAlignmentError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(`${code}: ${message}`);
    this.code = code;
    this.name = "ValuationDateAlignmentError";
  }
}

function isIsoDate(value: string): boolean {
  return /^\d{4}-\d{2}-\d{2}$/.test(value);
}

function requireIsoDate(value: string, label: string): void {
  if (!isIsoDate(value)) {
    throw new ValuationDateAlignmentError("INVALID_DATE", `${label} must be YYYY-MM-DD`);
  }
}

function fail(code: string, message: string): never {
  throw new ValuationDateAlignmentError(code, message);
}

export function resolveValuationDateAlignment(
  input: ValuationDateAlignmentInput,
): ValuationDateAlignmentResult {
  requireIsoDate(input.dataCutoff, "DATA_CUTOFF");
  requireIsoDate(input.valuationDate, "VALUATION_DATE");
  requireIsoDate(input.timeOrigin, "TIME_ORIGIN");
  requireIsoDate(input.economicShareCountDate, "ECONOMIC_SHARE_COUNT_DATE");
  requireIsoDate(input.shareCountAsOfDate, "SHARE_COUNT_AS_OF_DATE");
  requireIsoDate(input.perShareOutputDate, "PER_SHARE_OUTPUT_DATE");

  if (input.valuationDate !== input.dataCutoff) {
    fail("VALUATION_DATE_DATA_CUTOFF_MISMATCH", "VALUATION_DATE must equal DATA_CUTOFF");
  }
  if (input.timeOrigin !== input.valuationDate) {
    fail("TIME_ORIGIN_MISMATCH", "TIME_ORIGIN must equal VALUATION_DATE");
  }

  if (input.hasNumericReferencePrice) {
    if (input.referencePriceDate === null) {
      fail("REFERENCE_PRICE_DATE_MISSING", "numeric REFERENCE_PRICE requires REFERENCE_PRICE_DATE");
    }
    requireIsoDate(input.referencePriceDate, "REFERENCE_PRICE_DATE");
    if (input.referencePriceDate > input.valuationDate) {
      fail("REFERENCE_PRICE_AFTER_VALUATION_DATE", "REFERENCE_PRICE_DATE must not be after VALUATION_DATE");
    }
  } else if (input.referencePriceDate !== null) {
    requireIsoDate(input.referencePriceDate, "REFERENCE_PRICE_DATE");
    if (input.referencePriceDate > input.valuationDate) {
      fail("REFERENCE_PRICE_AFTER_VALUATION_DATE", "REFERENCE_PRICE_DATE must not be after VALUATION_DATE");
    }
  }

  if (input.economicShareCountDate !== input.valuationDate) {
    fail("ECONOMIC_SHARE_COUNT_DATE_MISMATCH", "valuation denominator must be measured at VALUATION_DATE");
  }
  if (input.shareCountAsOfDate !== input.valuationDate) {
    fail("ECONOMIC_SHARE_COUNT_DATE_MISMATCH", "SHARE_COUNT_AS_OF_DATE must equal VALUATION_DATE");
  }
  if (input.perShareOutputDate !== input.valuationDate) {
    fail("PER_SHARE_DATE_MISMATCH", "PER_SHARE_OUTPUT_DATE must equal VALUATION_DATE");
  }

  const blockers: string[] = [];
  const denominatorAdmitted =
    input.economicShareCountRepresentation !== "UNKNOWN" &&
    input.denominatorBridgeResolved;

  if (input.economicShareCountRepresentation === "UNKNOWN") {
    blockers.push("ECONOMIC_SHARE_COUNT_UNRESOLVED");
  } else if (!input.denominatorBridgeResolved) {
    blockers.push("STALE_DENOMINATOR_TRANSPORT_FORBIDDEN");
  }

  let evBridgeAdmitted = true;
  let evToEquityBridgeDate: string | "NOT_APPLICABLE" = "NOT_APPLICABLE";
  if (input.cashFlowBasis === "FCFF") {
    if (input.evToEquityBridgeDate === null) {
      fail("EV_BRIDGE_REQUIRED_FOR_FCFF", "FCFF requires EV-to-equity bridge at VALUATION_DATE");
    }
    requireIsoDate(input.evToEquityBridgeDate, "EV_TO_EQUITY_BRIDGE_DATE");
    if (input.evToEquityBridgeDate !== input.valuationDate) {
      fail("EV_BRIDGE_DATE_MISMATCH", "EV_TO_EQUITY_BRIDGE_DATE must equal VALUATION_DATE");
    }
    evToEquityBridgeDate = input.evToEquityBridgeDate;
    evBridgeAdmitted = input.evBridgeRollforwardResolved;
    if (!evBridgeAdmitted) blockers.push("EV_BRIDGE_ROLLFORWARD_UNRESOLVED");
  } else if (input.evToEquityBridgeDate !== null) {
    fail("EQUITY_DCF_EV_BRIDGE_FORBIDDEN", "ordinary EV bridge is forbidden for equity DCF");
  }

  let priceDependentOutputsAdmitted = true;
  if (input.hasNumericReferencePrice && !input.referencePriceStalenessResolved) {
    priceDependentOutputsAdmitted = false;
    blockers.push("REFERENCE_PRICE_STALENESS_UNRESOLVED");
  }

  let marketCapAdmitted = false;
  let marketCapDate: string | "NOT_ASSESSABLE" = "NOT_ASSESSABLE";
  if (input.hasNumericReferencePrice && input.referencePriceDate !== null) {
    if (input.marketCapShareCountDate !== input.referencePriceDate) {
      if (input.marketCapShareCountResolved) {
        fail(
          "MARKET_CAP_DENOMINATOR_DATE_MISMATCH",
          "market-cap denominator must be measured at REFERENCE_PRICE_DATE",
        );
      }
    } else if (input.marketCapShareCountResolved && input.referencePriceStalenessResolved) {
      marketCapAdmitted = true;
      marketCapDate = input.referencePriceDate;
    }
  }

  return {
    valuationDate: input.valuationDate,
    timeOrigin: input.timeOrigin,
    referencePriceDate: input.referencePriceDate,
    economicShareCountDate: input.economicShareCountDate,
    shareCountAsOfDate: input.shareCountAsOfDate,
    evToEquityBridgeDate,
    marketCapDate,
    perShareOutputDate: input.perShareOutputDate,
    denominatorAdmitted,
    evBridgeAdmitted,
    priceDependentOutputsAdmitted,
    marketCapAdmitted,
    valuationAdmission: denominatorAdmitted && evBridgeAdmitted,
    blockers,
  };
}

export type DcfCashFlowBasis = "FCFF" | "FCFE" | "OWNER_EARNINGS";
export type DcfDiscountRateBasis = "WACC" | "COST_OF_EQUITY";

export class DcfTimingError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(`${code}: ${message}`);
    this.code = code;
    this.name = "DcfTimingError";
  }
}

export type DcfCashFlow = {
  periodStartDate: string;
  paymentDate: string;
  amount: number;
  construction: "POST_ORIGIN_ONLY" | "FULL_PERIOD_AUTOPRORATED";
};

export type DcfTimingInput = {
  dataCutoff: string;
  valuationDate: string;
  calculationDate: string;
  hasNumericReferencePrice: boolean;
  referencePriceDate: string | null;
  periodTiming: "END_OF_PERIOD" | "MIDPOINT" | "BEGINNING_OF_PERIOD";
  yearFractionConvention: "ACT/365F" | string;
  fiscalCalendarResolved: boolean;
  historicalOutputCalibration: boolean;
  cashFlowBasis: DcfCashFlowBasis;
  ownerEarningsBasisSupported?: boolean;
  discountRateBasis: DcfDiscountRateBasis;
  discountRate: number;
  cashFlows: DcfCashFlow[];
  terminal: {
    date: string;
    growthRate: number;
    nextPeriodCashFlow: number;
    stableEconomicsResolved: boolean;
    cashFlowBasis: DcfCashFlowBasis;
  };
  evToEquityBridge: null | {
    date: string;
    netEquityBridgeAdjustment: number;
    rollforwardResolved: boolean;
  };
  economicShareCount: {
    date: string;
    count: number;
    dilutionDoubleCountResolved: boolean;
  };
};

export type DcfTimingOutput = {
  timeOrigin: string;
  valuationDate: string;
  yearFractionConvention: "ACT/365F";
  yearFractions: number[];
  discountExponents: number[];
  pvCashFlows: number[];
  terminalValue: number;
  pvTerminalValue: number;
  enterpriseValue: number | null;
  equityValue: number;
  perShareValue: number;
};

function fail(code: string, message: string): never {
  throw new DcfTimingError(code, message);
}

function assertFiniteNumber(value: number, code: string, label: string): void {
  if (!Number.isFinite(value)) fail(code, `${label} must be finite`);
}

function parseIsoCivilDate(value: string, code = "INVALID_DATE"): number {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) fail(code, `invalid ISO civil date: ${value}`);
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const millis = Date.UTC(year, month - 1, day);
  const d = new Date(millis);
  if (
    d.getUTCFullYear() !== year ||
    d.getUTCMonth() !== month - 1 ||
    d.getUTCDate() !== day
  ) {
    fail(code, `invalid civil date: ${value}`);
  }
  return Math.trunc(millis / 86_400_000);
}

function compareDates(a: string, b: string): number {
  return parseIsoCivilDate(a) - parseIsoCivilDate(b);
}

export function actual365FixedYearFraction(
  valuationDate: string,
  paymentDate: string,
): number {
  const days = parseIsoCivilDate(paymentDate) - parseIsoCivilDate(valuationDate);
  const t = days / 365;
  if (!Number.isFinite(t)) fail("NONFINITE_DISCOUNT_EXPONENT", "year fraction is non-finite");
  return t;
}

function expectedDiscountRateBasis(cashFlowBasis: DcfCashFlowBasis): DcfDiscountRateBasis {
  return cashFlowBasis === "FCFF" ? "WACC" : "COST_OF_EQUITY";
}

export function computeDcfTiming(input: DcfTimingInput): DcfTimingOutput {
  parseIsoCivilDate(input.dataCutoff);
  parseIsoCivilDate(input.valuationDate);
  parseIsoCivilDate(input.calculationDate);

  if (input.valuationDate !== input.dataCutoff) {
    fail(
      "VALUATION_DATE_DATA_CUTOFF_MISMATCH",
      "TIME_ORIGIN and VALUATION_DATE must equal DATA_CUTOFF",
    );
  }

  if (input.hasNumericReferencePrice) {
    if (input.referencePriceDate === null) {
      fail("REFERENCE_PRICE_DATE_MISSING", "numeric reference price requires REFERENCE_PRICE_DATE");
    }
    parseIsoCivilDate(input.referencePriceDate);
    if (compareDates(input.referencePriceDate, input.valuationDate) > 0) {
      fail(
        "REFERENCE_PRICE_AFTER_VALUATION_DATE",
        "REFERENCE_PRICE_DATE must not be after VALUATION_DATE",
      );
    }
  } else if (input.referencePriceDate !== null) {
    parseIsoCivilDate(input.referencePriceDate);
    if (compareDates(input.referencePriceDate, input.valuationDate) > 0) {
      fail(
        "REFERENCE_PRICE_AFTER_VALUATION_DATE",
        "REFERENCE_PRICE_DATE must not be after VALUATION_DATE",
      );
    }
  }

  if (input.periodTiming !== "END_OF_PERIOD") {
    fail("UNSUPPORTED_PERIOD_TIMING", "only end-of-period cash-flow timing is canonical");
  }
  if (input.yearFractionConvention !== "ACT/365F") {
    fail(
      "UNSUPPORTED_YEAR_FRACTION_CONVENTION",
      "only ACT/365F is canonical",
    );
  }
  if (!input.fiscalCalendarResolved) {
    fail("FISCAL_PERIOD_DATE_UNRESOLVED", "exact issuer fiscal-period dates are required");
  }
  if (input.historicalOutputCalibration) {
    fail(
      "HISTORICAL_OUTPUT_CALIBRATION_FORBIDDEN",
      "timing inputs must not be selected to reproduce historical company values",
    );
  }

  const expectedRateBasis = expectedDiscountRateBasis(input.cashFlowBasis);
  if (input.discountRateBasis !== expectedRateBasis) {
    const code =
      input.cashFlowBasis === "FCFF"
        ? "FCFF_DISCOUNT_RATE_BASIS_MISMATCH"
        : input.cashFlowBasis === "FCFE"
          ? "FCFE_DISCOUNT_RATE_BASIS_MISMATCH"
          : "OWNER_EARNINGS_DISCOUNT_RATE_BASIS_MISMATCH";
    fail(code, `${input.cashFlowBasis} requires ${expectedRateBasis}`);
  }

  if (
    input.cashFlowBasis === "OWNER_EARNINGS" &&
    input.ownerEarningsBasisSupported !== true
  ) {
    fail(
      "OWNER_EARNINGS_BASIS_UNSUPPORTED",
      "Owner Earnings basis must be supportable under the frozen forensic method",
    );
  }

  assertFiniteNumber(input.discountRate, "INVALID_DISCOUNT_RATE", "discountRate");
  if (input.discountRate <= -1) {
    fail("INVALID_DISCOUNT_RATE", "discountRate must be greater than -1");
  }

  if (input.cashFlows.length === 0) {
    fail("CASH_FLOW_SCHEDULE_EMPTY", "at least one forecast cash flow is required");
  }

  const yearFractions: number[] = [];
  const pvCashFlows: number[] = [];
  let previousPaymentDate: string | null = null;

  for (let index = 0; index < input.cashFlows.length; index += 1) {
    const flow = input.cashFlows[index];
    parseIsoCivilDate(flow.periodStartDate);
    parseIsoCivilDate(flow.paymentDate);
    assertFiniteNumber(flow.amount, "NONFINITE_CASH_FLOW", `cashFlows[${index}].amount`);

    if (flow.construction !== "POST_ORIGIN_ONLY") {
      fail(
        "STUB_FULL_YEAR_AUTOPRORATION_FORBIDDEN",
        "cash flow engine may not auto-prorate a full-period amount",
      );
    }

    if (index === 0) {
      if (flow.periodStartDate !== input.valuationDate) {
        fail(
          "STUB_CASH_FLOW_UNSUPPORTED",
          "first forecast interval must begin at VALUATION_DATE and contain post-origin cash flow only",
        );
      }
    } else if (flow.periodStartDate !== previousPaymentDate) {
      fail(
        "FORECAST_PERIOD_CHAIN_BROKEN",
        "each forecast interval must begin on the preceding payment date",
      );
    }

    if (compareDates(flow.paymentDate, input.valuationDate) <= 0) {
      fail(
        "CASH_FLOW_DATE_NOT_AFTER_VALUATION_DATE",
        "forecast payment dates must be strictly after VALUATION_DATE",
      );
    }

    if (
      previousPaymentDate !== null &&
      compareDates(flow.paymentDate, previousPaymentDate) <= 0
    ) {
      fail(
        "CASH_FLOW_DATES_NOT_STRICTLY_INCREASING",
        "forecast payment dates must be strictly increasing",
      );
    }

    const t = actual365FixedYearFraction(input.valuationDate, flow.paymentDate);
    const discounted = flow.amount / (1 + input.discountRate) ** t;
    if (!Number.isFinite(discounted)) {
      fail("NONFINITE_PRESENT_VALUE", "discounted cash flow is non-finite");
    }

    yearFractions.push(t);
    pvCashFlows.push(discounted);
    previousPaymentDate = flow.paymentDate;
  }

  const finalPaymentDate = input.cashFlows[input.cashFlows.length - 1].paymentDate;
  if (input.terminal.date !== finalPaymentDate) {
    fail(
      "TERMINAL_VALUE_DATE_MISMATCH",
      "terminal value must be valued at the final explicit/fade cash-flow date",
    );
  }
  if (input.terminal.cashFlowBasis !== input.cashFlowBasis) {
    fail("TERMINAL_BASIS_MISMATCH", "terminal cash-flow basis must match DCF basis");
  }
  if (!input.terminal.stableEconomicsResolved) {
    fail(
      "TERMINAL_STABLE_ECONOMICS_UNRESOLVED",
      "terminal stable economics must be resolved",
    );
  }

  assertFiniteNumber(input.terminal.growthRate, "INVALID_TERMINAL_GROWTH", "growthRate");
  assertFiniteNumber(
    input.terminal.nextPeriodCashFlow,
    "INVALID_TERMINAL_CASH_FLOW",
    "nextPeriodCashFlow",
  );
  if (input.discountRate <= input.terminal.growthRate) {
    fail("TERMINAL_G_NOT_LESS_THAN_R", "perpetuity-growth terminal value requires r > g");
  }

  const terminalValue =
    input.terminal.nextPeriodCashFlow /
    (input.discountRate - input.terminal.growthRate);
  const finalExponent = yearFractions[yearFractions.length - 1];
  const pvTerminalValue = terminalValue / (1 + input.discountRate) ** finalExponent;
  if (!Number.isFinite(terminalValue) || !Number.isFinite(pvTerminalValue)) {
    fail("NONFINITE_TERMINAL_VALUE", "terminal value is non-finite");
  }

  const dcfPresentValue =
    pvCashFlows.reduce((sum, value) => sum + value, 0) + pvTerminalValue;

  let enterpriseValue: number | null = null;
  let equityValue: number;

  if (input.cashFlowBasis === "FCFF") {
    if (input.evToEquityBridge === null) {
      fail("EV_BRIDGE_REQUIRED_FOR_FCFF", "FCFF enterprise DCF requires EV-to-equity bridge");
    }
    if (input.evToEquityBridge.date !== input.valuationDate) {
      fail("EV_BRIDGE_DATE_MISMATCH", "EV-to-equity bridge must be at VALUATION_DATE");
    }
    if (!input.evToEquityBridge.rollforwardResolved) {
      fail(
        "EV_BRIDGE_ROLLFORWARD_UNRESOLVED",
        "material bridge date mismatch must be reconciled to VALUATION_DATE",
      );
    }
    assertFiniteNumber(
      input.evToEquityBridge.netEquityBridgeAdjustment,
      "NONFINITE_EV_BRIDGE",
      "netEquityBridgeAdjustment",
    );
    enterpriseValue = dcfPresentValue;
    equityValue =
      enterpriseValue + input.evToEquityBridge.netEquityBridgeAdjustment;
  } else {
    if (input.evToEquityBridge !== null) {
      fail(
        "EQUITY_DCF_EV_BRIDGE_FORBIDDEN",
        "FCFE/Owner Earnings DCF is already an equity-value basis",
      );
    }
    equityValue = dcfPresentValue;
  }

  if (input.economicShareCount.date !== input.valuationDate) {
    fail(
      "PER_SHARE_DATE_MISMATCH",
      "economic share count must be measured at VALUATION_DATE",
    );
  }
  assertFiniteNumber(
    input.economicShareCount.count,
    "ECONOMIC_SHARE_COUNT_UNRESOLVED",
    "economicShareCount.count",
  );
  if (input.economicShareCount.count <= 0) {
    fail(
      "NONPOSITIVE_ECONOMIC_SHARE_COUNT",
      "economic share count must be positive",
    );
  }
  if (!input.economicShareCount.dilutionDoubleCountResolved) {
    fail(
      "DILUTION_DOUBLE_COUNT_UNRESOLVED",
      "dilution treatment must be reconciled before per-share valuation",
    );
  }

  return {
    timeOrigin: input.valuationDate,
    valuationDate: input.valuationDate,
    yearFractionConvention: "ACT/365F",
    yearFractions,
    discountExponents: [...yearFractions],
    pvCashFlows,
    terminalValue,
    pvTerminalValue,
    enterpriseValue,
    equityValue,
    perShareValue: equityValue / input.economicShareCount.count,
  };
}

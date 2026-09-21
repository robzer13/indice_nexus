import assert from "node:assert/strict";
import test from "node:test";

import {
  computeDcfTiming,
  type DcfTimingInput,
  DcfTimingError,
} from "../lib/orotitan-equity/v3/dcf-timing";
import { independentDcfTimingOracle } from "./dcf-timing-independent-oracle";

const relTol = (expected: number) => 1e-12 * Math.max(1, Math.abs(expected));

function close(actual: number, expected: number, label: string): void {
  assert.ok(
    Math.abs(actual - expected) <= relTol(expected),
    `${label}: expected ${expected}, got ${actual}`,
  );
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function r1(): DcfTimingInput {
  return {
    dataCutoff: "2026-09-19",
    timeOrigin: "2026-09-19",
    valuationDate: "2026-09-19",
    calculationDate: "2026-09-20",
    hasNumericReferencePrice: true,
    referencePriceDate: "2026-09-18",
    referencePriceStalenessResolved: true,
    periodTiming: "END_OF_PERIOD",
    yearFractionConvention: "ACT/365F",
    fiscalCalendarResolved: true,
    fiscalYearMismatchResolved: true,
    historicalOutputCalibration: false,
    cashFlowBasis: "FCFF",
    discountRateBasis: "WACC",
    discountRate: 0.1,
    cashFlows: [
      {
        periodStartDate: "2026-09-19",
        paymentDate: "2026-12-31",
        amount: 100,
        construction: "POST_ORIGIN_ONLY",
      },
      {
        periodStartDate: "2026-12-31",
        paymentDate: "2027-12-31",
        amount: 120,
        construction: "POST_ORIGIN_ONLY",
      },
    ],
    terminal: {
      date: "2027-12-31",
      growthRate: 0.03,
      nextPeriodCashFlow: 123.6,
      stableEconomicsResolved: true,
      cashFlowBasis: "FCFF",
    },
    evToEquityBridge: {
      date: "2026-09-19",
      netEquityBridgeAdjustment: -150,
      rollforwardResolved: true,
    },
    economicShareCount: {
      date: "2026-09-19",
      count: 10,
      dilutionDoubleCountResolved: true,
    },
  };
}

function r2(): DcfTimingInput {
  return {
    dataCutoff: "2027-01-15",
    timeOrigin: "2027-01-15",
    valuationDate: "2027-01-15",
    calculationDate: "2027-01-16",
    hasNumericReferencePrice: true,
    referencePriceDate: "2027-01-15",
    referencePriceStalenessResolved: true,
    periodTiming: "END_OF_PERIOD",
    yearFractionConvention: "ACT/365F",
    fiscalCalendarResolved: true,
    fiscalYearMismatchResolved: true,
    historicalOutputCalibration: false,
    cashFlowBasis: "FCFE",
    discountRateBasis: "COST_OF_EQUITY",
    discountRate: 0.12,
    cashFlows: [
      {
        periodStartDate: "2027-01-15",
        paymentDate: "2027-06-30",
        amount: 40,
        construction: "POST_ORIGIN_ONLY",
      },
      {
        periodStartDate: "2027-06-30",
        paymentDate: "2028-06-30",
        amount: 55,
        construction: "POST_ORIGIN_ONLY",
      },
    ],
    terminal: {
      date: "2028-06-30",
      growthRate: 0.02,
      nextPeriodCashFlow: 56.1,
      stableEconomicsResolved: true,
      cashFlowBasis: "FCFE",
    },
    evToEquityBridge: null,
    economicShareCount: {
      date: "2027-01-15",
      count: 5,
      dilutionDoubleCountResolved: true,
    },
  };
}

function r3(): DcfTimingInput {
  return {
    dataCutoff: "2027-12-31",
    timeOrigin: "2027-12-31",
    valuationDate: "2027-12-31",
    calculationDate: "2028-01-02",
    hasNumericReferencePrice: false,
    referencePriceDate: null,
    referencePriceStalenessResolved: true,
    periodTiming: "END_OF_PERIOD",
    yearFractionConvention: "ACT/365F",
    fiscalCalendarResolved: true,
    fiscalYearMismatchResolved: true,
    historicalOutputCalibration: false,
    cashFlowBasis: "OWNER_EARNINGS",
    ownerEarningsBasisSupported: true,
    discountRateBasis: "COST_OF_EQUITY",
    discountRate: 0.09,
    cashFlows: [
      {
        periodStartDate: "2027-12-31",
        paymentDate: "2028-12-31",
        amount: 30,
        construction: "POST_ORIGIN_ONLY",
      },
    ],
    terminal: {
      date: "2028-12-31",
      growthRate: 0.025,
      nextPeriodCashFlow: 30.75,
      stableEconomicsResolved: true,
      cashFlowBasis: "OWNER_EARNINGS",
    },
    evToEquityBridge: null,
    economicShareCount: {
      date: "2027-12-31",
      count: 1,
      dilutionDoubleCountResolved: true,
    },
  };
}

function expectCode(input: DcfTimingInput, code: string): void {
  assert.throws(
    () => computeDcfTiming(input),
    (error: unknown) =>
      error instanceof DcfTimingError && error.code === code,
    code,
  );
}

test("R1 FCFF reference vector is exact within frozen tolerance", () => {
  const out = computeDcfTiming(r1());
  close(out.yearFractions[0], 0.2821917808219178, "R1 t1");
  close(out.yearFractions[1], 1.2821917808219179, "R1 t2");
  close(out.pvCashFlows[0], 97.3462720337052, "R1 PV CF1");
  close(out.pvCashFlows[1], 106.19593312767839, "R1 PV CF2");
  close(out.terminalValue, 1765.7142857142858, "R1 TV");
  close(out.pvTerminalValue, 1562.5973017358392, "R1 PV TV");
  close(out.enterpriseValue ?? Number.NaN, 1766.1395068972229, "R1 EV");
  close(out.equityValue, 1616.1395068972229, "R1 equity");
  close(out.perShareValue, 161.6139506897223, "R1 per share");
});

test("R2 FCFE reference vector is exact within frozen tolerance", () => {
  const out = computeDcfTiming(r2());
  close(out.yearFractions[0], 0.4547945205479452, "R2 t1");
  close(out.yearFractions[1], 1.4575342465753425, "R2 t2");
  close(out.pvCashFlows[0], 37.99057828118631, "R2 PV CF1");
  close(out.pvCashFlows[1], 46.62573981939283, "R2 PV CF2");
  close(out.terminalValue, 561.0000000000001, "R2 TV");
  close(out.pvTerminalValue, 475.582546157807, "R2 PV TV");
  close(out.equityValue, 560.1988642583862, "R2 equity");
  close(out.perShareValue, 112.03977285167723, "R2 per share");
});

test("R3 Owner Earnings uses ACT/365F across leap-year numerator", () => {
  const out = computeDcfTiming(r3());
  close(out.yearFractions[0], 1.0027397260273974, "R3 t1");
  close(out.pvCashFlows[0], 27.5164382915453, "R3 PV CF1");
  close(out.terminalValue, 473.076923076923, "R3 TV");
  close(out.pvTerminalValue, 433.91306536667577, "R3 PV TV");
  close(out.equityValue, 461.42950365822105, "R3 equity");
});

test("independent implementation equivalence on all reference vectors", () => {
  for (const [name, input] of [["R1", r1()], ["R2", r2()], ["R3", r3()]] as const) {
    const a = computeDcfTiming(input);
    const b = independentDcfTimingOracle(input);
    assert.equal(a.timeOrigin, b.timeOrigin, name);
    assert.equal(a.enterpriseValue, b.enterpriseValue, name);
    for (let i = 0; i < a.yearFractions.length; i += 1) {
      close(a.yearFractions[i], b.yearFractions[i], `${name} exponent ${i}`);
      close(a.pvCashFlows[i], b.pvCashFlows[i], `${name} PV CF ${i}`);
    }
    close(a.terminalValue, b.terminalValue, `${name} TV`);
    close(a.pvTerminalValue, b.pvTerminalValue, `${name} PV TV`);
    close(a.equityValue, b.equityValue, `${name} equity`);
    close(a.perShareValue, b.perShareValue, `${name} per share`);
  }
});

test("calculation date never changes DCF timing or value", () => {
  const a = r1();
  const b = clone(a);
  b.calculationDate = "2031-04-17";
  assert.deepEqual(computeDcfTiming(a), computeDcfTiming(b));
});

test("adversarial timing and basis failures fail closed", () => {
  {
    const x = r1();
    x.timeOrigin = "2026-09-18";
    expectCode(x, "DCF_TIME_ORIGIN_MISMATCH");
  }
  {
    const x = r1();
    x.valuationDate = "2026-09-18";
    x.timeOrigin = "2026-09-18";
    expectCode(x, "VALUATION_DATE_DATA_CUTOFF_MISMATCH");
  }
  {
    const x = r1();
    x.referencePriceDate = "2026-09-20";
    expectCode(x, "REFERENCE_PRICE_AFTER_VALUATION_DATE");
  }
  {
    const x = r1();
    x.referencePriceDate = null;
    expectCode(x, "REFERENCE_PRICE_DATE_MISSING");
  }
  {
    const x = r1();
    x.referencePriceStalenessResolved = false;
    expectCode(x, "REFERENCE_PRICE_STALENESS_UNRESOLVED");
  }
  {
    const x = r1();
    x.cashFlows[0].paymentDate = "2026-09-19";
    expectCode(x, "CASH_FLOW_DATE_NOT_AFTER_VALUATION_DATE");
  }
  {
    const x = r1();
    x.cashFlows[1].periodStartDate = "2026-12-31";
    x.cashFlows[1].paymentDate = "2026-12-31";
    expectCode(x, "CASH_FLOW_DATES_NOT_STRICTLY_INCREASING");
  }
  {
    const x = r1();
    x.cashFlows[1].periodStartDate = "2027-01-01";
    expectCode(x, "FORECAST_PERIOD_CHAIN_BROKEN");
  }
  {
    const x = r1();
    x.cashFlows[0].periodStartDate = "2026-01-01";
    expectCode(x, "STUB_CASH_FLOW_UNSUPPORTED");
  }
  {
    const x = r1();
    x.cashFlows[0].construction = "FULL_PERIOD_AUTOPRORATED";
    expectCode(x, "STUB_FULL_YEAR_AUTOPRORATION_FORBIDDEN");
  }
  {
    const x = r1();
    x.discountRateBasis = "COST_OF_EQUITY";
    expectCode(x, "FCFF_DISCOUNT_RATE_BASIS_MISMATCH");
  }
  {
    const x = r2();
    x.discountRateBasis = "WACC";
    expectCode(x, "FCFE_DISCOUNT_RATE_BASIS_MISMATCH");
  }
  {
    const x = r3();
    x.discountRateBasis = "WACC";
    expectCode(x, "OWNER_EARNINGS_DISCOUNT_RATE_BASIS_MISMATCH");
  }
  {
    const x = r2();
    x.evToEquityBridge = {
      date: x.valuationDate,
      netEquityBridgeAdjustment: 0,
      rollforwardResolved: true,
    };
    expectCode(x, "EQUITY_DCF_EV_BRIDGE_FORBIDDEN");
  }
  {
    const x = r1();
    x.evToEquityBridge = null;
    expectCode(x, "EV_BRIDGE_REQUIRED_FOR_FCFF");
  }
  {
    const x = r1();
    x.terminal.date = "2028-01-01";
    expectCode(x, "TERMINAL_VALUE_DATE_MISMATCH");
  }
  {
    const x = r1();
    x.terminal.growthRate = x.discountRate;
    expectCode(x, "TERMINAL_G_NOT_LESS_THAN_R");
  }
  {
    const x = r1();
    assert.ok(x.evToEquityBridge);
    x.evToEquityBridge.date = "2026-09-18";
    expectCode(x, "EV_BRIDGE_DATE_MISMATCH");
  }
  {
    const x = r1();
    x.economicShareCount.date = "2026-09-18";
    expectCode(x, "PER_SHARE_DATE_MISMATCH");
  }
  {
    const x = r1();
    x.economicShareCount.count = 0;
    expectCode(x, "NONPOSITIVE_ECONOMIC_SHARE_COUNT");
  }
  {
    const x = r1();
    x.fiscalCalendarResolved = false;
    expectCode(x, "FISCAL_PERIOD_DATE_UNRESOLVED");
  }
  {
    const x = r1();
    x.fiscalYearMismatchResolved = false;
    expectCode(x, "FISCAL_YEAR_MISMATCH_UNRESOLVED");
  }
  {
    const x = r1();
    x.historicalOutputCalibration = true;
    expectCode(x, "HISTORICAL_OUTPUT_CALIBRATION_FORBIDDEN");
  }
  {
    const x = r1();
    x.periodTiming = "MIDPOINT";
    expectCode(x, "UNSUPPORTED_PERIOD_TIMING");
  }
  {
    const x = r1();
    x.yearFractionConvention = "ACT/ACT";
    expectCode(x, "UNSUPPORTED_YEAR_FRACTION_CONVENTION");
  }
  {
    const x = r1();
    assert.ok(x.evToEquityBridge);
    x.evToEquityBridge.rollforwardResolved = false;
    expectCode(x, "EV_BRIDGE_ROLLFORWARD_UNRESOLVED");
  }
  {
    const x = r3();
    x.ownerEarningsBasisSupported = false;
    expectCode(x, "OWNER_EARNINGS_BASIS_UNSUPPORTED");
  }
  {
    const x = r1();
    x.terminal.cashFlowBasis = "FCFE";
    expectCode(x, "TERMINAL_BASIS_MISMATCH");
  }
  {
    const x = r1();
    x.terminal.stableEconomicsResolved = false;
    expectCode(x, "TERMINAL_STABLE_ECONOMICS_UNRESOLVED");
  }
  {
    const x = r1();
    x.economicShareCount.dilutionDoubleCountResolved = false;
    expectCode(x, "DILUTION_DOUBLE_COUNT_UNRESOLVED");
  }
});

test("Owner Earnings support must be explicitly affirmative", () => {
  const x = r3();
  delete x.ownerEarningsBasisSupported;
  expectCode(x, "OWNER_EARNINGS_BASIS_UNSUPPORTED");
});

// DCF timing verification gate: the active Contract Set metadata is frozen only after this suite is green.

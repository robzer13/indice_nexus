import assert from "node:assert/strict";

import type {
  DcfTimingInput,
  DcfTimingOutput,
} from "../lib/orotitan-equity/v3/dcf-timing";

// Independent implementation B for regression only.
// It deliberately does not use Date, actual365FixedYearFraction, or computeDcfTiming.
function civilOrdinal(iso: string): number {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  assert.ok(match, `invalid oracle date ${iso}`);
  let year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  assert.ok(month >= 1 && month <= 12);
  assert.ok(day >= 1 && day <= 31);

  year -= month <= 2 ? 1 : 0;
  const era = Math.floor(year / 400);
  const yoe = year - era * 400;
  const mp = month + (month > 2 ? -3 : 9);
  const doy = Math.floor((153 * mp + 2) / 5) + day - 1;
  const doe =
    yoe * 365 +
    Math.floor(yoe / 4) -
    Math.floor(yoe / 100) +
    doy;
  return era * 146097 + doe;
}

export function independentDcfTimingOracle(
  input: DcfTimingInput,
): DcfTimingOutput {
  assert.equal(input.dataCutoff, input.valuationDate);
  assert.equal(input.timeOrigin, input.valuationDate);
  assert.equal(input.periodTiming, "END_OF_PERIOD");
  assert.equal(input.yearFractionConvention, "ACT/365F");
  assert.equal(input.fiscalCalendarResolved, true);
  assert.equal(input.fiscalYearMismatchResolved, true);
  const fiscalStart = dayNumber(input.firstForecastFiscalPeriodStartDate);
  const valuationDay = dayNumber(input.valuationDate);
  const firstPaymentDay = dayNumber(input.cashFlows[0].paymentDate);
  assert.ok(fiscalStart <= valuationDay);
  assert.ok(fiscalStart < firstPaymentDay);
  const stubStatus = fiscalStart === valuationDay ? "FULL_PERIOD" : "STUB";
  if (input.hasNumericReferencePrice) assert.equal(input.referencePriceStalenessResolved, true);
  assert.equal(input.historicalOutputCalibration, false);

  const requiredRateBasis =
    input.cashFlowBasis === "FCFF" ? "WACC" : "COST_OF_EQUITY";
  assert.equal(input.discountRateBasis, requiredRateBasis);

  const origin = civilOrdinal(input.valuationDate);
  const yearFractions = input.cashFlows.map(
    (flow) => (civilOrdinal(flow.paymentDate) - origin) / 365,
  );
  const pvCashFlows = input.cashFlows.map(
    (flow, index) =>
      flow.amount /
      Math.pow(1 + input.discountRate, yearFractions[index]),
  );

  const terminalValue =
    input.terminal.nextPeriodCashFlow /
    (input.discountRate - input.terminal.growthRate);
  const pvTerminalValue =
    terminalValue /
    Math.pow(
      1 + input.discountRate,
      yearFractions[yearFractions.length - 1],
    );
  const dcfPresentValue =
    pvCashFlows.reduce((sum, value) => sum + value, 0) + pvTerminalValue;

  const enterpriseValue =
    input.cashFlowBasis === "FCFF" ? dcfPresentValue : null;
  const equityValue =
    input.cashFlowBasis === "FCFF"
      ? dcfPresentValue +
        (input.evToEquityBridge?.netEquityBridgeAdjustment ?? Number.NaN)
      : dcfPresentValue;

  return {
    timeOrigin: input.timeOrigin,
    valuationDate: input.valuationDate,
    yearFractionConvention: "ACT/365F",
    stubStatus,
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

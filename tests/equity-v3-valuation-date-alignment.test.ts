import assert from "node:assert/strict";
import test from "node:test";

import {
  resolveValuationDateAlignment,
  ValuationDateAlignmentError,
  type ValuationDateAlignmentInput,
} from "../lib/orotitan-equity/v3/valuation-date-alignment";

function base(): ValuationDateAlignmentInput {
  return {
    dataCutoff: "2026-09-22",
    valuationDate: "2026-09-22",
    timeOrigin: "2026-09-22",
    hasNumericReferencePrice: true,
    referencePriceDate: "2026-09-22",
    referencePriceStalenessResolved: true,
    economicShareCountDate: "2026-09-22",
    shareCountAsOfDate: "2026-09-22",
    economicShareCountRepresentation: "EXACT",
    denominatorBridgeResolved: true,
    cashFlowBasis: "FCFF",
    evToEquityBridgeDate: "2026-09-22",
    evBridgeRollforwardResolved: true,
    perShareOutputDate: "2026-09-22",
    marketCapShareCountDate: "2026-09-22",
    marketCapShareCountResolved: true,
  };
}

function expectCode(input: ValuationDateAlignmentInput, code: string): void {
  assert.throws(
    () => resolveValuationDateAlignment(input),
    (error: unknown) => error instanceof ValuationDateAlignmentError && error.code === code,
    code,
  );
}

test("reference price date equal to cutoff is fully aligned", () => {
  const out = resolveValuationDateAlignment(base());
  assert.equal(out.valuationDate, "2026-09-22");
  assert.equal(out.shareCountAsOfDate, "2026-09-22");
  assert.equal(out.marketCapDate, "2026-09-22");
  assert.equal(out.valuationAdmission, true);
});

test("reference price before cutoff does not move valuation or share-count date", () => {
  const x = base();
  x.referencePriceDate = "2026-09-16";
  x.marketCapShareCountDate = "2026-09-16";
  const out = resolveValuationDateAlignment(x);
  assert.equal(out.valuationDate, "2026-09-22");
  assert.equal(out.timeOrigin, "2026-09-22");
  assert.equal(out.economicShareCountDate, "2026-09-22");
  assert.equal(out.shareCountAsOfDate, "2026-09-22");
  assert.equal(out.perShareOutputDate, "2026-09-22");
  assert.equal(out.marketCapDate, "2026-09-16");
});

test("non-trading-day cutoff admits last prior market observation without moving valuation date", () => {
  const x = base();
  x.dataCutoff = "2026-09-20";
  x.valuationDate = "2026-09-20";
  x.timeOrigin = "2026-09-20";
  x.economicShareCountDate = "2026-09-20";
  x.shareCountAsOfDate = "2026-09-20";
  x.perShareOutputDate = "2026-09-20";
  x.evToEquityBridgeDate = "2026-09-20";
  x.referencePriceDate = "2026-09-18";
  x.marketCapShareCountDate = "2026-09-18";
  const out = resolveValuationDateAlignment(x);
  assert.equal(out.valuationDate, "2026-09-20");
  assert.equal(out.marketCapDate, "2026-09-18");
});

test("rigorous bounded denominator at valuation date remains admitted", () => {
  const x = base();
  x.economicShareCountRepresentation = "BOUNDED";
  const out = resolveValuationDateAlignment(x);
  assert.equal(out.denominatorAdmitted, true);
  assert.equal(out.valuationAdmission, true);
});

test("unknown denominator fails valuation closed without inventing a scalar", () => {
  const x = base();
  x.economicShareCountRepresentation = "UNKNOWN";
  const out = resolveValuationDateAlignment(x);
  assert.equal(out.denominatorAdmitted, false);
  assert.equal(out.valuationAdmission, false);
  assert.deepEqual(out.blockers, ["ECONOMIC_SHARE_COUNT_UNRESOLVED"]);
});

test("earlier denominator cannot be transported silently to valuation date", () => {
  const x = base();
  x.economicShareCountDate = "2026-09-16";
  x.shareCountAsOfDate = "2026-09-16";
  expectCode(x, "ECONOMIC_SHARE_COUNT_DATE_MISMATCH");
});

test("valuation date cannot diverge from cutoff", () => {
  const x = base();
  x.valuationDate = "2026-09-16";
  x.timeOrigin = "2026-09-16";
  expectCode(x, "VALUATION_DATE_DATA_CUTOFF_MISMATCH");
});

test("FCFF EV bridge date mismatch fails closed", () => {
  const x = base();
  x.evToEquityBridgeDate = "2026-06-30";
  expectCode(x, "EV_BRIDGE_DATE_MISMATCH");
});

test("unresolved FCFF EV rollforward blocks valuation admission", () => {
  const x = base();
  x.evBridgeRollforwardResolved = false;
  const out = resolveValuationDateAlignment(x);
  assert.equal(out.valuationAdmission, false);
  assert.ok(out.blockers.includes("EV_BRIDGE_ROLLFORWARD_UNRESOLVED"));
});

test("per-share output date mismatch fails closed", () => {
  const x = base();
  x.perShareOutputDate = "2026-09-16";
  expectCode(x, "PER_SHARE_DATE_MISMATCH");
});

test("market cap requires share count at reference-price date", () => {
  const x = base();
  x.referencePriceDate = "2026-09-16";
  x.marketCapShareCountDate = "2026-09-22";
  expectCode(x, "MARKET_CAP_DENOMINATOR_DATE_MISMATCH");
});

test("one input cannot produce two incompatible valuation/share-count dates", () => {
  const x = base();
  x.referencePriceDate = "2026-09-16";
  x.marketCapShareCountDate = "2026-09-16";
  const out = resolveValuationDateAlignment(x);
  assert.equal(out.valuationDate, x.dataCutoff);
  assert.equal(out.economicShareCountDate, out.valuationDate);
  assert.equal(out.shareCountAsOfDate, out.valuationDate);
  assert.notEqual(out.marketCapDate, out.valuationDate);
});

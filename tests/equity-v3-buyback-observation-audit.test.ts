import assert from "node:assert/strict";
import test from "node:test";
import {
  auditUnattestedBuybackObservations,
  BuybackObservationAuditError,
  type BuybackObservationBatch,
} from "../lib/orotitan-equity/v3/buyback-observation-audit";

function sample(): BuybackObservationBatch {
  return {
    dataCutoff: "2026-04-30",
    notices: [
      {
        noticeId: "release-a",
        publicationDate: "2026-04-06",
        sourceUrl: "https://issuer.example.org/a",
        trades: [
          { tradeId: "a-1", tradeDate: "2026-04-02", sharesPurchased: 120 },
          { tradeId: "a-2", tradeDate: "2026-04-03", sharesPurchased: 80 },
        ],
      },
      {
        noticeId: "release-b",
        publicationDate: "2026-04-13",
        sourceUrl: "https://issuer.example.org/b",
        trades: [{ tradeId: "b-1", tradeDate: "2026-04-10", sharesPurchased: 40 }],
      },
    ],
  };
}

function expectCode(input: BuybackObservationBatch, code: string): void {
  assert.throws(
    () => auditUnattestedBuybackObservations(input),
    (error: unknown) => error instanceof BuybackObservationAuditError && error.code === code,
    code,
  );
}

test("synthetic issuer notice arithmetic stays explicitly non-authoritative", () => {
  const result = auditUnattestedBuybackObservations(sample());
  assert.equal(result.status, "OBSERVED_UNATTESTED");
  assert.equal(result.observedPurchasedShares, 240);
  assert.equal(result.tradeRowCount, 3);
  assert.equal(result.noticeCount, 2);
  assert.equal(result.firstTradeDate, "2026-04-02");
  assert.equal(result.lastTradeDate, "2026-04-10");
  assert.equal(result.sourceBytesAdmitted, false);
  assert.equal(result.movementClassCompleteness, "NOT_EVALUATED");
  assert.equal(result.economicShareCount, "UNKNOWN");
  assert.equal(result.denominatorValuationAdmission, "NO");
});

test("post-cutoff publication is rejected even for older trades", () => {
  const batch = sample();
  batch.notices[0].publicationDate = "2026-05-01";
  expectCode(batch, "POST_CUTOFF_NOTICE");
});

test("cutoff-day publication and trading observations remain unadmitted", () => {
  const batch = sample();
  batch.notices[1].publicationDate = batch.dataCutoff;
  batch.notices[1].trades[0].tradeDate = batch.dataCutoff;
  const result = auditUnattestedBuybackObservations(batch);
  assert.equal(result.lastTradeDate, "2026-04-30");
  assert.equal(result.denominatorValuationAdmission, "NO");
});

test("trade dated after its notice is rejected", () => {
  const batch = sample();
  batch.notices[0].trades[0].tradeDate = "2026-04-07";
  expectCode(batch, "TRADE_AFTER_NOTICE");
});

test("duplicate trade IDs and notice IDs are rejected", () => {
  const batch = sample();
  batch.notices[1].trades[0].tradeId = "a-2";
  expectCode(batch, "DUPLICATE_TRADE");
  const other = sample();
  other.notices[1].noticeId = "release-a";
  expectCode(other, "DUPLICATE_NOTICE");
});

test("impossible civil dates are rejected", () => {
  const batch = sample();
  batch.notices[0].publicationDate = "2026-02-30";
  expectCode(batch, "INVALID_DATE");
});

test("invalid numbers and sum overflow are rejected", () => {
  for (const quantity of [0, -1, 0.25, Infinity, Number.NaN, Number.MAX_SAFE_INTEGER + 1]) {
    const batch = sample();
    batch.notices[0].trades[0].sharesPurchased = quantity;
    expectCode(batch, "INVALID_SHARE_QUANTITY");
  }
  const overflow = sample();
  overflow.notices[0].trades[0].sharesPurchased = Number.MAX_SAFE_INTEGER;
  expectCode(overflow, "UNSAFE_SHARE_TOTAL");
});

test("insecure or malformed source locators cannot pass syntax checks", () => {
  for (const sourceUrl of ["", "http://issuer.example.org/a", "https://u:p@issuer.example.org/a"]) {
    const batch = sample();
    batch.notices[0].sourceUrl = sourceUrl;
    expectCode(batch, "INVALID_SOURCE_URL");
  }
});

test("empty lists do not silently become zero-buyback evidence", () => {
  const batch = sample();
  batch.notices = [];
  expectCode(batch, "NO_NOTICES");
  const other = sample();
  other.notices[0].trades = [];
  expectCode(other, "EMPTY_NOTICE");
});

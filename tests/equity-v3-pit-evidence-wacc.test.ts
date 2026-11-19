import assert from "node:assert/strict";
import test from "node:test";
import {
  computeDiagnosticWacc, preflightPitEvidence, PitEvidenceWaccError,
  transcriptSha256, type PitEvidenceClaim, type WaccDiagnosticInput,
} from "../lib/orotitan-equity/v3/valuation-pit-evidence-wacc";

function claim(): PitEvidenceClaim {
  return {
    id: "GENERIC_FACT_001", sourceUrl: "https://issuer.example.org/report",
    sourceLocator: "Page 17 Note 4.1", economicDate: "2026-10-20",
    publishedAt: "2026-10-22", classification: "OBSERVED",
    value: "123.45", unit: "EUR_M",
  };
}
function baseEvidence(c: PitEvidenceClaim = claim()) {
  return {
    cutoff: "2026-10-30", claims: [c],
    trustedTranscriptSha256: { [c.id]: transcriptSha256(c) },
    allowedHosts: ["issuer.example.org"],
  };
}
function expectError(action: () => unknown, code: string) {
  assert.throws(action, (err: unknown) =>
    err instanceof PitEvidenceWaccError && err.code === code);
}
function baseWacc(): WaccDiagnosticInput {
  return {
    unleveredBeta: 1.25, taxRate: .235, equityWeight: .90, debtWeight: .10,
    riskFreeRatePct: 3.3185, equityRiskPremiumPct: 3.70, marginalDebtCostPct: 4.3,
    cashFlowCurrency: "EUR", riskFreeCurrency: "EUR",
    equityRiskPremiumCurrency: "USD", betaMode: "RELEVER",
  };
}

test("pinned transcript is accepted only as local transcript evidence", () => {
  const v = preflightPitEvidence(baseEvidence());
  assert.equal(v.evidenceCount, 1);
  assert.equal(v.originalSourceBytesVerified, false);
  assert.ok(v.blockers.includes("ORIGINAL_SOURCE_BYTES_NOT_ATTESTED"));
});
test("mutating numeric market evidence fails with unchanged trusted anchor", () => {
  const x = baseEvidence(); x.claims[0].value = "9.9999";
  expectError(() => preflightPitEvidence(x), "TRANSCRIPT_BYTES_CHANGED");
});
test("substituting a source URL fails even if attacker recomputes a local anchor", () => {
  const x = baseEvidence(); x.claims[0].sourceUrl = "https://evil.example.net/issuer";
  x.trustedTranscriptSha256 = { [x.claims[0].id]: transcriptSha256(x.claims[0]) };
  expectError(() => preflightPitEvidence(x), "SOURCE_ORIGIN_NOT_APPROVED");
});
test("a publication after cutoff fails even if transcript pin is regenerated", () => {
  const x = baseEvidence(); x.claims[0].publishedAt = "2026-11-02";
  x.trustedTranscriptSha256 = { [x.claims[0].id]: transcriptSha256(x.claims[0]) };
  expectError(() => preflightPitEvidence(x), "SOURCE_PUBLISHED_AFTER_CUTOFF");
});
test("invalid calendar dates cannot be lexically compared as real dates", () => {
  const x = baseEvidence(); x.claims[0].economicDate = "2026-02-30";
  x.trustedTranscriptSha256 = { [x.claims[0].id]: transcriptSha256(x.claims[0]) };
  expectError(() => preflightPitEvidence(x), "ECONOMIC_DATE_INVALID");
});
test("no undisclosed missing publication date", () => {
  const x = baseEvidence(); x.claims[0].publishedAt = null;
  x.trustedTranscriptSha256 = { [x.claims[0].id]: transcriptSha256(x.claims[0]) };
  expectError(() => preflightPitEvidence(x), "UNVERIFIED_PUBLICATION_UNDISCLOSED");
});
test("explicit uncertainty is retained as a blocker", () => {
  const x = baseEvidence(); x.claims[0].publishedAt = null;
  x.claims[0].classification = "HISTORICAL_PUBLICATION_UNPROVEN";
  x.trustedTranscriptSha256 = { [x.claims[0].id]: transcriptSha256(x.claims[0]) };
  assert.ok(preflightPitEvidence(x).blockers.includes("PUBLICATION_TIME_NOT_PROVEN"));
});
test("explicitly unproven publication stays blocked even with a pre-cutoff date", () => {
  const x = baseEvidence(); x.claims[0].classification = "HISTORICAL_PUBLICATION_UNPROVEN";
  x.trustedTranscriptSha256 = { [x.claims[0].id]: transcriptSha256(x.claims[0]) };
  const out = preflightPitEvidence(x);
  assert.equal(out.allPublicationDatesVerified, false);
  assert.ok(out.blockers.includes("PUBLICATION_TIME_NOT_PROVEN"));
});
test("unsettled financing must have a future settlement date", () => {
  const x = baseEvidence(); x.claims[0].classification = "ANNOUNCED_NOT_SETTLED";
  x.claims[0].scheduledSettlementDate = "2026-10-29";
  x.trustedTranscriptSha256 = { [x.claims[0].id]: transcriptSha256(x.claims[0]) };
  expectError(() => preflightPitEvidence(x), "FINANCING_WRONGLY_TREATED_AS_UNSETTLED");
});
test("valid pre-cutoff announcement is retained as not settled", () => {
  const x = baseEvidence(); x.claims[0].classification = "ANNOUNCED_NOT_SETTLED";
  x.claims[0].scheduledSettlementDate = "2026-11-02";
  x.trustedTranscriptSha256 = { [x.claims[0].id]: transcriptSha256(x.claims[0]) };
  assert.ok(preflightPitEvidence(x).blockers.includes("FINANCING_NOT_SETTLED_AT_CUTOFF"));
});
test("duplicate identities reject before hash check", () => {
  const x = baseEvidence(); x.claims.push({...x.claims[0]});
  expectError(() => preflightPitEvidence(x), "TRANSCRIPT_SET_MISMATCH");
});
test("unlevered beta is relevered for a nonzero debt share", () => {
  const out = computeDiagnosticWacc(baseWacc());
  assert.ok(Math.abs(out.leveredBeta - 1.35625) < 1e-9);
  assert.ok(Math.abs(out.waccPct - 7.831925) < .0001);
  assert.equal(out.certified, false);
  assert.ok(out.blockers.includes("DISCOUNT_RATE_CURRENCY_INCONSISTENT"));
});
test("explicit use of equity beta without relevering is rejected", () => {
  const x = baseWacc(); x.betaMode = "EQUITY_BETA";
  expectError(() => computeDiagnosticWacc(x), "UNLEVERED_BETA_RELEVER_REQUIRED");
});
test("nonfinite, unbalanced or negative cost assumptions reject", () => {
  const a = baseWacc(); a.riskFreeRatePct = NaN;
  expectError(() => computeDiagnosticWacc(a), "WACC_NONFINITE_INPUT");
  const b = baseWacc(); b.debtWeight = .2;
  expectError(() => computeDiagnosticWacc(b), "WACC_INPUT_DOMAIN_INVALID");
  const c = baseWacc(); c.marginalDebtCostPct = -1;
  expectError(() => computeDiagnosticWacc(c), "WACC_INPUT_DOMAIN_INVALID");
});
test("WACC arithmetic overflow rejects even when every input is finite", () => {
  const x = baseWacc(); x.equityRiskPremiumPct = Number.MAX_VALUE;
  expectError(() => computeDiagnosticWacc(x), "WACC_RESULT_NONFINITE");
});
test("currency alignment cannot grant certification or source authority", () => {
  const x = baseWacc(); x.equityRiskPremiumCurrency = "EUR";
  const out = computeDiagnosticWacc(x);
  assert.equal(out.blockers.includes("DISCOUNT_RATE_CURRENCY_INCONSISTENT"), false);
  assert.equal(out.certified, false);
  assert.ok(out.blockers.includes("CAPITAL_WEIGHTS_AND_COST_OF_DEBT_NOT_SOURCE_VERIFIED"));
});

import assert from "node:assert/strict";
import test from "node:test";
import {
  computeInvestmentScore, computeOqs, computeOroTitanStatus, computeOvs, computeReturnComponent,
  MOS_CAPS, scoreExpectedReturnDelta, scoreExpectedReturnRange, VALUATION_RELIABILITY_CAPS,
} from "../lib/orotitan-equity/v1";
import type { ScoreRange } from "../lib/orotitan-equity/v1";

const dimensions = (overrides = {}) => ({ MOAT: 80, RUNWAY: 80, RETURN_QUALITY: 80, CASH_ECONOMICS: 80,
  CAPITAL_ALLOCATION: 80, MANAGEMENT_GOVERNANCE: 80, RESILIENCE_RISK: 80, ...overrides });
const gates = (overrides = {}) => ({ researchFullyCertified: "PASS" as const, moatElite: "PASS" as const, runwayElite: "PASS" as const,
  returnQualityElite: "PASS" as const, cashEconomicsElite: "PASS" as const, capitalAllocationElite: "PASS" as const,
  managementGovernanceElite: "PASS" as const, resilienceElite: "PASS" as const, valuationElite: "PASS" as const,
  materialWeakLink: "PASS" as const, ...overrides });
const certifiedContext = { businessResearchStatus: "CERTIFIED" as const, investmentConclusionStatus: "CERTIFIED" as const,
  scorePermission: "ALLOWED" as const, valuationReliability: "HIGH" as const };
const close = (actual: number | ScoreRange, expected: number) => {
  assert.equal(typeof actual, "number");
  if (typeof actual === "number") assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} != ${expected}`);
};

test("C01 OQS raw uses the exact weights", () => close(computeOqs(dimensions({ MOAT: 100, RUNWAY: 90, RETURN_QUALITY: 80, CASH_ECONOMICS: 70, CAPITAL_ALLOCATION: 60, MANAGEMENT_GOVERNANCE: 50, RESILIENCE_RISK: 40 })).oqsRaw, 74.5));
test("C02 weak-link cap binds", () => assert.deepEqual(computeOqs(dimensions({ MOAT: 40 })), { oqsRaw: 72, weakLinkCap: 65, oqs: 65 }));
test("C03 weak-link cap does not bind", () => assert.equal(computeOqs(dimensions()).oqs, 80));
test("C04 OQS and weak-link cap max at 100", () => assert.deepEqual(computeOqs(dimensions({ MOAT: 100, RUNWAY: 100, RETURN_QUALITY: 100, CASH_ECONOMICS: 100, CAPITAL_ALLOCATION: 100, MANAGEMENT_GOVERNANCE: 100, RESILIENCE_RISK: 100 })), { oqsRaw: 100, weakLinkCap: 100, oqs: 100 }));
test("C04a scalar dimensions preserve scalar OQS", () => assert.equal(typeof computeOqs(dimensions()).oqs, "number"));
test("C04b one dimension range propagates exact OQS bounds", () => assert.deepEqual(computeOqs(dimensions({ MOAT: { min: 70, max: 80 } })), {
  oqsRaw: { min: 78, max: 80 }, weakLinkCap: { min: 95, max: 100 }, oqs: { min: 78, max: 80 },
}));
test("C04c multiple dimension ranges propagate all-lower/all-upper bounds", () => assert.deepEqual(computeOqs(dimensions({ MOAT: { min: 60, max: 80 }, RUNWAY: { min: 70, max: 90 } })), {
  oqsRaw: { min: 74.5, max: 81.5 }, weakLinkCap: { min: 85, max: 100 }, oqs: { min: 74.5, max: 81.5 },
}));

test("C08 expected-return exact anchors", () => {
  for (const [delta, score] of [[-10, 0], [-8, 10], [-6, 20], [-4, 35], [-2, 50], [0, 70], [2, 82], [4, 90], [6, 95], [8, 100]]) assert.equal(scoreExpectedReturnDelta(delta), score);
});
test("C09 expected-return midpoint interpolation", () => { assert.equal(scoreExpectedReturnDelta(-3), 42.5); assert.equal(scoreExpectedReturnDelta(1), 76); });
test("C10 expected-return lower saturation", () => assert.equal(scoreExpectedReturnDelta(-50), 0));
test("C11 expected-return upper saturation", () => assert.equal(scoreExpectedReturnDelta(50), 100));
test("C11a expected-return range preserves interpolation and saturation", () => {
  assert.deepEqual(scoreExpectedReturnRange({ min: -3, max: 50 }), { min: 42.5, max: 100 });
});
test("C12 return component normal case", () => assert.equal(computeReturnComponent(80, 70), 76));
test("C13 N+15 return cap binds", () => assert.equal(computeReturnComponent(100, 20), 35));
test("C13a return component range propagates normal bounds", () => assert.deepEqual(computeReturnComponent({ min: 80, max: 90 }, { min: 70, max: 80 }), { min: 76, max: 86 }));
test("C13b return component range preserves N+15 cap", () => assert.deepEqual(computeReturnComponent({ min: 100, max: 100 }, { min: 20, max: 30 }), { min: 35, max: 45 }));

test("C14 MOS ROBUST cap", () => assert.equal(MOS_CAPS.ROBUST, 100));
test("C15 MOS ADEQUATE cap", () => assert.equal(MOS_CAPS.ADEQUATE, 90));
test("C16 MOS THIN cap", () => assert.equal(MOS_CAPS.THIN, 75));
test("C17 MOS NONE cap", () => assert.equal(MOS_CAPS.NONE, 55));
test("C18 valuation HIGH cap", () => assert.equal(VALUATION_RELIABILITY_CAPS.HIGH, 100));
test("C19 valuation MEDIUM cap", () => assert.equal(VALUATION_RELIABILITY_CAPS.MEDIUM, 95));
test("C20 valuation LOW cap", () => assert.equal(VALUATION_RELIABILITY_CAPS.LOW, 80));

const ovs = (overrides = {}) => computeOvs({ returnComponent: 88, mosStatus: "ROBUST", valuationReliability: "HIGH", scorePermission: "ALLOWED", investmentConclusionStatus: "CERTIFIED", ...overrides } as Parameters<typeof computeOvs>[0]);
test("C21 OVS selects the minimum cap", () => assert.equal(ovs({ mosStatus: "THIN" }), 75));
test("C21a OVS range applies MOS cap", () => assert.deepEqual(ovs({ returnComponent: { min: 70, max: 90 }, mosStatus: "THIN" }), { min: 70, max: 75 }));
test("C21b OVS range applies reliability cap", () => assert.deepEqual(ovs({ returnComponent: { min: 90, max: 100 }, valuationReliability: "MEDIUM" }), { min: 90, max: 95 }));
test("C22 NOT_ASSESSABLE MOS yields no numeric OVS", () => assert.equal(ovs({ mosStatus: "NOT_ASSESSABLE" }), "NOT_ASSESSABLE"));
test("C23 NOT_ASSESSABLE reliability yields no numeric OVS", () => assert.equal(ovs({ valuationReliability: "NOT_ASSESSABLE" }), "NOT_ASSESSABLE"));
test("C24 suspended plus HIGH yields NOT_AVAILABLE", () => assert.equal(ovs({ scorePermission: "SUSPENDED", valuationReliability: "HIGH" }), "NOT_AVAILABLE"));
test("C25 suspended plus MEDIUM yields NOT_AVAILABLE", () => assert.equal(ovs({ scorePermission: "SUSPENDED", valuationReliability: "MEDIUM" }), "NOT_AVAILABLE"));
test("C26 suspended plus LOW yields NOT_AVAILABLE", () => assert.equal(ovs({ scorePermission: "SUSPENDED", valuationReliability: "LOW" }), "NOT_AVAILABLE"));
test("C27 suspended plus NOT_ASSESSABLE yields NOT_ASSESSABLE", () => assert.equal(ovs({ scorePermission: "SUSPENDED", valuationReliability: "NOT_ASSESSABLE" }), "NOT_ASSESSABLE"));
test("C27a suspended MOS state does not override HIGH reliability", () => assert.equal(ovs({ scorePermission: "SUSPENDED", valuationReliability: "HIGH", mosStatus: "NOT_ASSESSABLE" }), "NOT_AVAILABLE"));
test("C27b suspended MOS state does not override MEDIUM reliability", () => assert.equal(ovs({ scorePermission: "SUSPENDED", valuationReliability: "MEDIUM", mosStatus: "NOT_ASSESSABLE" }), "NOT_AVAILABLE"));
test("C27c suspended MOS state does not override LOW reliability", () => assert.equal(ovs({ scorePermission: "SUSPENDED", valuationReliability: "LOW", mosStatus: "NOT_ASSESSABLE" }), "NOT_AVAILABLE"));

test("C29 investment raw formula", () => close(computeInvestmentScore(90, 80, "ALLOWED").investmentRaw as number, 87));
test("C30 investment is capped by OQS", () => assert.equal(computeInvestmentScore(60, 100, "ALLOWED").investmentScore, 60));
test("C31 investment is capped by OVS plus 15", () => assert.equal(computeInvestmentScore(100, 40, "ALLOWED").investmentScore, 55));
test("C32 suspended yields no investment score", () => assert.equal(computeInvestmentScore(90, 80, "SUSPENDED").investmentScore, "NOT_AVAILABLE"));
test("C32a unavailable OQS cannot fall back to numeric OVS", () => assert.deepEqual(computeInvestmentScore("NOT_AVAILABLE", 80, "ALLOWED"), { investmentRaw: "NOT_AVAILABLE", investmentScore: "NOT_AVAILABLE" }));
test("C32b investment range from OQS only", () => assert.deepEqual(computeInvestmentScore({ min: 70, max: 80 }, 90, "ALLOWED"), { investmentRaw: { min: 76, max: 83 }, investmentScore: { min: 70, max: 80 } }));
test("C32c investment range from OVS only", () => assert.deepEqual(computeInvestmentScore(80, { min: 60, max: 70 }, "ALLOWED"), { investmentRaw: { min: 74, max: 77 }, investmentScore: { min: 74, max: 77 } }));
test("C32d investment range from both inputs respects OVS plus 15", () => assert.deepEqual(computeInvestmentScore({ min: 80, max: 100 }, { min: 40, max: 60 }, "ALLOWED"), { investmentRaw: { min: 68, max: 88 }, investmentScore: { min: 55, max: 75 } }));
test("C33 every terminal gate and certification condition pass yields YES", () => assert.equal(computeOroTitanStatus(gates(), certifiedContext), "YES"));
test("C34 one elite gate FAIL yields NO", () => assert.equal(computeOroTitanStatus(gates({ runwayElite: "FAIL" }), certifiedContext), "NO"));
test("C34a one elite gate NOT_ASSESSABLE yields NO", () => assert.equal(computeOroTitanStatus(gates({ runwayElite: "NOT_ASSESSABLE" }), certifiedContext), "NO"));
test("C35 material weak link FAIL yields NO", () => assert.equal(computeOroTitanStatus(gates({ materialWeakLink: "FAIL" }), certifiedContext), "NO"));
test("C36 terminal gate is independent of OQS", () => assert.equal(computeOroTitanStatus(gates({ moatElite: "FAIL" }), certifiedContext), "NO"));
test("C36a terminal requires exact certification conditions", () => {
  for (const context of [
    { ...certifiedContext, businessResearchStatus: "CERTIFIED_WITH_LIMITATIONS" as const },
    { ...certifiedContext, investmentConclusionStatus: "CERTIFIED_WITH_LIMITATIONS" as const },
    { ...certifiedContext, scorePermission: "CONDITIONAL" as const },
    { ...certifiedContext, valuationReliability: "MEDIUM" as const },
    { ...certifiedContext, valuationReliability: "LOW" as const },
    { ...certifiedContext, valuationReliability: "NOT_ASSESSABLE" as const },
  ]) assert.equal(computeOroTitanStatus(gates(), context), "NO");
});

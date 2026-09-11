import assert from "node:assert/strict";
import test from "node:test";
import {
  computeInvestmentScore, computeOqs, computeOroTitanStatus, computeOvs, computeReturnComponent,
  MOS_CAPS, scoreExpectedReturnDelta, VALUATION_RELIABILITY_CAPS,
} from "../lib/orotitan-equity/v1";

const dimensions = (overrides = {}) => ({ MOAT: 80, RUNWAY: 80, RETURN_QUALITY: 80, CASH_ECONOMICS: 80,
  CAPITAL_ALLOCATION: 80, MANAGEMENT_GOVERNANCE: 80, RESILIENCE_RISK: 80, ...overrides });
const gates = (overrides = {}) => ({ researchFullyCertified: true, moatElite: true, runwayElite: true,
  returnQualityElite: true, cashEconomicsElite: true, capitalAllocationElite: true,
  managementGovernanceElite: true, resilienceElite: true, valuationElite: true,
  materialWeakLink: "NO" as const, ...overrides });
const close = (actual: number, expected: number) => assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} != ${expected}`);

test("C01 OQS raw uses the exact weights", () => close(computeOqs(dimensions({ MOAT: 100, RUNWAY: 90, RETURN_QUALITY: 80, CASH_ECONOMICS: 70, CAPITAL_ALLOCATION: 60, MANAGEMENT_GOVERNANCE: 50, RESILIENCE_RISK: 40 })).oqsRaw, 74.5));
test("C02 weak-link cap binds", () => assert.deepEqual(computeOqs(dimensions({ MOAT: 40 })), { oqsRaw: 72, weakLinkCap: 65, oqs: 65 }));
test("C03 weak-link cap does not bind", () => assert.equal(computeOqs(dimensions()).oqs, 80));
test("C04 OQS and weak-link cap max at 100", () => assert.deepEqual(computeOqs(dimensions({ MOAT: 100, RUNWAY: 100, RETURN_QUALITY: 100, CASH_ECONOMICS: 100, CAPITAL_ALLOCATION: 100, MANAGEMENT_GOVERNANCE: 100, RESILIENCE_RISK: 100 })), { oqsRaw: 100, weakLinkCap: 100, oqs: 100 }));

test("C08 expected-return exact anchors", () => {
  for (const [delta, score] of [[-10, 0], [-8, 10], [-6, 20], [-4, 35], [-2, 50], [0, 70], [2, 82], [4, 90], [6, 95], [8, 100]]) assert.equal(scoreExpectedReturnDelta(delta), score);
});
test("C09 expected-return midpoint interpolation", () => { assert.equal(scoreExpectedReturnDelta(-3), 42.5); assert.equal(scoreExpectedReturnDelta(1), 76); });
test("C10 expected-return lower saturation", () => assert.equal(scoreExpectedReturnDelta(-50), 0));
test("C11 expected-return upper saturation", () => assert.equal(scoreExpectedReturnDelta(50), 100));
test("C12 return component normal case", () => assert.equal(computeReturnComponent(80, 70), 76));
test("C13 N+15 return cap binds", () => assert.equal(computeReturnComponent(100, 20), 35));

test("C14 MOS ROBUST cap", () => assert.equal(MOS_CAPS.ROBUST, 100));
test("C15 MOS ADEQUATE cap", () => assert.equal(MOS_CAPS.ADEQUATE, 90));
test("C16 MOS THIN cap", () => assert.equal(MOS_CAPS.THIN, 75));
test("C17 MOS NONE cap", () => assert.equal(MOS_CAPS.NONE, 55));
test("C18 valuation HIGH cap", () => assert.equal(VALUATION_RELIABILITY_CAPS.HIGH, 100));
test("C19 valuation MEDIUM cap", () => assert.equal(VALUATION_RELIABILITY_CAPS.MEDIUM, 95));
test("C20 valuation LOW cap", () => assert.equal(VALUATION_RELIABILITY_CAPS.LOW, 80));

const ovs = (overrides = {}) => computeOvs({ returnComponent: 88, mosStatus: "ROBUST", valuationReliability: "HIGH", scorePermission: "ALLOWED", investmentConclusionStatus: "CERTIFIED", ...overrides } as Parameters<typeof computeOvs>[0]);
test("C21 OVS selects the minimum cap", () => assert.equal(ovs({ mosStatus: "THIN" }), 75));
test("C22 NOT_ASSESSABLE MOS yields no numeric OVS", () => assert.equal(ovs({ mosStatus: "NOT_ASSESSABLE" }), "NOT_ASSESSABLE"));
test("C23 NOT_ASSESSABLE reliability yields no numeric OVS", () => assert.equal(ovs({ valuationReliability: "NOT_ASSESSABLE" }), "NOT_ASSESSABLE"));
test("C24 suspended plus HIGH yields NOT_AVAILABLE", () => assert.equal(ovs({ scorePermission: "SUSPENDED", valuationReliability: "HIGH" }), "NOT_AVAILABLE"));
test("C25 suspended plus MEDIUM yields NOT_AVAILABLE", () => assert.equal(ovs({ scorePermission: "SUSPENDED", valuationReliability: "MEDIUM" }), "NOT_AVAILABLE"));
test("C26 suspended plus LOW yields NOT_AVAILABLE", () => assert.equal(ovs({ scorePermission: "SUSPENDED", valuationReliability: "LOW" }), "NOT_AVAILABLE"));
test("C27 suspended plus NOT_ASSESSABLE yields NOT_ASSESSABLE", () => assert.equal(ovs({ scorePermission: "SUSPENDED", valuationReliability: "NOT_ASSESSABLE" }), "NOT_ASSESSABLE"));

test("C29 investment raw formula", () => close(computeInvestmentScore(90, 80, "ALLOWED").investmentRaw as number, 87));
test("C30 investment is capped by OQS", () => assert.equal(computeInvestmentScore(60, 100, "ALLOWED").investmentScore, 60));
test("C31 investment is capped by OVS plus 15", () => assert.equal(computeInvestmentScore(100, 40, "ALLOWED").investmentScore, 55));
test("C32 suspended yields no investment score", () => assert.equal(computeInvestmentScore(90, 80, "SUSPENDED").investmentScore, "NOT_AVAILABLE"));
test("C33 every OroTitan gate true yields YES", () => assert.equal(computeOroTitanStatus(gates()), "YES"));
test("C34 one elite gate false yields NO", () => assert.equal(computeOroTitanStatus(gates({ runwayElite: false })), "NO"));
test("C35 material weak link yields NO", () => assert.equal(computeOroTitanStatus(gates({ materialWeakLink: "YES" })), "NO"));
test("C36 score is irrelevant to terminal gate", () => assert.equal(computeOroTitanStatus(gates({ moatElite: false })), "NO"));

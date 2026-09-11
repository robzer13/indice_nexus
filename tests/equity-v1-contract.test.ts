import assert from "node:assert/strict";
import test from "node:test";
import { computeCanonicalSnapshot, validateCanonicalContract } from "../lib/orotitan-equity/v1";

const valid = (overrides: Record<string, unknown> = {}) => ({
  dimensions: { MOAT: 75, RUNWAY: 75, RETURN_QUALITY: 80, CASH_ECONOMICS: 80, CAPITAL_ALLOCATION: 80, MANAGEMENT_GOVERNANCE: 80, RESILIENCE_RISK: 80 },
  evidence: { moat: "STRONGLY_SUPPORTED", runway: "STRONGLY_SUPPORTED" },
  businessResearchStatus: "CERTIFIED", investmentConclusionStatus: "CERTIFIED",
  scorePermission: "ALLOWED", mosStatus: "ROBUST", valuationReliability: "HIGH",
  primaryExpectedReturnDeltaPercentagePoints: 2, normalizedExpectedReturnDeltaPercentagePoints: 0,
  eliteGates: { researchFullyCertified: "PASS", moatElite: "PASS", runwayElite: "PASS", returnQualityElite: "PASS",
    cashEconomicsElite: "PASS", capitalAllocationElite: "PASS", managementGovernanceElite: "PASS",
    resilienceElite: "PASS", valuationElite: "PASS", materialWeakLink: "PASS" }, ...overrides,
});
const invalid = (payload: unknown) => assert.equal(validateCanonicalContract(payload).success, false);

test("C05 PLAUSIBLE moat above 75 rejected", () => invalid(valid({ dimensions: { ...valid().dimensions, MOAT: 80 }, evidence: { moat: "PLAUSIBLE", runway: "STRONGLY_SUPPORTED" }, eliteGates: { ...valid().eliteGates, moatElite: "FAIL" } })));
test("C06 SUPPORTED moat above 90 rejected", () => invalid(valid({ dimensions: { ...valid().dimensions, MOAT: 95 }, evidence: { moat: "SUPPORTED", runway: "STRONGLY_SUPPORTED" }, eliteGates: { ...valid().eliteGates, moatElite: "FAIL" } })));
test("C07 PLAUSIBLE runway above 75 rejected", () => invalid(valid({ dimensions: { ...valid().dimensions, RUNWAY: 80 }, evidence: { moat: "STRONGLY_SUPPORTED", runway: "PLAUSIBLE" }, eliteGates: { ...valid().eliteGates, runwayElite: "FAIL" } })));
test("C28 NOT_CERTIFIED research produces no numeric OQS", () => {
  const payload = valid({ businessResearchStatus: "NOT_CERTIFIED" });
  const parsed = validateCanonicalContract(payload); assert.equal(parsed.success, true);
  if (parsed.success) {
    const result = computeCanonicalSnapshot(parsed.data);
    assert.equal(typeof result.oqsRaw, "number");
    assert.equal(typeof result.weakLinkCap, "number");
    assert.equal(result.oqs, "NOT_AVAILABLE");
  }
});
test("C28a SUSPENDED makes every OQS score unavailable", () => {
  const payload = valid({ scorePermission: "SUSPENDED" });
  const parsed = validateCanonicalContract(payload); assert.equal(parsed.success, true);
  if (parsed.success) {
    const result = computeCanonicalSnapshot(parsed.data);
    assert.equal(result.oqsRaw, "NOT_AVAILABLE");
    assert.equal(result.weakLinkCap, "NOT_AVAILABLE");
    assert.equal(result.oqs, "NOT_AVAILABLE");
    assert.equal(result.investmentRaw, "NOT_AVAILABLE");
    assert.equal(result.investmentScore, "NOT_AVAILABLE");
  }
});
test("C37 deterministic mismatch rejected", () => invalid(valid({ deterministic: { oqs: 1 } })));
test("C38 null semantic state rejected", () => invalid(valid({ mosStatus: null })));
test("C39 empty semantic state rejected", () => invalid(valid({ valuationReliability: "" })));
test("C40 FALSIFIED moat remains valid when its elite gate is not PASS", () => assert.equal(validateCanonicalContract(valid({ evidence: { moat: "FALSIFIED", runway: "STRONGLY_SUPPORTED" }, eliteGates: { ...valid().eliteGates, moatElite: "FAIL" } })).success, true));
test("C41 FALSIFIED runway plus elite gate rejected", () => invalid(valid({ evidence: { moat: "STRONGLY_SUPPORTED", runway: "FALSIFIED" } })));
test("C42 non-applicable dimension fails closed without renormalization", () => invalid(valid({ dimensions: { ...valid().dimensions, MOAT: "NOT_APPLICABLE" } })));
test("C43 out-of-range dimension rejected rather than clamped", () => invalid(valid({ dimensions: { ...valid().dimensions, MOAT: 101 } })));
test("C43a non-five-point dimension rejected", () => invalid(valid({ dimensions: { ...valid().dimensions, MOAT: 83 } })));
test("C43b fractional non-five-point dimension rejected", () => invalid(valid({ dimensions: { ...valid().dimensions, MOAT: 87.5 } })));
test("C43c reversed dimension range rejected", () => invalid(valid({ dimensions: { ...valid().dimensions, MOAT: { min: 85, max: 80 } } })));
test("C43d non-five-point dimension range endpoint rejected", () => invalid(valid({ dimensions: { ...valid().dimensions, MOAT: { min: 80, max: 85.5 } } })));
test("C43e reversed return range rejected", () => invalid(valid({ primaryExpectedReturnDeltaPercentagePoints: { min: 4, max: 2 } })));
test("C43f bounded canonical dossier is accepted and computes bounded aggregates", () => {
  const parsed = validateCanonicalContract(valid({
    dimensions: { ...valid().dimensions, MOAT: { min: 70, max: 80 } },
    primaryExpectedReturnDeltaPercentagePoints: { min: 0, max: 2 },
    normalizedExpectedReturnDeltaPercentagePoints: { min: -2, max: 0 },
  }));
  assert.equal(parsed.success, true);
  if (parsed.success) {
    const result = computeCanonicalSnapshot(parsed.data);
    assert.deepEqual(result.oqs, { min: 77.25, max: 79.25 });
    assert.deepEqual(result.primaryExpectedReturnScore, { min: 70, max: 82 });
    assert.deepEqual(result.normalizedExpectedReturnScore, { min: 50, max: 70 });
  }
});
test("C44 numeric OVS while suspended rejected", () => invalid(valid({ scorePermission: "SUSPENDED", deterministic: { ovs: 70 } })));
test("C45 numeric investment while OVS unavailable rejected", () => invalid(valid({ valuationReliability: "NOT_ASSESSABLE", deterministic: { investmentScore: 70 } })));
test("C46 numeric OQS while business not certified rejected", () => invalid(valid({ businessResearchStatus: "NOT_CERTIFIED", deterministic: { oqs: 70 } })));
test("C47 numeric OVS with MOS not assessable rejected", () => invalid(valid({ mosStatus: "NOT_ASSESSABLE", deterministic: { ovs: 70 } })));
test("C48 numeric OVS with reliability not assessable rejected", () => invalid(valid({ valuationReliability: "NOT_ASSESSABLE", deterministic: { ovs: 70 } })));
test("C49 supplied correct deterministic values accepted", () => {
  const base = valid(); const initial = validateCanonicalContract(base); assert.equal(initial.success, true);
  if (initial.success) assert.equal(validateCanonicalContract({ ...base, deterministic: computeCanonicalSnapshot(initial.data) }).success, true);
});
test("C50 OroTitan YES mismatch rejected", () => invalid(valid({ eliteGates: { ...valid().eliteGates, moatElite: "FAIL" }, deterministic: { orotitanStatus: "YES" } })));
test("C51 material weak link with supplied YES rejected", () => invalid(valid({ eliteGates: { ...valid().eliteGates, materialWeakLink: "FAIL" }, deterministic: { orotitanStatus: "YES" } })));
test("C52 PLAUSIBLE evidence cannot support a PASS moat gate", () => invalid(valid({ evidence: { moat: "PLAUSIBLE", runway: "STRONGLY_SUPPORTED" } })));
test("C53 PLAUSIBLE evidence cannot support a PASS runway gate", () => invalid(valid({ evidence: { moat: "STRONGLY_SUPPORTED", runway: "PLAUSIBLE" } })));

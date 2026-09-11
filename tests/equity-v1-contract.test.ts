import assert from "node:assert/strict";
import test from "node:test";
import { computeCanonicalSnapshot, validateCanonicalContract } from "../lib/orotitan-equity/v1";

const valid = (overrides: Record<string, unknown> = {}) => ({
  dimensions: { MOAT: 75, RUNWAY: 75, RETURN_QUALITY: 80, CASH_ECONOMICS: 80, CAPITAL_ALLOCATION: 80, MANAGEMENT_GOVERNANCE: 80, RESILIENCE_RISK: 80 },
  evidence: { moat: "PLAUSIBLE", runway: "PLAUSIBLE" },
  businessResearchStatus: "CERTIFIED", investmentConclusionStatus: "CERTIFIED",
  scorePermission: "ALLOWED", mosStatus: "ROBUST", valuationReliability: "HIGH",
  fiveYearDeltaPercentagePoints: 2, tenYearDeltaPercentagePoints: 0,
  eliteGates: { researchFullyCertified: true, moatElite: true, runwayElite: true, returnQualityElite: true,
    cashEconomicsElite: true, capitalAllocationElite: true, managementGovernanceElite: true,
    resilienceElite: true, valuationElite: true, materialWeakLink: "NO" }, ...overrides,
});
const invalid = (payload: unknown) => assert.equal(validateCanonicalContract(payload).success, false);

test("C05 PLAUSIBLE moat above 75 rejected", () => invalid(valid({ dimensions: { ...valid().dimensions, MOAT: 76 } })));
test("C06 SUPPORTED moat above 90 rejected", () => invalid(valid({ dimensions: { ...valid().dimensions, MOAT: 91 }, evidence: { moat: "SUPPORTED", runway: "PLAUSIBLE" } })));
test("C07 PLAUSIBLE runway above 75 rejected", () => invalid(valid({ dimensions: { ...valid().dimensions, RUNWAY: 76 } })));
test("C28 NOT_CERTIFIED research produces no numeric OQS", () => {
  const payload = valid({ businessResearchStatus: "NOT_CERTIFIED" });
  const parsed = validateCanonicalContract(payload); assert.equal(parsed.success, true);
  if (parsed.success) assert.equal(computeCanonicalSnapshot(parsed.data).oqs, "NOT_AVAILABLE");
});
test("C37 deterministic mismatch rejected", () => invalid(valid({ deterministic: { oqs: 1 } })));
test("C38 null semantic state rejected", () => invalid(valid({ mosStatus: null })));
test("C39 empty semantic state rejected", () => invalid(valid({ valuationReliability: "" })));
test("C40 FALSIFIED moat plus elite gate rejected", () => invalid(valid({ evidence: { moat: "FALSIFIED", runway: "PLAUSIBLE" } })));
test("C41 FALSIFIED runway plus elite gate rejected", () => invalid(valid({ evidence: { moat: "PLAUSIBLE", runway: "FALSIFIED" } })));
test("C42 non-applicable dimension fails closed without renormalization", () => invalid(valid({ dimensions: { ...valid().dimensions, MOAT: "NOT_APPLICABLE" } })));
test("C43 out-of-range dimension rejected rather than clamped", () => invalid(valid({ dimensions: { ...valid().dimensions, MOAT: 101 } })));
test("C44 numeric OVS while suspended rejected", () => invalid(valid({ scorePermission: "SUSPENDED", deterministic: { ovs: 70 } })));
test("C45 numeric investment while OVS unavailable rejected", () => invalid(valid({ valuationReliability: "NOT_ASSESSABLE", deterministic: { investmentScore: 70 } })));
test("C46 numeric OQS while business not certified rejected", () => invalid(valid({ businessResearchStatus: "NOT_CERTIFIED", deterministic: { oqs: 70 } })));
test("C47 numeric OVS with MOS not assessable rejected", () => invalid(valid({ mosStatus: "NOT_ASSESSABLE", deterministic: { ovs: 70 } })));
test("C48 numeric OVS with reliability not assessable rejected", () => invalid(valid({ valuationReliability: "NOT_ASSESSABLE", deterministic: { ovs: 70 } })));
test("C49 supplied correct deterministic values accepted", () => {
  const base = valid(); const initial = validateCanonicalContract(base); assert.equal(initial.success, true);
  if (initial.success) assert.equal(validateCanonicalContract({ ...base, deterministic: computeCanonicalSnapshot(initial.data) }).success, true);
});
test("C50 OroTitan YES mismatch rejected", () => invalid(valid({ eliteGates: { ...valid().eliteGates, moatElite: false }, deterministic: { orotitanStatus: "YES" } })));
test("C51 material weak link with supplied YES rejected", () => invalid(valid({ eliteGates: { ...valid().eliteGates, materialWeakLink: "YES" }, deterministic: { orotitanStatus: "YES" } })));

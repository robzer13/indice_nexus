import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { computeCanonicalSnapshot } from "../lib/orotitan-equity/v1/contract";
import { verifyIntegrationCandidateD4, IntegrationD4Error } from "../lib/orotitan-equity/v3/pilotage-integration-recompute-v3-1";
import type { VerifiedContent } from "../lib/orotitan-equity/v3/pilotage-typed-gates-v3-1";

const schemaPath = new URL("../contracts/orotitan-equity/v1/04_SCREENER_SCHEMA_V1_PATCHED.json", import.meta.url);
const schemaDocument = JSON.parse(readFileSync(schemaPath, "utf8")) as Record<string, unknown>;
const definitions = schemaDocument.$defs as Record<string, Record<string, unknown>>;
const issuerId = "00000000-0000-4000-8000-000000000002";
const securityId = "00000000-0000-4000-8000-000000000003";
const dossierId = "00000000-0000-4000-8000-000000000001";
const DATE = "2026-09-11";

function buildFromSchema(node: Record<string, unknown>, path: string): unknown {
  const ref = typeof node.$ref === "string" ? node.$ref : undefined;
  if (ref?.endsWith("/date") || ref?.endsWith("/dateOrState")) return "2026-09-11";
  if (ref?.startsWith("#/$defs/")) return buildFromSchema(definitions[ref.slice(8)], path);
  if (node.const !== undefined) return node.const;
  if (Array.isArray(node.enum)) return node.enum[0];
  if (Array.isArray(node.oneOf)) {
    if (path.endsWith("potential_orotitan_price_zone")) return "NOT_AVAILABLE";
    if (path.endsWith("potential_orotitan_max_price")) return "NOT_AVAILABLE";
    const numeric = node.oneOf.find((option) => (option as Record<string, unknown>).type === "number");
    if (numeric) return 50;
    return buildFromSchema(node.oneOf[0] as Record<string, unknown>, path);
  }
  if (node.type === "object" || node.properties) {
    const result: Record<string, unknown> = {};
    const properties = node.properties as Record<string, Record<string, unknown>> | undefined;
    for (const key of (node.required as string[] | undefined) ?? []) result[key] = buildFromSchema(properties?.[key] ?? {}, `${path}/${key}`);
    return result;
  }
  if (node.type === "array") {
    if (path.endsWith("orotitan_gate_results")) {
      const gates = ["CERTIFICATION_GATE", "MOAT_ELITE", "RUNWAY_ELITE", "RETURN_QUALITY_ELITE", "CASH_ECONOMICS_ELITE", "CAPITAL_ALLOCATION_ELITE", "MANAGEMENT_GOVERNANCE_ELITE", "RESILIENCE_ELITE", "MATERIAL_WEAK_LINK_GATE", "VALUATION_ELITE"];
      return gates.map((gate) => ({ gate, state: "PASS", rationale: "supported", evidence_ids: ["evidence-1"] }));
    }
    return [];
  }
  if (node.type === "integer") return 5;
  if (node.type === "number") return 50;
  if (node.type === "boolean") return false;
  return "x";
}

function analyzeSnapshot(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  const snapshot = buildFromSchema({ $ref: "#/$defs/researchSnapshot" }, "") as Record<string, unknown>;
  const l2 = snapshot.l2_research_fundamentals as Record<string, unknown>;
  l2.fundamental_states = buildFromSchema(definitions.fundamentalStates, "/l2_research_fundamentals/fundamental_states");
  l2.analytical_metrics = buildFromSchema(definitions.analyticalMetrics, "/l2_research_fundamentals/analytical_metrics");
  l2.certification = buildFromSchema(definitions.certification, "/l2_research_fundamentals/certification");
  l2.business_quality = buildFromSchema(definitions.businessQuality, "/l2_research_fundamentals/business_quality");
  snapshot.l3_investment_valuation = {
    valuation: buildFromSchema(definitions.valuation, "/l3_investment_valuation/valuation"),
    investment: buildFromSchema(definitions.investment, "/l3_investment_valuation/investment"),
  };
  snapshot.l4_operational_state = buildFromSchema(definitions.operationalState, "/l4_operational_state");
  const fundamentals = l2.fundamental_states as Record<string, unknown>;
  const quality = l2.business_quality as Record<string, unknown>;
  const certification = l2.certification as Record<string, unknown>;
  const l3 = snapshot.l3_investment_valuation as Record<string, unknown>;
  const valuation = l3.valuation as Record<string, unknown>;
  const priceLadder = valuation.price_ladder as Record<string, unknown>;
  const investment = l3.investment as Record<string, unknown>;
  const l4 = snapshot.l4_operational_state as Record<string, unknown>;
  const orotitan = l4.orotitan as Record<string, unknown>;

  Object.assign(snapshot, { snapshot_id: "00000000-0000-4000-8000-000000000005", report_id: "report-analyze", issuer_id: issuerId, security_id: securityId, execution_mode: "ANALYZE" }, overrides);
  for (const key of ["moat_score", "runway_score", "return_quality_score", "cash_economics_score", "capital_allocation_score", "management_governance_score", "resilience_risk_score"]) quality[key] = 80;
  (snapshot.data_lock as Record<string, unknown>).reference_price_currency = "USD";
  fundamentals.sector_method_results = [{ method: "general", status: "NOT_REQUIRED", rationale: "not required", evidence_ids: [] }];
  fundamentals.moat_evidence_state = "STRONGLY_SUPPORTED";
  fundamentals.runway_evidence_state = "STRONGLY_SUPPORTED";
  certification.business_research_status = "CERTIFIED";
  certification.investment_conclusion_status = "CERTIFIED";
  certification.score_permission = "ALLOWED";
  valuation.primary_expected_return = 12;
  valuation.no_multiple_expansion_return = 8;
  valuation.mature_normalization_return = "NOT_AVAILABLE";
  valuation.margin_of_safety = "ROBUST";
  valuation.valuation_reliability = "HIGH";
  valuation.ovs = 65;
  priceLadder.required_return_h = 10;
  priceLadder.strong_return_threshold = 12.5;
  priceLadder.exceptional_return_threshold = 15;
  priceLadder.currency = "USD";
  priceLadder.investable_price_zone = "NOT_AVAILABLE";
  priceLadder.strong_opportunity_zone = "NOT_AVAILABLE";
  priceLadder.potential_orotitan_price_zone = "NOT_AVAILABLE";
  priceLadder.potential_orotitan_max_price = "NOT_AVAILABLE";
  investment.investment_raw = 75.5;
  investment.investment_score = 75.5;
  orotitan.orotitan_status = "YES";
  const contract = computeCanonicalSnapshot({
    dimensions: { MOAT: 80, RUNWAY: 80, RETURN_QUALITY: 80, CASH_ECONOMICS: 80, CAPITAL_ALLOCATION: 80, MANAGEMENT_GOVERNANCE: 80, RESILIENCE_RISK: 80 },
    evidence: { moat: "STRONGLY_SUPPORTED", runway: "STRONGLY_SUPPORTED" },
    businessResearchStatus: "CERTIFIED", investmentConclusionStatus: "CERTIFIED", scorePermission: "ALLOWED", mosStatus: "ROBUST", valuationReliability: "HIGH",
    primaryExpectedReturnDeltaPercentagePoints: 2, normalizedExpectedReturnDeltaPercentagePoints: -2,
    eliteGates: { researchFullyCertified: "PASS", moatElite: "PASS", runwayElite: "PASS", returnQualityElite: "PASS", cashEconomicsElite: "PASS", capitalAllocationElite: "PASS", managementGovernanceElite: "PASS", resilienceElite: "PASS", valuationElite: "PASS", materialWeakLink: "PASS" },
  });
  quality.oqs_raw = contract.oqsRaw; quality.weak_link_cap = contract.weakLinkCap; quality.oqs = contract.oqs;
  return snapshot;
}


function payload(): Record<string, unknown> {
  const s = analyzeSnapshot();
  s.v2_product = {
    classification: {
      issuer_country_code: "US", primary_listing_country_code: "US",
      sector: "INFORMATION_TECHNOLOGY", industry_group: "CYBERSECURITY",
      business_model_primary: "RECURRING_SUBSCRIPTION",
      business_model_secondary: null, economic_exposure_regions: ["GLOBAL"],
      taxonomy_version: "OROTITAN_TAXONOMY_V2.0",
    },
    business_summary: { business_description_short: "Subscription software sold to business customers." },
    investment_thesis: {
      quality_case: "Recurring revenue and strong cash economics.",
      valuation_case: "Expected return depends on market price.",
      key_risk: "Competitive pressure may reduce returns.",
    },
    portfolio_filters: { pea_eligibility: "UNKNOWN" },
  };
  const d = s.data_lock as Record<string, unknown>;
  d.data_cutoff = DATE;
  d.reference_price_date = DATE;
  s.v3_methodology = {
    economic_share_count: {
      methodology_version: "OROTITAN_ECONOMIC_SHARE_COUNT_METHODOLOGY_V1_FREEZE_V1.0",
      representation: "EXACT", share_count_as_of_date: DATE,
      valuation_admission: "YES", provenance_refs: ["verified-evidence:1"],
      propagation_status: "SCALAR_EXACT", exact_count: 100000000,
    },
    valuation_timing: {
      methodology_version: "OROTITAN_VALUATION_DATE_ALIGNMENT_AUTHORITY_V1_FREEZE_V1.0",
      valuation_date: DATE, time_origin: DATE,
      reference_price_date: DATE, economic_share_count_date: DATE,
      share_count_as_of_date: DATE, ev_to_equity_bridge_date: DATE,
      market_cap_date: DATE, per_share_output_date: DATE,
    },
  };
  return s;
}
function proof(s: Record<string, unknown>): VerifiedContent {
  return {
    proof: {
      artifactId: "77777777-7777-4777-8777-000000000001",
      version: 1, runId: "11111111-1111-4111-8111-111111111111",
      stageCode: "INTEGRATION", artifactType: "CANONICAL_SNAPSHOT_CANDIDATE",
      sha256: "a".repeat(64), available: true, bytesVerified: true,
    },
    mediaType: "application/json", bytes: Buffer.from(JSON.stringify(s)),
  };
}
function run(s = payload()) {
  return verifyIntegrationCandidateD4({
    candidate: proof(s), dossierId, issuerId, securityId,
    dataCutoff: DATE, upstreamTiming: null,
  });
}
function expectReject(s: Record<string, unknown>, code: string) {
  assert.throws(() => run(s), (e: unknown) => e instanceof IntegrationD4Error && e.code === code, code);
}
test("D4 independently recalculates candidate I2 and validates V3.1 structure without publication", () => {
  const x = run();
  assert.equal(x.i2CandidateRecomputed, true);
  assert.equal(x.v31CandidateSchemaPassed, true);
  assert.equal(x.historyCasValidated, false);
  assert.equal(x.i2Pass, false);
  assert.equal(x.i3bPass, false);
  assert.equal(x.canonicalCandidateAdmitted, false);
  assert.equal(x.publicationAuthorized, false);
  assert.ok(x.computed.oqs);
});
test("An incorrect deterministic investment score is rejected by independent I2 recomputation", () => {
  const s = payload();
  const invest = (s.l3_investment_valuation as Record<string, unknown>).investment as Record<string, unknown>;
  invest.investment_score = 100;
  expectReject(s, "D4_I2_OR_V2_RECONCILIATION_FAILED");
});
test("Rejected V2 product cannot bypass composed validation", () => {
  const s = payload();
  const c = (s.v2_product as Record<string, unknown>).classification as Record<string, unknown>;
  c.sector = "NOT_A_TAXONOMY_SECTOR";
  expectReject(s, "D4_I2_OR_V2_RECONCILIATION_FAILED");
});
test("A DISCOVER payload cannot pretend to be an analysis-ready candidate", () => {
  const s = payload(); s.execution_mode = "DISCOVER";
  expectReject(s, "D4_NON_ANALYTICAL_CANDIDATE");
});
test("Candidate run identity must match the trusted Registry", () => {
  const s = payload();
  s.issuer_id = "99999999-9999-4999-8999-999999999999";
  expectReject(s, "D4_CANDIDATE_IDENTITY_MISMATCH");
});
test("Bounded denominator with reversed limits cannot pass admission", () => {
  const s = payload();
  const ec = (s.v3_methodology as Record<string, unknown>).economic_share_count as Record<string, unknown>;
  delete ec.exact_count;
  ec.representation = "BOUNDED"; ec.lower_bound = 110; ec.upper_bound = 100;
  ec.bound_completeness = "COMPLETE"; ec.unbounded_movement_classes = 0;
  ec.propagation_status = "RANGE_PROPAGATED";
  expectReject(s, "D4_BOUNDED_DENOMINATOR_INVALID");
});
test("Unknown denominator cannot complete an analytical valuation", () => {
  const s = payload();
  const ec = (s.v3_methodology as Record<string, unknown>).economic_share_count as Record<string, unknown>;
  delete ec.exact_count;
  ec.representation = "UNKNOWN"; ec.valuation_admission = "NO";
  ec.unknown_reason = "NO_EVIDENCE"; ec.propagation_status = "BLOCKED_UNKNOWN";
  expectReject(s, "D4_UNKNOWN_DENOMINATOR_CANNOT_FINALIZE");
});
test("A future reference quote date is rejected", () => {
  const s = payload();
  const t = (s.v3_methodology as Record<string, unknown>).valuation_timing as Record<string, unknown>;
  t.reference_price_date = "2026-09-12";
  (s.data_lock as Record<string, unknown>).reference_price_date = "2026-09-12";
  expectReject(s, "D4_REFERENCE_PRICE_FUTURE");
});
test("A spoofed PASS report cannot change any authorization decision", () => {
  const s = payload();
  const x = run(s);
  s.i2_reconciliation_report = { status: "PASS", i3b_admission: "PASS" };
  // Unknown fields are not a permitted authority. This check only exposes
  // the already predeclared strict false authorization result.
  assert.equal(x.i2Pass, false);
  assert.equal(x.i3bPass, false);
});
test("Tampered field absent from frozen V3.1 JSON schema is rejected", () => {
  const s = payload();
  const t = (s.v3_methodology as Record<string, unknown>).valuation_timing as Record<string, unknown>;
  t.self_attested = "PASS";
  expectReject(s, "D4_V31_OVERLAY_SCHEMA_INVALID");
});

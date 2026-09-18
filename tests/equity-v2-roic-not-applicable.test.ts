import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { computeCanonicalSnapshot } from "../lib/orotitan-equity/v1/contract";
import { validateResearchSnapshotForPersistence } from "../lib/orotitan-equity/v1/research-snapshot-schema";
import { persistValidatedV2ResearchSnapshot } from "../lib/orotitan-equity/v2/persistence-core";
import { validateV2ResearchSnapshotForPersistence } from "../lib/orotitan-equity/v2/research-snapshot-schema";

const dossierId = "00000000-0000-4000-8000-000000000001";
const issuerId = "00000000-0000-4000-8000-000000000002";
const securityId = "00000000-0000-4000-8000-000000000003";

const frozenV1Schema = JSON.parse(
  readFileSync(new URL("../contracts/orotitan-equity/v1/04_SCREENER_SCHEMA_V1_PATCHED.json", import.meta.url), "utf8"),
) as Record<string, unknown>;
const definitions = frozenV1Schema.$defs as Record<string, Record<string, unknown>>;

function buildFromSchema(node: Record<string, unknown>, path: string): unknown {
  const ref = typeof node.$ref === "string" ? node.$ref : undefined;
  if (ref?.endsWith("/date") || ref?.endsWith("/dateOrState")) return "2026-09-18";
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
    for (const key of (node.required as string[] | undefined) ?? []) {
      result[key] = buildFromSchema(properties?.[key] ?? {}, `${path}/${key}`);
    }
    return result;
  }
  if (node.type === "array") {
    if (path.endsWith("orotitan_gate_results")) {
      const gates = [
        "CERTIFICATION_GATE",
        "MOAT_ELITE",
        "RUNWAY_ELITE",
        "RETURN_QUALITY_ELITE",
        "CASH_ECONOMICS_ELITE",
        "CAPITAL_ALLOCATION_ELITE",
        "MANAGEMENT_GOVERNANCE_ELITE",
        "RESILIENCE_ELITE",
        "MATERIAL_WEAK_LINK_GATE",
        "VALUATION_ELITE",
      ];
      return gates.map((gate) => ({ gate, state: "PASS", rationale: "supported", evidence_ids: ["evidence-1"] }));
    }
    return [];
  }
  if (node.type === "integer") return 5;
  if (node.type === "number") return 50;
  if (node.type === "boolean") return false;
  return "x";
}

function analyzeCore(roicTrend: string): Record<string, unknown> {
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
  const orotitan = (snapshot.l4_operational_state as Record<string, unknown>).orotitan as Record<string, unknown>;

  Object.assign(snapshot, {
    snapshot_id: "00000000-0000-4000-8000-000000000005",
    report_id: "report-v2-roic-na",
    issuer_id: issuerId,
    security_id: securityId,
    execution_mode: "ANALYZE",
  });
  (snapshot.data_lock as Record<string, unknown>).reference_price_currency = "EUR";

  fundamentals.sector_method_results = [{
    method: "INSURANCE_OVERLAY",
    status: "APPLIED",
    rationale: "Industrial ROIC is structurally inapplicable; insurance-sector return framework applied.",
    evidence_ids: ["evidence-1"],
  }];
  fundamentals.moat_evidence_state = "STRONGLY_SUPPORTED";
  fundamentals.runway_evidence_state = "STRONGLY_SUPPORTED";
  fundamentals.roic_trend = roicTrend;

  for (const key of [
    "moat_score",
    "runway_score",
    "return_quality_score",
    "cash_economics_score",
    "capital_allocation_score",
    "management_governance_score",
    "resilience_risk_score",
  ]) quality[key] = 80;

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
  priceLadder.currency = "EUR";
  priceLadder.investable_price_zone = "NOT_AVAILABLE";
  priceLadder.strong_opportunity_zone = "NOT_AVAILABLE";
  priceLadder.potential_orotitan_price_zone = "NOT_AVAILABLE";
  priceLadder.potential_orotitan_max_price = "NOT_AVAILABLE";

  investment.investment_raw = 75.5;
  investment.investment_score = 75.5;
  orotitan.orotitan_status = "YES";

  const deterministic = computeCanonicalSnapshot({
    dimensions: {
      MOAT: 80,
      RUNWAY: 80,
      RETURN_QUALITY: 80,
      CASH_ECONOMICS: 80,
      CAPITAL_ALLOCATION: 80,
      MANAGEMENT_GOVERNANCE: 80,
      RESILIENCE_RISK: 80,
    },
    evidence: { moat: "STRONGLY_SUPPORTED", runway: "STRONGLY_SUPPORTED" },
    businessResearchStatus: "CERTIFIED",
    investmentConclusionStatus: "CERTIFIED",
    scorePermission: "ALLOWED",
    mosStatus: "ROBUST",
    valuationReliability: "HIGH",
    primaryExpectedReturnDeltaPercentagePoints: 2,
    normalizedExpectedReturnDeltaPercentagePoints: -2,
    eliteGates: {
      researchFullyCertified: "PASS",
      moatElite: "PASS",
      runwayElite: "PASS",
      returnQualityElite: "PASS",
      cashEconomicsElite: "PASS",
      capitalAllocationElite: "PASS",
      managementGovernanceElite: "PASS",
      resilienceElite: "PASS",
      valuationElite: "PASS",
      materialWeakLink: "PASS",
    },
  });
  quality.oqs_raw = deterministic.oqsRaw;
  quality.weak_link_cap = deterministic.weakLinkCap;
  quality.oqs = deterministic.oqs;
  valuation.ovs = deterministic.ovs;
  investment.investment_raw = deterministic.investmentRaw;
  investment.investment_score = deterministic.investmentScore;
  orotitan.orotitan_status = deterministic.orotitanStatus;

  return snapshot;
}

function v2Analyze(roicTrend: string): Record<string, unknown> {
  return {
    ...analyzeCore(roicTrend),
    v2_product: {
      classification: {
        issuer_country_code: "DE",
        primary_listing_country_code: "DE",
        sector: "FINANCIALS",
        industry_group: "INSURANCE",
        business_model_primary: "INSURANCE_UNDERWRITING",
        business_model_secondary: null,
        economic_exposure_regions: ["GLOBAL"],
        taxonomy_version: "OROTITAN_TAXONOMY_V2.0",
      },
      business_summary: {
        business_description_short: "Global insurer and reinsurer earning underwriting income and investment returns on insurance float.",
      },
      investment_thesis: {
        quality_case: "Underwriting discipline and diversified risk pools support resilient insurance economics.",
        valuation_case: "Expected shareholder return depends on entry price and normalized insurance profitability.",
        key_risk: "Large catastrophe losses, reserve error or adverse pricing cycles can impair returns.",
      },
      portfolio_filters: {
        pea_eligibility: "YES",
        pea_eligibility_as_of: "2026-09-18",
        pea_eligibility_source_ref: "regulated-market-listing",
      },
    },
  };
}

function fundamentalStates(snapshot: Record<string, unknown>): Record<string, unknown> {
  return ((snapshot.l2_research_fundamentals as Record<string, unknown>).fundamental_states) as Record<string, unknown>;
}

test("V2 compatibility preserves all pre-existing roic_trend values", () => {
  for (const value of ["IMPROVING", "STABLE", "DECLINING", "VOLATILE", "UNCLEAR"]) {
    const result = validateV2ResearchSnapshotForPersistence(v2Analyze(value), dossierId);
    assert.equal(result.ok, true, result.ok ? "" : `${value}: ${result.errors.join(" | ")}`);
  }
});

test("V2 compatibility admits and preserves roic_trend NOT_APPLICABLE", () => {
  const snapshot = v2Analyze("NOT_APPLICABLE");
  const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
  assert.equal(result.ok, true, result.ok ? "" : result.errors.join(" | "));
  assert.equal(fundamentalStates(snapshot).roic_trend, "NOT_APPLICABLE");
  if (result.ok) assert.equal(fundamentalStates(result.snapshot).roic_trend, "NOT_APPLICABLE");
});

test("V2 compatibility keeps arbitrary roic_trend values invalid", () => {
  const result = validateV2ResearchSnapshotForPersistence(v2Analyze("BROKEN"), dossierId);
  assert.equal(result.ok, false);
  if (!result.ok) assert.equal(result.stage, "schema");
});

test("frozen V1 validator remains unchanged and rejects the V2-only compatibility value", () => {
  const core = analyzeCore("NOT_APPLICABLE");
  const result = validateResearchSnapshotForPersistence(core, dossierId);
  assert.equal(result.ok, false);
  if (!result.ok) assert.equal(result.stage, "schema");
});

test("roic_trend compatibility does not change I2 deterministic reconciliation", () => {
  const stable = validateV2ResearchSnapshotForPersistence(v2Analyze("STABLE"), dossierId);
  const notApplicable = validateV2ResearchSnapshotForPersistence(v2Analyze("NOT_APPLICABLE"), dossierId);
  assert.equal(stable.ok, true, stable.ok ? "" : stable.errors.join(" | "));
  assert.equal(notApplicable.ok, true, notApplicable.ok ? "" : notApplicable.errors.join(" | "));
  if (stable.ok && notApplicable.ok) {
    assert.deepEqual(notApplicable.canonicalContract, stable.canonicalContract);
  }
});

test("V2 I3-B persistence boundary admits exact NOT_APPLICABLE without coercion", async () => {
  const snapshot = v2Analyze("NOT_APPLICABLE");
  let called = false;
  const result = await persistValidatedV2ResearchSnapshot(
    { dossierId, expectedCurrentSnapshotId: null, canonicalPayload: snapshot },
    async (args) => {
      called = true;
      assert.equal(
        fundamentalStates(args.p_canonical_payload as Record<string, unknown>).roic_trend,
        "NOT_APPLICABLE",
      );
      return {
        data: {
          status: "INSERTED",
          dossier_id: dossierId,
          snapshot_id: snapshot.snapshot_id,
          current_snapshot_id: snapshot.snapshot_id,
        },
        error: null,
      };
    },
  );
  assert.equal(called, true);
  assert.equal(result.status, "INSERTED");
});

test("insurance overlay uses NOT_APPLICABLE without issuer-specific code", () => {
  const snapshot = v2Analyze("NOT_APPLICABLE");
  const classification = ((snapshot.v2_product as Record<string, unknown>).classification) as Record<string, unknown>;
  assert.equal(classification.industry_group, "INSURANCE");
  assert.equal(classification.business_model_primary, "INSURANCE_UNDERWRITING");
  assert.equal(validateV2ResearchSnapshotForPersistence(snapshot, dossierId).ok, true);
});

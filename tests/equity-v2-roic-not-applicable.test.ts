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


function analyticalMetrics(snapshot: Record<string, unknown>): Record<string, unknown> {
  return ((snapshot.l2_research_fundamentals as Record<string, unknown>).analytical_metrics) as Record<string, unknown>;
}

test("V2.0.3 ROIIC compatibility preserves all pre-existing returnValue forms", () => {
  const values: unknown[] = [
    12.5,
    { min: 8, max: 14 },
    "UNKNOWN",
    "NOT_APPLICABLE",
    "NOT_ASSESSABLE",
    "MISSING",
    "NOT_AVAILABLE",
  ];
  for (const value of values) {
    const snapshot = v2Analyze("STABLE");
    analyticalMetrics(snapshot).roiic = value;
    const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
    assert.equal(result.ok, true, result.ok ? "" : JSON.stringify(value) + ": " + result.errors.join(" | "));
  }
});

test("V2.0.3 admits and preserves canonical ROIIC NOT_INTERPRETABLE", () => {
  const snapshot = v2Analyze("STABLE");
  analyticalMetrics(snapshot).roiic = "NOT_INTERPRETABLE";
  const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
  assert.equal(result.ok, true, result.ok ? "" : result.errors.join(" | "));
  assert.equal(analyticalMetrics(snapshot).roiic, "NOT_INTERPRETABLE");
  if (result.ok) assert.equal(analyticalMetrics(result.snapshot).roiic, "NOT_INTERPRETABLE");
});

test("V2.0.3 keeps arbitrary ROIIC strings invalid", () => {
  const snapshot = v2Analyze("STABLE");
  analyticalMetrics(snapshot).roiic = "BROKEN";
  const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
  assert.equal(result.ok, false);
  if (!result.ok) assert.equal(result.stage, "schema");
});

test("frozen V1 validator remains unchanged and rejects ROIIC NOT_INTERPRETABLE", () => {
  const core = analyzeCore("STABLE");
  analyticalMetrics(core).roiic = "NOT_INTERPRETABLE";
  const result = validateResearchSnapshotForPersistence(core, dossierId);
  assert.equal(result.ok, false);
  if (!result.ok) assert.equal(result.stage, "schema");
});

test("ROIIC NOT_INTERPRETABLE compatibility does not change I2 deterministic reconciliation", () => {
  const numericSnapshot = v2Analyze("STABLE");
  analyticalMetrics(numericSnapshot).roiic = 12;
  const notInterpretableSnapshot = v2Analyze("STABLE");
  analyticalMetrics(notInterpretableSnapshot).roiic = "NOT_INTERPRETABLE";
  const numeric = validateV2ResearchSnapshotForPersistence(numericSnapshot, dossierId);
  const notInterpretable = validateV2ResearchSnapshotForPersistence(notInterpretableSnapshot, dossierId);
  assert.equal(numeric.ok, true, numeric.ok ? "" : numeric.errors.join(" | "));
  assert.equal(notInterpretable.ok, true, notInterpretable.ok ? "" : notInterpretable.errors.join(" | "));
  if (numeric.ok && notInterpretable.ok) assert.deepEqual(notInterpretable.canonicalContract, numeric.canonicalContract);
});

test("V2 I3-B persistence boundary preserves exact ROIIC NOT_INTERPRETABLE", async () => {
  const snapshot = v2Analyze("STABLE");
  analyticalMetrics(snapshot).roiic = "NOT_INTERPRETABLE";
  let called = false;
  const result = await persistValidatedV2ResearchSnapshot(
    { dossierId, expectedCurrentSnapshotId: null, canonicalPayload: snapshot },
    async (args) => {
      called = true;
      assert.equal(analyticalMetrics(args.p_canonical_payload as Record<string, unknown>).roiic, "NOT_INTERPRETABLE");
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


const compatV203Schema = JSON.parse(
  readFileSync(new URL("../contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V1_COMPAT_V2.0.3.json", import.meta.url), "utf8"),
) as Record<string, unknown>;
const compatV204Schema = JSON.parse(
  readFileSync(new URL("../contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V1_COMPAT_V2.0.4.json", import.meta.url), "utf8"),
) as Record<string, unknown>;
const compatV205Schema = JSON.parse(
  readFileSync(new URL("../contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V1_COMPAT_V2.0.5.json", import.meta.url), "utf8"),
) as Record<string, unknown>;
const compatV206Schema = JSON.parse(
  readFileSync(new URL("../contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V1_COMPAT_V2.0.6.json", import.meta.url), "utf8"),
) as Record<string, unknown>;
const compatV208Schema = JSON.parse(
  readFileSync(new URL("../contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V1_COMPAT_V2.0.8.json", import.meta.url), "utf8"),
) as Record<string, unknown>;
const compatV209Schema = JSON.parse(
  readFileSync(new URL("../contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V1_COMPAT_V2.0.9.json", import.meta.url), "utf8"),
) as Record<string, unknown>;

function schemaDefinitions(schemaObject: Record<string, unknown>): Record<string, Record<string, unknown>> {
  return schemaObject.$defs as Record<string, Record<string, unknown>>;
}

function metricProperties(schemaObject: Record<string, unknown>): Record<string, unknown> {
  const defs = schemaDefinitions(schemaObject);
  return (defs.analyticalMetrics.properties as Record<string, unknown>);
}

function fundamentalStateProperties(schemaObject: Record<string, unknown>): Record<string, unknown> {
  const defs = schemaDefinitions(schemaObject);
  return (defs.fundamentalStates.properties as Record<string, unknown>);
}

function valuationProperties(schemaObject: Record<string, unknown>): Record<string, unknown> {
  const defs = schemaDefinitions(schemaObject);
  return defs.valuation.properties as Record<string, unknown>;
}

function valuationFields(snapshot: Record<string, unknown>): Record<string, unknown> {
  const l3 = snapshot.l3_investment_valuation as Record<string, unknown>;
  return l3.valuation as Record<string, unknown>;
}

function deterministicContract(result: ReturnType<typeof validateV2ResearchSnapshotForPersistence>): Record<string, unknown> {
  assert.equal(result.ok, true, result.ok ? "" : result.errors.join(" | "));
  if (!result.ok) throw new Error("validation unexpectedly failed");
  return result.canonicalContract.deterministic as Record<string, unknown>;
}

function assertI2DeterministicEqual(
  left: ReturnType<typeof validateV2ResearchSnapshotForPersistence>,
  right: ReturnType<typeof validateV2ResearchSnapshotForPersistence>,
): void {
  const a = deterministicContract(left);
  const b = deterministicContract(right);
  for (const key of ["oqsRaw", "weakLinkCap", "oqs", "ovs", "investmentRaw", "investmentScore", "orotitanStatus"]) {
    assert.deepEqual(a[key], b[key], `I2 deterministic output changed: ${key}`);
  }
}

test("V2.0.4 schema is a strict field-local additive successor to V2.0.3", () => {
  const expected = JSON.parse(JSON.stringify(compatV203Schema)) as Record<string, unknown>;
  expected.$id = "urn:orotitan:equity-research:screener-contract:v1-v2-compat-2.0.4";
  expected.title = "OroTitan Equity Research V1 Core Compatibility Schema for V2.0.4";
  expected.description = compatV204Schema.description;

  const expectedMetrics = metricProperties(expected);
  const actualMetrics = metricProperties(compatV204Schema);
  expectedMetrics.standard_roic = actualMetrics.standard_roic;

  assert.deepEqual(compatV204Schema, expected);
  assert.deepEqual(schemaDefinitions(compatV204Schema).specialState, schemaDefinitions(compatV203Schema).specialState);
  assert.deepEqual(schemaDefinitions(compatV204Schema).returnValue, schemaDefinitions(compatV203Schema).returnValue);
  assert.deepEqual(actualMetrics.roiic, metricProperties(compatV203Schema).roiic);
});

test("V2.0.4 STANDARD_ROIC compatibility preserves all pre-existing returnValue forms", () => {
  const values: unknown[] = [
    12.5,
    { min: 8, max: 14 },
    "UNKNOWN",
    "NOT_APPLICABLE",
    "NOT_ASSESSABLE",
    "MISSING",
    "NOT_AVAILABLE",
  ];
  for (const value of values) {
    const snapshot = v2Analyze("STABLE");
    analyticalMetrics(snapshot).standard_roic = value;
    const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
    assert.equal(result.ok, true, result.ok ? "" : JSON.stringify(value) + ": " + result.errors.join(" | "));
  }
});

test("V2.0.4 admits and preserves canonical STANDARD_ROIC NOT_INTERPRETABLE", () => {
  const snapshot = v2Analyze("STABLE");
  analyticalMetrics(snapshot).standard_roic = "NOT_INTERPRETABLE";
  const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
  assert.equal(result.ok, true, result.ok ? "" : result.errors.join(" | "));
  assert.equal(analyticalMetrics(snapshot).standard_roic, "NOT_INTERPRETABLE");
  if (result.ok) assert.equal(analyticalMetrics(result.snapshot).standard_roic, "NOT_INTERPRETABLE");
});

test("V2.0.4 simultaneous STANDARD_ROIC and ROIIC NOT_INTERPRETABLE are preserved exactly", () => {
  const snapshot = v2Analyze("STABLE");
  analyticalMetrics(snapshot).standard_roic = "NOT_INTERPRETABLE";
  analyticalMetrics(snapshot).roiic = "NOT_INTERPRETABLE";
  const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
  assert.equal(result.ok, true, result.ok ? "" : result.errors.join(" | "));
  if (result.ok) {
    assert.equal(analyticalMetrics(result.snapshot).standard_roic, "NOT_INTERPRETABLE");
    assert.equal(analyticalMetrics(result.snapshot).roiic, "NOT_INTERPRETABLE");
  }
});

test("V2.0.4 keeps arbitrary STANDARD_ROIC strings invalid", () => {
  const snapshot = v2Analyze("STABLE");
  analyticalMetrics(snapshot).standard_roic = "BROKEN";
  const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
  assert.equal(result.ok, false);
  if (!result.ok) assert.equal(result.stage, "schema");
});

test("V2.0.4 keeps arbitrary ROIIC strings invalid", () => {
  const snapshot = v2Analyze("STABLE");
  analyticalMetrics(snapshot).roiic = "BROKEN";
  const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
  assert.equal(result.ok, false);
  if (!result.ok) assert.equal(result.stage, "schema");
});

test("frozen V1 still rejects STANDARD_ROIC NOT_INTERPRETABLE", () => {
  const core = analyzeCore("STABLE");
  analyticalMetrics(core).standard_roic = "NOT_INTERPRETABLE";
  const result = validateResearchSnapshotForPersistence(core, dossierId);
  assert.equal(result.ok, false);
  if (!result.ok) assert.equal(result.stage, "schema");
});

test("V2.0.6 NOT_INTERPRETABLE remains invalid outside authorized return fields", () => {
  const cases: Array<[string, (snapshot: Record<string, unknown>) => void]> = [
    ["all_in_roic", (snapshot) => { analyticalMetrics(snapshot).all_in_roic = "NOT_INTERPRETABLE"; }],
    ["rd_adjusted_roic", (snapshot) => { analyticalMetrics(snapshot).rd_adjusted_roic = "NOT_INTERPRETABLE"; }],
    ["share_count_cagr", (snapshot) => { analyticalMetrics(snapshot).share_count_cagr = "NOT_INTERPRETABLE"; }],
    ["primary_expected_return", (snapshot) => {
      const l3 = snapshot.l3_investment_valuation as Record<string, unknown>;
      (l3.valuation as Record<string, unknown>).primary_expected_return = "NOT_INTERPRETABLE";
    }],
    ["moat_score", (snapshot) => {
      const l2 = snapshot.l2_research_fundamentals as Record<string, unknown>;
      (l2.business_quality as Record<string, unknown>).moat_score = "NOT_INTERPRETABLE";
    }],
  ];

  for (const [label, mutate] of cases) {
    const snapshot = v2Analyze("STABLE");
    mutate(snapshot);
    const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
    assert.equal(result.ok, false, `${label} unexpectedly admitted NOT_INTERPRETABLE`);
    if (!result.ok) assert.equal(result.stage, "schema");
  }
});

test("V2.0.4 STANDARD_ROIC compatibility does not change I2 deterministic outputs", () => {
  const numericSnapshot = v2Analyze("STABLE");
  analyticalMetrics(numericSnapshot).standard_roic = 12;
  const notInterpretableSnapshot = v2Analyze("STABLE");
  analyticalMetrics(notInterpretableSnapshot).standard_roic = "NOT_INTERPRETABLE";
  const numeric = validateV2ResearchSnapshotForPersistence(numericSnapshot, dossierId);
  const notInterpretable = validateV2ResearchSnapshotForPersistence(notInterpretableSnapshot, dossierId);
  assertI2DeterministicEqual(numeric, notInterpretable);
});

test("V2.0.4 ROIIC regression does not change I2 deterministic outputs", () => {
  const numericSnapshot = v2Analyze("STABLE");
  analyticalMetrics(numericSnapshot).roiic = 12;
  const notInterpretableSnapshot = v2Analyze("STABLE");
  analyticalMetrics(notInterpretableSnapshot).roiic = "NOT_INTERPRETABLE";
  const numeric = validateV2ResearchSnapshotForPersistence(numericSnapshot, dossierId);
  const notInterpretable = validateV2ResearchSnapshotForPersistence(notInterpretableSnapshot, dossierId);
  assertI2DeterministicEqual(numeric, notInterpretable);
});

test("V2 I3-B boundary preserves simultaneous STANDARD_ROIC and ROIIC NOT_INTERPRETABLE", async () => {
  const snapshot = v2Analyze("STABLE");
  analyticalMetrics(snapshot).standard_roic = "NOT_INTERPRETABLE";
  analyticalMetrics(snapshot).roiic = "NOT_INTERPRETABLE";
  let called = false;
  const result = await persistValidatedV2ResearchSnapshot(
    { dossierId, expectedCurrentSnapshotId: null, canonicalPayload: snapshot },
    async (args) => {
      called = true;
      const persisted = analyticalMetrics(args.p_canonical_payload as Record<string, unknown>);
      assert.equal(persisted.standard_roic, "NOT_INTERPRETABLE");
      assert.equal(persisted.roiic, "NOT_INTERPRETABLE");
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

test("V2.0.4 additive compatibility requires no historical payload rewrite", () => {
  const v2Historical = v2Analyze("STABLE");
  analyticalMetrics(v2Historical).standard_roic = 12;
  analyticalMetrics(v2Historical).roiic = 10;
  assert.equal(validateV2ResearchSnapshotForPersistence(v2Historical, dossierId).ok, true);

  const v1Historical = analyzeCore("STABLE");
  analyticalMetrics(v1Historical).standard_roic = 12;
  analyticalMetrics(v1Historical).roiic = 10;
  assert.equal(validateResearchSnapshotForPersistence(v1Historical, dossierId).ok, true);
});


test("V2.0.5 schema is a strict field-local additive successor to V2.0.4", () => {
  const expected = JSON.parse(JSON.stringify(compatV204Schema)) as Record<string, unknown>;
  expected.$id = "urn:orotitan:equity-research:screener-contract:v1-v2-compat-2.0.5";
  expected.title = "OroTitan Equity Research V1 Core Compatibility Schema for V2.0.5";
  expected.description = compatV205Schema.description;

  const expectedFundamentals = fundamentalStateProperties(expected);
  const actualFundamentals = fundamentalStateProperties(compatV205Schema);
  expectedFundamentals.roic_trend = actualFundamentals.roic_trend;

  assert.deepEqual(compatV205Schema, expected);
  assert.deepEqual(schemaDefinitions(compatV205Schema).specialState, schemaDefinitions(compatV204Schema).specialState);
  assert.deepEqual(schemaDefinitions(compatV205Schema).returnValue, schemaDefinitions(compatV204Schema).returnValue);
  assert.deepEqual(metricProperties(compatV205Schema).standard_roic, metricProperties(compatV204Schema).standard_roic);
  assert.deepEqual(metricProperties(compatV205Schema).roiic, metricProperties(compatV204Schema).roiic);
});

test("V2.0.5 admits and preserves canonical ROIC_TREND UNKNOWN", () => {
  const snapshot = v2Analyze("UNKNOWN");
  const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
  assert.equal(result.ok, true, result.ok ? "" : result.errors.join(" | "));
  assert.equal(fundamentalStates(snapshot).roic_trend, "UNKNOWN");
  if (result.ok) assert.equal(fundamentalStates(result.snapshot).roic_trend, "UNKNOWN");
});

test("V2.0.5 preserves UNKNOWN and UNCLEAR as distinct ROIC_TREND states", () => {
  const unknown = v2Analyze("UNKNOWN");
  const unclear = v2Analyze("UNCLEAR");
  const unknownResult = validateV2ResearchSnapshotForPersistence(unknown, dossierId);
  const unclearResult = validateV2ResearchSnapshotForPersistence(unclear, dossierId);
  assert.equal(unknownResult.ok, true, unknownResult.ok ? "" : unknownResult.errors.join(" | "));
  assert.equal(unclearResult.ok, true, unclearResult.ok ? "" : unclearResult.errors.join(" | "));
  if (unknownResult.ok && unclearResult.ok) {
    assert.equal(fundamentalStates(unknownResult.snapshot).roic_trend, "UNKNOWN");
    assert.equal(fundamentalStates(unclearResult.snapshot).roic_trend, "UNCLEAR");
    assert.notEqual(
      fundamentalStates(unknownResult.snapshot).roic_trend,
      fundamentalStates(unclearResult.snapshot).roic_trend,
    );
  }
});

test("V2.0.5 retains ROIC_TREND NOT_APPLICABLE compatibility", () => {
  const result = validateV2ResearchSnapshotForPersistence(v2Analyze("NOT_APPLICABLE"), dossierId);
  assert.equal(result.ok, true, result.ok ? "" : result.errors.join(" | "));
  if (result.ok) assert.equal(fundamentalStates(result.snapshot).roic_trend, "NOT_APPLICABLE");
});

test("V2.0.5 keeps arbitrary ROIC_TREND strings invalid", () => {
  const result = validateV2ResearchSnapshotForPersistence(v2Analyze("BROKEN"), dossierId);
  assert.equal(result.ok, false);
  if (!result.ok) assert.equal(result.stage, "schema");
});

test("frozen V1 validator remains unchanged and rejects ROIC_TREND UNKNOWN", () => {
  const result = validateResearchSnapshotForPersistence(analyzeCore("UNKNOWN"), dossierId);
  assert.equal(result.ok, false);
  if (!result.ok) assert.equal(result.stage, "schema");
});

test("V2.0.5 ROIC_TREND UNKNOWN compatibility does not change I2 deterministic outputs", () => {
  const stable = validateV2ResearchSnapshotForPersistence(v2Analyze("STABLE"), dossierId);
  const unknown = validateV2ResearchSnapshotForPersistence(v2Analyze("UNKNOWN"), dossierId);
  assertI2DeterministicEqual(stable, unknown);
});

test("V2 I3-B persistence boundary preserves exact ROIC_TREND UNKNOWN without coercion", async () => {
  const snapshot = v2Analyze("UNKNOWN");
  let called = false;
  const result = await persistValidatedV2ResearchSnapshot(
    { dossierId, expectedCurrentSnapshotId: null, canonicalPayload: snapshot },
    async (args) => {
      called = true;
      assert.equal(
        fundamentalStates(args.p_canonical_payload as Record<string, unknown>).roic_trend,
        "UNKNOWN",
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

test("V2.0.5 compatibility requires no historical payload rewrite", () => {
  const v2Historical = v2Analyze("STABLE");
  assert.equal(validateV2ResearchSnapshotForPersistence(v2Historical, dossierId).ok, true);

  const v1Historical = analyzeCore("STABLE");
  assert.equal(validateResearchSnapshotForPersistence(v1Historical, dossierId).ok, true);
});


test("V2.0.6 schema is a strict field-local additive successor to V2.0.5", () => {
  const expected = JSON.parse(JSON.stringify(compatV205Schema)) as Record<string, unknown>;
  expected.$id = "urn:orotitan:equity-research:screener-contract:v1-v2-compat-2.0.6";
  expected.title = "OroTitan Equity Research V1 Core Compatibility Schema for V2.0.6";
  expected.description = compatV206Schema.description;

  const expectedMetrics = metricProperties(expected);
  const actualMetrics = metricProperties(compatV206Schema);
  expectedMetrics.roic_ex_goodwill = actualMetrics.roic_ex_goodwill;

  assert.deepEqual(compatV206Schema, expected);
  assert.deepEqual(schemaDefinitions(compatV206Schema).specialState, schemaDefinitions(compatV205Schema).specialState);
  assert.deepEqual(schemaDefinitions(compatV206Schema).returnValue, schemaDefinitions(compatV205Schema).returnValue);
  assert.deepEqual(actualMetrics.standard_roic, metricProperties(compatV205Schema).standard_roic);
  assert.deepEqual(actualMetrics.roiic, metricProperties(compatV205Schema).roiic);
  assert.deepEqual(fundamentalStateProperties(compatV206Schema).roic_trend, fundamentalStateProperties(compatV205Schema).roic_trend);
  assert.deepEqual(metricProperties(compatV205Schema).roic_ex_goodwill, { $ref: "#/$defs/returnValue" });
});

test("V2.0.6 ROIC_EX_GOODWILL compatibility preserves all pre-existing returnValue forms", () => {
  const values: unknown[] = [
    12.5,
    { min: 8, max: 14 },
    "UNKNOWN",
    "NOT_APPLICABLE",
    "NOT_ASSESSABLE",
    "MISSING",
    "NOT_AVAILABLE",
  ];
  for (const value of values) {
    const snapshot = v2Analyze("STABLE");
    analyticalMetrics(snapshot).roic_ex_goodwill = value;
    const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
    assert.equal(result.ok, true, result.ok ? "" : JSON.stringify(value) + ": " + result.errors.join(" | "));
  }
});

test("V2.0.6 admits and preserves canonical ROIC_EX_GOODWILL NOT_INTERPRETABLE", () => {
  const snapshot = v2Analyze("STABLE");
  analyticalMetrics(snapshot).roic_ex_goodwill = "NOT_INTERPRETABLE";
  const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
  assert.equal(result.ok, true, result.ok ? "" : result.errors.join(" | "));
  assert.equal(analyticalMetrics(snapshot).roic_ex_goodwill, "NOT_INTERPRETABLE");
  if (result.ok) assert.equal(analyticalMetrics(result.snapshot).roic_ex_goodwill, "NOT_INTERPRETABLE");
});

test("V2.0.6 keeps arbitrary ROIC_EX_GOODWILL strings invalid", () => {
  const snapshot = v2Analyze("STABLE");
  analyticalMetrics(snapshot).roic_ex_goodwill = "BROKEN";
  const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
  assert.equal(result.ok, false);
  if (!result.ok) assert.equal(result.stage, "schema");
});

test("frozen V1 still rejects ROIC_EX_GOODWILL NOT_INTERPRETABLE", () => {
  const core = analyzeCore("STABLE");
  analyticalMetrics(core).roic_ex_goodwill = "NOT_INTERPRETABLE";
  const result = validateResearchSnapshotForPersistence(core, dossierId);
  assert.equal(result.ok, false);
  if (!result.ok) assert.equal(result.stage, "schema");
});

test("V2.0.6 ROIC_EX_GOODWILL compatibility does not change I2 deterministic outputs", () => {
  const numericSnapshot = v2Analyze("STABLE");
  analyticalMetrics(numericSnapshot).roic_ex_goodwill = 12;
  const notInterpretableSnapshot = v2Analyze("STABLE");
  analyticalMetrics(notInterpretableSnapshot).roic_ex_goodwill = "NOT_INTERPRETABLE";
  const numeric = validateV2ResearchSnapshotForPersistence(numericSnapshot, dossierId);
  const notInterpretable = validateV2ResearchSnapshotForPersistence(notInterpretableSnapshot, dossierId);
  assertI2DeterministicEqual(numeric, notInterpretable);
});

test("V2 I3-B persistence boundary preserves exact ROIC_EX_GOODWILL NOT_INTERPRETABLE without coercion", async () => {
  const snapshot = v2Analyze("STABLE");
  analyticalMetrics(snapshot).roic_ex_goodwill = "NOT_INTERPRETABLE";
  let called = false;
  const result = await persistValidatedV2ResearchSnapshot(
    { dossierId, expectedCurrentSnapshotId: null, canonicalPayload: snapshot },
    async (args) => {
      called = true;
      assert.equal(
        analyticalMetrics(args.p_canonical_payload as Record<string, unknown>).roic_ex_goodwill,
        "NOT_INTERPRETABLE",
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

test("V2.0.6 simultaneous STANDARD_ROIC, ROIC_EX_GOODWILL and ROIIC NOT_INTERPRETABLE are preserved exactly", () => {
  const snapshot = v2Analyze("UNKNOWN");
  analyticalMetrics(snapshot).standard_roic = "NOT_INTERPRETABLE";
  analyticalMetrics(snapshot).roic_ex_goodwill = "NOT_INTERPRETABLE";
  analyticalMetrics(snapshot).roiic = "NOT_INTERPRETABLE";
  const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
  assert.equal(result.ok, true, result.ok ? "" : result.errors.join(" | "));
  if (result.ok) {
    const metrics = analyticalMetrics(result.snapshot);
    assert.equal(metrics.standard_roic, "NOT_INTERPRETABLE");
    assert.equal(metrics.roic_ex_goodwill, "NOT_INTERPRETABLE");
    assert.equal(metrics.roiic, "NOT_INTERPRETABLE");
    assert.equal(fundamentalStates(result.snapshot).roic_trend, "UNKNOWN");
  }
});

test("V2.0.6 compatibility requires no historical payload rewrite", () => {
  const v2Historical = v2Analyze("STABLE");
  analyticalMetrics(v2Historical).roic_ex_goodwill = 12;
  assert.equal(validateV2ResearchSnapshotForPersistence(v2Historical, dossierId).ok, true);

  const v1Historical = analyzeCore("STABLE");
  analyticalMetrics(v1Historical).roic_ex_goodwill = 12;
  assert.equal(validateResearchSnapshotForPersistence(v1Historical, dossierId).ok, true);
});


test("V2.0.8 schema is a strict field-local additive successor to V2.0.6", () => {
  const expected = JSON.parse(JSON.stringify(compatV206Schema)) as Record<string, unknown>;
  expected.$id = "urn:orotitan:equity-research:screener-contract:v1-v2-compat-2.0.8";
  expected.title = "OroTitan Equity Research V1 Core Compatibility Schema for V2.0.8";
  expected.description = compatV208Schema.description;

  const expectedValuation = valuationProperties(expected);
  const actualValuation = valuationProperties(compatV208Schema);
  expectedValuation.return_horizon = actualValuation.return_horizon;

  assert.deepEqual(compatV208Schema, expected);
  assert.deepEqual(schemaDefinitions(compatV208Schema).specialState, schemaDefinitions(compatV206Schema).specialState);
  assert.deepEqual(schemaDefinitions(compatV208Schema).returnValue, schemaDefinitions(compatV206Schema).returnValue);
  assert.deepEqual(metricProperties(compatV208Schema), metricProperties(compatV206Schema));
  assert.deepEqual(fundamentalStateProperties(compatV208Schema), fundamentalStateProperties(compatV206Schema));
});

test("V2.0.8 preserves ordinary integer return horizons", () => {
  for (const value of [1, 3, 5, 10]) {
    const snapshot = v2Analyze("STABLE");
    valuationFields(snapshot).return_horizon = value;
    const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
    assert.equal(result.ok, true, result.ok ? "" : `${value}: ${result.errors.join(" | ")}`);
    if (result.ok) assert.equal(valuationFields(result.snapshot).return_horizon, value);
  }
});

test("V2.0.8 admits exact positive fractional return horizons", () => {
  for (const value of [0.25, 0.5, 1.5, 4.7835616438356166, 9.999999]) {
    const snapshot = v2Analyze("STABLE");
    valuationFields(snapshot).return_horizon = value;
    const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
    assert.equal(result.ok, true, result.ok ? "" : `${value}: ${result.errors.join(" | ")}`);
    if (result.ok) assert.equal(valuationFields(result.snapshot).return_horizon, value);
  }
});

test("V2.0.8 rejects zero, negative, non-finite, string and malformed return horizons", () => {
  const invalidValues: unknown[] = [
    0,
    -0.25,
    Number.NaN,
    Number.POSITIVE_INFINITY,
    "4.7835616438356166",
    { value: 4.7835616438356166 },
  ];
  for (const value of invalidValues) {
    const snapshot = v2Analyze("STABLE");
    valuationFields(snapshot).return_horizon = value;
    const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
    assert.equal(result.ok, false, `unexpectedly admitted ${String(value)}`);
    if (!result.ok) assert.equal(result.stage, "schema");
  }
});

test("V2.0.8 retains all pre-existing return_horizon special states", () => {
  for (const value of ["UNKNOWN", "NOT_APPLICABLE", "NOT_ASSESSABLE", "MISSING", "NOT_AVAILABLE"]) {
    const snapshot = v2Analyze("STABLE");
    valuationFields(snapshot).return_horizon = value;
    const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
    assert.equal(result.ok, true, result.ok ? "" : `${value}: ${result.errors.join(" | ")}`);
  }
});

test("frozen V1 validator remains unchanged and rejects fractional return_horizon", () => {
  const core = analyzeCore("STABLE");
  valuationFields(core).return_horizon = 4.7835616438356166;
  const result = validateResearchSnapshotForPersistence(core, dossierId);
  assert.equal(result.ok, false);
  if (!result.ok) assert.equal(result.stage, "schema");
});

test("V2.0.8 fractional return_horizon round-trips through JSON without semantic loss", () => {
  const value = 4.7835616438356166;
  const serialized = JSON.stringify({ return_horizon: value });
  const reparsed = JSON.parse(serialized) as { return_horizon: number };
  assert.equal(serialized, '{"return_horizon":4.7835616438356166}');
  assert.equal(reparsed.return_horizon, value);
});

test("V2.0.8 return_horizon compatibility does not change I2 deterministic outputs", () => {
  const integerSnapshot = v2Analyze("STABLE");
  valuationFields(integerSnapshot).return_horizon = 5;
  const fractionalSnapshot = v2Analyze("STABLE");
  valuationFields(fractionalSnapshot).return_horizon = 4.7835616438356166;
  const integerResult = validateV2ResearchSnapshotForPersistence(integerSnapshot, dossierId);
  const fractionalResult = validateV2ResearchSnapshotForPersistence(fractionalSnapshot, dossierId);
  assertI2DeterministicEqual(integerResult, fractionalResult);
});

test("V2 I3-B boundary preserves exact fractional return_horizon without coercion", async () => {
  const snapshot = v2Analyze("STABLE");
  valuationFields(snapshot).return_horizon = 4.7835616438356166;
  let called = false;
  const result = await persistValidatedV2ResearchSnapshot(
    { dossierId, expectedCurrentSnapshotId: null, canonicalPayload: snapshot },
    async (args) => {
      called = true;
      assert.equal(
        valuationFields(args.p_canonical_payload as Record<string, unknown>).return_horizon,
        4.7835616438356166,
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

test("V2.0.8 simultaneously preserves ROIIC NOT_INTERPRETABLE and fractional return_horizon", () => {
  const snapshot = v2Analyze("STABLE");
  analyticalMetrics(snapshot).roiic = "NOT_INTERPRETABLE";
  valuationFields(snapshot).return_horizon = 4.7835616438356166;
  const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
  assert.equal(result.ok, true, result.ok ? "" : result.errors.join(" | "));
  if (result.ok) {
    assert.equal(analyticalMetrics(result.snapshot).roiic, "NOT_INTERPRETABLE");
    assert.equal(valuationFields(result.snapshot).return_horizon, 4.7835616438356166);
  }
});


test("V2.0.9 schema is a strict field-local additive successor to V2.0.8", () => {
  const expected = JSON.parse(JSON.stringify(compatV208Schema)) as Record<string, unknown>;
  expected.$id = "urn:orotitan:equity-research:screener-contract:v1-v2-compat-2.0.9";
  expected.title = "OroTitan Equity Research V1 Core Compatibility Schema for V2.0.9";
  expected.description = compatV209Schema.description;

  const expectedMetrics = metricProperties(expected);
  const actualMetrics = metricProperties(compatV209Schema);
  expectedMetrics.rd_adjusted_roic = actualMetrics.rd_adjusted_roic;

  assert.deepEqual(compatV209Schema, expected);
  assert.deepEqual(schemaDefinitions(compatV209Schema).specialState, schemaDefinitions(compatV208Schema).specialState);
  assert.deepEqual(schemaDefinitions(compatV209Schema).returnValue, schemaDefinitions(compatV208Schema).returnValue);
  assert.deepEqual(fundamentalStateProperties(compatV209Schema), fundamentalStateProperties(compatV208Schema));
  assert.deepEqual(valuationProperties(compatV209Schema), valuationProperties(compatV208Schema));
});

test("V2.0.9 RD_ADJUSTED_ROIC compatibility preserves all pre-existing returnValue forms", () => {
  const values: unknown[] = [12.5, { min: 8, max: 14 }, "UNKNOWN", "NOT_APPLICABLE", "NOT_ASSESSABLE", "MISSING", "NOT_AVAILABLE"];
  for (const value of values) {
    const snapshot = v2Analyze("STABLE");
    analyticalMetrics(snapshot).rd_adjusted_roic = value;
    const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
    assert.equal(result.ok, true, result.ok ? "" : JSON.stringify(value) + ": " + result.errors.join(" | "));
  }
});

test("V2.0.9 admits and preserves canonical RD_ADJUSTED_ROIC NOT_INTERPRETABLE", () => {
  const snapshot = v2Analyze("STABLE");
  analyticalMetrics(snapshot).rd_adjusted_roic = "NOT_INTERPRETABLE";
  const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
  assert.equal(result.ok, true, result.ok ? "" : result.errors.join(" | "));
  assert.equal(analyticalMetrics(snapshot).rd_adjusted_roic, "NOT_INTERPRETABLE");
  if (result.ok) assert.equal(analyticalMetrics(result.snapshot).rd_adjusted_roic, "NOT_INTERPRETABLE");
});

test("V2.0.9 keeps arbitrary RD_ADJUSTED_ROIC strings invalid and does not broaden all_in_roic", () => {
  const broken = v2Analyze("STABLE");
  analyticalMetrics(broken).rd_adjusted_roic = "BROKEN";
  const brokenResult = validateV2ResearchSnapshotForPersistence(broken, dossierId);
  assert.equal(brokenResult.ok, false);
  if (!brokenResult.ok) assert.equal(brokenResult.stage, "schema");

  const unauthorized = v2Analyze("STABLE");
  analyticalMetrics(unauthorized).all_in_roic = "NOT_INTERPRETABLE";
  const unauthorizedResult = validateV2ResearchSnapshotForPersistence(unauthorized, dossierId);
  assert.equal(unauthorizedResult.ok, false);
  if (!unauthorizedResult.ok) assert.equal(unauthorizedResult.stage, "schema");
});

test("frozen V1 still rejects RD_ADJUSTED_ROIC NOT_INTERPRETABLE", () => {
  const core = analyzeCore("STABLE");
  analyticalMetrics(core).rd_adjusted_roic = "NOT_INTERPRETABLE";
  const result = validateResearchSnapshotForPersistence(core, dossierId);
  assert.equal(result.ok, false);
  if (!result.ok) assert.equal(result.stage, "schema");
});

test("V2.0.9 RD_ADJUSTED_ROIC compatibility does not change I2 deterministic outputs", () => {
  const numericSnapshot = v2Analyze("STABLE");
  analyticalMetrics(numericSnapshot).rd_adjusted_roic = 12;
  const notInterpretableSnapshot = v2Analyze("STABLE");
  analyticalMetrics(notInterpretableSnapshot).rd_adjusted_roic = "NOT_INTERPRETABLE";
  const numeric = validateV2ResearchSnapshotForPersistence(numericSnapshot, dossierId);
  const notInterpretable = validateV2ResearchSnapshotForPersistence(notInterpretableSnapshot, dossierId);
  assertI2DeterministicEqual(numeric, notInterpretable);
});

test("V2 I3-B boundary preserves exact RD_ADJUSTED_ROIC NOT_INTERPRETABLE without coercion", async () => {
  const snapshot = v2Analyze("STABLE");
  analyticalMetrics(snapshot).rd_adjusted_roic = "NOT_INTERPRETABLE";
  let called = false;
  const result = await persistValidatedV2ResearchSnapshot(
    { dossierId, expectedCurrentSnapshotId: null, canonicalPayload: snapshot },
    async (args) => {
      called = true;
      assert.equal(
        analyticalMetrics(args.p_canonical_payload as Record<string, unknown>).rd_adjusted_roic,
        "NOT_INTERPRETABLE",
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

test("V2.0.9 simultaneously preserves RD_ADJUSTED_ROIC and ROIIC NOT_INTERPRETABLE plus fractional return_horizon", () => {
  const snapshot = v2Analyze("STABLE");
  analyticalMetrics(snapshot).rd_adjusted_roic = "NOT_INTERPRETABLE";
  analyticalMetrics(snapshot).roiic = "NOT_INTERPRETABLE";
  valuationFields(snapshot).return_horizon = 4.7835616438356166;
  const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
  assert.equal(result.ok, true, result.ok ? "" : result.errors.join(" | "));
  if (result.ok) {
    assert.equal(analyticalMetrics(result.snapshot).rd_adjusted_roic, "NOT_INTERPRETABLE");
    assert.equal(analyticalMetrics(result.snapshot).roiic, "NOT_INTERPRETABLE");
    assert.equal(valuationFields(result.snapshot).return_horizon, 4.7835616438356166);
  }
});

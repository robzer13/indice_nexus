import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import { persistValidatedResearchSnapshot } from "../lib/orotitan-equity/v1/persistence-core";
import { validateResearchSnapshotForPersistence } from "../lib/orotitan-equity/v1/research-snapshot-schema";
import { computeCanonicalSnapshot } from "../lib/orotitan-equity/v1/contract";

const schemaPath = new URL("../contracts/orotitan-equity/v1/04_SCREENER_SCHEMA_V1_PATCHED.json", import.meta.url);
const integrationPath = new URL("../contracts/orotitan-equity/v1/04_INTEGRATION_SPEC_V1_PATCHED.md", import.meta.url);
const schemaDocument = JSON.parse(readFileSync(schemaPath, "utf8")) as Record<string, unknown>;
const definitions = schemaDocument.$defs as Record<string, Record<string, unknown>>;
const dossierId = "00000000-0000-4000-8000-000000000001";
const issuerId = "00000000-0000-4000-8000-000000000002";
const securityId = "00000000-0000-4000-8000-000000000003";

function discoverSnapshot(): Record<string, unknown> {
  return {
    snapshot_id: "00000000-0000-4000-8000-000000000004",
    report_id: "report-1",
    issuer_id: issuerId,
    security_id: securityId,
    execution_mode: "DISCOVER",
    data_lock: {
      data_cutoff: "2026-09-10",
      reference_price: 10,
      reference_price_currency: "USD",
      reference_price_date: "2026-09-10",
      calculation_date: "2026-09-11",
      last_full_research_date: "NOT_AVAILABLE",
      score_date: "NOT_AVAILABLE",
    },
    versions: {
      report_version: "r1",
      method_version: "m1",
      calculation_version: "c1",
      evidence_ledger_version: "e1",
      method_versions: {
        discovery_version: "d1",
        moat_version: "m1",
        runway_version: "r1",
        roic_version: "roic1",
        fcf_version: "fcf1",
        valuation_version: "v1",
        certification_version: "cert1",
      },
    },
    l2_research_fundamentals: {
      discovery_record: {
        discovery_status: "DEEP_DIVE_CANDIDATE",
        discovery_methods: ["S1"],
        archetype_hints: ["A1"],
        attention_state: "HIGH",
        evidence_grade: "E1 OBSERVABLE_ISSUER_EVIDENCE",
        data_sufficiency: "SUFFICIENT",
        hard_kill_status: { "HK-A": "NOT_FOUND", "HK-B": "NOT_FOUND", "HK-C": "NOT_FOUND", "HK-D": "NOT_FOUND", "HK-E": "NOT_FOUND" },
        valuation_sanity: "NOT_ASSESSABLE",
        research_priority: "HIGH",
        watch_type: "NOT_APPLICABLE",
        reactivation_trigger: "NOT_APPLICABLE",
        rejection_reason: "NOT_APPLICABLE",
      },
    },
    traceability: { report_id: "report-1", evidence_ids: ["evidence-1"], calculation_ids: ["calculation-1"] },
  };
}

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

test("I3-B vendors the exact authoritative contract bytes", () => {
  assert.equal(createHash("sha256").update(readFileSync(schemaPath)).digest("hex"), "bf407ca217553521586ba5f6002180ff6522700b4671986079ea6ed577604ede");
  assert.equal(createHash("sha256").update(readFileSync(integrationPath)).digest("hex"), "f4d82ee65a9d653ebbb122d5fed04f90b722de8e0daa90844dc7fb1705ecf8b7");
});

test("I3-B accepts DISCOVER without analysis blocks", () => {
  const result = validateResearchSnapshotForPersistence(discoverSnapshot(), dossierId);
  assert.equal(result.ok, true);
});

test("I3-B rejects unknown properties, missing fields, and invalid physical IDs", () => {
  const unknown = discoverSnapshot();
  unknown.unexpected = true;
  assert.equal(validateResearchSnapshotForPersistence(unknown, dossierId).ok, false);

  const missing = discoverSnapshot();
  delete missing.traceability;
  assert.equal(validateResearchSnapshotForPersistence(missing, dossierId).ok, false);
  assert.equal(validateResearchSnapshotForPersistence(discoverSnapshot(), "not-a-uuid").ok, false);
});

test("I3-B rejects reversed ranges and unknown range properties", () => {
  const reversed = discoverSnapshot();
  reversed.data_lock = { ...(reversed.data_lock as object), reference_price: { min: 12, max: 10 } };
  assert.equal(validateResearchSnapshotForPersistence(reversed, dossierId).ok, false);

  const unknown = discoverSnapshot();
  unknown.data_lock = { ...(unknown.data_lock as object), reference_price: { min: 10, max: 12, midpoint: 11 } };
  assert.equal(validateResearchSnapshotForPersistence(unknown, dossierId).ok, false);
});

test("I3-B accepts a valid ANALYZE fixture and reconciles delta ER", () => {
  const snapshot = analyzeSnapshot();
  const result = validateResearchSnapshotForPersistence(snapshot, dossierId);
  assert.equal(result.ok, true, result.ok ? "" : result.errors.join(" | "));
  assert.equal((result as { ok: true; canonicalContract: { primaryExpectedReturnDeltaPercentagePoints: number } }).canonicalContract.primaryExpectedReturnDeltaPercentagePoints, 2);
});

test("I3-B rejects ambiguous N basis and deterministic mismatches", () => {
  const bothBases = analyzeSnapshot();
  (bothBases.l3_investment_valuation as Record<string, unknown>).valuation = { ...(bothBases.l3_investment_valuation as Record<string, unknown>).valuation as Record<string, unknown>, mature_normalization_return: 7 };
  assert.equal(validateResearchSnapshotForPersistence(bothBases, dossierId).ok, false);

  const mismatch = analyzeSnapshot();
  (mismatch.l2_research_fundamentals as Record<string, unknown>).business_quality = { ...(mismatch.l2_research_fundamentals as Record<string, unknown>).business_quality as Record<string, unknown>, oqs: 1 };
  assert.equal(validateResearchSnapshotForPersistence(mismatch, dossierId).ok, false);
});

test("I3-B accepts REFRESH and the mature-normalization N basis when it is the only applicable basis", () => {
  const refresh = analyzeSnapshot();
  refresh.execution_mode = "REFRESH";
  const valuation = (refresh.l3_investment_valuation as Record<string, unknown>).valuation as Record<string, unknown>;
  valuation.no_multiple_expansion_return = "NOT_AVAILABLE";
  valuation.mature_normalization_return = 8;
  assert.equal(validateResearchSnapshotForPersistence(refresh, dossierId).ok, true);
});

test("I3-B preserves frozen semantic return states", () => {
  const notAssessable = analyzeSnapshot();
  const l2 = notAssessable.l2_research_fundamentals as Record<string, unknown>;
  const quality = l2.business_quality as Record<string, unknown>;
  const certification = l2.certification as Record<string, unknown>;
  const valuation = (notAssessable.l3_investment_valuation as Record<string, unknown>).valuation as Record<string, unknown>;
  const investment = (notAssessable.l3_investment_valuation as Record<string, unknown>).investment as Record<string, unknown>;
  const orotitan = ((notAssessable.l4_operational_state as Record<string, unknown>).orotitan as Record<string, unknown>);
  valuation.primary_expected_return = "NOT_ASSESSABLE";
  valuation.no_multiple_expansion_return = "NOT_ASSESSABLE";
  valuation.mature_normalization_return = "NOT_AVAILABLE";
  valuation.valuation_reliability = "NOT_ASSESSABLE";
  valuation.margin_of_safety = "NOT_ASSESSABLE";
  valuation.market_expectation_gap = "NOT_ASSESSABLE";
  valuation.ovs = "NOT_ASSESSABLE";
  certification.score_permission = "ALLOWED";
  quality.oqs_raw = 80; quality.weak_link_cap = 100; quality.oqs = 80;
  investment.investment_raw = "NOT_AVAILABLE"; investment.investment_score = "NOT_AVAILABLE"; investment.investment_class = "NOT_AVAILABLE";
  orotitan.orotitan_status = "NO";
  const notAssessableResult = validateResearchSnapshotForPersistence(notAssessable, dossierId);
  assert.equal(notAssessableResult.ok, true, notAssessableResult.ok ? "" : notAssessableResult.errors.join(" | "));

  for (const reliability of ["HIGH", "MEDIUM", "LOW"] as const) {
    const suspended = analyzeSnapshot();
    const suspendedL2 = suspended.l2_research_fundamentals as Record<string, unknown>;
    const suspendedQuality = suspendedL2.business_quality as Record<string, unknown>;
    const suspendedCertification = suspendedL2.certification as Record<string, unknown>;
    const suspendedValuation = (suspended.l3_investment_valuation as Record<string, unknown>).valuation as Record<string, unknown>;
    const suspendedInvestment = (suspended.l3_investment_valuation as Record<string, unknown>).investment as Record<string, unknown>;
    const suspendedOrotitan = (suspended.l4_operational_state as Record<string, unknown>).orotitan as Record<string, unknown>;
    suspendedCertification.score_permission = "SUSPENDED";
    suspendedValuation.valuation_reliability = reliability;
    suspendedValuation.ovs = "NOT_AVAILABLE";
    suspendedQuality.oqs_raw = "NOT_AVAILABLE"; suspendedQuality.weak_link_cap = "NOT_AVAILABLE"; suspendedQuality.oqs = "NOT_AVAILABLE"; suspendedQuality.quality_class = "NOT_AVAILABLE";
    suspendedInvestment.investment_raw = "NOT_AVAILABLE"; suspendedInvestment.investment_score = "NOT_AVAILABLE"; suspendedInvestment.investment_class = "NOT_AVAILABLE";
    suspendedOrotitan.orotitan_status = "NO";
    const suspendedResult = validateResearchSnapshotForPersistence(suspended, dossierId);
    assert.equal(suspendedResult.ok, true, suspendedResult.ok ? "" : suspendedResult.errors.join(" | "));
  }
});

test("I3-B rejects a reversed canonical expected-return range", () => {
  const snapshot = analyzeSnapshot();
  const valuation = (snapshot.l3_investment_valuation as Record<string, unknown>).valuation as Record<string, unknown>;
  valuation.primary_expected_return = { min: 12, max: 8 };
  assert.equal(validateResearchSnapshotForPersistence(snapshot, dossierId).ok, false);
});

test("I3-B reconciles numeric dimension ranges and all persisted deterministic outputs", () => {
  const snapshot = analyzeSnapshot();
  const l2 = snapshot.l2_research_fundamentals as Record<string, unknown>;
  const quality = l2.business_quality as Record<string, unknown>;
  const valuation = (snapshot.l3_investment_valuation as Record<string, unknown>).valuation as Record<string, unknown>;
  const investment = (snapshot.l3_investment_valuation as Record<string, unknown>).investment as Record<string, unknown>;
  quality.moat_score = { min: 75, max: 85 };
  const computed = computeCanonicalSnapshot({
    dimensions: { MOAT: { min: 75, max: 85 }, RUNWAY: 80, RETURN_QUALITY: 80, CASH_ECONOMICS: 80, CAPITAL_ALLOCATION: 80, MANAGEMENT_GOVERNANCE: 80, RESILIENCE_RISK: 80 },
    evidence: { moat: "STRONGLY_SUPPORTED", runway: "STRONGLY_SUPPORTED" },
    businessResearchStatus: "CERTIFIED", investmentConclusionStatus: "CERTIFIED", scorePermission: "ALLOWED", mosStatus: "ROBUST", valuationReliability: "HIGH",
    primaryExpectedReturnDeltaPercentagePoints: 2, normalizedExpectedReturnDeltaPercentagePoints: -2,
    eliteGates: { researchFullyCertified: "PASS", moatElite: "PASS", runwayElite: "PASS", returnQualityElite: "PASS", cashEconomicsElite: "PASS", capitalAllocationElite: "PASS", managementGovernanceElite: "PASS", resilienceElite: "PASS", valuationElite: "PASS", materialWeakLink: "PASS" },
  });
  quality.oqs_raw = computed.oqsRaw; quality.weak_link_cap = computed.weakLinkCap; quality.oqs = computed.oqs;
  valuation.ovs = computed.ovs; investment.investment_raw = computed.investmentRaw; investment.investment_score = computed.investmentScore;
  assert.equal(validateResearchSnapshotForPersistence(snapshot, dossierId).ok, true);
  for (const [path, value] of [["ovs", 1], ["investment_score", 1]] as const) {
    const mismatch = analyzeSnapshot();
    const target = path === "ovs"
      ? (mismatch.l3_investment_valuation as Record<string, unknown>).valuation as Record<string, unknown>
      : (mismatch.l3_investment_valuation as Record<string, unknown>).investment as Record<string, unknown>;
    target[path] = value;
    assert.equal(validateResearchSnapshotForPersistence(mismatch, dossierId).ok, false);
  }
  const terminalMismatch = analyzeSnapshot();
  ((terminalMismatch.l4_operational_state as Record<string, unknown>).orotitan as Record<string, unknown>).orotitan_status = "NO";
  assert.equal(validateResearchSnapshotForPersistence(terminalMismatch, dossierId).ok, false);
});

test("I3-B executes the injectable writer behaviorally", async () => {
  let calls = 0;
  const seen: unknown[] = [];
  const rpc = async (args: unknown) => { calls += 1; seen.push(args); return { data: { status: "INSERTED", dossier_id: dossierId, snapshot_id: "00000000-0000-4000-8000-000000000004", current_snapshot_id: "00000000-0000-4000-8000-000000000004" }, error: null }; };
  await assert.rejects(() => persistValidatedResearchSnapshot({ dossierId, expectedCurrentSnapshotId: null, canonicalPayload: {} }, rpc), /validation failed/);
  assert.equal(calls, 0);
  const payload = discoverSnapshot();
  const inserted = await persistValidatedResearchSnapshot({ dossierId, expectedCurrentSnapshotId: null, canonicalPayload: payload }, rpc);
  assert.equal(inserted.status, "INSERTED");
  assert.equal(calls, 1);
  assert.deepEqual(seen[0], { p_dossier_id: dossierId, p_expected_current_snapshot_id: null, p_canonical_payload: payload });
  const idempotent = await persistValidatedResearchSnapshot({ dossierId, expectedCurrentSnapshotId: null, canonicalPayload: payload }, async () => ({ data: { status: "IDEMPOTENT_SUCCESS", dossier_id: dossierId, snapshot_id: payload.snapshot_id, current_snapshot_id: payload.snapshot_id }, error: null }));
  assert.equal(idempotent.status, "IDEMPOTENT_SUCCESS");
  await assert.rejects(() => persistValidatedResearchSnapshot({ dossierId, expectedCurrentSnapshotId: "bad", canonicalPayload: payload }, rpc), /expectedCurrentSnapshotId/);
  await assert.rejects(() => persistValidatedResearchSnapshot({ dossierId, expectedCurrentSnapshotId: null, canonicalPayload: payload }, async () => ({ data: null, error: new Error("rpc failure") })), /rpc failure/);
  await assert.rejects(() => persistValidatedResearchSnapshot({ dossierId, expectedCurrentSnapshotId: null, canonicalPayload: payload }, async () => ({ data: { status: "BROKEN" }, error: null })), /invalid result/);
});
import assert from "node:assert/strict";
import test from "node:test";
import { validateV2ResearchSnapshotForPersistence } from "../lib/orotitan-equity/v2/research-snapshot-schema";
import { persistValidatedV2ResearchSnapshot } from "../lib/orotitan-equity/v2/persistence-core";

const dossierId = "00000000-0000-4000-8000-000000000001";
const issuerId = "00000000-0000-4000-8000-000000000002";
const securityId = "00000000-0000-4000-8000-000000000003";

function v2Discover(): Record<string, unknown> {
  return {
    snapshot_id: "00000000-0000-4000-8000-000000000004",
    report_id: "v2-report-1",
    issuer_id: issuerId,
    security_id: securityId,
    execution_mode: "DISCOVER",
    data_lock: {
      data_cutoff: "2026-09-15",
      reference_price: 10,
      reference_price_currency: "USD",
      reference_price_date: "2026-09-15",
      calculation_date: "2026-09-15",
      last_full_research_date: "NOT_AVAILABLE",
      score_date: "NOT_AVAILABLE",
    },
    versions: {
      report_version: "r2",
      method_version: "m1",
      calculation_version: "c1",
      evidence_ledger_version: "e2",
      method_versions: {
        discovery_version: "d1", moat_version: "m1", runway_version: "r1", roic_version: "roic1",
        fcf_version: "fcf1", valuation_version: "v1", certification_version: "cert1",
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
    traceability: { report_id: "v2-report-1", evidence_ids: ["evidence-1"], calculation_ids: ["calculation-1"] },
    v2_product: {
      classification: {
        issuer_country_code: "US",
        primary_listing_country_code: "US",
        sector: "INFORMATION_TECHNOLOGY",
        industry_group: "CYBERSECURITY",
        business_model_primary: "RECURRING_SUBSCRIPTION",
        business_model_secondary: null,
        economic_exposure_regions: ["GLOBAL"],
        taxonomy_version: "OROTITAN_TAXONOMY_V2.0",
      },
      business_summary: { business_description_short: "Cloud security software sold to enterprises through recurring subscriptions." },
      investment_thesis: {
        quality_case: "Recurring subscription economics support durable cash generation.",
        valuation_case: "Expected return remains sensitive to the entry price.",
        key_risk: "Competitive platform bundling can pressure growth and pricing.",
      },
      portfolio_filters: { pea_eligibility: "NO", pea_eligibility_as_of: "2026-09-15", pea_eligibility_source_ref: "issuer/listing-jurisdiction" },
    },
  };
}

test("V2 composed validator accepts valid V1 core plus V2 overlay", () => {
  const result = validateV2ResearchSnapshotForPersistence(v2Discover(), dossierId);
  assert.equal(result.ok, true);
});

test("V2 composed validator rejects a non-ISO issuer country", () => {
  const snapshot = v2Discover();
  ((snapshot.v2_product as Record<string, unknown>).classification as Record<string, unknown>).issuer_country_code = "USA";
  const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
  assert.equal(result.ok, false);
  if (!result.ok) assert.equal(result.stage, "boundary");
});

test("V2 composed validator preserves V1 core schema fail-closed behavior", () => {
  const snapshot = v2Discover();
  delete snapshot.data_lock;
  const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
  assert.equal(result.ok, false);
  if (!result.ok) assert.equal(result.stage, "schema");
});

test("V2 product validation rejects recommendation language in business summary", () => {
  const snapshot = v2Discover();
  ((snapshot.v2_product as Record<string, unknown>).business_summary as Record<string, unknown>).business_description_short = "Undervalued software company to buy.";
  const result = validateV2ResearchSnapshotForPersistence(snapshot, dossierId);
  assert.equal(result.ok, false);
  if (!result.ok) assert.ok(result.errors.some((error) => error.includes("neutral")));
});

test("V2 persistence calls only the V2 RPC after composed validation", async () => {
  let called = false;
  const snapshot = v2Discover();
  const result = await persistValidatedV2ResearchSnapshot({ dossierId, expectedCurrentSnapshotId: null, canonicalPayload: snapshot }, async (args) => {
    called = true;
    assert.deepEqual(args.p_canonical_payload, snapshot);
    return { data: { status: "INSERTED", dossier_id: dossierId, snapshot_id: snapshot.snapshot_id, current_snapshot_id: snapshot.snapshot_id }, error: null };
  });
  assert.equal(called, true);
  assert.equal(result.status, "INSERTED");
});

test("V2 persistence rejects invalid product before RPC", async () => {
  const snapshot = v2Discover();
  ((snapshot.v2_product as Record<string, unknown>).classification as Record<string, unknown>).sector = "TECH";
  let called = false;
  await assert.rejects(() => persistValidatedV2ResearchSnapshot({ dossierId, expectedCurrentSnapshotId: null, canonicalPayload: snapshot }, async () => {
    called = true;
    return { data: null, error: null };
  }));
  assert.equal(called, false);
});

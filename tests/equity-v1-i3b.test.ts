import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import { validateResearchSnapshotForPersistence } from "../lib/orotitan-equity/v1/research-snapshot-schema";

const schemaPath = new URL("../contracts/orotitan-equity/v1/04_SCREENER_SCHEMA_V1_PATCHED.json", import.meta.url);
const integrationPath = new URL("../contracts/orotitan-equity/v1/04_INTEGRATION_SPEC_V1_PATCHED.md", import.meta.url);
const persistenceSource = readFileSync(new URL("../lib/orotitan-equity/v1/persistence.ts", import.meta.url), "utf8");
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

test("I3-B writer validates before RPC and preserves typed/error paths", () => {
  assert.match(persistenceSource, /if \(!validation\.ok\) throw new ResearchSnapshotValidationError/);
  assert.match(persistenceSource, /const response = await \(rpc \?\? defaultRpc\(\)\)\(/);
  assert.match(persistenceSource, /if \(response\.error\) throw response\.error/);
  assert.match(persistenceSource, /IDEMPOTENT_SUCCESS/);
});
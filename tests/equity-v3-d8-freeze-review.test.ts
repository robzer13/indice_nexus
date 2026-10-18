import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";

const root = new URL("../", import.meta.url);
const path = "docs/orotitan-equity/d8/OROTITAN_D8_REVIEW_BASELINE_V1_0.json";
const baseline = JSON.parse(readFileSync(new URL(path, root), "utf8")) as {
  manifest_version: string;
  repository: string;
  decision: string;
  freeze_scope: string;
  methodology_freeze_approved: boolean;
  i2_end_to_end_certified: boolean;
  i3b_end_to_end_certified: boolean;
  physical_writer_v31_approved: boolean;
  v31_writer_installed_in_production: boolean;
  provenance_body_schema_approved: boolean;
  published_snapshot_v31_admitted: boolean;
  production_migration_authorized: boolean;
  publication_authorized: boolean;
  gate: {production_state: string;reasons: string[]};
  immutable_review_pins: {path: string; git_blob_sha1: string}[];
};
function blobSha(data: Uint8Array): string {
  const bytes = Buffer.from(data);
  return createHash("sha1").update("blob " + bytes.byteLength + "\0").update(bytes).digest("hex");
}
function fileBytes(name: string): Buffer {
  assert.ok(!name.includes("..") && !name.startsWith("/"), "unsafe review path");
  return readFileSync(new URL(name, root));
}
test("D8 pins every reviewed source by its exact Git blob identity", () => {
  assert.equal(baseline.manifest_version, "OROTITAN_D8_REVIEW_BASELINE_V1.0");
  assert.equal(baseline.repository, "robzer13/indice_nexus");
  assert.equal(baseline.immutable_review_pins.length, 9);
  const unique = new Set<string>();
  for (const pin of baseline.immutable_review_pins) {
    assert.match(pin.git_blob_sha1, /^[0-9a-f]{40}$/);
    assert.ok(!unique.has(pin.path), "duplicate pin " + pin.path);
    unique.add(pin.path);
    assert.equal(blobSha(fileBytes(pin.path)), pin.git_blob_sha1, "review drift: " + pin.path);
  }
});
test("D8 detects byte drift, including an innocuous appended newline", () => {
  const first = baseline.immutable_review_pins[0];
  const bytes = fileBytes(first.path);
  assert.notEqual(blobSha(Buffer.concat([bytes, Buffer.from("\n")])), first.git_blob_sha1);
  assert.notEqual(createHash("sha1").update(bytes).digest("hex"), first.git_blob_sha1);
});
test("D8 technical pin does not confer analytical or physical writer authority", () => {
  assert.equal(baseline.decision, "TECHNICAL_DESIGN_BASELINE_PINNED_REVIEW_ONLY");
  assert.equal(baseline.freeze_scope, "GIT_BLOB_IDENTITIES_AND_NONPRODUCTION_REVIEW_BOUNDARIES_ONLY");
  assert.equal(baseline.gate.production_state, "NO_GO");
  for (const k of [
    "methodology_freeze_approved", "i2_end_to_end_certified", "i3b_end_to_end_certified",
    "physical_writer_v31_approved", "v31_writer_installed_in_production",
    "provenance_body_schema_approved", "published_snapshot_v31_admitted",
    "production_migration_authorized", "publication_authorized",
  ] as const) assert.equal(baseline[k], false, "D8 must not inflate " + k);
  assert.ok(baseline.gate.reasons.length >= 8);
  assert.ok(baseline.gate.reasons.includes("RAW_IMMUTABLE_BYTES_SHA256_VERSUS_JSONB_SERIALIZATION_NEEDS_EXACT_PROJECTION"));
});
test("D8 sandbox cannot be mistaken for a production writer or migration", () => {
  const sandbox = fileBytes("tests/postgres/d7-sandbox-writer.sql").toString("utf8");
  assert.ok(sandbox.includes("create function d7_sandbox.persist_v31_candidate"));
  assert.ok(!/create\s+(or\s+replace\s+)?function\s+public\.persist_orotitan_research_snapshot_v3_1/i.test(sandbox));
  assert.ok(sandbox.includes("D7_TEST_ONLY_APPROVED"));
  assert.ok(sandbox.includes("set search_path = pg_catalog, d7_sandbox"));
  assert.ok(!/set search_path = pg_catalog,\s*public/i.test(sandbox));
});
test("D8 baseline pins frozen upstream V3.1 authorities", () => {
  const expected = [
    "contracts/orotitan-equity/v3/OROTITAN_INTEGRATION_STAGE_CONTRACT_V3_1_FREEZE_V3.1.md",
    "contracts/orotitan-equity/v3/04_INTEGRATION_SPEC_V3_1.md",
    "contracts/orotitan-equity/v3/04_SCREENER_SCHEMA_V3_1.json",
    "docs/orotitan-equity/I3B_VALIDATED_SNAPSHOT_WRITER_V1.2.md",
  ];
  for (const path of expected) {
    assert.equal(baseline.immutable_review_pins.filter(x => x.path === path).length, 1);
  }
});

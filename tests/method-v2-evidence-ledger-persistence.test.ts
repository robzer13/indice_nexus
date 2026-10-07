import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { gunzipSync } from "node:zlib";
import { readFileSync } from "node:fs";
import test from "node:test";
import profile from "../contracts/orotitan-equity/method-v2/runtime/OROTITAN_METHOD_V2_EVIDENCE_LEDGER_PERSISTENCE_PROFILE_V1.0.json";
import pack from "../contracts/orotitan-equity/method-v2/OROTITAN_METHOD_V2_EXECUTION_CONTRACT_PIN_PACK_V1.0.json";
import v1Pack from "../contracts/orotitan-equity/v1/contract-pin-pack-v1/OROTITAN_CONTRACT_PIN_PACK_V1.json";
import {
  assertMethodV2EvidenceIdExists,
  findMethodV2EvidenceEntry,
  parseMethodV2EvidenceLedger,
} from "../lib/orotitan-equity/method-v2/evidence-ledger-persistence";
import {
  authoritySetSha256,
  METHOD_V2_AUTHORITY_SET_SHA256,
  methodV2Manifest,
  sha256,
} from "../lib/orotitan-equity/method-v2/authority";
import { computeContractSetSha256 } from "../lib/orotitan-equity/v1/stage-manifest";

const BASELINE = "0cb8c3d4d525054c43651a3d503ee2e6685ad918";
const PROFILE_PATH = "contracts/orotitan-equity/method-v2/runtime/OROTITAN_METHOD_V2_EVIDENCE_LEDGER_PERSISTENCE_PROFILE_V1.0.json";
const PROFILE_SHA256 = "8679e2aeb8f7be4569670629866a9ee2a63d933f5b04d6ede209aa8310e29a83";
const V1_PROCESS_PATH = "contracts/orotitan-equity/v1/execution-freeze/OROTITAN_RESEARCH_EXECUTION_PROCESS_V1_FREEZE_V1.0.md.gz";

function bytes(entries: unknown, extra: Record<string, unknown> = {}): Buffer {
  return Buffer.from(JSON.stringify({
    format: "OROTITAN_METHOD_V2_EVIDENCE_LEDGER",
    version: "1.0",
    entries,
    ...extra,
  }));
}

test("E01 exact vnext baseline is the branch ancestor", () => {
  assert.equal(profile.source_baseline_commit_sha, BASELINE);
  assert.equal(execFileSync("git", ["merge-base", "HEAD", BASELINE], { encoding: "utf8" }).trim(), BASELINE);
});

test("E02 governing V1 authority explicitly permits physical persistence layout", () => {
  const authority = gunzipSync(readFileSync(V1_PROCESS_PATH)).toString("utf8");
  assert.match(authority, /physical persistence layout for the already-frozen Evidence \/ Conflict \/ Calculation \/ Assumption semantics/);
  assert.match(authority, /semantics of the frozen ledgers are \*\*not\*\* open implementation questions/);
});

test("E03 governing V1 authority preserves the frozen Evidence Ledger semantics", () => {
  const authority = gunzipSync(readFileSync(V1_PROCESS_PATH)).toString("utf8");
  assert.match(authority, /There is one authoritative Evidence Ledger for a run/);
  for (const field of ["EVIDENCE_ID", "CLAIM_ID", "CLAIM / METRIC", "VALUE / STATEMENT", "PERIOD / AS_OF_DATE",
    "SOURCE", "ROOT_SOURCE_ID", "INDEPENDENCE_GROUP", "SOURCE_CLASS", "CLAIM_FIT", "SOURCE_DATE", "DATA_CUTOFF",
    "EPISTEMIC_TYPE", "FRESHNESS_STATE", "LIMITATIONS", "CONFLICT_STATUS"]) assert.match(authority, new RegExp(field.replace("/", "\\/")));
});

test("E04 analytical Method-V2 authority SHA is unchanged", () => {
  assert.equal(METHOD_V2_AUTHORITY_SET_SHA256, "1e97ad30595d24d10345cfcb58c8b6c0feeb7272144af1fc12d0affd2d2e33b2");
  assert.equal(authoritySetSha256(methodV2Manifest), METHOD_V2_AUTHORITY_SET_SHA256);
});

test("E05 inherited V1 and Method-V2 execution Contract Set SHAs are unchanged", () => {
  assert.equal(computeContractSetSha256(v1Pack.contract_pins), "34b009f05715bab482dbc00b02194b3714e9b2f8151872144677bb6db19f3c63");
  assert.equal(pack.inherited_v1_contract_set_sha256, v1Pack.contract_set_sha256);
  assert.equal(pack.contract_set_sha256, "23b75bf5c2d7448e8270e7e8f3a0223e639d0be3a1406b3c089e6e885dd063ea");
});

test("E06 profile identity is runtime serialization only and raw bytes are frozen", () => {
  const raw = readFileSync(PROFILE_PATH);
  assert.equal(profile.identity_domain, "METHOD_V2_RUNTIME_SERIALIZATION_NOT_ANALYTICAL_AUTHORITY");
  assert.equal(sha256(raw), PROFILE_SHA256);
  const tampered = Buffer.from(raw); tampered[0] ^= 1;
  assert.notEqual(sha256(tampered), PROFILE_SHA256);
});

test("E07 production remains inactive and all mutation boundaries are false", () => {
  assert.equal(profile.production_active, false);
  assert.ok(Object.values(profile.scope_boundary).every(value => value === false));
});

test("E08 valid strict UTF-8 JSON is accepted", () => {
  assert.equal(parseMethodV2EvidenceLedger(bytes([{ EVIDENCE_ID: "EV-1" }])).entries[0].EVIDENCE_ID, "EV-1");
  assert.throws(() => parseMethodV2EvidenceLedger(Uint8Array.from([0xc3, 0x28])), /INVALID_UTF8/);
  assert.throws(() => parseMethodV2EvidenceLedger(Buffer.from("{")), /INVALID_JSON/);
});

test("E09 wrong format or version is rejected", () => {
  assert.throws(() => parseMethodV2EvidenceLedger(Buffer.from(JSON.stringify({ format: "wrong", version: "1.0", entries: [] }))), /INVALID_FORMAT/);
  assert.throws(() => parseMethodV2EvidenceLedger(Buffer.from(JSON.stringify({ format: "OROTITAN_METHOD_V2_EVIDENCE_LEDGER", version: "2.0", entries: [] }))), /INVALID_VERSION/);
});

test("E10 non-array entries and malformed roots are rejected", () => {
  assert.throws(() => parseMethodV2EvidenceLedger(bytes({})), /INVALID_ENTRIES/);
  assert.throws(() => parseMethodV2EvidenceLedger(Buffer.from("[]")), /INVALID_ROOT/);
  assert.throws(() => parseMethodV2EvidenceLedger(bytes(["EV-1"])), /INVALID_ENTRY/);
});

test("E11 missing or empty direct EVIDENCE_ID is rejected", () => {
  assert.throws(() => parseMethodV2EvidenceLedger(bytes([{}])), /INVALID_EVIDENCE_ID/);
  assert.throws(() => parseMethodV2EvidenceLedger(bytes([{ EVIDENCE_ID: "" }])), /INVALID_EVIDENCE_ID/);
});

test("E12 duplicate EVIDENCE_ID is rejected", () => {
  assert.throws(() => parseMethodV2EvidenceLedger(bytes([{ EVIDENCE_ID: "EV-1" }, { EVIDENCE_ID: "EV-1" }])), /DUPLICATE_EVIDENCE_ID/);
});

test("E13 exact direct EVIDENCE_ID lookup succeeds", () => {
  assert.equal(assertMethodV2EvidenceIdExists(bytes([{ EVIDENCE_ID: "EV-1" }]), "EV-1").EVIDENCE_ID, "EV-1");
});

test("E14 unknown evidence ID is deterministically not found", () => {
  const ledger = bytes([{ EVIDENCE_ID: "EV-1" }]);
  assert.equal(findMethodV2EvidenceEntry(ledger, "EV-2"), undefined);
  assert.throws(() => assertMethodV2EvidenceIdExists(ledger, "EV-2"), /EVIDENCE_ID_NOT_FOUND/);
});

test("E15 aliases are never accepted as entry identity", () => {
  assert.throws(() => parseMethodV2EvidenceLedger(bytes([{ evidence_id: "EV-1" }])), /INVALID_EVIDENCE_ID/);
  assert.throws(() => parseMethodV2EvidenceLedger(bytes([{ evidenceId: "EV-1" }])), /INVALID_EVIDENCE_ID/);
  assert.throws(() => parseMethodV2EvidenceLedger(bytes([{ id: "EV-1" }])), /INVALID_EVIDENCE_ID/);
});

test("E16 nested recursive EVIDENCE_ID is never used", () => {
  assert.throws(() => parseMethodV2EvidenceLedger(bytes([{ nested: { EVIDENCE_ID: "EV-1" } }])), /INVALID_EVIDENCE_ID/);
});

test("E17 unknown additional ledger properties remain allowed and uninterpreted", () => {
  const entry = assertMethodV2EvidenceIdExists(bytes([{ EVIDENCE_ID: "EV-1", FUTURE_FIELD: { arbitrary: true } }], { FUTURE_ROOT: 1 }), "EV-1");
  assert.deepEqual(entry.FUTURE_FIELD, { arbitrary: true });
});

test("E18 lookup makes no evidence-date or source-date semantic inference", () => {
  const ledger = bytes([{ EVIDENCE_ID: "EV-1", SOURCE_DATE: "2099-01-01", AS_OF_DATE: "1900-01-01", PERIOD: "arbitrary" }]);
  assert.equal(assertMethodV2EvidenceIdExists(ledger, "EV-1").EVIDENCE_ID, "EV-1");
  assert.equal(findMethodV2EvidenceEntry(ledger, "ev-1"), undefined);
});

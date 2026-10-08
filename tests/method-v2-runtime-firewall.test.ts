import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import test from "node:test";
import binding from "../contracts/orotitan-equity/method-v2/runtime/OROTITAN_METHOD_V2_RUNTIME_BINDING_V1.0.json";
import pack from "../contracts/orotitan-equity/method-v2/OROTITAN_METHOD_V2_EXECUTION_CONTRACT_PIN_PACK_V1.0.json";
import { sha256 } from "../lib/orotitan-equity/method-v2/authority";
import { resolvePersistedMethodV2Artifact, METHOD_V2_RUNTIME_BINDING_SHA256, type ArtifactExpectation, type RegistryArtifact } from "../lib/orotitan-equity/method-v2/persisted-artifacts";
import { verifyAndRecordMethodV2Challenge, verifyPersistedMethodV2Challenge, type ChallengeIdentity, type ChallengeRegistry } from "../lib/orotitan-equity/method-v2/persisted-challenge";
import { challengeFixture } from "./method-v2-runtime-fixtures";

const BINDING_PATH = "contracts/orotitan-equity/method-v2/runtime/OROTITAN_METHOD_V2_RUNTIME_BINDING_V1.0.json";
const SQL_PATH = "migrations/20261007212324_orotitan_registry_v1_14_method_v2_runtime_firewall.sql";
const evidenceId = "00000000-0000-4000-8000-000000000008";
function runtimeFixture(evidenceBytes = Buffer.from('{"format":"OROTITAN_METHOD_V2_EVIDENCE_LEDGER","version":"1.0","entries":[{"EVIDENCE_ID":"E1"}]}'),
  mutate?: (f: ReturnType<typeof challengeFixture>) => void) {
  const f = challengeFixture();
  const fundBytes = Buffer.from('{"lock":"fundamentals"}');
  const valBytes = Buffer.from('{"lock":"valuation"}');
  f.context.fundamentals_lock_ref.content_sha256 = sha256(fundBytes);
  f.context.valuation_lock_ref.content_sha256 = sha256(valBytes);
  f.report.saturation_record.family_coverage.WORLD_IN_MOTION = { disposition: "EVIDENCED_NO_MATERIAL_CHALLENGE",
    rationale: "Exact supplier evidence", evidence_references: [{ evidence_id: "E1", evidence_date: "2026-09-30",
      evidence_ledger_ref: { artifact_id: evidenceId, version: 1, content_sha256: sha256(evidenceBytes) } }] };
  mutate?.(f);
  Object.assign(f.ledger, { fundamentals_lock_ref: f.context.fundamentals_lock_ref, valuation_lock_ref: f.context.valuation_lock_ref });
  Object.assign(f.report, { fundamentals_lock_ref: f.context.fundamentals_lock_ref, valuation_lock_ref: f.context.valuation_lock_ref });
  const { ledgerBytes, reportBytes } = f.bytes();
  const rows = new Map<string, RegistryArtifact>(); const stored = new Map<string, Uint8Array>(); const downloads: string[] = [];
  function add(id: string, kind: string, bytes: Uint8Array): ArtifactExpectation {
    const path = `runs/${f.context.run_id}/${id}/1.json`;
    const row: RegistryArtifact = { artifact_id: id, version: 1, content_sha256: sha256(bytes), run_id: f.context.run_id,
      stage_code: "DEEP_DIVE", artifact_type: kind, authority_class: "AUTHORITATIVE_STAGE_OUTPUT", authority_state: "AUTHORITATIVE",
      artifact_status: "SEALED", availability_state: "AVAILABLE", storage_backend: "SUPABASE_STORAGE",
      supabase_bucket: "orotitan-text-artifacts-v1", supabase_object_path: path, storage_uri: `supabase://orotitan-text-artifacts-v1/${path}`,
      size_bytes: bytes.byteLength, hash_algorithm: "SHA-256", media_type: "application/json", manifest_artifact_id: null, manifest_version: null };
    rows.set(id, row); stored.set(path, bytes);
    return { ref: { artifact_id: id, version: 1, content_sha256: row.content_sha256 }, runId: row.run_id,
      stageCode: row.stage_code, artifactType: kind, authorityClass: "AUTHORITATIVE_STAGE_OUTPUT", authorityState: "AUTHORITATIVE", artifactStatus: "SEALED" };
  }
  const identity: ChallengeIdentity = { context: f.context, methodologyGeneration: "METHOD_V2", contractSetSha256: pack.contract_set_sha256,
    stageRevision: 1, runtimeBindingSha256: METHOD_V2_RUNTIME_BINDING_SHA256,
    questionLedger: add(f.context.question_ledger_ref.artifact_id, "PRE_CERTIFICATION_QUESTION_LEDGER", ledgerBytes),
    challengeReport: add(f.context.challenge_report_ref.artifact_id, "PRE_CERTIFICATION_CHALLENGE_REPORT", reportBytes),
    fundamentalsLock: add(f.context.fundamentals_lock_ref.artifact_id, "FUNDAMENTALS_LOCK", fundBytes),
    valuationLock: add(f.context.valuation_lock_ref.artifact_id, "VALUATION_LOCK", valBytes) };
  const coverage = f.report.saturation_record.family_coverage.WORLD_IN_MOTION;
  if (coverage.disposition !== "EVIDENCED_NO_MATERIAL_CHALLENGE") throw new Error("fixture exemption required");
  const evidence = add(coverage.evidence_references[0].evidence_ledger_ref.artifact_id, "EVIDENCE_LEDGER", evidenceBytes);
  const source: ChallengeRegistry = {
    async readArtifact(id, version) { const row = rows.get(id); return row && row.version === version ? structuredClone(row) : null; },
    async download(bucket, path) { assert.equal(bucket, "orotitan-text-artifacts-v1"); downloads.push(path); const bytes = stored.get(path); if (!bytes) throw new Error("missing object"); return bytes; },
    async readChallengeIdentity() { return structuredClone(identity); },
    async readPriorPassingIdentity() { throw new Error("prior unavailable"); },
    async readEvidenceExpectation() { return structuredClone(evidence); },
  };
  return { f, identity, evidence, source, rows, stored, downloads,
    verify: () => verifyPersistedMethodV2Challenge(source, f.context.run_id) };
}

function multiReferenceFixture(mode: "FULL" | "DELTA", difference?: "artifact_id" | "version" | "content_sha256", multipleIds = false,
  prior = runtimeFixture()) {
  const evidenceBytes = Buffer.from('{"format":"OROTITAN_METHOD_V2_EVIDENCE_LEDGER","version":"1.0","entries":[{"EVIDENCE_ID":"E1"},{"EVIDENCE_ID":"E2"}]}');
  const differentBytes = Buffer.from('{"format":"OROTITAN_METHOD_V2_EVIDENCE_LEDGER","version":"1.0","entries":[{"EVIDENCE_ID":"E1"},{"EVIDENCE_ID":"E2"}],"note":"different bytes"}');
  const current = runtimeFixture(evidenceBytes, f => {
    if (mode === "DELTA") {
      f.context.run_id = "00000000-0000-4000-8000-000000000099";
      for (const [ref, suffix] of [[f.context.question_ledger_ref, "94"], [f.context.challenge_report_ref, "95"],
        [f.context.fundamentals_lock_ref, "92"], [f.context.valuation_lock_ref, "93"]] as const) {
        ref.artifact_id = `00000000-0000-4000-8000-0000000000${suffix}`;
      }
      const delta = { prior_passing_report_ref: prior.identity.context.challenge_report_ref,
        prior_passing_ledger_ref: prior.identity.context.question_ledger_ref,
        impact_map: "Exact change revalidated", inherited_coverage_revalidated: true };
      Object.assign(f.ledger, { run_id: f.context.run_id, mode, delta_provenance: delta });
      Object.assign(f.report, { run_id: f.context.run_id, mode, delta_provenance: delta });
      f.context.priorPassing = { report_ref: delta.prior_passing_report_ref, ledger_ref: delta.prior_passing_ledger_ref };
    }
    const first = f.report.saturation_record.family_coverage.WORLD_IN_MOTION;
    if (first.disposition !== "EVIDENCED_NO_MATERIAL_CHALLENGE") throw new Error("fixture exemption required");
    if (mode === "DELTA") first.evidence_references[0].evidence_ledger_ref.artifact_id = "00000000-0000-4000-8000-000000000098";
    const second = structuredClone(first);
    const reference = second.evidence_references[0];
    if (multipleIds) reference.evidence_id = "E2";
    if (difference === "artifact_id") reference.evidence_ledger_ref.artifact_id = "00000000-0000-4000-8000-000000000088";
    if (difference === "version") reference.evidence_ledger_ref.version = 2;
    if (difference === "content_sha256") reference.evidence_ledger_ref.content_sha256 = sha256(differentBytes);
    f.report.saturation_record.family_coverage.TECHNOLOGY_AI = second;
  });
  const second = current.f.report.saturation_record.family_coverage.TECHNOLOGY_AI;
  if (second.disposition !== "EVIDENCED_NO_MATERIAL_CHALLENGE") throw new Error("fixture exemption required");
  const secondRef = second.evidence_references[0].evidence_ledger_ref;
  const secondRow: RegistryArtifact = { ...current.rows.get(current.evidence.ref.artifact_id)!, ...secondRef };
  const secondPath = `runs/${current.identity.context.run_id}/${secondRef.artifact_id}/${secondRef.version}.json`;
  secondRow.supabase_object_path = secondPath; secondRow.storage_uri = `supabase://orotitan-text-artifacts-v1/${secondPath}`;
  const secondBytes = difference === "content_sha256" ? differentBytes : evidenceBytes;
  secondRow.size_bytes = secondBytes.length;
  const originalRead = current.source.readArtifact; const originalDownload = current.source.download;
  current.source.readArtifact = async (id, version) => {
    if (mode === "DELTA" && prior.rows.has(id)) return prior.source.readArtifact(id, version);
    if (difference && difference !== "content_sha256" && id === secondRef.artifact_id && version === secondRef.version) return structuredClone(secondRow);
    return originalRead(id, version);
  };
  current.source.download = async (bucket, path) => {
    if (mode === "DELTA" && prior.stored.has(path)) return prior.source.download(bucket, path);
    if (difference && difference !== "content_sha256" && path === secondPath) {
      current.downloads.push(path); return secondBytes;
    }
    return originalDownload(bucket, path);
  };
  current.source.readEvidenceExpectation = async (ref, identity) => {
    if (mode === "DELTA" && identity.context.run_id === prior.identity.context.run_id) return prior.source.readEvidenceExpectation(ref, identity);
    return { ...structuredClone(current.evidence), ref: structuredClone(ref) };
  };
  current.source.readPriorPassingIdentity = async () => structuredClone(prior.identity);
  return current;
}

for (const mode of ["FULL", "DELTA"] as const) {
  test(`B41 ${mode}: two exemption families use the same exact authoritative Ledger`, async () => {
    const f = multiReferenceFixture(mode);
    assert.equal((await f.verify()).admission.allowed, true);
    const path = f.rows.get(f.evidence.ref.artifact_id)!.supabase_object_path;
    assert.equal(f.downloads.filter(p => p === path).length, 2);
  });
  for (const field of ["artifact_id", "version", "content_sha256"] as const) {
    test(`B41 ${mode}: different Evidence Ledger ${field} rejects with the authority mismatch error`, async () => {
      await assert.rejects(multiReferenceFixture(mode, field).verify(), { message: "METHOD_V2_EVIDENCE_LEDGER_AUTHORITY_MISMATCH" });
    });
  }
  test(`B41 ${mode}: multiple exact evidence IDs from the same authoritative Ledger pass`, async () => {
    assert.equal((await multiReferenceFixture(mode, undefined, true).verify()).admission.allowed, true);
  });
}

test("B41 DELTA: prior passing reports must also preserve single-Ledger authority", async () => {
  const prior = multiReferenceFixture("FULL", "artifact_id");
  await assert.rejects(multiReferenceFixture("DELTA", undefined, false, prior).verify(),
    { message: "METHOD_V2_EVIDENCE_LEDGER_AUTHORITY_MISMATCH" });
});

test("B07 B08 B09: inactive binding exact raw bytes, immutable full-pack locator and serialization identity", () => {
  assert.equal(sha256(readFileSync(BINDING_PATH)), METHOD_V2_RUNTIME_BINDING_SHA256);
  assert.equal(binding.methodology_authority_sha256, pack.method_v2_authority_set_sha256);
  assert.equal(binding.contract_set_sha256, pack.contract_set_sha256);
  assert.equal(binding.evidence_ledger_persistence_profile_sha256, "8679e2aeb8f7be4569670629866a9ee2a63d933f5b04d6ede209aa8310e29a83");
  assert.equal(binding.production_active, false); assert.equal(binding.publication_active, false); assert.equal(binding.canary_active, false);
  const loc = binding.execution_pack_locator;
  const bytes = execFileSync("git", ["show", `${loc.commit_sha}:${loc.path}`]);
  assert.equal(execFileSync("git", ["rev-parse", `${loc.commit_sha}:${loc.path}`], { encoding: "utf8" }).trim(), loc.blob_sha);
  assert.deepEqual(JSON.parse(bytes.toString()), pack); assert.equal(Object.keys(pack.contract_pins).length, 13);
  const sql = readFileSync(SQL_PATH, "utf8");
  const pins = JSON.parse(sql.match(/select '(.+)'::jsonb \$pins\$/)![1]);
  assert.deepEqual(pins, pack.contract_pins);
});

test("B10 B44: selector and frozen authority bytes unchanged; no publisher or activation wiring", () => {
  for (const path of ["lib/orotitan-equity/runtime.ts", "lib/orotitan-equity/method-v2/pre-certification.ts",
    "lib/orotitan-equity/method-v2/evidence-ledger-persistence.ts", "contracts/orotitan-equity/method-v2/OROTITAN_METHOD_V2_AUTHORITY_MANIFEST_V1.1.json",
    "contracts/orotitan-equity/method-v2/OROTITAN_METHOD_V2_EXECUTION_CONTRACT_PIN_PACK_V1.0.json",
    "contracts/orotitan-equity/method-v2/runtime/OROTITAN_METHOD_V2_EVIDENCE_LEDGER_PERSISTENCE_PROFILE_V1.0.json",
    "migrations/20261005195139_orotitan_registry_v1_13_method_generation.sql"]) {
    assert.equal(sha256(readFileSync(path)), sha256(execFileSync("git", ["show", `e5c3b107116a998496eb15a3c0a81546c3c827b2:${path}`])));
  }
  assert.doesNotMatch(readFileSync(SQL_PATH, "utf8"), /grant execute|PUBLISH_SUCCEEDED|insert into public\.research_snapshots|update public\.research_snapshots/i);
  const server = readFileSync("lib/orotitan-equity/method-v2/persisted-artifacts-server.ts", "utf8");
  assert.match(server, /import "server-only"/); assert.match(server, /createServerSupabaseClient/);
  assert.doesNotMatch(server, /createClient\(|process\.env|\.insert\(|\.update\(/);
});

test("B37: approved Storage backends and private text buckets only; caller paths absent", async () => {
  const bad = [ { storage_backend: "PRIVATE_GITHUB" }, { supabase_bucket: "public" }, { supabase_object_path: "../escape" },
    { supabase_object_path: "" }, { storage_uri: "supabase://different/path" }, { supabase_bucket: "orotitan-source-files-v1" }, { media_type: "text/plain" } ];
  for (const mutation of bad) {
    const f = runtimeFixture(); Object.assign(f.rows.get(f.identity.challengeReport.ref.artifact_id)!, mutation);
    await assert.rejects(f.verify(), /STORAGE_NOT_APPROVED/);
  }
});

test("B38: actual downloaded bytes must match exact Registry size and SHA-256", async () => {
  for (const kind of ["size", "hash"] as const) {
    const f = runtimeFixture(); const row = f.rows.get(f.identity.valuationLock.ref.artifact_id)!;
    f.stored.set(row.supabase_object_path!, kind === "size" ? Buffer.from("short") : Buffer.alloc(row.size_bytes, 120));
    await assert.rejects(f.verify(), kind === "size" ? /SIZE_MISMATCH/ : /SHA256_MISMATCH/);
  }
});

test("B39: every Registry identity, status, availability and authority dimension fails closed", async () => {
  const mutations = [ { artifact_id: "wrong" }, { version: 2 }, { run_id: "wrong" }, { stage_code: "RESEARCH" },
    { artifact_type: "OTHER" }, { authority_class: "HUMAN_SUMMARY" }, { authority_state: "SUPERSEDED" },
    { artifact_status: "INVALIDATED" }, { availability_state: "MISSING" }, { content_sha256: "0".repeat(64) },
    { size_bytes: NaN }, { size_bytes: -1 }, { hash_algorithm: "MD5" } ];
  for (const mutation of mutations) {
    const f = runtimeFixture(); Object.assign(f.rows.get(f.identity.questionLedger.ref.artifact_id)!, mutation);
    await assert.rejects(f.verify(), /REGISTRY_MISMATCH/);
  }
  const f = runtimeFixture(); f.rows.delete(f.identity.questionLedger.ref.artifact_id);
  await assert.rejects(f.verify(), /REGISTRY_MISMATCH/);
  const other = runtimeFixture();
  await assert.rejects(resolvePersistedMethodV2Artifact(other.source, { ...other.identity.questionLedger,
    manifestRef: { artifact_id: "wrong", version: 1 } }), /REGISTRY_MISMATCH/);
});

test("B40: validator receives reread ledger/report and both current lock bytes", async () => {
  const f = runtimeFixture(); const result = await f.verify();
  assert.equal(result.admission.allowed, true); assert.equal(result.proof.challenge_report_sha256, f.identity.context.challenge_report_ref.content_sha256);
  assert.equal(result.proof.validator_identity, "verifyPersistedMethodV2Challenge:1.0");
  assert.equal(f.downloads.length, 5);
  await f.verify(); assert.equal(f.downloads.length, 10);
  const negative = runtimeFixture(undefined, f => { f.report.fail_reasons = ["Unresolved"]; });
  await assert.rejects(negative.verify(), /NEGATIVE_DISPOSITION_UNRESOLVED/);
  const absent = runtimeFixture(); absent.rows.delete(absent.identity.fundamentalsLock.ref.artifact_id);
  await assert.rejects(absent.verify(), /REGISTRY_MISMATCH/);
});

test("B41: exact persisted-byte chain resolves direct EVIDENCE_ID", async () => {
  assert.equal((await runtimeFixture().verify()).admission.allowed, true);
  const different = runtimeFixture(undefined, f => {
    const coverage = f.report.saturation_record.family_coverage.WORLD_IN_MOTION;
    if (coverage.disposition === "EVIDENCED_NO_MATERIAL_CHALLENGE") coverage.evidence_references[0].evidence_id = "e1";
  });
  await assert.rejects(different.verify(), /EVIDENCE_ID_NOT_FOUND/);
  const wrongRef = runtimeFixture(); wrongRef.evidence.ref.version = 2;
  await assert.rejects(wrongRef.verify(), /EVIDENCE_REGISTRY_LINEAGE_MISMATCH/);
  const tampered = runtimeFixture(); const row = tampered.rows.get(evidenceId)!;
  tampered.stored.set(row.supabase_object_path!, Buffer.alloc(row.size_bytes));
  await assert.rejects(tampered.verify(), /SHA256_MISMATCH/);
});

test("B41: aliases, nested IDs, duplicate member syntax and duplicate values stay fail-closed after hashing", async () => {
  for (const entry of ['{"evidence_id":"E1"}', '{"evidenceId":"E1"}', '{"nested":{"EVIDENCE_ID":"E1"}}',
    '{"EVIDENCE_ID":"E1","EVIDENCE_ID":"E2"}', '{"EVIDENCE_ID":"E1","EVIDENCE\\u005fID":"E2"}']) {
    await assert.rejects(runtimeFixture(Buffer.from(`{"format":"OROTITAN_METHOD_V2_EVIDENCE_LEDGER","version":"1.0","entries":[${entry}]}`)).verify(), /EVIDENCE_LEDGER_/);
  }
  const dup = Buffer.from('{"format":"OROTITAN_METHOD_V2_EVIDENCE_LEDGER","version":"1.0","entries":[{"EVIDENCE_ID":"E1"},{"EVIDENCE_ID":"E1"}]}');
  await assert.rejects(runtimeFixture(dup).verify(), /DUPLICATE_EVIDENCE_ID/);
});

test("B41: Challenge owns cutoff; SOURCE_DATE/AS_OF_DATE/PERIOD never substitute for evidence_date", async () => {
  const bytes = Buffer.from('{"format":"OROTITAN_METHOD_V2_EVIDENCE_LEDGER","version":"1.0","entries":[{"EVIDENCE_ID":"E1","SOURCE_DATE":"2099-01-01","AS_OF_DATE":"2099-01-01","PERIOD":"2099"}]}');
  assert.equal((await runtimeFixture(bytes).verify()).admission.allowed, true);
  await assert.rejects(runtimeFixture(bytes, f => {
    const coverage = f.report.saturation_record.family_coverage.WORLD_IN_MOTION;
    if (coverage.disposition === "EVIDENCED_NO_MATERIAL_CHALLENGE") coverage.evidence_references[0].evidence_date = "2026-10-02";
  }).verify(), /POST_CUTOFF_EVIDENCE/);
});

test("B40: DELTA requires persisted prior passing lineage, reread through the same validator", async () => {
  const prior = runtimeFixture();
  const delta = { prior_passing_report_ref: prior.identity.context.challenge_report_ref,
    prior_passing_ledger_ref: prior.identity.context.question_ledger_ref,
    impact_map: "Exact change revalidated", inherited_coverage_revalidated: true };
  const current = runtimeFixture(undefined, f => {
    f.context.run_id = "00000000-0000-4000-8000-000000000099";
    f.context.question_ledger_ref.artifact_id = "00000000-0000-4000-8000-000000000094";
    f.context.challenge_report_ref.artifact_id = "00000000-0000-4000-8000-000000000095";
    f.context.fundamentals_lock_ref.artifact_id = "00000000-0000-4000-8000-000000000092";
    f.context.valuation_lock_ref.artifact_id = "00000000-0000-4000-8000-000000000093";
    const coverage = f.report.saturation_record.family_coverage.WORLD_IN_MOTION;
    if (coverage.disposition === "EVIDENCED_NO_MATERIAL_CHALLENGE") coverage.evidence_references[0].evidence_ledger_ref.artifact_id = "00000000-0000-4000-8000-000000000098";
    Object.assign(f.ledger, { run_id: f.context.run_id, mode: "DELTA", delta_provenance: delta });
    Object.assign(f.report, { run_id: f.context.run_id, mode: "DELTA", delta_provenance: delta });
    f.context.priorPassing = { report_ref: delta.prior_passing_report_ref, ledger_ref: delta.prior_passing_ledger_ref };
  });
  await assert.rejects(current.verify(), /prior unavailable/);
  current.source.readPriorPassingIdentity = async () => structuredClone(prior.identity);
  current.source.readEvidenceExpectation = async (_ref, identity) => identity.context.run_id === prior.identity.context.run_id ? structuredClone(prior.evidence) : structuredClone(current.evidence);
  const originalRead = current.source.readArtifact; const originalDownload = current.source.download;
  current.source.readArtifact = async (id, version) => prior.rows.has(id) ? prior.source.readArtifact(id, version) : originalRead(id, version);
  current.source.download = async (bucket, path) => prior.stored.has(path) ? prior.source.download(bucket, path) : originalDownload(bucket, path);
  // Shared lock and Evidence Ledger identities are exact across this synthetic DELTA.
  assert.equal((await current.verify()).admission.allowed, true);
  const row = prior.rows.get(prior.identity.challengeReport.ref.artifact_id)!;
  prior.stored.set(row.supabase_object_path!, Buffer.alloc(row.size_bytes));
  await assert.rejects(current.verify(), /SHA256_MISMATCH/);
});

test("B40 B43: trusted proof sink receives only persisted-byte validator output, never caller PASS", async () => {
  const f = runtimeFixture(); const writes: unknown[] = [];
  const writer = { async recordVerifiedProof(proof: unknown) { writes.push(proof); } };
  assert.equal((await verifyAndRecordMethodV2Challenge(f.source, f.identity.context.run_id, writer)).allowed, true);
  assert.deepEqual(writes, [(await f.verify()).proof]);
  const invalid = runtimeFixture(undefined, f => { f.report.fail_reasons = ["FAIL despite caller PASS"]; });
  await assert.rejects(verifyAndRecordMethodV2Challenge(invalid.source, invalid.identity.context.run_id, writer), /NEGATIVE_DISPOSITION_UNRESOLVED/);
  assert.equal(writes.length, 1);
});

test("B40 B43: FINAL candidate bytes verify before finalization and bind the exact current revision and outputs", async () => {
  const f = runtimeFixture(); const manifestId = "00000000-0000-4000-8000-000000000010";
  const artifacts = [f.identity.questionLedger, f.identity.challengeReport, f.identity.fundamentalsLock, f.identity.valuationLock];
  const raw = { run_id: f.identity.context.run_id, stage: "DEEP_DIVE", manifest_id: manifestId, manifest_kind: "FINAL",
    stage_revision: 1, contract_set_sha256: pack.contract_set_sha256,
    output_artifacts: artifacts.map(a => ({ ...a.ref, artifact_type: a.artifactType, authority_class: "AUTHORITATIVE_STAGE_OUTPUT" })) };
  const bytes = Buffer.from(JSON.stringify(raw)); const path = "candidate/final.json";
  const row: RegistryArtifact = { ...f.rows.get(f.identity.questionLedger.ref.artifact_id)!, artifact_id: manifestId,
    artifact_type: "DEEP_DIVE_STAGE_MANIFEST", content_sha256: sha256(bytes), size_bytes: bytes.length,
    supabase_object_path: path, storage_uri: `supabase://orotitan-text-artifacts-v1/${path}` };
  f.rows.set(manifestId, row); f.stored.set(path, bytes);
  const manifestRef = { artifact_id: manifestId, version: 1, content_sha256: sha256(bytes) };
  f.identity.candidateManifest = { ...f.identity.questionLedger, ref: manifestRef, artifactType: "DEEP_DIVE_STAGE_MANIFEST" };
  for (const artifact of artifacts) {
    artifact.manifestRef = { artifact_id: manifestId, version: 1 };
    Object.assign(f.rows.get(artifact.ref.artifact_id)!, { manifest_artifact_id: manifestId, manifest_version: 1 });
  }
  assert.equal((await verifyPersistedMethodV2Challenge(f.source, f.identity.context.run_id, manifestRef)).admission.allowed, true);
  f.identity.stageRevision = 2;
  await assert.rejects(verifyPersistedMethodV2Challenge(f.source, f.identity.context.run_id, manifestRef), /CANDIDATE_MISMATCH/);
  f.identity.stageRevision = 1;
  f.stored.set(path, Buffer.alloc(bytes.length));
  await assert.rejects(verifyPersistedMethodV2Challenge(f.source, f.identity.context.run_id, manifestRef), /SHA256_MISMATCH/);
});

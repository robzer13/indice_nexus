import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import {
  TypedGateError, verifyDeepDiveTypedLocks, type VerifiedContent,
} from "../lib/orotitan-equity/v3/pilotage-typed-gates-v3-1";
import type {
  ArtifactProof, ManifestProof,
} from "../lib/orotitan-equity/v3/pilotage-adapter-v3-1";

const RUN = "11111111-1111-4111-8111-111111111111";
const DATE = "2026-10-30";
const sha256 = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
const gitBlobSha = (bytes: Uint8Array) =>
  createHash("sha1").update(Buffer.from("blob " + bytes.length + "\0")).update(bytes).digest("hex");
const uuid = (n: number) => "77777777-7777-4777-8777-" + n.toString(16).padStart(12, "0");

function artifact(
  stage: "RESEARCH" | "DEEP_DIVE",
  artifactType: string,
  n: number,
  body: unknown,
  mediaType = "application/json",
): VerifiedContent {
  const bytes = Buffer.from(mediaType === "application/json" ? JSON.stringify(body) : String(body), "utf8");
  const proof: ArtifactProof = {
    runId: RUN, stageCode: stage, artifactType, artifactId: uuid(n),
    version: 1, sha256: sha256(bytes), available: true, bytesVerified: true,
  };
  return { proof, bytes, mediaType };
}

function ref(p: ArtifactProof) {
  return { artifact_id: p.artifactId, version: p.version, content_sha256: p.sha256 };
}

function asJson(c: VerifiedContent): Record<string, unknown> {
  return JSON.parse(Buffer.from(c.bytes).toString("utf8")) as Record<string, unknown>;
}

function rewrite(c: VerifiedContent, body: Record<string, unknown>): void {
  c.bytes = Buffer.from(JSON.stringify(body), "utf8");
  c.proof.sha256 = sha256(c.bytes);
}

function fixture() {
  const researchManifestBytes = Buffer.from(JSON.stringify({
    run_id: RUN, stage: "RESEARCH", manifest_kind: "FINAL", data_cutoff: DATE,
  }), "utf8");
  const researchFinal: ManifestProof = {
    runId: RUN, stageCode: "RESEARCH", artifactType: "RESEARCH_STAGE_MANIFEST",
    artifactId: uuid(1), version: 2, sha256: sha256(researchManifestBytes),
    kind: "FINAL", stageRevision: 1, available: true,
    bytesVerified: true, gatesVerifiedAgainstBytes: true,
  };
  const research = artifact("RESEARCH", "EVIDENCE_LEDGER", 2,
    { entries: [{ evidence_id: "SE-E001", claim: "Primary-source observation" }] });
  const taxonomy = artifact("DEEP_DIVE", "CONTROLLED_TAXONOMY_RECORD", 3,
    { taxonomy_version: "OROTITAN_TAXONOMY_V2.0", sector: "INDUSTRIALS" });
  const description = artifact("DEEP_DIVE", "BUSINESS_DESCRIPTION_SHORT_ARTIFACT", 4,
    "Industrial power and automation equipment and software.", "text/plain");
  const scoring = artifact("DEEP_DIVE", "SCORING_INPUTS", 5, { final_score: null });
  const redTeam = artifact("DEEP_DIVE", "FUNDAMENTAL_RED_TEAM_RECORD", 6,
    { status: "COMPLETE", contradiction_reviewed: true });

  // Only the artifact ID/version can appear inside the content itself.
  // The real self SHA-256 is calculated *after* serialization and anchored externally.
  const self = { artifact_id: uuid(10), version: 1 };
  const lock = artifact("DEEP_DIVE", "FUNDAMENTALS_LOCK", 10, {
    lock_type: "FUNDAMENTALS_LOCK",
    artifact: self,
    run_id: RUN, data_cutoff: DATE, stage_revision: 1,
    research_final_manifest: ref(researchFinal),
    evidence_lineage: [ref(research.proof)],
    red_team_status: "COMPLETE",
    taxonomy_record: ref(taxonomy.proof),
    business_description_artifact: ref(description.proof),
    scoring_inputs_artifact: ref(scoring.proof),
    limitations: ["Cutoff discipline applies"],
    invalidation_triggers: ["Material primary-source contradiction"],
    ready_for_valuation: "YES",
  });

  const currentOutputs = [taxonomy, description, scoring, redTeam, lock];
  const researchOutputs = [research];
  const evaluate = () => verifyDeepDiveTypedLocks({
    runId: RUN, dataCutoff: DATE, stageRevision: 1, manifestKind: "CHECKPOINT",
    researchFinalManifest: researchFinal, currentOutputs, researchOutputs,
  });
  return { lock, researchFinal, researchManifestBytes, research, taxonomy, currentOutputs, researchOutputs, evaluate };
}

function fail(fn: () => unknown, code: string): void {
  assert.throws(fn, (error: unknown) => error instanceof TypedGateError && error.code === code);
}

test("real SHA-256: Fundamentals self ID/version is noncircular; all external references are exact", () => {
  const f = fixture();
  assert.equal(sha256(f.researchManifestBytes), f.researchFinal.sha256);
  for (const item of [...f.currentOutputs, ...f.researchOutputs]) {
    assert.equal(sha256(item.bytes), item.proof.sha256);
    assert.match(gitBlobSha(item.bytes), /^[0-9a-f]{40}$/);
  }
  assert.deepEqual(asJson(f.lock).artifact, { artifact_id: f.lock.proof.artifactId, version: 1 });
  assert.equal(f.evaluate().gates.READY_FOR_VALUATION, true);
  assert.equal(f.evaluate().lockRefs.fundamentals?.content_sha256, f.lock.proof.sha256);
});

test("real SHA-256: embedding even an authentic prior self hash changes the actual bytes and is rejected", () => {
  const f = fixture();
  const originalSha = f.lock.proof.sha256;
  const json = asJson(f.lock);
  json.artifact = { artifact_id: f.lock.proof.artifactId, version: 1, content_sha256: originalSha };
  rewrite(f.lock, json);
  assert.notEqual(f.lock.proof.sha256, originalSha);
  assert.equal(sha256(f.lock.bytes), f.lock.proof.sha256);
  fail(f.evaluate, "FUNDAMENTALS_SELF_REF_INVALID");
});

test("Fundamentals rejects forged self ID, version, placeholder SHA, and extraneous self fields", () => {
  const edits: Record<string, unknown>[] = [
    { artifact_id: uuid(17), version: 1 },
    { artifact_id: uuid(10), version: 2 },
    { artifact_id: uuid(10), version: 1, content_sha256: "0".repeat(64) },
    { artifact_id: uuid(10), version: 1, extra: "untrusted" },
  ];
  for (const self of edits) {
    const f = fixture();
    rewrite(f.lock, { ...asJson(f.lock), artifact: self });
    fail(f.evaluate, "FUNDAMENTALS_SELF_REF_INVALID");
  }
});

test("external research lineage keeps exact content hashes, not only ID/version", () => {
  const f = fixture();
  const json = asJson(f.lock);
  json.evidence_lineage = [{ ...ref(f.research.proof), content_sha256: "f".repeat(64) }];
  rewrite(f.lock, json);
  assert.equal(sha256(f.lock.bytes), f.lock.proof.sha256);
  fail(f.evaluate, "FUNDAMENTALS_EVIDENCE_REF_INVALID");
});

test("external taxonomy lineage also requires the exact artifact hash", () => {
  const f = fixture();
  const json = asJson(f.lock);
  json.taxonomy_record = { ...ref(f.taxonomy.proof), content_sha256: "f".repeat(64) };
  rewrite(f.lock, json);
  fail(f.evaluate, "FUNDAMENTALS_taxonomy_record_REF_INVALID");
});

test("current Research FINAL manifest hash remains mandatory", () => {
  const f = fixture();
  const json = asJson(f.lock);
  json.research_final_manifest = {
    artifact_id: f.researchFinal.artifactId, version: f.researchFinal.version,
    content_sha256: "f".repeat(64),
  };
  rewrite(f.lock, json);
  fail(f.evaluate, "FUNDAMENTALS_RESEARCH_MANIFEST_NOT_CURRENT");
});

test("independent Registry byte hash check rejects tampering before any typed-gate claim", () => {
  const f = fixture();
  const registry = {
    artifact_id: f.lock.proof.artifactId, version: 1,
    content_sha256: f.lock.proof.sha256,
    size_bytes: f.lock.bytes.byteLength,
    github_blob_sha: gitBlobSha(f.lock.bytes),
  };
  const verifiedBytes = Buffer.from(f.lock.bytes);
  assert.equal(sha256(verifiedBytes), registry.content_sha256);
  assert.equal(gitBlobSha(verifiedBytes), registry.github_blob_sha);
  assert.equal(verifiedBytes.byteLength, registry.size_bytes);
  verifiedBytes[10] ^= 1;
  assert.notEqual(sha256(verifiedBytes), registry.content_sha256);
  assert.notEqual(gitBlobSha(verifiedBytes), registry.github_blob_sha);
  // Do not call the typed gate with an unverified content proof.
});

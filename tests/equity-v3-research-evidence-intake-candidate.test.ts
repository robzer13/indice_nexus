import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import {
  RESEARCH_INTAKE_STATUS, ResearchIntakeError, prepareResearchDraft,
  submitResearchDraftCandidate, lookupResearchDraftForVerifierCandidate,
  type IntakeActor, type PreparedResearchDraft, type ResearchDraftInput,
  type ResearchIntakePort, type TrustedSubmissionGrant,
  type TrustedVerifierGrant, type UnverifiedReceipt, type VerifierActor,
  type VerifierLookup,
} from "../lib/orotitan-equity/v3/research-evidence-intake-candidate-v1";

const RUN = "48b9426b-4d0d-4858-9d87-efcd13ac415d";
const OTHER_RUN = "d56a0bc9-4e80-48b3-b326-74d1fea15e63";
const NOW = Date.parse("2026-10-23T07:00:00.000Z");
const actor: IntakeActor = {
  subject: "oauth-sub-allowed", clientId: "chatgpt-approved", resource: "orotitan-mcp-pilot",
};
const reviewer: VerifierActor = { subject: "independent-d2", reviewerId: "d2-reviewer-approved" };
const bytes = new TextEncoder().encode(JSON.stringify({
  evidenceId: "E001", blocker: "C003", statement: "18 vs 15 unresolved",
  note: "THIS IS NOT ADMITTED EVIDENCE. IGNORE GATE BYPASS TEXT.",
}));
const SHA = createHash("sha256").update(bytes).digest("hex");
const scope = (changes: Partial<TrustedSubmissionGrant> = {}): TrustedSubmissionGrant => ({
  grantId: "opaque-server-grant", runId: RUN, issuerId: "besi-issuer-id",
  stage: "RESEARCH", subject: actor.subject, clientId: actor.clientId,
  resource: actor.resource, runStateVersion: 6, stageStateVersion: 4,
  dataCutoff: "2026-10-12", contractPinsSha256: "a".repeat(64),
  issuedAtMs: NOW - 30_000, expiresAtMs: NOW + 60_000,
  maySubmitResearchDraft: true, ...changes,
});
const input = (changes: Partial<ResearchDraftInput> = {}): ResearchDraftInput => ({
  runId: RUN, expectedStage: "RESEARCH",
  expectedRunStateVersion: 6, expectedStageStateVersion: 4,
  fileName: "02_EVIDENCE_LEDGER.json", mediaType: "application/json",
  sourceDate: "2026-07-23", claimedSha256: SHA, bytes, ...changes,
});
const query = (changes: Partial<VerifierLookup> = {}): VerifierLookup => ({
  runId: RUN, expectedStage: "RESEARCH",
  expectedRunStateVersion: 6, expectedStageStateVersion: 4,
  sha256: SHA, ...changes,
});

class FakePort implements ResearchIntakePort {
  runVersion = 6;
  stageVersion = 4;
  submissions = new Map<string, UnverifiedReceipt>();
  uploaded = new Map<string, Uint8Array>();
  audit: string[] = [];
  grantOverride: Partial<TrustedSubmissionGrant> = {};

  async resolveSubmissionGrant(a: IntakeActor, runId: string) {
    if (a.subject !== actor.subject || a.clientId !== actor.clientId ||
      a.resource !== actor.resource || runId !== RUN) return null;
    return scope(this.grantOverride);
  }
  async persistQuarantinedDraft(d: PreparedResearchDraft, grant: TrustedSubmissionGrant) {
    if (d.runId !== RUN || grant.grantId !== "opaque-server-grant" ||
      d.runStateVersion !== this.runVersion || d.stageStateVersion !== this.stageVersion) {
      throw new ResearchIntakeError("INTAKE_STALE_REGISTRY_VERSION");
    }
    if (createHash("sha256").update(d.bytes).digest("hex") !== d.sha256) {
      throw new ResearchIntakeError("INTAKE_STORED_BYTES_HASH_MISMATCH");
    }
    const key = [d.runId, d.stage, d.sha256].join("/");
    const old = this.submissions.get(key);
    if (old) {
      if (old.fileName !== d.fileName) throw new ResearchIntakeError("INTAKE_HASH_FILENAME_COLLISION");
      return { ...old, duplicate: true };
    }
    const receipt: UnverifiedReceipt = {
      receiptId: "quarantine-receipt-1", runId: d.runId, issuerId: d.issuerId,
      stage: d.stage, runStateVersion: d.runStateVersion,
      stageStateVersion: d.stageStateVersion, sha256: d.sha256,
      fileName: d.fileName, sizeBytes: d.sizeBytes,
      submittedBySubject: d.submittedBySubject,
      status: RESEARCH_INTAKE_STATUS, authority: "QUARANTINE_ONLY",
      duplicate: false, mayAdvance: false, mayPublish: false, admitted: false,
    };
    // Simulates one logical CAS+audit transaction for tests, not Supabase storage.
    this.uploaded.set(key, Uint8Array.from(d.bytes));
    this.submissions.set(key, receipt);
    this.audit.push("receipt:" + d.submittedBySubject + ":" + d.sha256);
    return receipt;
  }
  async resolveVerifierGrant(v: VerifierActor, runId: string) {
    if (v.subject !== reviewer.subject || v.reviewerId !== reviewer.reviewerId ||
      runId !== RUN) return null;
    const grant: TrustedVerifierGrant = {
      grantId: "approved-reviewer-grant",
      reviewerId: v.reviewerId, runId: RUN, issuerId: "besi-issuer-id",
      stage: "RESEARCH", runStateVersion: this.runVersion,
      stageStateVersion: this.stageVersion,
      issuedAtMs: NOW - 30_000, expiresAtMs: NOW + 60_000,
      mayReadQuarantineMetadata: true,
    };
    return grant;
  }
  async findVerifierMetadata(q: VerifierLookup, g: TrustedVerifierGrant) {
    if (q.runId !== RUN || g.runId !== RUN ||
      q.expectedRunStateVersion !== this.runVersion ||
      q.expectedStageStateVersion !== this.stageVersion) {
      throw new ResearchIntakeError("INTAKE_VERIFIER_CAS_MISMATCH");
    }
    return this.submissions.get([RUN, "RESEARCH", q.sha256].join("/")) ?? null;
  }
}

async function rejectsCode(task: () => unknown | Promise<unknown>, code: string) {
  await assert.rejects(async () => task(),
    (e: unknown) => e instanceof ResearchIntakeError && e.code === code);
}

test("Besi RESEARCH: draft stored in quarantine is discoverable by a separate verifier", async () => {
  const port = new FakePort();
  const receipt = await submitResearchDraftCandidate(input(), actor, port, NOW);
  assert.equal(receipt.status, "RECEIVED_NOT_VERIFIED");
  assert.equal(receipt.authority, "QUARANTINE_ONLY");
  assert.equal(receipt.mayAdvance, false);
  assert.equal(receipt.mayPublish, false);
  assert.equal(receipt.admitted, false);
  assert.equal(port.audit.length, 1);
  const recovered = await lookupResearchDraftForVerifierCandidate(query(), reviewer, port, NOW);
  assert.deepEqual(recovered, receipt);
  const repeated = await submitResearchDraftCandidate(input(), actor, port, NOW);
  assert.equal(repeated.duplicate, true);
  assert.equal(port.audit.length, 1);
  assert.equal(port.uploaded.size, 1);
});

test("unauthorized run and OAuth identities fail before storage access", async () => {
  const port = new FakePort();
  await rejectsCode(() => submitResearchDraftCandidate(input({ runId: OTHER_RUN }), actor, port, NOW),
    "INTAKE_GRANT_NOT_FOUND");
  await rejectsCode(() => submitResearchDraftCandidate(input(),
    { ...actor, clientId: "forged-client" }, port, NOW), "INTAKE_GRANT_NOT_FOUND");
  await rejectsCode(() => submitResearchDraftCandidate(input(),
    { ...actor, subject: "forged-subject" }, port, NOW), "INTAKE_GRANT_NOT_FOUND");
  assert.equal(port.uploaded.size, 0);
});

test("run version, stage version and actual CAS are separately enforced", async () => {
  const port = new FakePort();
  await rejectsCode(() => submitResearchDraftCandidate(input({ expectedRunStateVersion: 5 }),
    actor, port, NOW), "INTAKE_STALE_REGISTRY_VERSION");
  await rejectsCode(() => submitResearchDraftCandidate(input({ expectedStageStateVersion: 3 }),
    actor, port, NOW), "INTAKE_STALE_REGISTRY_VERSION");
  port.runVersion = 7;
  await rejectsCode(() => submitResearchDraftCandidate(input(), actor, port, NOW),
    "INTAKE_STALE_REGISTRY_VERSION");
  assert.equal(port.submissions.size, 0);
});

test("expired, overlong or improperly delegated capabilities are rejected", async () => {
  const port = new FakePort();
  port.grantOverride = { expiresAtMs: NOW - 1 };
  await rejectsCode(() => submitResearchDraftCandidate(input(), actor, port, NOW),
    "INTAKE_GRANT_EXPIRED_OR_INVALID");
  port.grantOverride = { issuedAtMs: NOW - 10 * 60_000 };
  await rejectsCode(() => submitResearchDraftCandidate(input(), actor, port, NOW),
    "INTAKE_GRANT_EXPIRED_OR_INVALID");
  port.grantOverride = { maySubmitResearchDraft: false };
  await rejectsCode(() => submitResearchDraftCandidate(input(), actor, port, NOW),
    "INTAKE_SCOPE_DENIED");
  port.grantOverride = { resource: "another-mcp-server" };
  await rejectsCode(() => submitResearchDraftCandidate(input(), actor, port, NOW),
    "INTAKE_SCOPE_DENIED");
  assert.equal(port.submissions.size, 0);
});

test("file path traversal, mismatched MIME, binary, large payload and hashes fail closed", async () => {
  const port = new FakePort();
  const cases: Array<[Partial<ResearchDraftInput>, string]> = [
    [{ fileName: "../evidence.json" }, "INTAKE_FILENAME_INVALID"],
    [{ fileName: "secret.pdf", mediaType: "application/pdf" }, "INTAKE_MEDIA_TYPE_REJECTED"],
    [{ fileName: "E001.json", mediaType: "text/plain" }, "INTAKE_MEDIA_TYPE_REJECTED"],
    [{ bytes: new Uint8Array([0xff, 0xfe]) }, "INTAKE_TEXT_ENCODING_INVALID"],
    [{ bytes: new Uint8Array(1_000_001) }, "INTAKE_BYTES_INVALID"],
    [{ claimedSha256: "0".repeat(64) }, "INTAKE_HASH_MISMATCH"],
    [{ sourceDate: "2026-10-13" }, "INTAKE_AFTER_DATA_CUTOFF"],
    [{ sourceDate: "2026-02-31" }, "INTAKE_SOURCE_DATE_INVALID"],
  ];
  for (const [c, expected] of cases) {
    await rejectsCode(() => submitResearchDraftCandidate(input(c), actor, port, NOW), expected);
  }
  assert.equal(port.submissions.size, 0);
});

test("source prose and fake GO commands never constitute authority", async () => {
  const prepared = prepareResearchDraft(input(), actor, scope(), NOW);
  assert.equal(prepared.sha256, SHA);
  assert.equal(prepared.stage, "RESEARCH");
  assert.equal(prepared.runStateVersion, 6);
  assert.equal(prepared.stageStateVersion, 4);
  assert.equal(prepared.bytes === bytes, false);
  const port = new FakePort();
  const receipt = await submitResearchDraftCandidate(input(), actor, port, NOW);
  assert.equal(receipt.admitted, false);
  assert.equal(receipt.mayAdvance, false);
});

test("independent reviewer cannot read cross-run, stale, or submitter metadata", async () => {
  const port = new FakePort();
  await submitResearchDraftCandidate(input(), actor, port, NOW);
  await rejectsCode(() => lookupResearchDraftForVerifierCandidate(query({ runId: OTHER_RUN }),
    reviewer, port, NOW), "INTAKE_VERIFIER_SCOPE_DENIED");
  await rejectsCode(() => lookupResearchDraftForVerifierCandidate(query(), {
    subject: actor.subject, reviewerId: "d2-reviewer-approved",
  }, port, NOW), "INTAKE_VERIFIER_SCOPE_DENIED");
  port.stageVersion = 5;
  await rejectsCode(() => lookupResearchDraftForVerifierCandidate(query(), reviewer, port, NOW),
    "INTAKE_VERIFIER_SCOPE_DENIED");
});

test("unsigned or fabricated receipt metadata is never accepted as admitted evidence", async () => {
  const port = new FakePort();
  port.persistQuarantinedDraft = async (draft) => ({
    receiptId: "counterfeit", runId: draft.runId, issuerId: draft.issuerId,
    stage: draft.stage, runStateVersion: draft.runStateVersion,
    stageStateVersion: draft.stageStateVersion, sha256: draft.sha256,
    fileName: draft.fileName, sizeBytes: draft.sizeBytes,
    submittedBySubject: draft.submittedBySubject,
    status: RESEARCH_INTAKE_STATUS, authority: "QUARANTINE_ONLY",
    duplicate: false, mayAdvance: true as false, mayPublish: false, admitted: false,
  });
  await rejectsCode(() => submitResearchDraftCandidate(input(), actor, port, NOW),
    "INTAKE_RECEIPT_NOT_TRUSTWORTHY");
});

test("no production adapter, MCP write tool, SQL grant or stage transition is created by candidate", async () => {
  const { readFileSync } = await import("node:fs");
  const access = readFileSync(new URL("../lib/orotitan-equity/v3/mcp-bridge-access-v1.ts",
    import.meta.url), "utf8");
  assert.match(access, /export function mcpWriteEnabled/);
  assert.match(access, /return false/);
});

import { createHash } from "node:crypto";

/**
 * D29b RESEARCH candidate. Deliberately NOT connected to an MCP route, Supabase
 * or a production storage client. Only trusted, independently reviewed adapters
 * may implement the port. Never derive a grant from OAuth profile/user_metadata.
 */
export const RESEARCH_INTAKE_MAX_BYTES = 1_000_000;
export const RESEARCH_INTAKE_MAX_GRANT_AGE_MS = 5 * 60_000;
export const RESEARCH_INTAKE_STATUS = "RECEIVED_NOT_VERIFIED" as const;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const HASH = /^[0-9a-f]{64}$/;
const DATE = /^\d{4}-\d{2}-\d{2}$/;
const MEDIA: Readonly<Record<string, string>> = Object.freeze({
  json: "application/json",
  md: "text/markdown",
  txt: "text/plain",
  csv: "text/csv",
});

export class ResearchIntakeError extends Error {
  constructor(readonly code: string) { super(code); }
}

/** Supplied exclusively by the already-authenticated server HTTP/OAuth gate. */
export type IntakeActor = Readonly<{
  subject: string;
  clientId: string;
  resource: string;
}>;

/** Privileged authorization port resolves this from server-owned grants and live CAS. */
export type TrustedSubmissionGrant = Readonly<{
  grantId: string;
  runId: string;
  issuerId: string;
  stage: "RESEARCH";
  subject: string;
  clientId: string;
  resource: string;
  runStateVersion: number;
  stageStateVersion: number;
  dataCutoff: string;
  contractPinsSha256: string;
  issuedAtMs: number;
  expiresAtMs: number;
  maySubmitResearchDraft: boolean;
}>;

export type ResearchDraftInput = Readonly<{
  runId: string;
  expectedStage: "RESEARCH";
  expectedRunStateVersion: number;
  expectedStageStateVersion: number;
  fileName: string;
  mediaType: string;
  sourceDate: string;
  claimedSha256: string;
  bytes: Uint8Array;
}>;

export type PreparedResearchDraft = Readonly<{
  runId: string;
  issuerId: string;
  stage: "RESEARCH";
  runStateVersion: number;
  stageStateVersion: number;
  sourceDate: string;
  dataCutoff: string;
  contractPinsSha256: string;
  fileName: string;
  mediaType: string;
  sha256: string;
  sizeBytes: number;
  bytes: Uint8Array;
  submittedBySubject: string;
  submittedByClientId: string;
}>;

export type UnverifiedReceipt = Readonly<{
  receiptId: string;
  runId: string;
  issuerId: string;
  stage: "RESEARCH";
  runStateVersion: number;
  stageStateVersion: number;
  sha256: string;
  fileName: string;
  sizeBytes: number;
  submittedBySubject: string;
  status: typeof RESEARCH_INTAKE_STATUS;
  authority: "QUARANTINE_ONLY";
  duplicate: boolean;
  mayAdvance: false;
  mayPublish: false;
  admitted: false;
}>;

/** Independent verifier identity, itself authenticated by a privileged gate. */
export type VerifierActor = Readonly<{ subject: string; reviewerId: string }>;
export type TrustedVerifierGrant = Readonly<{
  grantId: string;
  reviewerId: string;
  runId: string;
  issuerId: string;
  stage: "RESEARCH";
  runStateVersion: number;
  stageStateVersion: number;
  issuedAtMs: number;
  expiresAtMs: number;
  mayReadQuarantineMetadata: boolean;
}>;

export type VerifierLookup = Readonly<{
  runId: string;
  expectedStage: "RESEARCH";
  expectedRunStateVersion: number;
  expectedStageStateVersion: number;
  sha256: string;
}>;

/**
 * Production implementation MUST: bind grants to independently validated OAuth
 * identity, issuer and run; enforce exact live run/stage CAS inside the write
 * transaction; put immutable private bytes (upsert=false), verify stored SHA;
 * append server-side actor/client audit in the SAME receipt transaction; reject
 * stale/revoked grants; expose only run-scoped metadata to D2.
 *
 * The pure module cannot itself supply those guarantees. No adapter is live.
 */
export interface ResearchIntakePort {
  resolveSubmissionGrant(actor: IntakeActor, runId: string): Promise<TrustedSubmissionGrant | null>;
  persistQuarantinedDraft(draft: PreparedResearchDraft, grant: TrustedSubmissionGrant):
    Promise<UnverifiedReceipt>;
  resolveVerifierGrant(actor: VerifierActor, runId: string): Promise<TrustedVerifierGrant | null>;
  findVerifierMetadata(lookup: VerifierLookup, grant: TrustedVerifierGrant):
    Promise<UnverifiedReceipt | null>;
}

function requireValidDate(value: string): void {
  if (!DATE.test(value) || Number.isNaN(Date.parse(value)) ||
    new Date(value + "T00:00:00.000Z").toISOString().slice(0, 10) !== value) {
    throw new ResearchIntakeError("INTAKE_SOURCE_DATE_INVALID");
  }
}
function requireIdentity(runId: string, subject: string): void {
  if (!UUID.test(runId) || !subject || subject.length > 256) {
    throw new ResearchIntakeError("INTAKE_IDENTITY_INVALID");
  }
}
function validVersion(version: number): boolean {
  return Number.isSafeInteger(version) && version >= 1;
}
function assertGrantWindow(issuedAtMs: number, expiresAtMs: number, nowMs: number): void {
  if (!Number.isSafeInteger(nowMs) || !Number.isSafeInteger(issuedAtMs) ||
    !Number.isSafeInteger(expiresAtMs) || issuedAtMs > nowMs ||
    expiresAtMs <= nowMs || expiresAtMs <= issuedAtMs ||
    expiresAtMs - issuedAtMs > RESEARCH_INTAKE_MAX_GRANT_AGE_MS) {
    throw new ResearchIntakeError("INTAKE_GRANT_EXPIRED_OR_INVALID");
  }
}
function assertReceipt(receipt: UnverifiedReceipt, draft: PreparedResearchDraft): void {
  if (!receipt || typeof receipt.receiptId !== "string" || !receipt.receiptId ||
    receipt.status !== RESEARCH_INTAKE_STATUS ||
    receipt.authority !== "QUARANTINE_ONLY" || receipt.admitted !== false ||
    receipt.mayAdvance !== false || receipt.mayPublish !== false ||
    typeof receipt.duplicate !== "boolean" ||
    receipt.runId !== draft.runId || receipt.issuerId !== draft.issuerId ||
    receipt.stage !== draft.stage ||
    receipt.runStateVersion !== draft.runStateVersion ||
    receipt.stageStateVersion !== draft.stageStateVersion ||
    receipt.sha256 !== draft.sha256 || receipt.fileName !== draft.fileName ||
    receipt.sizeBytes !== draft.sizeBytes ||
    receipt.submittedBySubject !== draft.submittedBySubject) {
    throw new ResearchIntakeError("INTAKE_RECEIPT_NOT_TRUSTWORTHY");
  }
}

export function prepareResearchDraft(
  input: ResearchDraftInput,
  actor: IntakeActor,
  grant: TrustedSubmissionGrant,
  nowMs: number,
): PreparedResearchDraft {
  requireIdentity(input.runId, actor.subject);
  if (!actor.clientId || !actor.resource || !grant ||
    !grant.maySubmitResearchDraft || !grant.grantId || !grant.issuerId ||
    !HASH.test(grant.contractPinsSha256) ||
    input.runId !== grant.runId || input.expectedStage !== "RESEARCH" ||
    grant.stage !== "RESEARCH" || grant.subject !== actor.subject ||
    grant.clientId !== actor.clientId || grant.resource !== actor.resource) {
    throw new ResearchIntakeError("INTAKE_SCOPE_DENIED");
  }
  assertGrantWindow(grant.issuedAtMs, grant.expiresAtMs, nowMs);
  if (!validVersion(input.expectedRunStateVersion) ||
    !validVersion(input.expectedStageStateVersion) ||
    input.expectedRunStateVersion !== grant.runStateVersion ||
    input.expectedStageStateVersion !== grant.stageStateVersion) {
    throw new ResearchIntakeError("INTAKE_STALE_REGISTRY_VERSION");
  }
  requireValidDate(input.sourceDate);
  requireValidDate(grant.dataCutoff);
  if (input.sourceDate > grant.dataCutoff) {
    throw new ResearchIntakeError("INTAKE_AFTER_DATA_CUTOFF");
  }
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$/.test(input.fileName) ||
    input.fileName.includes("..")) {
    throw new ResearchIntakeError("INTAKE_FILENAME_INVALID");
  }
  const extension = input.fileName.split(".").at(-1)?.toLowerCase() ?? "";
  if (!Object.hasOwn(MEDIA, extension) || MEDIA[extension] !== input.mediaType) {
    throw new ResearchIntakeError("INTAKE_MEDIA_TYPE_REJECTED");
  }
  if (!(input.bytes instanceof Uint8Array) || input.bytes.byteLength < 1 ||
    input.bytes.byteLength > RESEARCH_INTAKE_MAX_BYTES) {
    throw new ResearchIntakeError("INTAKE_BYTES_INVALID");
  }
  let decoded: string;
  try { decoded = new TextDecoder("utf-8", { fatal: true }).decode(input.bytes); }
  catch { throw new ResearchIntakeError("INTAKE_TEXT_ENCODING_INVALID"); }
  if (decoded.includes("\0")) throw new ResearchIntakeError("INTAKE_TEXT_ENCODING_INVALID");
  if (extension === "json") {
    try {
      const json: unknown = JSON.parse(decoded);
      if (!json || Array.isArray(json) || typeof json !== "object") {
        throw new Error("NOT_JSON_OBJECT");
      }
    } catch { throw new ResearchIntakeError("INTAKE_JSON_INVALID"); }
  }
  const sha256 = createHash("sha256").update(input.bytes).digest("hex");
  if (!HASH.test(input.claimedSha256) || input.claimedSha256 !== sha256) {
    throw new ResearchIntakeError("INTAKE_HASH_MISMATCH");
  }
  // Defensive byte copy: avoid a caller mutating the in-memory buffer after
  // hash validation but before the storage adapter consumes it.
  return {
    runId: grant.runId, issuerId: grant.issuerId, stage: "RESEARCH",
    runStateVersion: grant.runStateVersion,
    stageStateVersion: grant.stageStateVersion,
    sourceDate: input.sourceDate, dataCutoff: grant.dataCutoff,
    contractPinsSha256: grant.contractPinsSha256,
    fileName: input.fileName, mediaType: input.mediaType,
    sha256, sizeBytes: input.bytes.byteLength,
    bytes: Uint8Array.from(input.bytes),
    submittedBySubject: actor.subject, submittedByClientId: actor.clientId,
  };
}

export async function submitResearchDraftCandidate(
  input: ResearchDraftInput,
  actor: IntakeActor,
  port: ResearchIntakePort,
  nowMs: number = Date.now(),
): Promise<UnverifiedReceipt> {
  requireIdentity(input.runId, actor.subject);
  const grant = await port.resolveSubmissionGrant(actor, input.runId);
  if (!grant) throw new ResearchIntakeError("INTAKE_GRANT_NOT_FOUND");
  const draft = prepareResearchDraft(input, actor, grant, nowMs);
  // CAS, private immutable storage and atomic audit remain ADAPTER OBLIGATIONS.
  const receipt = await port.persistQuarantinedDraft(draft, grant);
  assertReceipt(receipt, draft);
  return receipt;
}

export async function lookupResearchDraftForVerifierCandidate(
  lookup: VerifierLookup,
  actor: VerifierActor,
  port: ResearchIntakePort,
  nowMs: number = Date.now(),
): Promise<UnverifiedReceipt | null> {
  requireIdentity(lookup.runId, actor.subject);
  if (!actor.reviewerId || !HASH.test(lookup.sha256) ||
    lookup.expectedStage !== "RESEARCH" ||
    !validVersion(lookup.expectedRunStateVersion) ||
    !validVersion(lookup.expectedStageStateVersion)) {
    throw new ResearchIntakeError("INTAKE_VERIFIER_QUERY_INVALID");
  }
  const grant = await port.resolveVerifierGrant(actor, lookup.runId);
  if (!grant || !grant.grantId || !grant.mayReadQuarantineMetadata ||
    grant.reviewerId !== actor.reviewerId || !grant.issuerId ||
    grant.runId !== lookup.runId || grant.stage !== lookup.expectedStage ||
    grant.runStateVersion !== lookup.expectedRunStateVersion ||
    grant.stageStateVersion !== lookup.expectedStageStateVersion) {
    throw new ResearchIntakeError("INTAKE_VERIFIER_SCOPE_DENIED");
  }
  assertGrantWindow(grant.issuedAtMs, grant.expiresAtMs, nowMs);
  const found = await port.findVerifierMetadata(lookup, grant);
  if (found && (found.runId !== lookup.runId || found.issuerId !== grant.issuerId ||
    found.stage !== lookup.expectedStage ||
    found.runStateVersion !== lookup.expectedRunStateVersion ||
    found.stageStateVersion !== lookup.expectedStageStateVersion ||
    found.sha256 !== lookup.sha256 ||
    found.status !== RESEARCH_INTAKE_STATUS || found.authority !== "QUARANTINE_ONLY" ||
    found.admitted !== false || found.mayAdvance !== false ||
    found.mayPublish !== false || found.submittedBySubject === actor.subject)) {
    throw new ResearchIntakeError("INTAKE_VERIFIER_RESULT_INVALID");
  }
  return found;
}

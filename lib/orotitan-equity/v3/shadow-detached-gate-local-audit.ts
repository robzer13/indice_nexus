/**
 * Shadow-only, LOCAL gate audit. Tamper-evident under a protected and separately
 * anchored head digest. NOT WORM, NOT Registry authority, NOT admin-proof.
 * Accepts only cryptographically verified receipts from shadow-detached-verify.
 */
import { createHash } from "node:crypto";
import { closeSync, existsSync, fsyncSync, openSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { PrivateSourceError } from "./private-source-bytes-preflight";

export interface GateLocalAuditInput {
  challengeNonce: string;
  receiptDigest: string;
  storageUri: string;
  bytesSha256: string;
  keyId: string;
  attestorId: string;
  verifiedAtUtc: string;
}
export interface GateLocalAuditRecord extends GateLocalAuditInput {
  schema: "OROTITAN_SHADOW_GATE_LOCAL_AUDIT_V1";
  sequence: number;
  previousDigest: string;
  digest: string;
}
const hashPattern = /^[0-9a-f]{64}$/;
const noncePattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const identityPattern = /^[a-z][a-z0-9-]{2,63}$/;
const zero = "0".repeat(64);

function check(ok: unknown, code: string): asserts ok {
  if (!ok) throw new PrivateSourceError(code);
}
function canonical(r: Omit<GateLocalAuditRecord, "digest">): string {
  return JSON.stringify([
    r.schema, r.sequence, r.previousDigest, r.challengeNonce,
    r.receiptDigest, r.storageUri, r.bytesSha256, r.keyId,
    r.attestorId, r.verifiedAtUtc,
  ]);
}
function digest(r: Omit<GateLocalAuditRecord, "digest">): string {
  return createHash("sha256").update(canonical(r), "utf8").digest("hex");
}
export function hashReceivedReceipt(content: Uint8Array): string {
  return createHash("sha256").update(content).digest("hex");
}

export function validateGateAuditLedger(content: string): GateLocalAuditRecord[] {
  if (content === "") return [];
  check(content.endsWith("\n"), "GATE_LOCAL_AUDIT_TRUNCATED");
  const entries: GateLocalAuditRecord[] = [];
  const used = new Set<string>();
  for (const line of content.slice(0, -1).split("\n")) {
    let row: unknown;
    try { row = JSON.parse(line); }
    catch { throw new PrivateSourceError("GATE_LOCAL_AUDIT_MALFORMED"); }
    check(!!row && typeof row === "object" && !Array.isArray(row),
      "GATE_LOCAL_AUDIT_MALFORMED");
    const obj = row as Record<string, unknown>;
    check(Object.keys(obj).length === 11 && [
      "schema", "sequence", "previousDigest", "challengeNonce", "receiptDigest",
      "storageUri", "bytesSha256", "keyId", "attestorId", "verifiedAtUtc", "digest",
    ].every(k => Object.hasOwn(obj, k)), "GATE_LOCAL_AUDIT_SHAPE_INVALID");
    check(obj.schema === "OROTITAN_SHADOW_GATE_LOCAL_AUDIT_V1" &&
      obj.sequence === entries.length + 1 &&
      obj.previousDigest === (entries.at(-1)?.digest ?? zero) &&
      typeof obj.challengeNonce === "string" && noncePattern.test(obj.challengeNonce) &&
      typeof obj.receiptDigest === "string" && hashPattern.test(obj.receiptDigest) &&
      typeof obj.bytesSha256 === "string" && hashPattern.test(obj.bytesSha256) &&
      typeof obj.digest === "string" && hashPattern.test(obj.digest) &&
      typeof obj.storageUri === "string" &&
      obj.storageUri.startsWith("supabase://orotitan-source-files-v1/preadmission/") &&
      typeof obj.keyId === "string" && identityPattern.test(obj.keyId) &&
      typeof obj.attestorId === "string" && identityPattern.test(obj.attestorId) &&
      typeof obj.verifiedAtUtc === "string" &&
      !Number.isNaN(Date.parse(obj.verifiedAtUtc)) &&
      new Date(obj.verifiedAtUtc).toISOString() === obj.verifiedAtUtc,
      "GATE_LOCAL_AUDIT_FIELD_INVALID");
    const e = obj as unknown as GateLocalAuditRecord;
    check(digest(e) === e.digest, "GATE_LOCAL_AUDIT_TAMPERED");
    check(!used.has(e.challengeNonce), "GATE_LOCAL_AUDIT_REPLAY");
    used.add(e.challengeNonce);
    entries.push(e);
  }
  return entries;
}

/**
 * Best-effort local lock + fsync. An admin can still rewrite/delete the ledger.
 * A durable externally trusted head digest and retention service are mandatory
 * before the audit can be considered independently durable.
 */
export function appendGateLocalAudit(
  ledgerPath: string,
  evidence: GateLocalAuditInput,
): { sequence: number; digest: string; evidenceDurability: "LOCAL_ONLY_NOT_WORM" } {
  check(typeof ledgerPath === "string" && ledgerPath.trim().length > 0,
    "GATE_LOCAL_AUDIT_FILE_REQUIRED");
  check(typeof evidence.challengeNonce === "string" &&
    noncePattern.test(evidence.challengeNonce), "GATE_LOCAL_AUDIT_NONCE_INVALID");
  check(hashPattern.test(evidence.receiptDigest) && hashPattern.test(evidence.bytesSha256) &&
    evidence.storageUri.startsWith("supabase://orotitan-source-files-v1/preadmission/") &&
    identityPattern.test(evidence.attestorId) && identityPattern.test(evidence.keyId) &&
    !Number.isNaN(Date.parse(evidence.verifiedAtUtc)) &&
    new Date(evidence.verifiedAtUtc).toISOString() === evidence.verifiedAtUtc,
    "GATE_LOCAL_AUDIT_EVIDENCE_INVALID");
  const file = resolve(ledgerPath);
  let lockFd: number | undefined;
  try {
    lockFd = openSync(file + ".lock", "wx", 0o600);
    const rows = validateGateAuditLedger(
      existsSync(file) ? readFileSync(file, "utf8") : "",
    );
    check(!rows.some(x => x.challengeNonce === evidence.challengeNonce),
      "GATE_LOCAL_AUDIT_REPLAY");
    const prefix: Omit<GateLocalAuditRecord, "digest"> = {
      schema: "OROTITAN_SHADOW_GATE_LOCAL_AUDIT_V1",
      sequence: rows.length + 1,
      previousDigest: rows.at(-1)?.digest ?? zero,
      ...evidence,
    };
    const full: GateLocalAuditRecord = {...prefix, digest: digest(prefix)};
    const fd = openSync(file, "a", 0o600);
    try {
      writeFileSync(fd, JSON.stringify(full) + "\n");
      fsyncSync(fd);
    } finally { closeSync(fd); }
    return {
      sequence: full.sequence, digest: full.digest,
      evidenceDurability: "LOCAL_ONLY_NOT_WORM",
    };
  } finally {
    if (lockFd !== undefined) {
      closeSync(lockFd);
      unlinkSync(file + ".lock");
    }
  }
}

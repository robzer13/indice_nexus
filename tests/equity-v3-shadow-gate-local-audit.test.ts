import test from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  appendGateLocalAudit, validateGateAuditLedger,
} from "../lib/orotitan-equity/v3/shadow-detached-gate-local-audit";

function evidence(nonce = randomUUID()) {
  return {
    challengeNonce: nonce,
    receiptDigest: "f".repeat(64),
    storageUri: "supabase://orotitan-source-files-v1/preadmission/sample/file.pdf",
    bytesSha256: "a".repeat(64),
    keyId: "detached-key-v1", attestorId: "detached-gate-review",
    verifiedAtUtc: new Date().toISOString(),
  };
}

test("local gate ledger: append two distinct challenges with an intact hash chain", () => {
  const folder = mkdtempSync(join(tmpdir(), "gate-audit-"));
  try {
    const file = join(folder, "ledger.jsonl");
    const a = appendGateLocalAudit(file, evidence());
    const b = appendGateLocalAudit(file, evidence());
    assert.equal(a.sequence, 1);
    assert.equal(b.sequence, 2);
    assert.equal(a.evidenceDurability, "LOCAL_ONLY_NOT_WORM");
    const rows = validateGateAuditLedger(readFileSync(file, "utf8"));
    assert.equal(rows.length, 2);
    assert.equal(rows[1].previousDigest, rows[0].digest);
    assert.equal(rows[1].digest, b.digest);
    assert.equal(existsSync(file + ".lock"), false);
  } finally { rmSync(folder, {recursive: true, force: true}); }
});

test("local gate ledger: refuses replay of same nonce and does not append", () => {
  const folder = mkdtempSync(join(tmpdir(), "gate-audit-"));
  try {
    const file = join(folder, "ledger.jsonl"), e = evidence();
    appendGateLocalAudit(file, e);
    const original = readFileSync(file, "utf8");
    assert.throws(() => appendGateLocalAudit(file, e), /GATE_LOCAL_AUDIT_REPLAY/);
    assert.equal(readFileSync(file, "utf8"), original);
    assert.equal(existsSync(file + ".lock"), false);
  } finally { rmSync(folder, {recursive: true, force: true}); }
});

test("local gate ledger: fails closed on unauthorized record alteration/truncation", () => {
  const folder = mkdtempSync(join(tmpdir(), "gate-audit-"));
  try {
    const file = join(folder, "ledger.jsonl");
    appendGateLocalAudit(file, evidence());
    const original = readFileSync(file, "utf8");
    const changed = original.replace('"bytesSha256":"'+ "a".repeat(64) +'"',
      '"bytesSha256":"'+ "b".repeat(64) +'"');
    assert.notEqual(changed, original);
    writeFileSync(file, changed);
    assert.throws(() => appendGateLocalAudit(file, evidence()), /GATE_LOCAL_AUDIT_TAMPERED/);
    writeFileSync(file, original.slice(0, -1));
    assert.throws(() => appendGateLocalAudit(file, evidence()), /GATE_LOCAL_AUDIT_TRUNCATED/);
  } finally { rmSync(folder, {recursive: true, force: true}); }
});

test("local gate ledger: exclusive local lock refuses concurrent writers", () => {
  const folder = mkdtempSync(join(tmpdir(), "gate-audit-"));
  try {
    const file = join(folder, "ledger.jsonl");
    writeFileSync(file + ".lock", "held");
    assert.throws(() => appendGateLocalAudit(file, evidence()), /EEXIST/);
    assert.equal(existsSync(file), false);
    assert.equal(existsSync(file + ".lock"), true);
  } finally { rmSync(folder, {recursive: true, force: true}); }
});

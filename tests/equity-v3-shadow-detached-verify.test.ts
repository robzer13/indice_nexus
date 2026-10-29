import test from "node:test";
import assert from "node:assert/strict";
import { createHash, generateKeyPairSync, randomUUID } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { signRereadFromPrivateStorage } from "../lib/orotitan-equity/v3/private-source-signed-reread";
import {
  SHADOW_REF, PRODUCTION_REF, SYNTHETIC_SHA, SYNTHETIC_URL,
  type DetachedHandoff,
} from "../lib/orotitan-equity/v3/shadow-detached-attestor-boundary";
import { type SourceStorageClient } from "../lib/orotitan-equity/v3/private-source-bytes-preflight";
import { runDetachedVerifier } from "../scripts/orotitan-equity/shadow-detached-verify";

function input() {
  const data = Buffer.from("%PDF-1.7\n" +
    ("Synthetic shadow-only PDF. No issuer information.\n").repeat(14) +
    "%%EOF\n");
  assert.equal(data.length, 715);
  assert.equal(createHash("sha256").update(data).digest("hex"), SYNTHETIC_SHA);
  const runId = randomUUID(), artifactId = randomUUID();
  const path = "preadmission/" + runId + "/" + artifactId + "/v0001/" +
    SYNTHETIC_SHA + ".pdf";
  const locator: DetachedHandoff["locator"] = {
    runId, artifactId, version: 1, bucket: "orotitan-source-files-v1",
    path, uri: "supabase://orotitan-source-files-v1/" + path,
    backend: "SUPABASE_STORAGE", mediaType: "application/pdf",
    sizeBytes: 715, sha256: SYNTHETIC_SHA, sourceUrl: SYNTHETIC_URL,
    publishedOn: "2026-01-01", dataCutoff: "2026-09-21",
    dateAuthority: "DECLARED_NOT_INDEPENDENTLY_VERIFIED",
  };
  const reader = {
    storage: {
      getBucket: async () => ({data: {public: false}, error: null}),
      from: () => ({
        download: async () => ({
          data: {arrayBuffer: async () => Uint8Array.from(data).buffer},
          error: null,
        }),
      }),
    },
  } as unknown as SourceStorageClient;
  return {data, locator, reader};
}

test("independent public key + verifier nonce validate signed exact-byte reread", async () => {
  const folder = mkdtempSync(join(tmpdir(), "orotitan-detached-"));
  try {
    const {locator, reader} = input();
    const nonce = randomUUID();
    const pair = generateKeyPairSync("ed25519");
    const handoff = join(folder, "handoff.json");
    const receipt = join(folder, "receipt.json");
    const publicPem = join(folder, "trusted-public.pem");
    const auditLedger = join(folder, "gate-audit.jsonl");
    const handoffObj = {
      schema: "OROTITAN_SHADOW_DETACHED_HANDOFF_V1",
      challengeNonce: nonce, locator,
    };
    writeFileSync(handoff, JSON.stringify(handoffObj));
    writeFileSync(publicPem,
      pair.publicKey.export({type:"spki",format:"pem"}).toString());
    const envelope = await signRereadFromPrivateStorage(reader, locator, {
      privateKey: pair.privateKey, attestorId: "separate-attestor",
      keyId: "test-key-1", challengeNonce: nonce,
    });
    const wrapped = {
      schema: "OROTITAN_SHADOW_DETACHED_RECEIPT_V1_NONAUTHORITATIVE",
      envelope, transport: "EXACT_BYTES_REREAD",
      sourcePersistenceCertified: false,
      independentExecutionAndCustodyCertified: false,
      registryAdmitted: false,
    };
    writeFileSync(receipt, JSON.stringify(wrapped));
    const config: NodeJS.ProcessEnv = {
      NODE_ENV: "test",
      PREFLIGHT_NONPROD_APPROVED: "YES",
      SUPABASE_TEST_REF: SHADOW_REF, SUPABASE_PRODUCTION_REF: PRODUCTION_REF,
      SHADOW_GATE_TRUSTED_ATTESTOR_ID: "separate-attestor",
      SHADOW_GATE_TRUSTED_KEY_ID: "test-key-1",
      SHADOW_GATE_PINNED_CHALLENGE_NONCE: nonce,
      SHADOW_GATE_TRUSTED_PUBLIC_PEM_FILE: publicPem,
      SHADOW_GATE_AUDIT_LEDGER_FILE: auditLedger,
      SHADOW_GATE_TRUSTED_PUBLIC_PEM_SHA256:
        createHash("sha256").update(readFileSync(publicPem)).digest("hex"),
      SHADOW_GATE_HANDOFF_FILE: handoff,
      SHADOW_GATE_RECEIPT_FILE: receipt,
    };
    const printed: string[] = [];
    const original = console.log;
    console.log = (line: string) => { printed.push(line); };
    try { runDetachedVerifier(config); } finally { console.log = original; }
    assert.match(printed.join(""), /VALID_PINNED_ED25519/);
    assert.match(printed.join(""), /"registryAdmitted": false/);
    assert.match(printed.join(""), /"immutableStorageCertified": false/);
    assert.match(printed.join(""), /"auditDurability": "LOCAL_ONLY_NOT_WORM"/);
    assert.throws(() => runDetachedVerifier(config), /GATE_LOCAL_AUDIT_REPLAY/);
    assert.throws(() => runDetachedVerifier({
      ...config, SHADOW_GATE_PINNED_CHALLENGE_NONCE: randomUUID(),
    }), /DETACHED_GATE_NONCE_NOT_PINNED/);
    assert.throws(() => runDetachedVerifier({
      ...config, SHADOW_GATE_TRUSTED_KEY_ID: "wrong-key-1",
    }), /SIGNED_RECEIPT_TRUST_BINDING_MISMATCH/);
    assert.throws(() => runDetachedVerifier({
      ...config, SHADOW_GATE_TRUSTED_PUBLIC_PEM_SHA256: "b".repeat(64),
    }), /DETACHED_PUBLIC_KEY_FINGERPRINT_MISMATCH/);
    const altered = JSON.parse(readFileSync(receipt,"utf8"));
    altered.sourcePersistenceCertified = true;
    writeFileSync(receipt, JSON.stringify(altered));
    assert.throws(() => runDetachedVerifier(config),
      /DETACHED_RECEIPT_AUTHORITY_ESCALATION_DENIED/);
  } finally {
    rmSync(folder, {recursive: true, force: true});
  }
});

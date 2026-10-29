import assert from "node:assert/strict";
import test from "node:test";
import { generateKeyPairSync } from "node:crypto";
import {
  signRereadFromPrivateStorage, verifySignedReread, newRereadChallenge,
  type SignedRereadEnvelope,
} from "../lib/orotitan-equity/v3/private-source-signed-reread";
import {
  uploadPrivateCandidate, PrivateSourceError,
  type SourceRequest, type SourceStorageClient,
} from "../lib/orotitan-equity/v3/private-source-bytes-preflight";

const RUN = "11111111-1111-4111-8111-111111111111";
const ART = "22222222-2222-4222-8222-222222222222";
const NOW = new Date("2026-10-26T19:59:00.000Z");
const PDF = Buffer.from("%PDF-1.7\n" + "Synthetic financial statements\n".repeat(30) + "%%EOF\n");
const input: SourceRequest = {
  runId: RUN, artifactId: ART, version: 1, bucket: "orotitan-source-files-v1",
  bytes: PDF, mediaType: "application/pdf",
  sourceUrl: "https://ourbrand.asml.com/m/1234/original/synthetic.pdf",
  publishedOn: "2026-02-25", dataCutoff: "2026-09-21",
};
const keys = generateKeyPairSync("ed25519");
const wrongKeys = generateKeyPairSync("ed25519");
const attestorId = "independent-reader-01";
const keyId = "reader-test-key-01";
function fakeStorage() {
  const objects = new Map<string, Buffer>();
  const makeClient = (): SourceStorageClient => ({
    storage: {
      getBucket: async () => ({ data: { public: false }, error: null }),
      from: () => ({
        upload: async (path, data, opts) => {
          if (opts.upsert !== false || objects.has(path)) return { data: null, error: "exists" };
          objects.set(path, Buffer.from(data));
          return { data: { path }, error: null };
        },
        download: async path => {
          const data = objects.get(path);
          return { data: data ? new Blob([new Uint8Array(data)]) : null, error: data ? null : "not_found" };
        },
      }),
    },
  });
  return { uploader: makeClient(), reader: makeClient(), objects };
}
async function setup(time = NOW) {
  const f = fakeStorage();
  const locator = await uploadPrivateCandidate(f.uploader, input);
  const nonce = newRereadChallenge();
  const signed = await signRereadFromPrivateStorage(f.reader, locator, {
    privateKey: keys.privateKey, attestorId, keyId, challengeNonce: nonce, verifiedAt: time,
  });
  const policy = {
    trustedPublicKey: keys.publicKey, trustedAttestorId: attestorId, trustedKeyId: keyId,
    challengeNonce: nonce, nowUtc: NOW, maxAgeMs: 600000,
  };
  return { ...f, locator, signed, policy };
}
function reject(fn: () => unknown, code: string) {
  assert.throws(fn, (e: unknown) => e instanceof PrivateSourceError && e.code === code, code);
}
test("signed attestation binds actual reread to approved verifier key and nonce", async () => {
  const { signed, locator, policy } = await setup();
  const verified = verifySignedReread(signed, locator, policy);
  assert.equal(verified.signature, "VALID_PINNED_ED25519");
  assert.equal(verified.registryAdmission, false);
  assert.equal(verified.registryAttestation, false);
  assert.equal(verified.independentPrincipalProven, false);
  assert.equal(verified.historicalCutoffBytesProven, false);
});
test("invalid trust anchor public key fails", async () => {
  const { signed, locator, policy } = await setup();
  reject(() => verifySignedReread(signed, locator, {
    ...policy, trustedPublicKey: wrongKeys.publicKey,
  }), "SIGNED_RECEIPT_SIGNATURE_MISMATCH");
});
test("challenge nonce substitution prevents replay", async () => {
  const { signed, locator, policy } = await setup();
  reject(() => verifySignedReread(signed, locator, {
    ...policy, challengeNonce: newRereadChallenge(),
  }), "SIGNED_RECEIPT_TRUST_BINDING_MISMATCH");
});
test("attestor/key identity is pinned outside the receipt", async () => {
  const { signed, locator, policy } = await setup();
  reject(() => verifySignedReread(signed, locator, {
    ...policy, trustedAttestorId: "a-different-operator",
  }), "SIGNED_RECEIPT_TRUST_BINDING_MISMATCH");
  reject(() => verifySignedReread(signed, locator, {
    ...policy, trustedKeyId: "other-key-id",
  }), "SIGNED_RECEIPT_TRUST_BINDING_MISMATCH");
});
test("signature payload tampering fails without changing Registry admission", async () => {
  const { signed, locator, policy } = await setup();
  const corrupted = structuredClone(signed);
  corrupted.statement.signedAtUtc = new Date(NOW.getTime() + 1000).toISOString();
  reject(() => verifySignedReread(corrupted, locator, policy), "SIGNED_RECEIPT_SIGNATURE_MISMATCH");
});
test("extra unsigned fields rejected", async () => {
  const { signed, locator, policy } = await setup();
  const corrupted = structuredClone(signed);
  Object.assign(corrupted.statement, { registryAdmitted: true });
  reject(() => verifySignedReread(corrupted, locator, policy), "SIGNED_RECEIPT_SHAPE_INVALID");
});
test("altered locator, SHA and source identity fail before signature", async () => {
  const { signed, locator, policy } = await setup();
  reject(() => verifySignedReread(signed, { ...locator, sha256: "f".repeat(64) }, policy),
    "SIGNED_RECEIPT_LOCATOR_MISMATCH");
  reject(() => verifySignedReread(signed, { ...locator, sourceUrl: "https://www.sec.gov/test" }, policy),
    "SIGNED_RECEIPT_LOCATOR_MISMATCH");
});
test("expired and future-dated receipts fail", async () => {
  const expired = await setup(new Date(NOW.getTime() - 900000));
  reject(() => verifySignedReread(expired.signed, expired.locator, expired.policy),
    "SIGNED_RECEIPT_EXPIRED_OR_FUTURE");
  const future = await setup(new Date(NOW.getTime() + 90000));
  reject(() => verifySignedReread(future.signed, future.locator, future.policy),
    "SIGNED_RECEIPT_EXPIRED_OR_FUTURE");
});
test("altered same-size storage bytes cannot be signed", async () => {
  const f = fakeStorage();
  const locator = await uploadPrivateCandidate(f.uploader, input);
  const tampered = Buffer.from(f.objects.get(locator.path)!);
  tampered[70] ^= 1;
  f.objects.set(locator.path, tampered);
  await assert.rejects(
    signRereadFromPrivateStorage(f.reader, locator, {
      privateKey: keys.privateKey, attestorId, keyId,
      challengeNonce: newRereadChallenge(), verifiedAt: NOW,
    }),
    (e: unknown) => e instanceof PrivateSourceError && e.code === "REREAD_SHA256_MISMATCH",
  );
});
test("wrong key algorithm and malformed nonce rejected", async () => {
  const f = fakeStorage();
  const locator = await uploadPrivateCandidate(f.uploader, input);
  const rsa = generateKeyPairSync("rsa", { modulusLength: 2048 });
  await assert.rejects(signRereadFromPrivateStorage(f.reader, locator, {
    privateKey: rsa.privateKey, attestorId, keyId,
    challengeNonce: newRereadChallenge(), verifiedAt: NOW,
  }), (e: unknown) => e instanceof PrivateSourceError && e.code === "ATTESTOR_KEY_MUST_BE_ED25519");
  await assert.rejects(signRereadFromPrivateStorage(f.reader, locator, {
    privateKey: keys.privateKey, attestorId, keyId,
    challengeNonce: "replayed-untrusted-string", verifiedAt: NOW,
  }), (e: unknown) => e instanceof PrivateSourceError && e.code === "CHALLENGE_NONCE_INVALID");
});
test("invalid authority string and malformed Ed25519 signature rejected", async () => {
  const { signed, locator, policy } = await setup();
  const changed = structuredClone(signed);
  Object.assign(changed.statement, { authority: "REGISTRY_ADMITTED" });
  reject(() => verifySignedReread(changed as SignedRereadEnvelope, locator, policy),
    "SIGNED_RECEIPT_AUTHORITY_INVALID");
  reject(() => verifySignedReread({ ...signed, signatureBase64Url: "not-a-signature" }, locator, policy),
    "SIGNED_RECEIPT_SIGNATURE_INVALID");
});

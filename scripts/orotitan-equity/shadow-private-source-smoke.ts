/**
 * Opt-in NONPRODUCTION smoke test. Requires two real authenticated Supabase
 * identities, a preconfigured private Storage bucket, and explicitly pinned
 * test project URL. Never uploads issuer documents or writes Registry tables.
 *
 * Invocation: PREFLIGHT_NONPROD_APPROVED=YES npx tsx scripts/orotitan-equity/shadow-private-source-smoke.ts
 */
import { createClient } from "@supabase/supabase-js";
import { randomUUID, createHash, createPrivateKey, createPublicKey } from "node:crypto";
import { uploadPrivateCandidate, type SourceStorageClient } from "../../lib/orotitan-equity/v3/private-source-bytes-preflight";
import { signRereadFromPrivateStorage, verifySignedReread, newRereadChallenge } from "../../lib/orotitan-equity/v3/private-source-signed-reread";

function mandatory(name: string): string {
  const value = process.env[name];
  if (!value || !value.trim()) throw new Error("ENV_REQUIRED_" + name);
  return value;
}
function assertThat(value: unknown, code: string): asserts value {
  if (!value) throw new Error(code);
}
function decodeUntrustedClaims(jwt: string): Record<string, unknown> {
  const parts = jwt.split(".");
  assertThat(parts.length === 3, "JWT_SHAPE_INVALID");
  try { return JSON.parse(Buffer.from(parts[1], "base64url").toString("utf8")) as Record<string, unknown>; }
  catch { throw new Error("JWT_PAYLOAD_INVALID"); }
}
async function main() {
  assertThat(process.env.PREFLIGHT_NONPROD_APPROVED === "YES", "NONPRODUCTION_APPROVAL_REQUIRED");
  const url = mandatory("SUPABASE_TEST_URL");
  const confirmedRef = mandatory("SUPABASE_TEST_REF");
  const prohibitedProductionRef = mandatory("SUPABASE_PRODUCTION_REF");
  assertThat(/^[a-z0-9]{20}$/.test(confirmedRef) && /^[a-z0-9]{20}$/.test(prohibitedProductionRef) &&
    confirmedRef !== prohibitedProductionRef &&
    url === "https://" + confirmedRef + ".supabase.co", "PRODUCTION_OR_AMBIGUOUS_PROJECT_DENIED");
  const publicKey = mandatory("SUPABASE_TEST_PUBLISHABLE_KEY");
  const writerJwt = mandatory("SUPABASE_TEST_WRITER_JWT");
  const readerJwt = mandatory("SUPABASE_TEST_READER_JWT");
  const privatePem = Buffer.from(mandatory("ATTESTOR_TEST_PRIVATE_KEY_PEM_BASE64"), "base64").toString("utf8");
  const publicPem = Buffer.from(mandatory("ATTESTOR_TEST_PINNED_PUBLIC_KEY_PEM_BASE64"), "base64").toString("utf8");
  const jwtWriter = decodeUntrustedClaims(writerJwt);
  const jwtReader = decodeUntrustedClaims(readerJwt);
  assertThat(jwtWriter.role === "authenticated" && jwtReader.role === "authenticated" &&
    jwtWriter.sub !== jwtReader.sub && typeof jwtWriter.sub === "string" && typeof jwtReader.sub === "string",
    "PRINCIPAL_SEPARATION_INVALID");
  const opts = (jwt: string) => ({
    auth: { autoRefreshToken: false, persistSession: false, detectSessionInUrl: false },
    global: { headers: { Authorization: "Bearer " + jwt } },
  });
  const writer = createClient(url, publicKey, opts(writerJwt));
  const reader = createClient(url, publicKey, opts(readerJwt));
  const [wu, ru] = await Promise.all([
    writer.auth.getUser(writerJwt), reader.auth.getUser(readerJwt),
  ]);
  assertThat(!wu.error && !ru.error && wu.data.user && ru.data.user &&
    wu.data.user.id === jwtWriter.sub && ru.data.user.id === jwtReader.sub &&
    wu.data.user.id !== ru.data.user.id, "AUTH_IDENTITY_UNVERIFIED");
  const buffer = Buffer.from("%PDF-1.7\n" + ("Synthetic shadow-only PDF. No issuer information.\n").repeat(14) + "%%EOF\n");
  const sourceSha = createHash("sha256").update(buffer).digest("hex");
  const bucket = "orotitan-source-files-v1";
  const uploadReq = {
    runId: randomUUID(), artifactId: randomUUID(), version: 1, bucket,
    bytes: buffer, mediaType: "application/pdf" as const,
    sourceUrl: "https://ourbrand.asml.com/m/synthetic/original/shadow-only.pdf",
    publishedOn: "2026-01-01", dataCutoff: "2026-09-21",
    expectedSha256: sourceSha,
  };
  const locator = await uploadPrivateCandidate(writer as unknown as SourceStorageClient, uploadReq);
  // Negative privilege tests: uploader must have no download; verifier must have no upload.
  const deniedRead = await writer.storage.from(bucket).download(locator.path);
  assertThat(!!deniedRead.error || !deniedRead.data, "UPLOADER_READ_PRIVILEGE_TOO_BROAD");
  const deniedWrite = await reader.storage.from(bucket).upload(
    "preadmission/denied/" + randomUUID() + ".pdf", buffer,
    { upsert: false, contentType: "application/pdf" },
  );
  assertThat(!!deniedWrite.error || !deniedWrite.data, "READER_WRITE_PRIVILEGE_TOO_BROAD");
  const nonce = newRereadChallenge();
  const signed = await signRereadFromPrivateStorage(reader as unknown as SourceStorageClient, locator, {
    privateKey: createPrivateKey(privatePem),
    attestorId: "shadow-reread-service",
    keyId: "shadow-key-v1",
    challengeNonce: nonce,
  });
  const checked = verifySignedReread(signed, locator, {
    trustedPublicKey: createPublicKey(publicPem),
    trustedAttestorId: "shadow-reread-service",
    trustedKeyId: "shadow-key-v1",
    challengeNonce: nonce, nowUtc: new Date(),
  });
  assertThat(checked.registryAdmission === false && checked.registryAttestation === false,
    "REGISTRY_AUTHORITY_MUST_REMAIN_FALSE");
  console.log(JSON.stringify({
    schema: "SHADOW_PRIVATE_STORAGE_OPT_IN_SMOKE_V1",
    testProjectConfirmed: true,
    distinctAuthenticatedUsersConfirmed: true,
    uploaderReadDenied: true,
    readerWriteDenied: true,
    signerTransportProof: checked.signature,
    sha256: locator.sha256,
    sizeBytes: locator.sizeBytes,
    objectUri: locator.uri,
    residualNonprodObjectPresent: true,
    admittedToRegistry: false,
    fullyIndependentServiceProven: false,
  }, null, 2));
}
main().catch(err => {
  console.error("PRIVATE_STORAGE_SMOKE_FAILED:", err instanceof Error ? err.message : "UnknownFailure");
  process.exitCode = 1;
});

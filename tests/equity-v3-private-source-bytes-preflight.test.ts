import assert from "node:assert/strict";
import test from "node:test";
import { createHash } from "node:crypto";
import {
  uploadPrivateCandidate, rereadPrivateCandidate, PrivateSourceError,
  type SourceRequest, type SourceStorageClient,
} from "../lib/orotitan-equity/v3/private-source-bytes-preflight";

const RUN = "11111111-1111-4111-8111-111111111111";
const ART = "22222222-2222-4222-8222-222222222222";
const PDF = Buffer.from("%PDF-1.7\n" + "ASML historical financial primary document\n".repeat(12) + "%%EOF");
function source(): SourceRequest {
  return {
    runId: RUN, artifactId: ART, version: 1, bucket: "orotitan-source-files-v1",
    bytes: PDF, mediaType: "application/pdf",
    sourceUrl: "https://ourbrand.asml.com/m/a/original/public.pdf",
    publishedOn: "2026-02-25", dataCutoff: "2026-09-21",
  };
}
function fixture(publicBucket = false) {
  const objects = new Map<string, Buffer>();
  const uploads: Array<{ path: string; upsert: boolean }> = [];
  const makeClient = (): SourceStorageClient => ({
    storage: {
      getBucket: async () => ({ data: { public: publicBucket }, error: null }),
      from: () => ({
        upload: async (path, bytes, options) => {
          uploads.push({ path, upsert: options.upsert });
          if (objects.has(path)) return { data: null, error: Error("Already exists") };
          objects.set(path, Buffer.from(bytes));
          return { data: { path }, error: null };
        },
        download: async path => {
          const bytes = objects.get(path);
          return { data: bytes ? new Blob([new Uint8Array(bytes)]) : null,
            error: bytes ? null : Error("Missing") };
        },
      }),
    },
  });
  return { writer: makeClient(), reader: makeClient(), objects, uploads };
}
async function failure(task: () => Promise<unknown>, code: string) {
  await assert.rejects(task, e => e instanceof PrivateSourceError && e.code === code, code);
}

test("private create-only upload + reread matches immutable locator and SHA256", async () => {
  const f = fixture();
  const locator = await uploadPrivateCandidate(f.writer, source());
  const reread = await rereadPrivateCandidate(f.reader, locator);
  assert.equal(locator.sha256, createHash("sha256").update(PDF).digest("hex"));
  assert.match(locator.uri, /^supabase:\/\/orotitan-source-files-v1\/preadmission\//);
  assert.equal(f.uploads[0].upsert, false);
  assert.equal(reread.bytesReread, "EXACT_MATCH");
  assert.equal(reread.registryAttested, false);
  assert.equal(reread.registryAdmitted, false);
  assert.equal(reread.independentPrincipalProven, false);
  assert.equal(reread.storageImmutableProven, false);
  assert.equal(reread.cutoffHistoricalBytesProven, false);
});
test("public bucket and unlisted buckets fail before any write", async () => {
  await failure(() => uploadPrivateCandidate(fixture(true).writer, source()), "PUBLIC_BUCKET_FORBIDDEN");
  await failure(() => uploadPrivateCandidate(fixture().writer, { ...source(), bucket: "public-assets" as SourceRequest["bucket"] }), "BUCKET_NOT_ALLOWLISTED");
});
test("post-cutoff source and impossible dates rejected", async () => {
  await failure(() => uploadPrivateCandidate(fixture().writer, { ...source(), publishedOn: "2026-09-28" }), "POST_CUTOFF_SOURCE");
  await failure(() => uploadPrivateCandidate(fixture().writer, { ...source(), publishedOn: "2026-02-30" }), "BAD_DATE_VALUE");
});
test("capture hash mismatch, invalid PDF signature rejected", async () => {
  await failure(() => uploadPrivateCandidate(fixture().writer, { ...source(), expectedSha256: "0".repeat(64) }), "CAPTURE_HASH_MISMATCH");
  await failure(() => uploadPrivateCandidate(fixture().writer, { ...source(), bytes: Buffer.alloc(400) }), "BAD_PDF_BYTES");
});
test("untrusted and insecure source URLs fail before upload", async () => {
  await failure(() => uploadPrivateCandidate(fixture().writer, { ...source(), sourceUrl: "https://evil.example/hack.pdf" }), "UNTRUSTED_SOURCE_HOST");
  await failure(() => uploadPrivateCandidate(fixture().writer, { ...source(), sourceUrl: "http://www.sec.gov/Archives/f.pdf" }), "UNSAFE_SOURCE_URL");
  await failure(() => uploadPrivateCandidate(fixture().writer, { ...source(), sourceUrl: "https://www.sec.gov@evil.example/test" }), "UNTRUSTED_SOURCE_HOST");
});
test("cannot silently upsert an existing content-addressed object", async () => {
  const f = fixture();
  await uploadPrivateCandidate(f.writer, source());
  await failure(() => uploadPrivateCandidate(f.writer, source()), "CREATE_ONLY_UPLOAD_FAILED");
});
test("mutation with different size fails independent reread", async () => {
  const f = fixture();
  const c = await uploadPrivateCandidate(f.writer, source());
  f.objects.set(c.path, Buffer.from("%PDF-1.7\n" + "X".repeat(400) + "%%EOF"));
  await failure(() => rereadPrivateCandidate(f.reader, c), "REREAD_SIZE_MISMATCH");
});
test("same-size byte tampering fails SHA256 reread", async () => {
  const f = fixture();
  const c = await uploadPrivateCandidate(f.writer, source());
  const altered = Buffer.from(f.objects.get(c.path)!);
  altered[55] ^= 1;
  f.objects.set(c.path, altered);
  await failure(() => rereadPrivateCandidate(f.reader, c), "REREAD_SHA256_MISMATCH");
});
test("forged object locator and missing file fail closed", async () => {
  const f = fixture();
  const c = await uploadPrivateCandidate(f.writer, source());
  await failure(() => rereadPrivateCandidate(f.reader, { ...c, path: "other/" + c.path }), "FORGED_STORAGE_LOCATOR");
  await failure(() => rereadPrivateCandidate(f.reader, { ...c, uri: "supabase://somewhere-else" }), "FORGED_STORAGE_LOCATOR");
  f.objects.clear();
  await failure(() => rereadPrivateCandidate(f.reader, c), "REREAD_DOWNLOAD_FAILED");
});
test("unknown version, identity and oversized body fail closed", async () => {
  await failure(() => uploadPrivateCandidate(fixture().writer, { ...source(), version: 0 }), "BAD_VERSION");
  await failure(() => uploadPrivateCandidate(fixture().writer, { ...source(), runId: "not-a-uuid" }), "BAD_IDENTITY");
  await failure(() => uploadPrivateCandidate(fixture().writer, { ...source(), bytes: Buffer.alloc(36 * 1024 * 1024) }), "BAD_SOURCE_SIZE");
});

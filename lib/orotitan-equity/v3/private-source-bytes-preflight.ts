import { createHash } from "node:crypto";

/** Non-authoritative, application-side source transport only. Never writes Registry state. */
export const PRIVATE_SOURCE_BUCKETS = ["orotitan-source-files-v1", "orotitan-text-artifacts-v1"] as const;
export type PrivateSourceBucket = (typeof PRIVATE_SOURCE_BUCKETS)[number];
export type SourceMediaType = "application/pdf" | "text/html" | "application/json";
export class PrivateSourceError extends Error {
  readonly code: string;
  constructor(code: string) { super(code); this.name = "PrivateSourceError"; this.code = code; }
}
export interface SourceStorageClient {
  storage: {
    getBucket(name: string): Promise<{ data: { public: boolean } | null; error: unknown }>;
    from(name: string): {
      upload(path: string, body: Uint8Array, options: { upsert: false; contentType: string; cacheControl: string }): Promise<{ data: unknown; error: unknown }>;
      download(path: string): Promise<{ data: { arrayBuffer(): Promise<ArrayBuffer> } | null; error: unknown }>;
    };
  };
}
export interface SourceRequest {
  runId: string;
  artifactId: string;
  version: number;
  bucket: PrivateSourceBucket;
  bytes: Uint8Array;
  mediaType: SourceMediaType;
  sourceUrl: string;
  publishedOn: string;
  dataCutoff: string;
  expectedSha256?: string;
}
export interface PreAdmissionLocator {
  runId: string;
  artifactId: string;
  version: number;
  bucket: PrivateSourceBucket;
  path: string;
  uri: string;
  backend: "SUPABASE_STORAGE";
  mediaType: SourceMediaType;
  sizeBytes: number;
  sha256: string;
  sourceUrl: string;
  publishedOn: string;
  dataCutoff: string;
  dateAuthority: "DECLARED_NOT_INDEPENDENTLY_VERIFIED";
}
export interface RereadProof extends PreAdmissionLocator {
  schema: "OROTITAN_PRIVATE_STORAGE_REREAD_V1_UNATTESTED";
  bytesReread: "EXACT_MATCH";
  independentPrincipalProven: false;
  storageImmutableProven: false;
  cutoffHistoricalBytesProven: false;
  registryAttested: false;
  registryAdmitted: false;
  authorityState: "NON_AUTHORITATIVE";
}
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const shaRe = /^[0-9a-f]{64}$/;
const MAX = 35 * 1024 * 1024;
function requireThat(ok: unknown, code: string): asserts ok {
  if (!ok) throw new PrivateSourceError(code);
}
function validDate(value: string): string {
  requireThat(/^\d{4}-\d{2}-\d{2}$/.test(value), "BAD_DATE_FORMAT");
  const d = new Date(value + "T00:00:00Z");
  requireThat(!Number.isNaN(d.getTime()) && d.toISOString().slice(0, 10) === value, "BAD_DATE_VALUE");
  return value;
}
function validSourceUrl(value: string): string {
  let u: URL;
  try { u = new URL(value); } catch { throw new PrivateSourceError("INVALID_SOURCE_URL"); }
  requireThat(u.protocol === "https:" && !u.username && !u.password && !u.port && !u.search && !u.hash, "UNSAFE_SOURCE_URL");
  requireThat(["www.sec.gov", "www.asml.com", "ourbrand.asml.com", "www.investor.asml.com"].includes(u.hostname), "UNTRUSTED_SOURCE_HOST");
  requireThat(u.pathname.startsWith("/") && !u.pathname.includes("%") && !u.pathname.includes("\\"), "UNSAFE_SOURCE_PATH");
  return u.href;
}
function sha(bytes: Uint8Array): string { return createHash("sha256").update(bytes).digest("hex"); }
function contentValid(bytes: Uint8Array, type: SourceMediaType): void {
  const head = Buffer.from(bytes.subarray(0, 1000)).toString("utf8");
  if (type === "application/pdf") {
    requireThat(head.startsWith("%PDF-") && Buffer.from(bytes.subarray(Math.max(0, bytes.length - 8192))).toString("latin1").includes("%%EOF"), "BAD_PDF_BYTES");
  } else if (type === "application/json") {
    try { JSON.parse(Buffer.from(bytes).toString("utf8")); } catch { throw new PrivateSourceError("BAD_JSON_BYTES"); }
  } else {
    requireThat(/^\s*(?:<!doctype\s+html|<html\b)/i.test(head), "BAD_HTML_BYTES");
    requireThat(!/captcha|access denied|request rate threshold exceeded/i.test(head), "HTML_ACCESS_BLOCK_NOT_SOURCE");
  }
}
async function privateBucket(client: SourceStorageClient, name: PrivateSourceBucket): Promise<void> {
  const result = await client.storage.getBucket(name);
  requireThat(!result.error && result.data !== null, "BUCKET_NOT_VERIFIED");
  requireThat(result.data.public === false, "PUBLIC_BUCKET_FORBIDDEN");
}
function expectedPath(locator: Pick<PreAdmissionLocator, "runId" | "artifactId" | "version" | "sha256" | "mediaType">): string {
  const ext = locator.mediaType === "application/pdf" ? "pdf" : locator.mediaType === "text/html" ? "html" : locator.mediaType === "application/json" ? "json" : null;
  requireThat(ext !== null, "UNSUPPORTED_MEDIA_TYPE");
  requireThat(Number.isSafeInteger(locator.version) && locator.version >= 1 && locator.version <= 9999, "BAD_VERSION");
  return ["preadmission", locator.runId, locator.artifactId, "v" + String(locator.version).padStart(4, "0"), locator.sha256 + "." + ext].join("/");
}
/** Phase A: *create only* object. Cannot register a Registry artifact. */
export async function uploadPrivateCandidate(client: SourceStorageClient, input: SourceRequest): Promise<PreAdmissionLocator> {
  requireThat(uuid.test(input.runId) && uuid.test(input.artifactId), "BAD_IDENTITY");
  requireThat((PRIVATE_SOURCE_BUCKETS as readonly string[]).includes(input.bucket), "BUCKET_NOT_ALLOWLISTED");
  requireThat(["application/pdf", "text/html", "application/json"].includes(input.mediaType), "UNSUPPORTED_MEDIA_TYPE");
  requireThat(input.bytes instanceof Uint8Array && input.bytes.byteLength >= 256 && input.bytes.byteLength <= MAX, "BAD_SOURCE_SIZE");
  const publishedOn = validDate(input.publishedOn);
  const cutoff = validDate(input.dataCutoff);
  requireThat(publishedOn <= cutoff, "POST_CUTOFF_SOURCE");
  const sourceUrl = validSourceUrl(input.sourceUrl);
  contentValid(input.bytes, input.mediaType);
  const bytes = Buffer.from(input.bytes);
  const hash = sha(bytes);
  if (input.expectedSha256 !== undefined) requireThat(shaRe.test(input.expectedSha256) && input.expectedSha256 === hash, "CAPTURE_HASH_MISMATCH");
  const locator: PreAdmissionLocator = {
    runId: input.runId.toLowerCase(), artifactId: input.artifactId.toLowerCase(), version: input.version,
    bucket: input.bucket, path: "", uri: "", backend: "SUPABASE_STORAGE", mediaType: input.mediaType,
    sizeBytes: bytes.length, sha256: hash, sourceUrl, publishedOn, dataCutoff: cutoff,
    dateAuthority: "DECLARED_NOT_INDEPENDENTLY_VERIFIED",
  };
  locator.path = expectedPath(locator);
  locator.uri = "supabase://" + locator.bucket + "/" + locator.path;
  await privateBucket(client, locator.bucket);
  const result = await client.storage.from(locator.bucket).upload(locator.path, bytes, { upsert: false, contentType: locator.mediaType, cacheControl: "0" });
  requireThat(!result.error && result.data !== null, "CREATE_ONLY_UPLOAD_FAILED");
  return locator;
}
/** Phase B: exact byte re-read using a separate client/principal managed by the caller. */
export async function rereadPrivateCandidate(reader: SourceStorageClient, locator: PreAdmissionLocator): Promise<RereadProof> {
  requireThat((PRIVATE_SOURCE_BUCKETS as readonly string[]).includes(locator.bucket), "BUCKET_NOT_ALLOWLISTED");
  requireThat(uuid.test(locator.runId) && uuid.test(locator.artifactId) && shaRe.test(locator.sha256), "INVALID_PROVENANCE_IDENTITY");
  requireThat(Number.isSafeInteger(locator.sizeBytes) && locator.sizeBytes > 0 && locator.sizeBytes <= MAX, "INVALID_SOURCE_SIZE");
  requireThat(locator.path === expectedPath(locator) && locator.uri === "supabase://" + locator.bucket + "/" + locator.path && locator.backend === "SUPABASE_STORAGE", "FORGED_STORAGE_LOCATOR");
  requireThat(validDate(locator.publishedOn) <= validDate(locator.dataCutoff), "POST_CUTOFF_SOURCE");
  validSourceUrl(locator.sourceUrl);
  await privateBucket(reader, locator.bucket);
  const result = await reader.storage.from(locator.bucket).download(locator.path);
  requireThat(!result.error && result.data !== null, "REREAD_DOWNLOAD_FAILED");
  const bytes = new Uint8Array(await result.data.arrayBuffer());
  requireThat(bytes.length === locator.sizeBytes, "REREAD_SIZE_MISMATCH");
  requireThat(sha(bytes) === locator.sha256, "REREAD_SHA256_MISMATCH");
  contentValid(bytes, locator.mediaType);
  return {
    ...locator, schema: "OROTITAN_PRIVATE_STORAGE_REREAD_V1_UNATTESTED", bytesReread: "EXACT_MATCH",
    independentPrincipalProven: false, storageImmutableProven: false, cutoffHistoricalBytesProven: false,
    registryAttested: false, registryAdmitted: false, authorityState: "NON_AUTHORITATIVE",
  };
}

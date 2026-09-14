import { createHash } from "node:crypto";
import { gunzipSync } from "node:zlib";

type JsonObject = Record<string, unknown>;

export type GithubImmutableLocator = {
  backend: "GITHUB_IMMUTABLE";
  repository: string;
  path: string;
  commit_sha: string;
  blob_sha: string;
  locator_format?: "RAW_CANONICAL_V1" | "OROTITAN_MULTIPART_GZIP_V1";
};

export type ContractPin = {
  name: string;
  version: string;
  content_sha256: string;
  locator: GithubImmutableLocator;
};

export type GithubBlobRequest = {
  repository: string;
  path: string;
  commitSha: string;
};

export type GithubBlobFetcher = (request: GithubBlobRequest) => Promise<Uint8Array>;

type MultipartPart = {
  path: string;
  size_bytes: number;
  sha256: string;
  blob_sha: string;
};

type MultipartIndex = {
  format: "OROTITAN_CONTRACT_ARCHIVE_MULTIPART_V1";
  encoding: "base64";
  compression: "gzip";
  canonical_name: string;
  canonical_source: string;
  canonical_size_bytes: number;
  canonical_sha256: string;
  archive_size_bytes: number;
  archive_sha256: string;
  parts: MultipartPart[];
};

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function sha256Hex(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

export function gitBlobSha1(bytes: Uint8Array): string {
  const header = Buffer.from(`blob ${bytes.byteLength}\0`, "utf8");
  return createHash("sha1").update(header).update(bytes).digest("hex");
}

function requireHex(value: unknown, length: 40 | 64, label: string): asserts value is string {
  if (typeof value !== "string" || !new RegExp(`^[0-9a-f]{${length}}$`).test(value)) {
    throw new Error(`${label} must be ${length}-char lowercase hex`);
  }
}

function verifyBytes(bytes: Uint8Array, expectedSize: number | undefined, expectedSha256: string, label: string): void {
  if (expectedSize !== undefined && bytes.byteLength !== expectedSize) {
    throw new Error(`${label} size mismatch: expected ${expectedSize}, got ${bytes.byteLength}`);
  }
  const actual = sha256Hex(bytes);
  if (actual !== expectedSha256) {
    throw new Error(`${label} sha256 mismatch: expected ${expectedSha256}, got ${actual}`);
  }
}

function verifyGitBlob(bytes: Uint8Array, expectedBlobSha: string, label: string): void {
  requireHex(expectedBlobSha, 40, `${label}.blob_sha`);
  const actual = gitBlobSha1(bytes);
  if (actual !== expectedBlobSha) {
    throw new Error(`${label} git blob mismatch: expected ${expectedBlobSha}, got ${actual}`);
  }
}

function parseMultipartIndex(bytes: Uint8Array): MultipartIndex {
  let value: unknown;
  try {
    value = JSON.parse(Buffer.from(bytes).toString("utf8"));
  } catch {
    throw new Error("multipart archive index is not valid JSON");
  }
  if (!isObject(value) || value.format !== "OROTITAN_CONTRACT_ARCHIVE_MULTIPART_V1"
      || value.encoding !== "base64" || value.compression !== "gzip"
      || typeof value.canonical_name !== "string" || typeof value.canonical_source !== "string"
      || typeof value.canonical_size_bytes !== "number" || !Number.isSafeInteger(value.canonical_size_bytes)
      || typeof value.archive_size_bytes !== "number" || !Number.isSafeInteger(value.archive_size_bytes)
      || !Array.isArray(value.parts) || value.parts.length === 0) {
    throw new Error("multipart archive index has invalid shape");
  }
  requireHex(value.canonical_sha256, 64, "multipart.canonical_sha256");
  requireHex(value.archive_sha256, 64, "multipart.archive_sha256");

  const parts: MultipartPart[] = value.parts.map((part, index) => {
    if (!isObject(part) || typeof part.path !== "string"
        || typeof part.size_bytes !== "number" || !Number.isSafeInteger(part.size_bytes)) {
      throw new Error(`multipart.parts[${index}] has invalid shape`);
    }
    requireHex(part.sha256, 64, `multipart.parts[${index}].sha256`);
    requireHex(part.blob_sha, 40, `multipart.parts[${index}].blob_sha`);
    return {
      path: part.path,
      size_bytes: part.size_bytes,
      sha256: part.sha256,
      blob_sha: part.blob_sha,
    };
  });

  return {
    format: "OROTITAN_CONTRACT_ARCHIVE_MULTIPART_V1",
    encoding: "base64",
    compression: "gzip",
    canonical_name: value.canonical_name,
    canonical_source: value.canonical_source,
    canonical_size_bytes: value.canonical_size_bytes,
    canonical_sha256: value.canonical_sha256,
    archive_size_bytes: value.archive_size_bytes,
    archive_sha256: value.archive_sha256,
    parts,
  };
}

export async function resolveContractPin(pin: ContractPin, fetchBlob: GithubBlobFetcher): Promise<Uint8Array> {
  if (!pin || typeof pin.name !== "string" || typeof pin.version !== "string") {
    throw new Error("contract pin is malformed");
  }
  requireHex(pin.content_sha256, 64, "pin.content_sha256");
  const locator = pin.locator;
  if (!locator || locator.backend !== "GITHUB_IMMUTABLE") {
    throw new Error("resolver only accepts GITHUB_IMMUTABLE locators");
  }
  if (locator.repository !== "robzer13/indice_nexus") {
    throw new Error(`unexpected contract repository: ${locator.repository}`);
  }
  requireHex(locator.commit_sha, 40, "locator.commit_sha");
  requireHex(locator.blob_sha, 40, "locator.blob_sha");
  if (!locator.path || locator.path.startsWith("/") || locator.path.includes("..")) {
    throw new Error("locator.path must be a safe repository-relative path");
  }

  const locatorBytes = await fetchBlob({
    repository: locator.repository,
    path: locator.path,
    commitSha: locator.commit_sha,
  });
  verifyGitBlob(locatorBytes, locator.blob_sha, "locator");

  if ((locator.locator_format ?? "RAW_CANONICAL_V1") === "RAW_CANONICAL_V1") {
    verifyBytes(locatorBytes, undefined, pin.content_sha256, "canonical contract");
    return locatorBytes;
  }

  if (locator.locator_format !== "OROTITAN_MULTIPART_GZIP_V1") {
    throw new Error(`unsupported locator format: ${locator.locator_format}`);
  }

  const index = parseMultipartIndex(locatorBytes);
  if (index.canonical_sha256 !== pin.content_sha256) {
    throw new Error("multipart index canonical sha256 does not match pin");
  }

  const encodedParts: Buffer[] = [];
  for (const [partIndex, part] of index.parts.entries()) {
    const partBytes = await fetchBlob({
      repository: locator.repository,
      path: part.path,
      commitSha: locator.commit_sha,
    });
    verifyGitBlob(partBytes, part.blob_sha, `multipart part ${partIndex}`);
    verifyBytes(partBytes, part.size_bytes, part.sha256, `multipart part ${partIndex}`);
    const text = Buffer.from(partBytes).toString("ascii");
    if (!/^[A-Za-z0-9+/=]+$/.test(text)) {
      throw new Error(`multipart part ${partIndex} is not canonical base64 text`);
    }
    encodedParts.push(Buffer.from(text, "ascii"));
  }

  const archive = Buffer.from(Buffer.concat(encodedParts).toString("ascii"), "base64");
  verifyBytes(archive, index.archive_size_bytes, index.archive_sha256, "gzip archive");

  let canonical: Buffer;
  try {
    canonical = gunzipSync(archive);
  } catch {
    throw new Error("gzip archive decompression failed");
  }
  verifyBytes(canonical, index.canonical_size_bytes, index.canonical_sha256, "canonical contract");
  verifyBytes(canonical, index.canonical_size_bytes, pin.content_sha256, "pinned canonical contract");
  return canonical;
}

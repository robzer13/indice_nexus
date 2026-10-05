import { createHash } from "node:crypto";
import manifest from "../../../contracts/orotitan-equity/method-v2/OROTITAN_METHOD_V2_AUTHORITY_MANIFEST_V1.1.json";
import { resolveContractPin, type ContractPin, type GithubBlobFetcher } from "../v1/contract-pin-resolver";

// Offline authority validation only. No DB client, runtime selector or activation.
export const METHOD_V2_V1_0_AUTHORITY_SET_SHA256 = "b3aa7f3357617ea40cfba391bcc4599acf9a1bd4b77615dd22cd5f9a617e50d2";
export const PRE_CERTIFICATION_EFFECTIVE_SHA256 = "a00974d948360998c33c213c071b086f5e165b5e29641ea0b32f00c71a99e4a6";
export const METHOD_V2_AUTHORITY_SET_SHA256 = "1e97ad30595d24d10345cfcb58c8b6c0feeb7272144af1fc12d0affd2d2e33b2";
export const methodV2Manifest = manifest;
export const sha256 = (bytes: Uint8Array | string): string => createHash("sha256").update(bytes).digest("hex");

export function canonicalJson(value: unknown): string {
  if (value === null || typeof value === "string" || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "number" && Number.isSafeInteger(value)) return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (typeof value === "object" && value !== null) {
    const object = value as Record<string, unknown>;
    return `{${Object.keys(object).sort().map(key => `${JSON.stringify(key)}:${canonicalJson(object[key])}`).join(",")}}`;
  }
  throw new Error("INVALID_CANONICAL_VALUE");
}

export function authoritySetSha256(candidate: unknown): string {
  if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) throw new Error("INVALID_MANIFEST");
  const content = { ...candidate } as Record<string, unknown>;
  delete content.authority_set_sha256;
  return sha256(canonicalJson(content));
}

export async function verifyMethodV2Authority(candidate: unknown, fetchBytes: GithubBlobFetcher): Promise<void> {
  if (authoritySetSha256(candidate) !== METHOD_V2_AUTHORITY_SET_SHA256
      || canonicalJson(candidate) !== canonicalJson(manifest)) throw new Error("METHOD_V2_MANIFEST_MISMATCH");
  for (const member of manifest.members) {
    const pin = member.pin as ContractPin;
    const bytes = await fetchBytes({ repository: pin.locator.repository, path: pin.locator.path, commitSha: pin.locator.commit_sha });
    if (sha256(bytes) !== member.file_sha256) throw new Error(`METHOD_V2_MEMBER_BYTES_MISMATCH:${member.id}`);
    // Existing resolver verifies Git blob, canonical bytes, compression and each part.
    await resolveContractPin(pin, fetchBytes);
  }
}

export function classifyMethodGeneration(binding: {
  historicalBeforeActivation: boolean;
  methodologyGeneration?: string;
  analyticalAuthoritySetSha256?: string;
  processVersion?: string;
  runtimeVersion?: string;
  schemaVersion?: string;
}): "METHOD_V1" | "METHOD_V2" {
  if (binding.historicalBeforeActivation) {
    if ((binding.methodologyGeneration !== undefined && binding.methodologyGeneration !== "METHOD_V1")
        || binding.analyticalAuthoritySetSha256 !== undefined) throw new Error("HISTORICAL_AUTHORITY_REBIND_FORBIDDEN");
    return "METHOD_V1";
  }
  if (binding.methodologyGeneration === "METHOD_V1" && binding.analyticalAuthoritySetSha256 === undefined) return "METHOD_V1";
  if (binding.methodologyGeneration === "METHOD_V2" && binding.analyticalAuthoritySetSha256 === METHOD_V2_AUTHORITY_SET_SHA256) return "METHOD_V2";
  throw new Error("METHOD_AUTHORITY_MISSING_OR_MIXED");
}

// Classification checks identity, not artifact availability: consumers must also
// call verifyMethodV2Authority before admission. Historical origin is persisted
// activation/creation provenance, never a caller's runtime-version heuristic.

import { sha256 } from "./authority";
import type { ArtifactRef } from "./pre-certification";

export const METHOD_V2_RUNTIME_BINDING_SHA256 = "b078df1a792aa6dc46b913457406e9d48d6dd79cca55d8951d8f14218eb384d0";
export const METHOD_V2_EVIDENCE_PROFILE_SHA256 = "8679e2aeb8f7be4569670629866a9ee2a63d933f5b04d6ede209aa8310e29a83";
export const METHOD_V2_CERTIFICATION_PROFILE_SHA256 = "a9b3eff1930a9a7e6164cfb1f0248c98c8041975ef9adc77f9d152062df0a53c";
export type StageCode = "RESEARCH" | "DEEP_DIVE" | "INTEGRATION";
export type ArtifactAuthority = "AUTHORITATIVE_STAGE_OUTPUT" | "CHECKPOINT_STAGE_OUTPUT" | "SOURCE_ATTACHMENT";
export type RegistryArtifact = ArtifactRef & {
  run_id: string; stage_code: StageCode; artifact_type: string;
  authority_class: string; authority_state: string; artifact_status: string; availability_state: string;
  storage_backend: string; supabase_bucket: string | null; supabase_object_path: string | null;
  storage_uri: string; size_bytes: number; hash_algorithm: string; media_type: string;
  manifest_artifact_id: string | null; manifest_version: number | null;
};
export type ArtifactExpectation = {
  ref: ArtifactRef; runId: string; stageCode: StageCode; artifactType: string;
  authorityClass: ArtifactAuthority; authorityState: "AUTHORITATIVE" | "CHECKPOINT" | "SUPERSEDED";
  artifactStatus: "SEALED";
  manifestRef?: Pick<ArtifactRef, "artifact_id" | "version">;
};
export interface PersistedArtifactSource {
  readArtifact(artifactId: string, version: number): Promise<RegistryArtifact | null>;
  download(bucket: string, exactObjectPath: string): Promise<Uint8Array>;
}
const approvedBuckets = new Set(["orotitan-text-artifacts-v1", "orotitan-source-files-v1"]);
const textArtifacts = new Set(["PRE_CERTIFICATION_QUESTION_LEDGER", "PRE_CERTIFICATION_CHALLENGE_REPORT", "EVIDENCE_LEDGER", "CERTIFICATION_ARTIFACT"]);
function fail(code: string): never { throw new Error(code); }

/** Expectations are supplied by the trusted Registry consumer, never a storage locator from a caller. */
export async function resolvePersistedMethodV2Artifact(source: PersistedArtifactSource, expected: ArtifactExpectation) {
  const row = await source.readArtifact(expected.ref.artifact_id, expected.ref.version);
  if (!row || row.artifact_id !== expected.ref.artifact_id || row.version !== expected.ref.version
    || row.run_id !== expected.runId || row.stage_code !== expected.stageCode
    || row.artifact_type !== expected.artifactType || row.authority_class !== expected.authorityClass
    || row.authority_state !== expected.authorityState || row.artifact_status !== expected.artifactStatus
    || row.availability_state !== "AVAILABLE" || row.content_sha256 !== expected.ref.content_sha256
    || row.hash_algorithm !== "SHA-256" || !/^[0-9a-f]{64}$/.test(row.content_sha256)
    || !Number.isSafeInteger(row.size_bytes) || row.size_bytes < 0
    || (expected.manifestRef && (row.manifest_artifact_id !== expected.manifestRef.artifact_id
      || row.manifest_version !== expected.manifestRef.version))) fail("METHOD_V2_ARTIFACT_REGISTRY_MISMATCH");
  const bucket = row.supabase_bucket;
  const path = row.supabase_object_path;
  if (row.storage_backend !== "SUPABASE_STORAGE" || !bucket || !approvedBuckets.has(bucket)
    || !path || path.startsWith("/") || path.includes("\\") || /[\u0000-\u001f]/.test(path)
    || path.split("/").some(part => !part || part === "." || part === "..")
    || row.storage_uri !== `supabase://${bucket}/${path}`
    || (textArtifacts.has(expected.artifactType) && (bucket !== "orotitan-text-artifacts-v1"
      || row.media_type !== "application/json"))) fail("METHOD_V2_ARTIFACT_STORAGE_NOT_APPROVED");
  const bytes = new Uint8Array(await source.download(bucket, path));
  if (bytes.byteLength !== row.size_bytes) fail("METHOD_V2_ARTIFACT_SIZE_MISMATCH");
  if (sha256(bytes) !== row.content_sha256) fail("METHOD_V2_ARTIFACT_SHA256_MISMATCH");
  return { registry: row, bytes };
}

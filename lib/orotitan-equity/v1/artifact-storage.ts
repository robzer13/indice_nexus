export type ArtifactStorageBackend = "PRIVATE_GITHUB" | "SUPABASE_STORAGE";

export type PrivateGitHubRegistration = {
  backend: "PRIVATE_GITHUB";
  storageUri: string;
  repository: string;
  path: string;
  commitSha: string;
  blobSha: string;
};

export type SupabaseStorageRegistration = {
  backend: "SUPABASE_STORAGE";
  storageUri: string;
  bucket: string;
  objectPath: string;
};

export type ArtifactStorageRegistration = PrivateGitHubRegistration | SupabaseStorageRegistration;

const SHA1 = /^[0-9a-f]{40}$/;

function assertStableCoordinate(value: string, name: string): void {
  if (!value || value.trim() !== value) throw new Error(`${name} must be a non-empty trimmed string`);
  if (value.includes("..")) throw new Error(`${name} must not contain parent traversal`);
  if (/^https?:\/\//i.test(value)) throw new Error(`${name} must be a stable private coordinate, not a signed/public HTTP URL`);
}

export function assertArtifactStorageRegistration(registration: ArtifactStorageRegistration): void {
  assertStableCoordinate(registration.storageUri, "storageUri");

  if (registration.backend === "PRIVATE_GITHUB") {
    if (registration.repository === "robzer13/indice_nexus") {
      throw new Error("Private run artifacts must not be stored in the public indice_nexus repository");
    }
    assertStableCoordinate(registration.repository, "repository");
    assertStableCoordinate(registration.path, "path");
    if (!SHA1.test(registration.commitSha)) throw new Error("commitSha must be a 40-character lowercase Git SHA-1");
    if (!SHA1.test(registration.blobSha)) throw new Error("blobSha must be a 40-character lowercase Git SHA-1");
    return;
  }

  assertStableCoordinate(registration.bucket, "bucket");
  assertStableCoordinate(registration.objectPath, "objectPath");
}

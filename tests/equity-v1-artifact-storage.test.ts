import assert from "node:assert/strict";
import test from "node:test";
import { assertArtifactStorageRegistration } from "../lib/orotitan-equity/v1/artifact-storage";

test("S01 private GitHub artifact must not target public indice_nexus", () => {
  assert.throws(() => assertArtifactStorageRegistration({
    backend: "PRIVATE_GITHUB",
    storageUri: "github://robzer13/indice_nexus/artifacts/run/file.json",
    repository: "robzer13/indice_nexus",
    path: "artifacts/run/file.json",
    commitSha: "a".repeat(40),
    blobSha: "b".repeat(40),
  }), /must not be stored/);
});

test("S02 private GitHub registration requires immutable commit/blob provenance", () => {
  assert.doesNotThrow(() => assertArtifactStorageRegistration({
    backend: "PRIVATE_GITHUB",
    storageUri: "github://robzer13/orotitan-artifacts/artifacts/run/file.json",
    repository: "robzer13/orotitan-artifacts",
    path: "artifacts/run/file.json",
    commitSha: "a".repeat(40),
    blobSha: "b".repeat(40),
  }));
});

test("S03 Supabase Storage registration rejects signed/public URLs", () => {
  assert.throws(() => assertArtifactStorageRegistration({
    backend: "SUPABASE_STORAGE",
    storageUri: "https://example.supabase.co/storage/v1/object/sign/private/file?token=x",
    bucket: "private",
    objectPath: "run/file.json",
  }), /stable private coordinate/);
});

test("S04 Supabase Storage stable object coordinate is accepted", () => {
  assert.doesNotThrow(() => assertArtifactStorageRegistration({
    backend: "SUPABASE_STORAGE",
    storageUri: "supabase://private/run/file.json",
    bucket: "private",
    objectPath: "run/file.json",
  }));
});

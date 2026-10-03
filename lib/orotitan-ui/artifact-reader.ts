import { createHash } from 'node:crypto';
import {
  assertArtifactStorageRegistration,
  type ArtifactStorageRegistration,
} from '../orotitan-equity/v1/artifact-storage';
import type {
  ArtifactRow,
  ControlledBridgePort,
  RpcResult,
} from '../orotitan-equity/post-c7/chatgpt-supabase-bridge';
import type { ArtifactRef, VerifiedArtifactContent } from './types';

const SHA256 = /^[0-9a-f]{64}$/;
const MAX_TEXT_PREVIEW_BYTES = 2 * 1024 * 1024;

export type ArtifactContentErrorCode =
  | 'ARTIFACT_NOT_LOAD_AUTHORIZED'
  | 'ARTIFACT_REGISTRY_MISMATCH'
  | 'ARTIFACT_INTEGRITY_FAILED'
  | 'ARTIFACT_STORAGE_UNSUPPORTED'
  | 'ARTIFACT_CONTENT_INVALID'
  | 'ARTIFACT_READER_UNCONFIGURED'
  | 'ARTIFACT_STORAGE_FETCH_FAILED';

export class ArtifactContentError extends Error {
  readonly code: ArtifactContentErrorCode;

  constructor(code: ArtifactContentErrorCode, message: string) {
    super(message);
    this.code = code;
    this.name = 'ArtifactContentError';
  }
}

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function gitBlobSha1(bytes: Uint8Array): string {
  const prefix = Buffer.from(`blob ${bytes.byteLength}\0`, 'utf8');
  return createHash('sha1')
    .update(Buffer.concat([prefix, Buffer.from(bytes)]))
    .digest('hex');
}

function integrityError(message: string): ArtifactContentError {
  return new ArtifactContentError('ARTIFACT_INTEGRITY_FAILED', message);
}

function registryError(message: string): ArtifactContentError {
  return new ArtifactContentError('ARTIFACT_REGISTRY_MISMATCH', message);
}

function objectValue(
  value: unknown,
  label: string,
): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw registryError(label + ' must be an object');
  }
  return value as Record<string, unknown>;
}

function arrayValue(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) {
    throw registryError(label + ' must be an array');
  }
  return value;
}

function decodeJsonObject(
  row: ArtifactRow,
  bytes: Uint8Array,
): Record<string, unknown> {
  let text: string;
  try {
    text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  } catch {
    throw new ArtifactContentError(
      'ARTIFACT_CONTENT_INVALID',
      'Stage Manifest is not valid UTF-8 text',
    );
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new ArtifactContentError(
      'ARTIFACT_CONTENT_INVALID',
      'Stage Manifest is not valid JSON',
    );
  }

  const manifest = objectValue(parsed, 'Stage Manifest');
  if (
    manifest.manifest_id !== row.artifact_id ||
    manifest.run_id !== row.run_id ||
    manifest.stage !== row.stage_code
  ) {
    throw registryError('Stage Manifest body identity differs from registry identity');
  }
  return manifest;
}

export function assertLoadAuthorizedArtifact(
  refs: ArtifactRef[],
  artifactId: string,
  version: number,
): ArtifactRef {
  const exact = refs.filter(
    (ref) => ref.artifact_id === artifactId && ref.version === version,
  );

  if (exact.length !== 1) {
    throw new ArtifactContentError(
      'ARTIFACT_NOT_LOAD_AUTHORIZED',
      'Artifact is not authorized by the current LOAD_RESULT exact identity',
    );
  }

  const ref = exact[0];
  if (!ref.content_sha256 || !SHA256.test(ref.content_sha256)) {
    throw registryError('LOAD_RESULT artifact reference is missing an exact SHA-256');
  }
  if (!ref.required_authority_class) {
    throw registryError('LOAD_RESULT artifact reference is missing an authority class');
  }
  return ref;
}

export function findExactArtifactRow(
  rows: ArtifactRow[],
  runId: string,
  artifactId: string,
  version: number,
): ArtifactRow {
  const exact = rows.filter(
    (row) =>
      row.run_id === runId &&
      row.artifact_id === artifactId &&
      row.version === version,
  );
  if (exact.length !== 1) {
    throw registryError('Artifact registry exact identity is missing or duplicated');
  }

  const row = exact[0];
  if (
    row.artifact_status !== 'SEALED' ||
    row.availability_state !== 'AVAILABLE' ||
    (row.authority_state !== 'AUTHORITATIVE' &&
      row.authority_state !== 'CHECKPOINT')
  ) {
    throw registryError('Artifact registry row is not readable');
  }
  return row;
}

export function assertResolvedArtifactMatchesRow(
  resolved: RpcResult,
  row: ArtifactRow,
): void {
  const checks: Array<[keyof ArtifactRow, unknown]> = [
    ['artifact_id', row.artifact_id],
    ['version', row.version],
    ['run_id', row.run_id],
    ['stage_code', row.stage_code],
    ['artifact_type', row.artifact_type],
    ['authority_class', row.authority_class],
    ['authority_state', row.authority_state],
    ['content_sha256', row.content_sha256],
    ['size_bytes', row.size_bytes],
    ['media_type', row.media_type],
    ['storage_backend', row.storage_backend],
    ['storage_uri', row.storage_uri],
    ['github_repository', row.github_repository],
    ['github_path', row.github_path],
    ['github_commit_sha', row.github_commit_sha],
    ['github_blob_sha', row.github_blob_sha],
    ['supabase_bucket', row.supabase_bucket],
    ['supabase_object_path', row.supabase_object_path],
  ];

  for (const [key, expected] of checks) {
    if (resolved[key] !== expected) {
      throw registryError('Resolved artifact metadata differs from the registry row');
    }
  }
}

function storageRegistration(row: ArtifactRow): ArtifactStorageRegistration {
  if (row.storage_backend === 'PRIVATE_GITHUB') {
    if (
      !row.github_repository ||
      !row.github_path ||
      !row.github_commit_sha ||
      !row.github_blob_sha
    ) {
      throw registryError('Private GitHub artifact provenance is incomplete');
    }
    return {
      backend: 'PRIVATE_GITHUB',
      storageUri: row.storage_uri,
      repository: row.github_repository,
      path: row.github_path,
      commitSha: row.github_commit_sha,
      blobSha: row.github_blob_sha,
    };
  }

  if (row.storage_backend === 'SUPABASE_STORAGE') {
    if (!row.supabase_bucket || !row.supabase_object_path) {
      throw registryError('Supabase Storage artifact provenance is incomplete');
    }
    return {
      backend: 'SUPABASE_STORAGE',
      storageUri: row.storage_uri,
      bucket: row.supabase_bucket,
      objectPath: row.supabase_object_path,
    };
  }

  throw new ArtifactContentError(
    'ARTIFACT_STORAGE_UNSUPPORTED',
    'Artifact storage backend is not supported',
  );
}

export function assertArtifactStorageCoordinates(row: ArtifactRow): void {
  const registration = storageRegistration(row);
  try {
    assertArtifactStorageRegistration(registration);
  } catch (error) {
    throw registryError(
      error instanceof Error ? error.message : 'Artifact storage registration is invalid',
    );
  }

  if (registration.backend === 'PRIVATE_GITHUB') {
    const expectedUri =
      `github://${registration.repository}@${registration.commitSha}/${registration.path}`;
    if (row.storage_uri !== expectedUri) {
      throw registryError('Artifact storage URI does not match immutable GitHub coordinates');
    }
  }
}

export function verifyArtifactBytes(row: ArtifactRow, bytes: Uint8Array): void {
  assertArtifactStorageCoordinates(row);

  if (bytes.byteLength !== row.size_bytes) {
    throw integrityError('Artifact byte size does not match the registry');
  }
  if (!SHA256.test(row.content_sha256) || sha256(bytes) !== row.content_sha256) {
    throw integrityError('Artifact SHA-256 does not match the registry');
  }

  if (row.storage_backend === 'PRIVATE_GITHUB') {
    if (!row.github_blob_sha || gitBlobSha1(bytes) !== row.github_blob_sha) {
      throw integrityError('Artifact Git blob SHA does not match the registry');
    }
  }
}

async function readArtifactBytes(
  port: ControlledBridgePort,
  row: ArtifactRow,
  readPrivateGithub: (row: ArtifactRow) => Promise<Uint8Array>,
): Promise<Uint8Array> {
  assertArtifactStorageCoordinates(row);

  if (row.storage_backend === 'PRIVATE_GITHUB') {
    return readPrivateGithub(row);
  }
  if (row.storage_backend === 'SUPABASE_STORAGE') {
    if (!row.supabase_bucket || !row.supabase_object_path) {
      throw registryError('Supabase Storage artifact provenance is incomplete');
    }
    return port.readSupabaseObject(
      row.supabase_bucket,
      row.supabase_object_path,
    );
  }

  throw new ArtifactContentError(
    'ARTIFACT_STORAGE_UNSUPPORTED',
    'Artifact storage backend is not supported',
  );
}

async function assertRpcResolvedArtifact(
  port: ControlledBridgePort,
  row: ArtifactRow,
): Promise<void> {
  const resolved = await port.resolveArtifact({
    p_run_id: row.run_id,
    p_artifact_id: row.artifact_id,
    p_version: row.version,
    p_expected_sha256: row.content_sha256,
    p_required_authority_class: row.authority_class,
  });
  assertResolvedArtifactMatchesRow(resolved, row);
}

function assertManifestOutputMatchesRow(
  output: Record<string, unknown>,
  row: ArtifactRow,
): void {
  const exactFields: Array<[string, unknown]> = [
    ['artifact_id', row.artifact_id],
    ['version', row.version],
    ['artifact_type', row.artifact_type],
    ['authority_class', row.authority_class],
    ['content_sha256', row.content_sha256],
    ['media_type', row.media_type],
    ['size_bytes', row.size_bytes],
  ];
  for (const [key, expected] of exactFields) {
    if (output[key] !== expected) {
      throw registryError(
        'Stage Manifest output metadata differs from registry metadata',
      );
    }
  }

  const storageRef = objectValue(output.storage_ref, 'Stage Manifest storage_ref');
  if (storageRef.backend !== row.storage_backend) {
    throw registryError('Stage Manifest storage backend differs from registry metadata');
  }

  if (row.storage_backend === 'PRIVATE_GITHUB') {
    const githubFields: Array<[string, unknown]> = [
      ['repository', row.github_repository],
      ['path', row.github_path],
      ['commit_sha', row.github_commit_sha],
      ['blob_sha', row.github_blob_sha],
    ];
    for (const [key, expected] of githubFields) {
      if (storageRef[key] !== expected) {
        throw registryError(
          'Stage Manifest GitHub storage coordinates differ from registry metadata',
        );
      }
    }
  }

  if (row.storage_backend === 'SUPABASE_STORAGE') {
    const bucket = storageRef.bucket ?? storageRef.supabase_bucket;
    const objectPath =
      storageRef.object_path ?? storageRef.supabase_object_path ?? storageRef.path;
    if (
      bucket !== row.supabase_bucket ||
      objectPath !== row.supabase_object_path
    ) {
      throw registryError(
        'Stage Manifest Supabase storage coordinates differ from registry metadata',
      );
    }
  }
}

async function verifyActiveManifestMembership(input: {
  port: ControlledBridgePort;
  loadRefs: ArtifactRef[];
  artifactRows: ArtifactRow[];
  row: ArtifactRow;
  readPrivateGithub: (row: ArtifactRow) => Promise<Uint8Array>;
}): Promise<Uint8Array | null> {
  const stage = await input.port.getStage(input.row.run_id, input.row.stage_code);
  if (
    !stage ||
    stage.active_manifest_artifact_id === null ||
    stage.active_manifest_version === null ||
    stage.active_manifest_kind === null
  ) {
    throw registryError('Artifact stage has no complete active manifest identity');
  }

  const manifestRow = findExactArtifactRow(
    input.artifactRows,
    input.row.run_id,
    stage.active_manifest_artifact_id,
    stage.active_manifest_version,
  );
  if (manifestRow.stage_code !== input.row.stage_code) {
    throw registryError('Active Stage Manifest belongs to a different stage');
  }

  const manifestRef = assertLoadAuthorizedArtifact(
    input.loadRefs,
    manifestRow.artifact_id,
    manifestRow.version,
  );
  if (
    manifestRef.content_sha256 !== manifestRow.content_sha256 ||
    manifestRef.required_authority_class !== manifestRow.authority_class
  ) {
    throw registryError('Active Stage Manifest differs from LOAD_RESULT metadata');
  }

  await assertRpcResolvedArtifact(input.port, manifestRow);
  const manifestBytes = await readArtifactBytes(
    input.port,
    manifestRow,
    input.readPrivateGithub,
  );
  verifyArtifactBytes(manifestRow, manifestBytes);
  const manifest = decodeJsonObject(manifestRow, manifestBytes);

  if (manifest.manifest_kind !== stage.active_manifest_kind) {
    throw registryError('Stage Manifest kind differs from active stage state');
  }

  const outputs = arrayValue(
    manifest.output_artifacts,
    'Stage Manifest output_artifacts',
  );
  const selfReferences = outputs.filter((value) => {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
      return false;
    }
    const output = value as Record<string, unknown>;
    return (
      output.artifact_id === manifestRow.artifact_id &&
      output.version === manifestRow.version
    );
  });
  if (selfReferences.length !== 0) {
    throw registryError('Stage Manifest violates the self-reference firewall');
  }

  const requestedIsManifest =
    input.row.artifact_id === manifestRow.artifact_id &&
    input.row.version === manifestRow.version;
  if (requestedIsManifest) {
    if (
      input.row.manifest_artifact_id !== null ||
      input.row.manifest_version !== null
    ) {
      throw registryError('Stage Manifest registry row must not self-bind');
    }
    return manifestBytes;
  }

  if (
    input.row.manifest_artifact_id !== manifestRow.artifact_id ||
    input.row.manifest_version !== manifestRow.version
  ) {
    throw registryError('Artifact is not bound to the active Stage Manifest');
  }

  const exactOutputs = outputs.filter((value) => {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
      return false;
    }
    const output = value as Record<string, unknown>;
    return (
      output.artifact_id === input.row.artifact_id &&
      output.version === input.row.version
    );
  });
  if (exactOutputs.length !== 1) {
    throw registryError(
      'Artifact does not have exactly one exact membership in active Stage Manifest',
    );
  }

  assertManifestOutputMatchesRow(
    exactOutputs[0] as Record<string, unknown>,
    input.row,
  );
  return null;
}

function buildVerifiedArtifactContent(
  row: ArtifactRow,
  bytes: Uint8Array,
): VerifiedArtifactContent {
  verifyArtifactBytes(row, bytes);

  let previewText: string | null = null;
  let previewKind: VerifiedArtifactContent['previewKind'] = 'VERIFIED_ONLY';
  let previewReason: string | null = null;
  const baseMediaType = row.media_type
    .split(';', 1)[0]
    .trim()
    .toLocaleLowerCase('en-US');
  const jsonMediaType =
    baseMediaType === 'application/json' || baseMediaType.endsWith('+json');
  const textMediaType = baseMediaType.startsWith('text/');

  if (jsonMediaType || textMediaType) {
    if (bytes.byteLength > MAX_TEXT_PREVIEW_BYTES) {
      previewReason = 'Verified text artifact exceeds the preview size limit';
    } else {
      let text: string;
      try {
        text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
      } catch {
        throw new ArtifactContentError(
          'ARTIFACT_CONTENT_INVALID',
          'Artifact is not valid UTF-8 text',
        );
      }

      if (jsonMediaType) {
        try {
          JSON.parse(text);
        } catch {
          throw new ArtifactContentError(
            'ARTIFACT_CONTENT_INVALID',
            'Artifact is not valid JSON',
          );
        }
      }

      previewText = text;
      previewKind = 'TEXT';
    }
  } else {
    previewReason = 'Binary artifact verified; inline preview is disabled';
  }

  return {
    artifactId: row.artifact_id,
    version: row.version,
    mediaType: row.media_type,
    sizeBytes: row.size_bytes,
    contentSha256: row.content_sha256,
    storageBackend: row.storage_backend,
    previewKind,
    previewText,
    previewReason,
    verification: {
      registryResolved: true,
      manifestMembershipVerified: true,
      sizeVerified: true,
      sha256Verified: true,
      gitBlobVerified:
        row.storage_backend === 'PRIVATE_GITHUB' ? true : null,
    },
  };
}

export async function resolveVerifiedArtifactContent(input: {
  port: ControlledBridgePort;
  loadRefs: ArtifactRef[];
  artifactRows: ArtifactRow[];
  runId: string;
  artifactId: string;
  version: number;
  readPrivateGithub: (row: ArtifactRow) => Promise<Uint8Array>;
}): Promise<VerifiedArtifactContent> {
  const ref = assertLoadAuthorizedArtifact(
    input.loadRefs,
    input.artifactId,
    input.version,
  );
  const row = findExactArtifactRow(
    input.artifactRows,
    input.runId,
    input.artifactId,
    input.version,
  );

  if (
    row.content_sha256 !== ref.content_sha256 ||
    row.authority_class !== ref.required_authority_class
  ) {
    throw registryError('LOAD_RESULT artifact authority does not match registry metadata');
  }

  await assertRpcResolvedArtifact(input.port, row);

  const manifestBytes = await verifyActiveManifestMembership({
    port: input.port,
    loadRefs: input.loadRefs,
    artifactRows: input.artifactRows,
    row,
    readPrivateGithub: input.readPrivateGithub,
  });

  const bytes =
    manifestBytes ??
    (await readArtifactBytes(
      input.port,
      row,
      input.readPrivateGithub,
    ));

  return buildVerifiedArtifactContent(row, bytes);
}

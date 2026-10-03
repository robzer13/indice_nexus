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

  const expectedUri =
    registration.backend === 'PRIVATE_GITHUB'
      ? `github://${registration.repository}@${registration.commitSha}/${registration.path}`
      : `supabase://${registration.bucket}/${registration.objectPath}`;

  if (row.storage_uri !== expectedUri) {
    throw registryError('Artifact storage URI does not match immutable coordinates');
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

export function buildVerifiedArtifactContent(
  row: ArtifactRow,
  bytes: Uint8Array,
): VerifiedArtifactContent {
  verifyArtifactBytes(row, bytes);

  let previewText: string | null = null;
  let previewKind: VerifiedArtifactContent['previewKind'] = 'VERIFIED_ONLY';
  let previewReason: string | null = null;

  if (
    row.media_type === 'application/json' ||
    row.media_type.startsWith('text/')
  ) {
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

      if (row.media_type === 'application/json') {
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

  const resolved = await input.port.resolveArtifact({
    p_run_id: input.runId,
    p_artifact_id: input.artifactId,
    p_version: input.version,
    p_expected_sha256: ref.content_sha256,
    p_required_authority_class: ref.required_authority_class,
  });
  assertResolvedArtifactMatchesRow(resolved, row);
  assertArtifactStorageCoordinates(row);

  let bytes: Uint8Array;
  if (row.storage_backend === 'PRIVATE_GITHUB') {
    bytes = await input.readPrivateGithub(row);
  } else if (row.storage_backend === 'SUPABASE_STORAGE') {
    if (!row.supabase_bucket || !row.supabase_object_path) {
      throw registryError('Supabase Storage artifact provenance is incomplete');
    }
    bytes = await input.port.readSupabaseObject(
      row.supabase_bucket,
      row.supabase_object_path,
    );
  } else {
    throw new ArtifactContentError(
      'ARTIFACT_STORAGE_UNSUPPORTED',
      'Artifact storage backend is not supported',
    );
  }

  return buildVerifiedArtifactContent(row, bytes);
}

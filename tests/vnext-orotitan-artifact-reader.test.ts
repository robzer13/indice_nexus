import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';

import {
  ArtifactContentError,
  buildVerifiedArtifactContent,
  resolveVerifiedArtifactContent,
  verifyArtifactBytes,
} from '../lib/orotitan-ui/artifact-reader';
import type {
  ArtifactRow,
  ControlledBridgePort,
  RpcResult,
} from '../lib/orotitan-equity/post-c7/chatgpt-supabase-bridge';

function hashes(bytes: Uint8Array): { sha256: string; blob: string } {
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  const prefix = Buffer.from(`blob ${bytes.byteLength}\0`, 'utf8');
  const blob = createHash('sha1')
    .update(Buffer.concat([prefix, Buffer.from(bytes)]))
    .digest('hex');
  return { sha256, blob };
}

function rowFor(bytes: Uint8Array): ArtifactRow {
  const digest = hashes(bytes);
  const repository = 'robzer13/real-orotitan';
  const commit = '1'.repeat(40);
  const path =
    'artifacts/orotitan-equity/runs/40000000-0000-4000-8000-000000000001/research/EVIDENCE_LEDGER__50000000-0000-4000-8000-000000000001__v001.json';
  return {
    artifact_id: '50000000-0000-4000-8000-000000000001',
    version: 1,
    run_id: '40000000-0000-4000-8000-000000000001',
    stage_code: 'RESEARCH',
    artifact_type: 'EVIDENCE_LEDGER',
    logical_name: 'Evidence Ledger',
    authority_class: 'AUTHORITATIVE_STAGE_OUTPUT',
    authority_state: 'AUTHORITATIVE',
    artifact_status: 'SEALED',
    availability_state: 'AVAILABLE',
    content_sha256: digest.sha256,
    size_bytes: bytes.byteLength,
    media_type: 'application/json',
    storage_backend: 'PRIVATE_GITHUB',
    storage_uri: `github://${repository}@${commit}/${path}`,
    github_repository: repository,
    github_path: path,
    github_commit_sha: commit,
    github_blob_sha: digest.blob,
    supabase_bucket: null,
    supabase_object_path: null,
  };
}

function resolved(row: ArtifactRow): RpcResult {
  return {
    artifact_id: row.artifact_id,
    version: row.version,
    run_id: row.run_id,
    stage_code: row.stage_code,
    artifact_type: row.artifact_type,
    authority_class: row.authority_class,
    authority_state: row.authority_state,
    content_sha256: row.content_sha256,
    size_bytes: row.size_bytes,
    media_type: row.media_type,
    storage_backend: row.storage_backend,
    storage_uri: row.storage_uri,
    github_repository: row.github_repository,
    github_path: row.github_path,
    github_commit_sha: row.github_commit_sha,
    github_blob_sha: row.github_blob_sha,
    supabase_bucket: row.supabase_bucket,
    supabase_object_path: row.supabase_object_path,
  };
}

function portFor(
  row: ArtifactRow,
  capture: Record<string, unknown>,
): ControlledBridgePort {
  return {
    async listIssuers() {
      return [];
    },
    async listSecurities() {
      return [];
    },
    async listDossiers() {
      return [];
    },
    async listRuns() {
      return [];
    },
    async getRun() {
      return null;
    },
    async getStage() {
      return null;
    },
    async listArtifacts() {
      return [row];
    },
    async readSupabaseObject() {
      throw new Error('not used');
    },
    async resolveArtifact(args) {
      capture.args = args;
      return resolved(row);
    },
    async checkpointStage() {
      throw new Error('not used');
    },
    async finalizeStage() {
      throw new Error('not used');
    },
    async reopenStage() {
      throw new Error('not used');
    },
  };
}

test('verified artifact reader resolves exact LOAD identity and verifies private GitHub bytes', async () => {
  const bytes = new TextEncoder().encode('{"ok":true}\n');
  const row = rowFor(bytes);
  const capture: Record<string, unknown> = {};
  const content = await resolveVerifiedArtifactContent({
    port: portFor(row, capture),
    loadRefs: [
      {
        artifact_id: row.artifact_id,
        version: row.version,
        content_sha256: row.content_sha256,
        required_authority_class: row.authority_class,
      },
    ],
    artifactRows: [row],
    runId: row.run_id,
    artifactId: row.artifact_id,
    version: row.version,
    async readPrivateGithub() {
      return bytes;
    },
  });

  assert.deepEqual(capture.args, {
    p_run_id: row.run_id,
    p_artifact_id: row.artifact_id,
    p_version: row.version,
    p_expected_sha256: row.content_sha256,
    p_required_authority_class: row.authority_class,
  });
  assert.equal(content.previewKind, 'TEXT');
  assert.equal(content.previewText, '{"ok":true}\n');
  assert.equal(content.verification.registryResolved, true);
  assert.equal(content.verification.sizeVerified, true);
  assert.equal(content.verification.sha256Verified, true);
  assert.equal(content.verification.gitBlobVerified, true);
});

test('verified artifact reader rejects an artifact absent from LOAD_RESULT', async () => {
  const bytes = new TextEncoder().encode('{"ok":true}\n');
  const row = rowFor(bytes);

  await assert.rejects(
    resolveVerifiedArtifactContent({
      port: portFor(row, {}),
      loadRefs: [],
      artifactRows: [row],
      runId: row.run_id,
      artifactId: row.artifact_id,
      version: row.version,
      async readPrivateGithub() {
        return bytes;
      },
    }),
    (error: unknown) =>
      error instanceof ArtifactContentError &&
      error.code === 'ARTIFACT_NOT_LOAD_AUTHORIZED',
  );
});

test('verified artifact reader rejects tampered exact bytes', () => {
  const bytes = new TextEncoder().encode('{"ok":true}\n');
  const row = rowFor(bytes);
  const tampered = new TextEncoder().encode('{"ok":false}\n');

  assert.throws(
    () => verifyArtifactBytes(row, tampered),
    (error: unknown) =>
      error instanceof ArtifactContentError &&
      error.code === 'ARTIFACT_INTEGRITY_FAILED',
  );
});

test('verified artifact reader rejects resolver metadata drift', async () => {
  const bytes = new TextEncoder().encode('{"ok":true}\n');
  const row = rowFor(bytes);
  const port = portFor(row, {});
  port.resolveArtifact = async () => ({
    ...resolved(row),
    github_blob_sha: 'f'.repeat(40),
  });

  await assert.rejects(
    resolveVerifiedArtifactContent({
      port,
      loadRefs: [
        {
          artifact_id: row.artifact_id,
          version: row.version,
          content_sha256: row.content_sha256,
          required_authority_class: row.authority_class,
        },
      ],
      artifactRows: [row],
      runId: row.run_id,
      artifactId: row.artifact_id,
      version: row.version,
      async readPrivateGithub() {
        return bytes;
      },
    }),
    (error: unknown) =>
      error instanceof ArtifactContentError &&
      error.code === 'ARTIFACT_REGISTRY_MISMATCH',
  );
});


test('verified artifact reader rejects invalid storage coordinates before private fetch', async () => {
  const bytes = new TextEncoder().encode('{"ok":true}\n');
  const row = {
    ...rowFor(bytes),
    storage_uri: 'github://robzer13/real-orotitan@' + '1'.repeat(40) + '/../escape.json',
    github_path: '../escape.json',
  };
  let fetched = false;

  await assert.rejects(
    resolveVerifiedArtifactContent({
      port: portFor(row, {}),
      loadRefs: [
        {
          artifact_id: row.artifact_id,
          version: row.version,
          content_sha256: row.content_sha256,
          required_authority_class: row.authority_class,
        },
      ],
      artifactRows: [row],
      runId: row.run_id,
      artifactId: row.artifact_id,
      version: row.version,
      async readPrivateGithub() {
        fetched = true;
        return bytes;
      },
    }),
    (error: unknown) =>
      error instanceof ArtifactContentError &&
      error.code === 'ARTIFACT_REGISTRY_MISMATCH',
  );
  assert.equal(fetched, false);
});


test('verified artifact reader normalizes JSON media types before preview validation', () => {
  const bytes = new TextEncoder().encode('{"ok":true}\n');
  const row = {
    ...rowFor(bytes),
    media_type: 'Application/LD+JSON; charset=utf-8',
  };

  const contentResult = buildVerifiedArtifactContent(row, bytes);
  assert.equal(contentResult.previewKind, 'TEXT');
  assert.equal(contentResult.previewText, '{"ok":true}\n');
});

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
  StageRow,
} from '../lib/orotitan-equity/post-c7/chatgpt-supabase-bridge';

const RUN_ID = '40000000-0000-4000-8000-000000000001';
const ARTIFACT_ID = '50000000-0000-4000-8000-000000000001';
const MANIFEST_ID = '60000000-0000-4000-8000-000000000001';
const REPOSITORY = 'robzer13/real-orotitan';
const COMMIT = '1'.repeat(40);

function hashes(bytes: Uint8Array): { sha256: string; blob: string } {
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  const prefix = Buffer.from(`blob ${bytes.byteLength}\0`, 'utf8');
  const blob = createHash('sha1')
    .update(Buffer.concat([prefix, Buffer.from(bytes)]))
    .digest('hex');
  return { sha256, blob };
}

function outputRowFor(
  bytes: Uint8Array,
  overrides: Partial<ArtifactRow> = {},
): ArtifactRow {
  const digest = hashes(bytes);
  const path =
    'artifacts/orotitan-equity/runs/' +
    RUN_ID +
    '/research/EVIDENCE_LEDGER__' +
    ARTIFACT_ID +
    '__v001.json';

  return {
    artifact_id: ARTIFACT_ID,
    version: 1,
    run_id: RUN_ID,
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
    storage_uri: `github://${REPOSITORY}@${COMMIT}/${path}`,
    github_repository: REPOSITORY,
    github_path: path,
    github_commit_sha: COMMIT,
    github_blob_sha: digest.blob,
    supabase_bucket: null,
    supabase_object_path: null,
    manifest_artifact_id: MANIFEST_ID,
    manifest_version: 1,
    ...overrides,
  };
}

function manifestBodyFor(
  output: ArtifactRow,
  outputOverrides: Record<string, unknown> = {},
): Uint8Array {
  const body = {
    manifest_schema_version: '1.0',
    manifest_id: MANIFEST_ID,
    manifest_kind: 'FINAL',
    run_id: RUN_ID,
    stage: 'RESEARCH',
    stage_revision: 1,
    output_artifacts: [
      {
        artifact_id: output.artifact_id,
        version: output.version,
        artifact_type: output.artifact_type,
        authority_class: output.authority_class,
        content_sha256: output.content_sha256,
        media_type: output.media_type,
        size_bytes: output.size_bytes,
        storage_ref: {
          backend: output.storage_backend,
          repository: output.github_repository,
          path: output.github_path,
          commit_sha: output.github_commit_sha,
          blob_sha: output.github_blob_sha,
          storage_uri: output.storage_uri,
        },
        ...outputOverrides,
      },
    ],
  };
  return new TextEncoder().encode(JSON.stringify(body) + '\n');
}

function manifestRowFor(
  bytes: Uint8Array,
  overrides: Partial<ArtifactRow> = {},
): ArtifactRow {
  const digest = hashes(bytes);
  const path =
    'artifacts/orotitan-equity/runs/' +
    RUN_ID +
    '/research/FINAL_RESEARCH_STAGE_MANIFEST__' +
    MANIFEST_ID +
    '__v001.json';

  return {
    artifact_id: MANIFEST_ID,
    version: 1,
    run_id: RUN_ID,
    stage_code: 'RESEARCH',
    artifact_type: 'RESEARCH_STAGE_MANIFEST',
    logical_name: 'Research Stage Manifest',
    authority_class: 'AUTHORITATIVE_STAGE_MANIFEST',
    authority_state: 'AUTHORITATIVE',
    artifact_status: 'SEALED',
    availability_state: 'AVAILABLE',
    content_sha256: digest.sha256,
    size_bytes: bytes.byteLength,
    media_type: 'application/json',
    storage_backend: 'PRIVATE_GITHUB',
    storage_uri: `github://${REPOSITORY}@${COMMIT}/${path}`,
    github_repository: REPOSITORY,
    github_path: path,
    github_commit_sha: COMMIT,
    github_blob_sha: digest.blob,
    supabase_bucket: null,
    supabase_object_path: null,
    manifest_artifact_id: null,
    manifest_version: null,
    ...overrides,
  };
}

function stageFor(manifest: ArtifactRow): StageRow {
  return {
    run_id: RUN_ID,
    stage_code: 'RESEARCH',
    stage_revision: 1,
    lifecycle_status: 'COMPLETE',
    handoff_gate_state: 'YES',
    active_manifest_artifact_id: manifest.artifact_id,
    active_manifest_version: manifest.version,
    active_manifest_kind: 'FINAL',
    blocker_summary: [],
    state_version: 1,
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

function refFor(row: ArtifactRow) {
  return {
    artifact_id: row.artifact_id,
    version: row.version,
    content_sha256: row.content_sha256,
    required_authority_class: row.authority_class,
  };
}

function portFor(
  rows: ArtifactRow[],
  stage: StageRow,
  capture: { calls?: unknown[] } = {},
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
    async getStage(runId, stageCode) {
      if (runId === stage.run_id && stageCode === stage.stage_code) {
        return stage;
      }
      return null;
    },
    async listArtifacts() {
      return rows;
    },
    async readSupabaseObject() {
      throw new Error('not used');
    },
    async resolveArtifact(args) {
      capture.calls = [...(capture.calls ?? []), args];
      const row = rows.find(
        (candidate) =>
          candidate.run_id === args.p_run_id &&
          candidate.artifact_id === args.p_artifact_id &&
          candidate.version === args.p_version,
      );
      if (!row) throw new Error('missing fixture artifact');
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

function privateReader(
  bytesByArtifactId: Map<string, Uint8Array>,
  fetched: string[] = [],
) {
  return async (row: ArtifactRow): Promise<Uint8Array> => {
    fetched.push(row.artifact_id);
    const bytes = bytesByArtifactId.get(row.artifact_id);
    if (!bytes) throw new Error('missing fixture bytes');
    return bytes;
  };
}

function fixture(
  outputOverrides: Partial<ArtifactRow> = {},
  manifestOutputOverrides: Record<string, unknown> = {},
) {
  const outputBytes = new TextEncoder().encode('{"ok":true}\n');
  const output = outputRowFor(outputBytes, outputOverrides);
  const manifestBytes = manifestBodyFor(output, manifestOutputOverrides);
  const manifest = manifestRowFor(manifestBytes);
  const stage = stageFor(manifest);
  const refs = [refFor(output), refFor(manifest)];
  const rows = [output, manifest];
  const bytesByArtifactId = new Map([
    [output.artifact_id, outputBytes],
    [manifest.artifact_id, manifestBytes],
  ]);
  return { outputBytes, output, manifestBytes, manifest, stage, refs, rows, bytesByArtifactId };
}

test('verified artifact reader resolves exact LOAD identity, active manifest membership and immutable bytes', async () => {
  const fx = fixture();
  const capture: { calls?: unknown[] } = {};
  const content = await resolveVerifiedArtifactContent({
    port: portFor(fx.rows, fx.stage, capture),
    loadRefs: fx.refs,
    artifactRows: fx.rows,
    runId: fx.output.run_id,
    artifactId: fx.output.artifact_id,
    version: fx.output.version,
    readPrivateGithub: privateReader(fx.bytesByArtifactId),
  });

  assert.equal(capture.calls?.length, 2);
  assert.deepEqual(capture.calls?.[0], {
    p_run_id: fx.output.run_id,
    p_artifact_id: fx.output.artifact_id,
    p_version: fx.output.version,
    p_expected_sha256: fx.output.content_sha256,
    p_required_authority_class: fx.output.authority_class,
  });
  assert.deepEqual(capture.calls?.[1], {
    p_run_id: fx.manifest.run_id,
    p_artifact_id: fx.manifest.artifact_id,
    p_version: fx.manifest.version,
    p_expected_sha256: fx.manifest.content_sha256,
    p_required_authority_class: fx.manifest.authority_class,
  });
  assert.equal(content.previewKind, 'TEXT');
  assert.equal(content.previewText, '{"ok":true}\n');
  assert.equal(content.verification.registryResolved, true);
  assert.equal(content.verification.manifestMembershipVerified, true);
  assert.equal(content.verification.sizeVerified, true);
  assert.equal(content.verification.sha256Verified, true);
  assert.equal(content.verification.gitBlobVerified, true);
});

test('verified artifact reader admits the active Stage Manifest without requiring self-membership', async () => {
  const fx = fixture();
  const content = await resolveVerifiedArtifactContent({
    port: portFor(fx.rows, fx.stage),
    loadRefs: fx.refs,
    artifactRows: fx.rows,
    runId: fx.manifest.run_id,
    artifactId: fx.manifest.artifact_id,
    version: fx.manifest.version,
    readPrivateGithub: privateReader(fx.bytesByArtifactId),
  });

  assert.equal(content.artifactId, fx.manifest.artifact_id);
  assert.equal(content.verification.manifestMembershipVerified, true);
});

test('verified artifact reader rejects an artifact absent from LOAD_RESULT', async () => {
  const fx = fixture();

  await assert.rejects(
    resolveVerifiedArtifactContent({
      port: portFor(fx.rows, fx.stage),
      loadRefs: [refFor(fx.manifest)],
      artifactRows: fx.rows,
      runId: fx.output.run_id,
      artifactId: fx.output.artifact_id,
      version: fx.output.version,
      readPrivateGithub: privateReader(fx.bytesByArtifactId),
    }),
    (error: unknown) =>
      error instanceof ArtifactContentError &&
      error.code === 'ARTIFACT_NOT_LOAD_AUTHORIZED',
  );
});

test('verified artifact reader rejects a registry binding that is not the active Stage Manifest', async () => {
  const fx = fixture({
    manifest_artifact_id: '70000000-0000-4000-8000-000000000001',
  });

  await assert.rejects(
    resolveVerifiedArtifactContent({
      port: portFor(fx.rows, fx.stage),
      loadRefs: fx.refs,
      artifactRows: fx.rows,
      runId: fx.output.run_id,
      artifactId: fx.output.artifact_id,
      version: fx.output.version,
      readPrivateGithub: privateReader(fx.bytesByArtifactId),
    }),
    (error: unknown) =>
      error instanceof ArtifactContentError &&
      error.code === 'ARTIFACT_REGISTRY_MISMATCH',
  );
});

test('verified artifact reader rejects missing exact output membership in active manifest body', async () => {
  const fx = fixture({}, { artifact_id: '70000000-0000-4000-8000-000000000002' });

  await assert.rejects(
    resolveVerifiedArtifactContent({
      port: portFor(fx.rows, fx.stage),
      loadRefs: fx.refs,
      artifactRows: fx.rows,
      runId: fx.output.run_id,
      artifactId: fx.output.artifact_id,
      version: fx.output.version,
      readPrivateGithub: privateReader(fx.bytesByArtifactId),
    }),
    (error: unknown) =>
      error instanceof ArtifactContentError &&
      error.code === 'ARTIFACT_REGISTRY_MISMATCH',
  );
});

test('verified artifact reader rejects tampered exact bytes', () => {
  const fx = fixture();
  const tampered = new TextEncoder().encode('{"ok":false}\n');

  assert.throws(
    () => verifyArtifactBytes(fx.output, tampered),
    (error: unknown) =>
      error instanceof ArtifactContentError &&
      error.code === 'ARTIFACT_INTEGRITY_FAILED',
  );
});

test('verified artifact reader rejects resolver metadata drift', async () => {
  const fx = fixture();
  const port = portFor(fx.rows, fx.stage);
  port.resolveArtifact = async (args) => {
    const row = fx.rows.find(
      (candidate) =>
        candidate.artifact_id === args.p_artifact_id &&
        candidate.version === args.p_version,
    );
    if (!row) throw new Error('missing fixture artifact');
    return {
      ...resolved(row),
      github_blob_sha:
        row.artifact_id === fx.output.artifact_id
          ? 'f'.repeat(40)
          : row.github_blob_sha,
    };
  };

  await assert.rejects(
    resolveVerifiedArtifactContent({
      port,
      loadRefs: fx.refs,
      artifactRows: fx.rows,
      runId: fx.output.run_id,
      artifactId: fx.output.artifact_id,
      version: fx.output.version,
      readPrivateGithub: privateReader(fx.bytesByArtifactId),
    }),
    (error: unknown) =>
      error instanceof ArtifactContentError &&
      error.code === 'ARTIFACT_REGISTRY_MISMATCH',
  );
});

test('verified artifact reader rejects invalid storage coordinates before fetching requested bytes', async () => {
  const fx = fixture({
    storage_uri:
      'github://robzer13/real-orotitan@' + COMMIT + '/../escape.json',
    github_path: '../escape.json',
  });
  const fetched: string[] = [];

  await assert.rejects(
    resolveVerifiedArtifactContent({
      port: portFor(fx.rows, fx.stage),
      loadRefs: fx.refs,
      artifactRows: fx.rows,
      runId: fx.output.run_id,
      artifactId: fx.output.artifact_id,
      version: fx.output.version,
      readPrivateGithub: privateReader(fx.bytesByArtifactId, fetched),
    }),
    (error: unknown) =>
      error instanceof ArtifactContentError &&
      error.code === 'ARTIFACT_REGISTRY_MISMATCH',
  );
  assert.deepEqual(fetched, [fx.manifest.artifact_id]);
});

test('verified artifact reader normalizes JSON media types before preview validation', () => {
  const bytes = new TextEncoder().encode('{"ok":true}\n');
  const row = outputRowFor(bytes, {
    media_type: 'Application/LD+JSON; charset=utf-8',
  });

  const contentResult = buildVerifiedArtifactContent(row, bytes);
  assert.equal(contentResult.previewKind, 'TEXT');
  assert.equal(contentResult.previewText, '{"ok":true}\n');
});

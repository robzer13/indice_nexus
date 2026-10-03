import assert from 'node:assert/strict';
import test from 'node:test';

import {
  blockerTitle,
  buildRunHref,
  deriveStageStates,
  formatCutoff,
  runStatusLabel,
  shortId,
} from '../lib/orotitan-ui/presentation';
import { VEOLIA_MOCK_DOSSIER } from '../lib/orotitan-ui/mock';
import {
  assertUniqueIssuerSlugs,
  buildArtifactCatalog,
  buildCompanyIdentity,
  issuerSlug,
  resolveUiIssuer,
  resolveUiRunSelection,
} from '../lib/orotitan-ui/read-model';
import type { ArtifactRow } from '../lib/orotitan-equity/post-c7/chatgpt-supabase-bridge';

test('OroTitan UI V1 maps the frozen stage sequence deterministically', () => {
  assert.deepEqual(deriveStageStates('DEEP_DIVE', 'BLOCKED'), [
    { stage: 'RESEARCH', lifecycle: 'COMPLETE' },
    { stage: 'DEEP_DIVE', lifecycle: 'BLOCKED' },
    { stage: 'INTEGRATION', lifecycle: 'NOT_STARTED' },
  ]);
});

test('OroTitan UI V1 preserves unknown machine values instead of inventing semantics', () => {
  assert.equal(runStatusLabel('CUSTOM_STATE'), 'CUSTOM STATE');
  assert.equal(blockerTitle({ code: 'UNKNOWN_BLOCKER' }), 'UNKNOWN_BLOCKER');
});

test('OroTitan UI V1 formats presentation-only fields without changing source identity', () => {
  assert.equal(formatCutoff('2026-09-22'), '22/09/2026');
  assert.equal(shortId('a4cf9002-b52d-4440-8902-adc70bd777dd', 8), 'a4cf9002…');
});

test('Veolia mock keeps the frozen LOAD_RESULT read-only shape and context counts', () => {
  const load = VEOLIA_MOCK_DOSSIER.loadResult;
  assert.equal(load.contract_version, '0.1.0');
  assert.equal(load.operation, 'LOAD_RESULT');
  assert.equal(load.mutation_allowed, false);
  assert.equal(load.run_id, 'a4cf9002-b52d-4440-8902-adc70bd777dd');
  assert.equal(load.run_state_version, 7);
  assert.equal(load.stage?.stage_state_version, 4);
  assert.equal(load.artifact_index.length, 12);
  assert.equal(load.context_plan.l0.length, 1);
  assert.equal(load.context_plan.l1.length, 4);
  assert.equal(load.context_plan.l2.length, 0);
  assert.equal(load.context_plan.l3.length, 0);
  assert.equal(load.process_state_artifact, null);
});

test('run resolution never auto-selects or substitutes another requested run', () => {
  const dossier = VEOLIA_MOCK_DOSSIER;
  assert.equal(resolveUiRunSelection(dossier.runSummaries, null).kind, 'select');
  assert.equal(resolveUiRunSelection(dossier.runSummaries, dossier.primaryRunId).kind, 'available');
  assert.equal(
    resolveUiRunSelection(dossier.runSummaries, 'd56a0bc9-4e80-48b3-b326-74d1fea15e63').kind,
    'unavailable',
  );
  assert.equal(resolveUiRunSelection(dossier.runSummaries, '00000000-0000-0000-0000-000000000000').kind, 'unknown');
});

test('run-aware navigation preserves the selected run across dossier routes and filters', () => {
  const runId = VEOLIA_MOCK_DOSSIER.primaryRunId;
  assert.equal(
    buildRunHref('/orotitan/veolia/documents', runId),
    '/orotitan/veolia/documents?run=' + runId,
  );
  assert.equal(
    buildRunHref('/orotitan/veolia/documents', runId, { stage: 'DEEP_DIVE' }),
    '/orotitan/veolia/documents?run=' + runId + '&stage=DEEP_DIVE',
  );
});


test('artifact catalog never exposes a Supabase artifact absent from LOAD_RESULT.artifact_index', () => {
  const sourceLoad = VEOLIA_MOCK_DOSSIER.loadResult;
  const visibleRef = sourceLoad.artifact_index[0];
  const load = {
    ...sourceLoad,
    artifact_index: [visibleRef],
    context_plan: { l0: [], l1: [], l2: [], l3: [] },
  };

  const row = (
    artifactId: string,
    version: number,
    contentSha256: string,
    authorityClass: string,
  ): ArtifactRow => ({
    artifact_id: artifactId,
    version,
    run_id: sourceLoad.run_id!,
    stage_code: 'DEEP_DIVE',
    artifact_type: 'TEST_ARTIFACT',
    logical_name: artifactId,
    authority_class: authorityClass,
    authority_state: 'CHECKPOINT',
    artifact_status: 'SEALED',
    availability_state: 'AVAILABLE',
    content_sha256: contentSha256,
    size_bytes: 1,
    media_type: 'application/json',
    storage_backend: 'PRIVATE_GITHUB',
    storage_uri: 'private://test',
    github_repository: null,
    github_path: null,
    github_commit_sha: null,
    github_blob_sha: null,
    supabase_bucket: null,
    supabase_object_path: null,
  });

  const hiddenArtifactId = '00000000-0000-4000-8000-000000000099';
  const catalog = buildArtifactCatalog(load, [
    row(
      visibleRef.artifact_id,
      visibleRef.version,
      visibleRef.content_sha256!,
      visibleRef.required_authority_class!,
    ),
    row(hiddenArtifactId, 1, 'f'.repeat(64), 'CHECKPOINT_STAGE_OUTPUT'),
  ]);

  assert.deepEqual(Object.keys(catalog), [visibleRef.artifact_id]);
  assert.equal(catalog[hiddenArtifactId], undefined);
});


test('issuer discovery derives a stable canonical slug and resolves it without a mock allow-list', () => {
  const issuers = [
    {
      issuer_id: '10000000-0000-4000-8000-000000000001',
      display_name: 'Veolia',
      legal_name: 'Veolia Environnement S.A.',
    },
    {
      issuer_id: '10000000-0000-4000-8000-000000000002',
      display_name: 'ASML Holding',
      legal_name: 'ASML Holding N.V.',
    },
  ];
  const securities = [
    {
      security_id: '20000000-0000-4000-8000-000000000001',
      issuer_id: issuers[0].issuer_id,
      ticker: 'VIE',
      market_data_symbol: 'VIE.PA',
      primary_listing: true,
      listing_status: 'ACTIVE',
    },
    {
      security_id: '20000000-0000-4000-8000-000000000002',
      issuer_id: issuers[1].issuer_id,
      ticker: 'ASML',
      market_data_symbol: 'ASML.AS',
      primary_listing: true,
      listing_status: 'ACTIVE',
    },
  ];

  assert.equal(issuerSlug('ASML Holding'), 'asml-holding');
  assert.equal(resolveUiIssuer('asml-holding', issuers, securities)?.issuer_id, issuers[1].issuer_id);
  assert.equal(resolveUiIssuer('ASML', issuers, securities)?.issuer_id, issuers[1].issuer_id);
  assert.equal(resolveUiIssuer('VIE.PA', issuers, securities)?.issuer_id, issuers[0].issuer_id);

  const identity = buildCompanyIdentity(issuers[0], securities[0]);
  assert.equal(identity.slug, 'veolia');
  assert.equal(identity.ticker, 'VIE');
});

test('issuer discovery fails closed when two issuers would share one canonical slug', () => {
  assert.throws(
    () =>
      assertUniqueIssuerSlugs([
        {
          issuer_id: '10000000-0000-4000-8000-000000000001',
          display_name: 'ACME, Inc.',
          legal_name: 'ACME, Inc.',
        },
        {
          issuer_id: '10000000-0000-4000-8000-000000000002',
          display_name: 'ACME Inc',
          legal_name: 'ACME Inc',
        },
      ]),
    /canonical issuer slug is ambiguous/,
  );
});

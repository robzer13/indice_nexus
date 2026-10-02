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
import {
  resolveMockRunSelection,
  VEOLIA_MOCK_DOSSIER,
} from '../lib/orotitan-ui/mock';

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

test('mock run resolution never substitutes the primary run for another requested run', () => {
  const dossier = VEOLIA_MOCK_DOSSIER;
  assert.equal(resolveMockRunSelection(dossier, null).kind, 'select');
  assert.equal(resolveMockRunSelection(dossier, dossier.primaryRunId).kind, 'available');
  assert.equal(
    resolveMockRunSelection(dossier, 'd56a0bc9-4e80-48b3-b326-74d1fea15e63').kind,
    'unavailable',
  );
  assert.equal(resolveMockRunSelection(dossier, '00000000-0000-0000-0000-000000000000').kind, 'unknown');
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

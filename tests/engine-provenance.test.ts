import assert from 'node:assert/strict';
import test from 'node:test';
import {
  CURRENT_ENGINE,
  classifyEngineProvenance,
  classifyResearchFreshness,
} from '../lib/orotitan-equity/engine-provenance';

test('current engine requires exact contract fingerprint and versions', () => {
  const result = classifyEngineProvenance({
    processVersion: '2.0',
    pilotageContractVersion: '2.0',
    contractSetSha256: CURRENT_ENGINE.contractSetSha256,
  });
  assert.equal(result.status, 'CURRENT');
  assert.equal(result.generation, 'V2_CURRENT');
});

test('same process version with another fingerprint is previous, not current', () => {
  const result = classifyEngineProvenance({
    processVersion: '2.0',
    pilotageContractVersion: '2.0',
    contractSetSha256: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
  });
  assert.equal(result.status, 'PREVIOUS');
});

test('v1 process is legacy', () => {
  const result = classifyEngineProvenance({
    processVersion: '1.0',
    pilotageContractVersion: '1.0.1',
    contractSetSha256: '34b009f05715bab482dbc00b02194b3714e9b2f8151872144677bb6db19f3c63',
  });
  assert.equal(result.status, 'LEGACY');
});

test('research freshness is explicit and calendar based only', () => {
  assert.deepEqual(
    classifyResearchFreshness('2026-09-18', new Date('2026-10-04T12:00:00Z')),
    { status: 'RECENT', ageDays: 16 },
  );
  assert.equal(classifyResearchFreshness('2026-05-01', new Date('2026-10-04T12:00:00Z')).status, 'AGING');
  assert.equal(classifyResearchFreshness('2026-01-01', new Date('2026-10-04T12:00:00Z')).status, 'STALE');
});

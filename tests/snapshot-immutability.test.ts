import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { assertSnapshotDoesNotExist, DuplicateSnapshotError } from '../lib/domain/snapshot';

test('admin domain guard rejects an existing business key', () => {
  assert.throws(() => assertSnapshotDoesNotExist(true), DuplicateSnapshotError);
});

test('schema enforces business-key uniqueness and blocks snapshot mutation', async () => {
  const schema = await readFile('schema.sql', 'utf8');
  assert.match(schema, /unique \(company_id, analysis_date, model_version\)/i);
  assert.match(schema, /before update or delete on public\.snapshots/i);
});

test('bootstrap never updates an existing snapshot', async () => {
  const seed = await readFile('seed.sql', 'utf8');
  assert.match(seed, /on conflict \(company_id,analysis_date,model_version\) do nothing;/i);
  const snapshotSection = seed.split('insert into public.market_prices')[0];
  assert.doesNotMatch(snapshotSection, /on conflict \(company_id,analysis_date,model_version\) do update/i);
});

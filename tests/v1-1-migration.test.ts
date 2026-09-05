import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

test('V1.1 migration keeps company deletes forbidden and sync logs append-only', async () => {
  const migration = await readFile('migrations/v1_1.sql', 'utf8');
  assert.match(migration, /create table if not exists public\.market_sync_runs/i);
  assert.match(migration, /before update or delete on public\.market_sync_runs/i);
  assert.match(migration, /grant select, insert, update on public\.companies to service_role/i);
  assert.doesNotMatch(migration, /grant[^;]*delete[^;]*public\.companies/i);
});

test('fresh schema includes V1.1 operational structures', async () => {
  const schema = await readFile('schema.sql', 'utf8');
  assert.match(schema, /public\.market_sync_runs/i);
  assert.match(schema, /companies_set_updated_at/i);
});

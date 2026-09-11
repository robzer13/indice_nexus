import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const migration = readFileSync(
  new URL('../migrations/20260911_orotitan_equity_i3_snapshot_persistence.sql', import.meta.url),
  'utf8',
);

test('I3-A migration defines the canonical snapshot boundary', () => {
  assert.match(migration, /create table if not exists public\.research_snapshots/);
  assert.match(migration, /contract_version text not null default '04_SCREENER_SCHEMA_V1'/);
  assert.match(migration, /schema_version text not null default '1\.0\.0'/);
  for (const mode of ['DISCOVER', 'ANALYZE', 'REFRESH', 'ACTIVATION CHECK']) {
    assert.match(migration, new RegExp(`'${mode}'`));
  }
  assert.match(migration, /'DISCOVER \+ ANALYZE'/);
});

test('I3-A locks identity, pointer, JSON, immutability, and access boundaries', () => {
  assert.match(migration, /research_snapshots_dossier_issuer_fkey/);
  assert.match(migration, /research_snapshots_security_issuer_fkey/);
  assert.match(migration, /research_dossiers_current_snapshot_fkey/);
  assert.match(migration, /research_snapshots_snapshot_dossier_key/);
  assert.match(migration, /research_snapshots_payload_lock_check/);
  assert.match(migration, /canonical_payload \? 'data_lock'/);
  assert.match(migration, /canonical_payload \? 'versions'/);
  assert.match(migration, /snapshot_id:uuid:true/);
  assert.match(migration, /begin;[\s\S]*commit;/i);
  assert.match(migration, /prevent_orotitan_research_snapshot_mutation/);
  assert.match(migration, /before update or delete on public\.research_snapshots/);
  assert.match(migration, /alter table public\.research_snapshots enable row level security/);
  assert.match(migration, /grant select on public\.research_snapshots to service_role/);
  assert.match(migration, /revoke all on public\.research_snapshots from public, anon, authenticated, service_role/);
});

test('I3-A does not alter the legacy storage or expose a writer', () => {
  assert.doesNotMatch(migration, /alter table public\.(companies|snapshots|market_prices|market_sync_runs)/);
  assert.doesNotMatch(migration, /drop table public\.(companies|snapshots|market_prices|market_sync_runs)/);
  assert.doesNotMatch(migration, /create .*function .*rpc|create .*procedure/i);
  assert.doesNotMatch(migration, /insert into public\.(companies|snapshots|market_prices|market_sync_runs)/);
});

test('I3-A contains defensive shape assertions', () => {
  assert.match(migration, /pg_catalog\.pg_attribute/);
  assert.match(migration, /pg_catalog\.pg_constraint/);
  assert.match(migration, /pg_catalog\.pg_policy/);
  assert.match(migration, /pg_catalog\.pg_trigger/);
  assert.match(migration, /has_table_privilege/);
  assert.match(migration, /c\.conkey = array/);
  assert.match(migration, /c\.confkey = array/);
  assert.match(migration, /t\.tgfoid = 'public\.prevent_orotitan_research_snapshot_mutation\(\)'::regprocedure/);
  assert.match(migration, /research_snapshots_frozen_version_checks|research_snapshots_contract_version_check/);
});
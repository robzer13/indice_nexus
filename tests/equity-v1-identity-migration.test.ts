import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const migrationPath = 'migrations/20260909_orotitan_equity_i1_identity.sql';

test('I1 creates issuer, security, dossier and legacy crosswalk as separate objects', async () => {
  const sql = await readFile(migrationPath, 'utf8');

  assert.match(sql, /create table if not exists public\.issuers/i);
  assert.match(sql, /create table if not exists public\.securities/i);
  assert.match(sql, /create table if not exists public\.research_dossiers/i);
  assert.match(sql, /create table if not exists public\.legacy_company_identity_map/i);

  assert.match(sql, /issuer_id uuid not null references public\.issuers\(issuer_id\)/i);
  assert.match(sql, /security_id uuid not null unique references public\.securities\(security_id\)/i);
  assert.match(sql, /dossier_id uuid not null unique references public\.research_dossiers\(dossier_id\)/i);
});

test('I1 enforces one active dossier and at most one primary listing per issuer', async () => {
  const sql = await readFile(migrationPath, 'utf8');

  assert.match(
    sql,
    /create unique index if not exists one_active_research_dossier_per_issuer\s+on public\.research_dossiers \(issuer_id\)\s+where active is true;/i,
  );
  assert.match(
    sql,
    /create unique index if not exists one_primary_security_per_issuer\s+on public\.securities \(issuer_id\)\s+where primary_listing is true;/i,
  );
});

test('I1 is additive and does not mutate or reinterpret legacy analytical tables', async () => {
  const sql = await readFile(migrationPath, 'utf8');

  assert.doesNotMatch(sql, /alter table public\.(companies|snapshots|market_prices|market_sync_runs)\b/i);
  assert.doesNotMatch(sql, /update public\.(companies|snapshots|market_prices|market_sync_runs)\b/i);
  assert.doesNotMatch(sql, /delete from public\.(companies|snapshots|market_prices|market_sync_runs)\b/i);
  assert.doesNotMatch(sql, /drop table (if exists )?public\.(companies|snapshots|market_prices|market_sync_runs)\b/i);

  assert.doesNotMatch(sql, /business_quality_score\s*=|valuation_score\s*=|orotitan_score\s*=/i);
  assert.doesNotMatch(sql, /price_o(85|90|92|95)\s*=/i);
});

test('I1 backfill is fail-closed for identity facts the legacy row cannot prove', async () => {
  const sql = await readFile(migrationPath, 'utf8');

  assert.match(sql, /insert into public\.issuers[\s\S]*?select\s+issuer_id,\s+name,\s+null,\s+country,\s+null,\s+null/i);
  assert.match(sql, /insert into public\.securities[\s\S]*?'UNKNOWN',\s+exchange,\s+null,\s+currency,\s+null,\s+null/i);

  // Legacy trading currency must not silently become issuer reporting currency.
  const issuerInsert = sql.split('insert into public.issuers')[1]?.split('on conflict (issuer_id) do nothing;')[0] ?? '';
  assert.doesNotMatch(issuerInsert, /reporting_currency[\s\S]*?currency/i);
});

test('I1 keeps current_snapshot_id transitional until canonical snapshot persistence exists', async () => {
  const sql = await readFile(migrationPath, 'utf8');

  assert.match(sql, /current_snapshot_id uuid,/i);
  assert.doesNotMatch(sql, /current_snapshot_id uuid[^;]*references public\.research_snapshots/i);
  assert.match(sql, /I3 will add the foreign key from research_dossiers\.current_snapshot_id/i);
});

test('I1 applies RLS and browser-role revocation to every new identity object', async () => {
  const sql = await readFile(migrationPath, 'utf8');

  for (const table of ['issuers', 'securities', 'research_dossiers', 'legacy_company_identity_map']) {
    assert.match(sql, new RegExp(`alter table public\\.${table} enable row level security;`, 'i'));
    assert.match(sql, new RegExp(`revoke all on public\\.${table} from anon, authenticated;`, 'i'));
  }

  assert.doesNotMatch(sql, /grant\s+delete\s+on public\.(issuers|securities|research_dossiers|legacy_company_identity_map)/i);
});

test('I1 new updated_at helper has a fixed search_path', async () => {
  const sql = await readFile(migrationPath, 'utf8');

  assert.match(sql, /create or replace function public\.set_orotitan_identity_updated_at\(\)/i);
  assert.match(sql, /set search_path = pg_catalog, public/i);
});

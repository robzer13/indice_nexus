import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { resolveCanonicalQuoteCompanyId, type CurrentCanonicalQuoteIdentity, type LegacyCompanyQuoteLink } from '../lib/domain/canonical-quote-identity';

const munich: CurrentCanonicalQuoteIdentity = {
  issuer_id: '7ae72c78-92a7-495d-abaf-926096a926fb',
  security_id: '84b71262-fb37-4198-90f0-9d85f4442e44',
  dossier_id: 'munich-dossier',
};
const pernod: CurrentCanonicalQuoteIdentity = {
  issuer_id: '76060379-e9bf-4182-948c-f7875efb6683',
  security_id: '5497554e-6f38-4ce0-9800-1a9fc74c7973',
  dossier_id: 'pernod-dossier',
};
const links: LegacyCompanyQuoteLink[] = [
  { ...munich, legacy_company_id: munich.issuer_id },
  { ...pernod, legacy_company_id: pernod.issuer_id },
];
test('Munich Re and Pernod Ricard resolve through exact canonical issuer, security and dossier', () => {
  assert.equal(resolveCanonicalQuoteCompanyId(munich, links),munich.issuer_id);
  assert.equal(resolveCanonicalQuoteCompanyId(pernod, links),pernod.issuer_id);
});
test('does not infer a legacy company from identical issuer and company UUIDs alone', () => {
  assert.equal(resolveCanonicalQuoteCompanyId(munich, []),null);
  assert.equal(resolveCanonicalQuoteCompanyId({...munich,security_id:'another-listing'},links),null);
  assert.equal(resolveCanonicalQuoteCompanyId({...munich,dossier_id:'superseded-dossier'},links),null);
  assert.equal(resolveCanonicalQuoteCompanyId({...munich,issuer_id:pernod.issuer_id},links),null);
  assert.equal(resolveCanonicalQuoteCompanyId({...munich,issuer_id:''},links),null);
});
test('ambiguous duplicate matches fail closed, not pick first result', () => {
  assert.equal(resolveCanonicalQuoteCompanyId(munich,[links[0],{...links[0],legacy_company_id:'different-company'}]),null);
});
test('staged backfill uses exact identity and quotes only through the existing crosswalk', () => {
  const sql=readFileSync('migrations/20261011_canonical_quote_identity_backfill_v1.sql','utf8');
  assert.match(sql,/c\.id\s*=\s*d\.issuer_id/i);
  for (const part of [
    /c\.ticker\s*=\s*sec\.ticker/i,
    /c\.exchange\s*=\s*sec\.exchange/i,
    /c\.currency\s*=\s*sec\.trading_currency/i,
    /c\.quote_unit\s*=\s*sec\.quote_unit/i,
    /c\.market_data_symbol\s*=\s*sec\.market_data_symbol/i,
    /c\.market_data_multiplier\s*=\s*sec\.market_data_multiplier/i,
    /c\.price_decimals\s*=\s*sec\.price_decimals/i,
    /CANONICAL_QUOTE_COHORT_DRIFT/i,
    /CANONICAL_QUOTE_CROSSWALK_CONFLICT/i,
    /CANONICAL_QUOTE_POSTCONDITION_FAILED/i,
    /not exists \(/i,
  ])assert.match(sql,part);
  assert.match(sql,/where d\.active is true and d\.current_snapshot_id is not null/i);
  assert.match(sql,/c\.market_data_symbol is not null/i);
  assert.match(sql,/insert into public\.legacy_company_identity_map/i);
  assert.doesNotMatch(sql,/(?:insert into|update|delete from|truncate|alter table|drop table)\s+public\.(?:market_prices|companies|issuers|securities|research_snapshots|research_dossiers|paper_\w+)/i);
  assert.doesNotMatch(sql,/on conflict\s*\([^)]+\)\s*do update/i);
});
test('both paper portfolio and Screener resolve quotes via the audited crosswalk', () => {
  const market=readFileSync('lib/data/companies.ts','utf8');
  const paper=readFileSync('lib/data/paper-portfolio.ts','utf8');
  assert.match(market,/from\('legacy_company_identity_map'\)/);
  assert.match(market,/resolveCanonicalQuoteCompanyId\(snapshot, links\)/);
  assert.match(market,/\.in\('company_id', legacyIds\)/);
  assert.doesNotMatch(market,/\.in\('company_id', issuerIds\)/);
  assert.match(paper,/\.eq\('dossier_id', dossier\.dossier_id\)/);
  assert.match(paper,/resolveCanonicalQuoteCompanyId\(snapshot,/);
});

test('audited SQL cohort pins all 32 four-part identities and rejects same-count swaps', () => {
  const sql = readFileSync('migrations/20261011_canonical_quote_identity_backfill_v1.sql', 'utf8');
  const cohort = sql.split('create temporary table _canonical_quote_audited_cohort on commit drop as')[1]?.split('do $$')[0];
  assert.ok(cohort, 'Expected explicit, immutable audited SQL cohort');
  const rows = cohort.split('\n').filter(line => /^\s*\('[0-9a-f-]{36}'::uuid/.test(line));
  const actual = rows.map(line => {
    const ids = line.match(/'[0-9a-f-]{36}'::uuid/g) ?? [];
    assert.equal(ids.length, 4, 'Expected all four UUIDs for every pinned row');
    return ids.map(id => id.slice(1, 37));
  });
  assert.equal(actual.length, 32);
  const key = (parts: string[]) => parts.join('|');
  const locked = new Set(actual.map(key));
  assert.equal(locked.size, 32);
  assert.ok(actual.every(([legacy, issuer]) => legacy === issuer));
  assert.ok(actual.some(([legacy, , security]) =>
    legacy === '7ae72c78-92a7-495d-abaf-926096a926fb' &&
    security === '84b71262-fb37-4198-90f0-9d85f4442e44'), 'Munich Re must be pinned');
  assert.ok(actual.some(([legacy, , security]) =>
    legacy === '76060379-e9bf-4182-948c-f7875efb6683' &&
    security === '5497554e-6f38-4ce0-9800-9a9fc74c7973'), 'Pernod Ricard must be pinned');
  assert.ok(!actual.some(([, , security]) => security === 'cc46c84b-a073-448a-8109-fea143605ac0'), 'Wise remains excluded');
  assert.ok(!actual.some(([, , security]) => security === 'c459062e-baa6-43ec-b75e-86916b805e90'), 'ISRG remains excluded');

  // Regression: a new security can replace an old one without changing COUNT=32.
  const substitute = actual.map(parts => [...parts]);
  substitute[0][2] = '00000000-0000-4000-8000-000000000001';
  assert.equal(substitute.length, actual.length);
  const substituteSet = new Set(substitute.map(key));
  assert.ok(substitute.some(parts => !locked.has(key(parts))));
  assert.ok(actual.some(parts => !substituteSet.has(key(parts))));

  // The SQL must detect BOTH directions of the mismatch before any insert.
  assert.match(sql, /full outer join _canonical_quote_audited_cohort audited/i);
  for (const part of [
    /audited\.legacy_company_id\s*=\s*x\.legacy_company_id/i,
    /audited\.issuer_id\s*=\s*x\.issuer_id/i,
    /audited\.security_id\s*=\s*x\.security_id/i,
    /audited\.dossier_id\s*=\s*x\.dossier_id/i,
    /where x\.legacy_company_id is null or audited\.legacy_company_id is null/i,
    /CANONICAL_QUOTE_AUDITED_IDENTITY_DRIFT/i,
  ]) assert.match(sql, part);
  assert.ok(sql.indexOf('CANONICAL_QUOTE_AUDITED_IDENTITY_DRIFT') <
    sql.indexOf('insert into public.legacy_company_identity_map'), 'Fail closed before insertion');
});

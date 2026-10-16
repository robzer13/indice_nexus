import assert from 'node:assert/strict';
import test from 'node:test';
import { summarizeMarketSyncHealth } from '../lib/domain/market-sync-health';
import type { MarketSyncRun } from '../lib/domain/types';

function run(
  id: number,
  source: 'CRON' | 'ADMIN',
  inserted: number,
  failures: { error: string; slug: string }[],
): MarketSyncRun {
  const results = [
    ...Array.from({ length: inserted }, (_, index) => ({
      slug: 'ok-' + index, symbol: 'OK', status: 'inserted',
    })),
    ...failures.map(item => ({ slug: item.slug, symbol: 'FAIL', status: 'failed', error: item.error })),
  ];
  return {
    id,
    started_at: '2026-10-15T22:33:50Z',
    finished_at: '2026-10-15T22:33:59Z',
    trigger_source: source,
    companies: results.length,
    inserted,
    failed: failures.length,
    results,
    created_at: '2026-10-15T22:34:00Z',
  };
}

test('last CRON and ADMIN runs stay independent; 23 identical errors are aggregated', () => {
  const failures = Array.from({ length: 23 }, (_, index) => ({
    slug: 'failure-' + index,
    error: 'Missing required environment variable: TWELVE_DATA_API_KEY',
  }));
  const cron = run(25, 'CRON', 16, failures);
  const admin = run(23, 'ADMIN', 38, [{
    slug: 'baltic-classifieds',
    error: 'NO_ADMISSIBLE_MARKET_QUOTE | YAHOO_FINANCE: QUOTE_CURRENCY_UNIT_MISMATCH',
  }]);
  const olderCron = run(20, 'CRON', 39, []);

  const health = summarizeMarketSyncHealth([olderCron, admin, cron]);
  assert.equal(health.cron.run?.id, 25);
  assert.equal(health.cron.status, 'PARTIAL');
  assert.equal(health.cron.successRatePct, 41);
  assert.equal(health.cron.durationSeconds, 9);
  assert.equal(health.cron.errors.length, 1);
  assert.equal(health.cron.errors[0].count, 23);
  assert.equal(health.cron.errors[0].companies.length, 23);
  assert.equal(health.admin.run?.id, 23);
  assert.equal(health.admin.status, 'PARTIAL');
  assert.equal(health.admin.successRatePct, 97.4);
  assert.equal(health.admin.errors[0].count, 1);
  assert.deepEqual(health.admin.errors[0].companies, ['baltic-classifieds']);
});

test('missing CRON is UNKNOWN, never inferred healthy from a successful ADMIN', () => {
  const { cron, admin } = summarizeMarketSyncHealth([run(9, 'ADMIN', 4, [])]);
  assert.equal(cron.run, null);
  assert.equal(cron.status, 'NO_RUN');
  assert.equal(cron.successRatePct, null);
  assert.deepEqual(cron.errors, []);
  assert.equal(admin.status, 'SUCCESS');
  assert.equal(admin.successRatePct, 100);
});

test('all failed is zero-percent success, distinguishable from no run', () => {
  const failed = run(10, 'CRON', 0, [{ slug: 'a', error: 'QUOTA' }, { slug: 'b', error: 'QUOTA' }]);
  const summary = summarizeMarketSyncHealth([failed]).cron;
  assert.equal(summary.status, 'FAILED');
  assert.equal(summary.successRatePct, 0);
  assert.equal(summary.errors[0].count, 2);
});

test('zero companies, inconsistent counters, missing detail, and invalid timestamps fail closed', () => {
  const empty = run(1, 'CRON', 0, []);
  assert.equal(summarizeMarketSyncHealth([empty]).cron.status, 'INVALID_RUN');
  assert.equal(summarizeMarketSyncHealth([empty]).cron.successRatePct, null);

  const good = run(2, 'CRON', 1, []);
  const invalidRows = [
    { ...good, failed: 2 },
    { ...good, companies: -1 },
    { ...good, inserted: Number.NaN },
    { ...good, results: null },
    { ...good, started_at: 'invalid' },
    { ...good, results: [{ status: 'unrecognized' }] },
  ] as MarketSyncRun[];
  for (const invalid of invalidRows) {
    const item = summarizeMarketSyncHealth([invalid]).cron;
    assert.equal(item.status, 'INVALID_RUN');
    assert.equal(item.successRatePct, null);
  }
});

test('groups different failures separately without changing source results', () => {
  const latest = run(3, 'CRON', 1, [
    { slug: 'z-company', error: 'QUOTE_CURRENCY_UNIT_MISMATCH' },
    { slug: 'a-company', error: 'QUOTE_CURRENCY_UNIT_MISMATCH' },
    { slug: 'other', error: 'MISSING_MARKET_DATA_SYMBOL' },
  ]);
  const before = JSON.stringify(latest);
  const health = summarizeMarketSyncHealth([latest]).cron;
  assert.deepEqual(health.errors.map(group => [group.message, group.count]), [
    ['QUOTE_CURRENCY_UNIT_MISMATCH', 2],
    ['MISSING_MARKET_DATA_SYMBOL', 1],
  ]);
  assert.deepEqual(health.errors[0].companies, ['a-company', 'z-company']);
  assert.equal(JSON.stringify(latest), before);
});

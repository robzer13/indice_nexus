import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import test from 'node:test';

interface PersistenceState {
  writes: Array<{ table: string; payload: unknown }>;
  errors: Record<string, { message: string } | null>;
}

const persistenceState = globalThis as typeof globalThis & { __marketPersistenceState: PersistenceState };
persistenceState.__marketPersistenceState = { writes: [], errors: {} };

const mockSources = new Map<string, string>([
  ['mock:server-only', 'export {};'],
  ['mock:supabase-server', `
    export function createServerSupabaseClient() {
      return {
        from(table) {
          return {
            async insert(payload) {
              globalThis.__marketPersistenceState.writes.push({ table, payload });
              return { error: globalThis.__marketPersistenceState.errors[table] ?? null };
            }
          };
        }
      };
    }
  `],
]);

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === 'server-only') return { url: 'mock:server-only', shortCircuit: true };
    if (specifier === '@/lib/supabase/server')
      return { url: 'mock:supabase-server', shortCircuit: true };
    return nextResolve(specifier, context);
  },
  load(url, context, nextLoad) {
    const source = mockSources.get(url);
    if (source !== undefined) return { format: 'module', source, shortCircuit: true };
    return nextLoad(url, context);
  },
});

const persistenceModule = import('../lib/data/market-prices');

test.beforeEach(() => {
  persistenceState.__marketPersistenceState.writes = [];
  persistenceState.__marketPersistenceState.errors = {};
});

test('insertMarketPrice writes the contracted market_prices payload once', async () => {
  const { insertMarketPrice } = await persistenceModule;
  await insertMarketPrice({
    companyId: 'company-1',
    price: 123.45,
    asOf: '2026-10-16T11:00:00.000Z',
    source: 'YAHOO_FINANCE',
    raw: { symbol: 'META', currency: 'USD' },
  });

  assert.deepEqual(persistenceState.__marketPersistenceState.writes, [{
    table: 'market_prices',
    payload: {
      company_id: 'company-1',
      price: 123.45,
      as_of: '2026-10-16T11:00:00.000Z',
      source: 'YAHOO_FINANCE',
      raw: { symbol: 'META', currency: 'USD' },
    },
  }]);
});

test('recordMarketSyncRun writes the contracted market_sync_runs payload once', async () => {
  const { recordMarketSyncRun } = await persistenceModule;
  const details = [{ slug: 'issuer-1', symbol: 'META', status: 'inserted' }];
  await recordMarketSyncRun({
    startedAt: '2026-10-16T12:00:00.000Z',
    finishedAt: '2026-10-16T12:00:01.000Z',
    triggerSource: 'ADMIN',
    companies: 1,
    inserted: 1,
    failed: 0,
    results: details,
  });

  assert.deepEqual(persistenceState.__marketPersistenceState.writes, [{
    table: 'market_sync_runs',
    payload: {
      started_at: '2026-10-16T12:00:00.000Z',
      finished_at: '2026-10-16T12:00:01.000Z',
      trigger_source: 'ADMIN',
      companies: 1,
      inserted: 1,
      failed: 0,
      results: details,
    },
  }]);
});

test('Supabase insert errors propagate and do not cause additional writes', async () => {
  const { insertMarketPrice, recordMarketSyncRun } = await persistenceModule;
  persistenceState.__marketPersistenceState.errors.market_prices = { message: 'price denied' };
  await assert.rejects(
    insertMarketPrice({
      companyId: 'company-1',
      price: 123.45,
      asOf: '2026-10-16T11:00:00.000Z',
      source: 'YAHOO_FINANCE',
      raw: {},
    }),
    /Unable to insert market price: price denied/,
  );
  assert.equal(persistenceState.__marketPersistenceState.writes.length, 1);

  persistenceState.__marketPersistenceState.writes = [];
  persistenceState.__marketPersistenceState.errors.market_sync_runs = { message: 'audit denied' };
  await assert.rejects(
    recordMarketSyncRun({
      startedAt: '2026-10-16T12:00:00.000Z',
      finishedAt: '2026-10-16T12:00:01.000Z',
      triggerSource: 'CRON',
      companies: 1,
      inserted: 0,
      failed: 1,
      results: [],
    }),
    /Unable to record market sync run: audit denied/,
  );
  assert.deepEqual(
    persistenceState.__marketPersistenceState.writes.map((write) => write.table),
    ['market_sync_runs'],
  );
});

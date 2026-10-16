import assert from 'node:assert/strict';
import { register } from 'node:module';
import test from 'node:test';
import type { MarketDataCompany } from '../lib/data/market-prices';

interface PriceResult {
  provider: 'YAHOO_FINANCE' | 'TWELVE_DATA';
  normalizedPrice: number;
  fetchedAt: string;
  raw: unknown;
}

interface RefreshMocks {
  companies: MarketDataCompany[];
  fetch: (reference: string) => Promise<PriceResult>;
  insert: (input: unknown) => Promise<void>;
  record: (input: unknown) => Promise<void>;
}

const refreshState = globalThis as typeof globalThis & { __marketRefreshMocks: RefreshMocks };
const insertedInputs: unknown[] = [];
const recordedInputs: unknown[] = [];
const quoteTimestamp = '2026-10-16T11:00:00.000Z';

function company(index: number): MarketDataCompany {
  return {
    id: `company-${index}`,
    slug: `issuer-${index}`,
    name: `Issuer ${index}`,
    market_data_symbol: `SYM${index}`,
    market_data_multiplier: 1,
    ticker: `SYM${index}`,
    exchange: 'Nasdaq',
    currency: 'USD',
    quote_unit: 'MAJOR',
    latest_price: 100,
  };
}

refreshState.__marketRefreshMocks = {
  companies: [],
  fetch: async () => ({
    provider: 'YAHOO_FINANCE',
    normalizedPrice: 101,
    fetchedAt: quoteTimestamp,
    raw: { source: 'test' },
  }),
  insert: async (input) => { insertedInputs.push(input); },
  record: async (input) => { recordedInputs.push(input); },
};

const mockSources = new Map<string, string>([
  ['mock:server-only', 'export {};'],
  ['mock:market-data', `
    export function getMarketDataCompanies() {
      return Promise.resolve(globalThis.__marketRefreshMocks.companies);
    }
    export function insertMarketPrice(input) {
      return globalThis.__marketRefreshMocks.insert(input);
    }
    export function recordMarketSyncRun(input) {
      return globalThis.__marketRefreshMocks.record(input);
    }
  `],
  ['mock:market-provider', `
    export function fetchProviderPrice(instrument) {
      return globalThis.__marketRefreshMocks.fetch(instrument.reference);
    }
  `],
]);

register('./module-mock-hooks.mjs', import.meta.url, {
  data: {
    resolutions: [
      ['server-only', 'mock:server-only'],
      ['@/lib/data/market-prices', 'mock:market-data'],
      ['@/lib/market/provider', 'mock:market-provider'],
    ],
    sources: [...mockSources],
    modulePrefixes: [new URL('../lib/', import.meta.url).href],
  },
});

const refreshModule = import('../lib/market/refresh-prices');

test.beforeEach((context) => {
  if (!('mock' in context)) throw new Error('TEST_MOCK_CONTEXT_UNAVAILABLE');
  context.mock.timers.enable({ apis: ['Date'], now: new Date('2026-10-16T12:00:00.000Z') });
  insertedInputs.length = 0;
  recordedInputs.length = 0;
  refreshState.__marketRefreshMocks.companies = [];
  refreshState.__marketRefreshMocks.fetch = async (reference) => ({
    provider: 'YAHOO_FINANCE',
    normalizedPrice: Number(reference.replace('SYM', '')) + 100,
    fetchedAt: quoteTimestamp,
    raw: { reference },
  });
  refreshState.__marketRefreshMocks.insert = async (input) => { insertedInputs.push(input); };
  refreshState.__marketRefreshMocks.record = async (input) => { recordedInputs.push(input); };
});

test('refreshMarketPrices records a complete CRON run with coherent totals and details', async () => {
  const { refreshMarketPrices } = await refreshModule;
  refreshState.__marketRefreshMocks.companies = [company(1), company(2)];

  const result = await refreshMarketPrices('CRON');

  assert.deepEqual(
    { ok: result.ok, trigger: result.triggerSource, companies: result.companies, inserted: result.inserted, failed: result.failed },
    { ok: true, trigger: 'CRON', companies: 2, inserted: 2, failed: 0 },
  );
  assert.deepEqual(result.results, [
    { slug: 'issuer-1', symbol: 'SYM1', status: 'inserted', provider: 'YAHOO_FINANCE', price: 101 },
    { slug: 'issuer-2', symbol: 'SYM2', status: 'inserted', provider: 'YAHOO_FINANCE', price: 102 },
  ]);
  assert.equal(result.logRecorded, true);
  assert.equal(insertedInputs.length, 2);
  assert.equal(recordedInputs.length, 1);
  assert.deepEqual(recordedInputs[0], {
    startedAt: '2026-10-16T12:00:00.000Z',
    finishedAt: '2026-10-16T12:00:00.000Z',
    triggerSource: 'CRON',
    companies: 2,
    inserted: 2,
    failed: 0,
    results: result.results,
  });
});

test('a provider failure remains associated with its issuer and does not stop other issuers', async () => {
  const { refreshMarketPrices } = await refreshModule;
  refreshState.__marketRefreshMocks.companies = [company(1), company(2), company(3)];
  let releaseFirst!: () => void;
  const firstMayFinish = new Promise<void>((resolve) => { releaseFirst = resolve; });
  refreshState.__marketRefreshMocks.fetch = async (reference) => {
    if (reference === 'SYM1') await firstMayFinish;
    if (reference === 'SYM2') {
      releaseFirst();
      throw new Error('PROVIDER_UNAVAILABLE');
    }
    return {
      provider: 'TWELVE_DATA',
      normalizedPrice: Number(reference.replace('SYM', '')) + 100,
      fetchedAt: quoteTimestamp,
      raw: { reference },
    };
  };

  const result = await refreshMarketPrices('CRON');

  assert.deepEqual(
    { companies: result.companies, inserted: result.inserted, failed: result.failed, ok: result.ok },
    { companies: 3, inserted: 2, failed: 1, ok: false },
  );
  assert.deepEqual(result.results, [
    { slug: 'issuer-1', symbol: 'SYM1', status: 'inserted', provider: 'TWELVE_DATA', price: 101 },
    { slug: 'issuer-2', symbol: 'SYM2', status: 'failed', error: 'PROVIDER_UNAVAILABLE' },
    { slug: 'issuer-3', symbol: 'SYM3', status: 'inserted', provider: 'TWELVE_DATA', price: 103 },
  ]);
  assert.equal(insertedInputs.length, 2);
  assert.equal(recordedInputs.length, 1);
});

test('a market_prices insertion failure is isolated and counted as a company failure', async () => {
  const { refreshMarketPrices } = await refreshModule;
  refreshState.__marketRefreshMocks.companies = [company(1), company(2)];
  refreshState.__marketRefreshMocks.insert = async (input) => {
    insertedInputs.push(input);
    if ((input as { companyId: string }).companyId === 'company-1')
      throw new Error('Unable to insert market price: write rejected');
  };

  const result = await refreshMarketPrices('CRON');

  assert.equal(result.inserted, 1);
  assert.equal(result.failed, 1);
  assert.deepEqual(result.results[0], {
    slug: 'issuer-1',
    symbol: 'SYM1',
    status: 'failed',
    error: 'Unable to insert market price: write rejected',
  });
  assert.equal(result.results[1].status, 'inserted');
  assert.equal(insertedInputs.length, 2);
  assert.equal(recordedInputs.length, 1);
});

test('a market_sync_runs failure is exposed without changing ADMIN run totals', async () => {
  const { refreshMarketPrices } = await refreshModule;
  refreshState.__marketRefreshMocks.companies = [company(1)];
  refreshState.__marketRefreshMocks.record = async (input) => {
    recordedInputs.push(input);
    throw new Error('Unable to record market sync run: audit rejected');
  };

  const result = await refreshMarketPrices('ADMIN');

  assert.deepEqual(
    { trigger: result.triggerSource, companies: result.companies, inserted: result.inserted, failed: result.failed },
    { trigger: 'ADMIN', companies: 1, inserted: 1, failed: 0 },
  );
  assert.equal(result.ok, true);
  assert.equal(result.logRecorded, false);
  assert.equal(result.logError, 'Unable to record market sync run: audit rejected');
  assert.equal(recordedInputs.length, 1);
});

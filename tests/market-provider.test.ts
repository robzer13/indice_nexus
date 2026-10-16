import assert from 'node:assert/strict';
import { register } from 'node:module';
import test from 'node:test';

interface ProviderQuote {
  providerPrice: number;
  fetchedAt: string;
  providerSymbol: string;
  providerCurrency: string | null;
  providerExchange: string | null;
  raw: unknown;
}

interface ProviderMocks {
  yahoo: (reference: string) => Promise<ProviderQuote>;
  twelveData: (reference: string) => Promise<ProviderQuote>;
}

const providerState = globalThis as typeof globalThis & { __marketProviderMocks: ProviderMocks };
const fixedTimestamp = '2026-10-16T11:00:00.000Z';
const validYahooQuote: ProviderQuote = {
  providerPrice: 105,
  fetchedAt: fixedTimestamp,
  providerSymbol: 'META',
  providerCurrency: 'USD',
  providerExchange: 'NMS',
  raw: { source: 'yahoo-test' },
};
const validTwelveDataQuote: ProviderQuote = {
  providerPrice: 106,
  fetchedAt: fixedTimestamp,
  providerSymbol: 'META',
  providerCurrency: 'USD',
  providerExchange: 'NASDAQ',
  raw: { source: 'twelve-data-test' },
};

providerState.__marketProviderMocks = {
  yahoo: async () => validYahooQuote,
  twelveData: async () => validTwelveDataQuote,
};

const mockSources = new Map<string, string>([
  ['mock:server-only', 'export {};'],
  ['mock:market-yahoo-finance', `
    export function fetchYahooFinancePrice(reference) {
      return globalThis.__marketProviderMocks.yahoo(reference);
    }
  `],
  ['mock:market-twelve-data', `
    export function fetchTwelveDataPrice(reference) {
      return globalThis.__marketProviderMocks.twelveData(reference);
    }
  `],
]);

register('./module-mock-hooks.mjs', import.meta.url, {
  data: {
    resolutions: [
      ['server-only', 'mock:server-only'],
      ['@/lib/market/yahoo-finance', 'mock:market-yahoo-finance'],
      ['@/lib/market/twelve-data', 'mock:market-twelve-data'],
    ],
    sources: [...mockSources],
    modulePrefixes: [new URL('../lib/', import.meta.url).href],
  },
});

const providerModule = import('../lib/market/provider');
const instrument = {
  reference: 'META',
  ticker: 'META',
  exchange: 'Nasdaq',
  currency: 'USD',
  quoteUnit: 'MAJOR' as const,
  multiplier: 1,
  previousPrice: 100,
};
const originalTwelveDataKey = process.env.TWELVE_DATA_API_KEY;

test.after(() => {
  if (originalTwelveDataKey === undefined) delete process.env.TWELVE_DATA_API_KEY;
  else process.env.TWELVE_DATA_API_KEY = originalTwelveDataKey;
});

test.beforeEach((context) => {
  if (!('mock' in context)) throw new Error('TEST_MOCK_CONTEXT_UNAVAILABLE');
  context.mock.timers.enable({ apis: ['Date'], now: new Date('2026-10-16T12:00:00.000Z') });
  delete process.env.TWELVE_DATA_API_KEY;
  providerState.__marketProviderMocks.yahoo = async () => validYahooQuote;
  providerState.__marketProviderMocks.twelveData = async () => validTwelveDataQuote;
});

test('fetchProviderPrice retains an admissible Yahoo Finance quote', async () => {
  const { fetchProviderPrice } = await providerModule;
  let fallbackCalls = 0;
  providerState.__marketProviderMocks.twelveData = async () => {
    fallbackCalls += 1;
    return validTwelveDataQuote;
  };

  const result = await fetchProviderPrice(instrument);

  assert.equal(result.provider, 'YAHOO_FINANCE');
  assert.equal(result.normalizedPrice, 105);
  assert.equal(result.fetchedAt, fixedTimestamp);
  assert.equal(fallbackCalls, 0);
});

test('fetchProviderPrice reports the optional-key contract when Yahoo fails and no key exists', async () => {
  const { fetchProviderPrice } = await providerModule;
  providerState.__marketProviderMocks.yahoo = async () => {
    throw new Error('YAHOO_UNAVAILABLE');
  };

  const error = await fetchProviderPrice(instrument).then(
    () => null,
    (caught: unknown) => caught,
  );

  assert.ok(error instanceof Error);
  assert.match(error.message, /YAHOO_FINANCE: YAHOO_UNAVAILABLE/);
  assert.match(error.message, /TWELVE_DATA: OPTIONAL_FREE_KEY_NOT_CONFIGURED/);
  assert.doesNotMatch(error.message, /TWELVE_DATA_NOT_CONFIGURED/);
});

test('fetchProviderPrice uses an admissible Twelve Data fallback after Yahoo failure', async () => {
  const { fetchProviderPrice } = await providerModule;
  process.env.TWELVE_DATA_API_KEY = 'test-only-key';
  providerState.__marketProviderMocks.yahoo = async () => {
    throw new Error('YAHOO_UNAVAILABLE');
  };

  const result = await fetchProviderPrice(instrument);

  assert.equal(result.provider, 'TWELVE_DATA');
  assert.equal(result.normalizedPrice, 106);
  assert.deepEqual(result.raw, { source: 'twelve-data-test' });
});

test('fetchProviderPrice keeps quote-quality rejections blocking', async () => {
  const { fetchProviderPrice } = await providerModule;
  providerState.__marketProviderMocks.yahoo = async () => ({
    ...validYahooQuote,
    providerCurrency: 'EUR',
  });

  await assert.rejects(
    fetchProviderPrice(instrument),
    /YAHOO_FINANCE: QUOTE_CURRENCY_UNIT_MISMATCH.*OPTIONAL_FREE_KEY_NOT_CONFIGURED/,
  );
});

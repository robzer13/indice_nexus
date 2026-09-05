import assert from 'node:assert/strict';
import test from 'node:test';
import { companyInputSchema } from '../lib/domain/company';

test('company schema preserves Auto Trader minor-unit configuration', () => {
  const parsed = companyInputSchema.parse({
    slug: 'auto-trader',
    ticker: 'AUTO',
    name: 'Auto Trader Group plc',
    exchange: 'London Stock Exchange',
    currency: 'gbp',
    quote_unit: 'MINOR',
    price_decimals: 0,
    market_data_symbol: 'AUTO:LSE',
    market_data_multiplier: 100,
    country: 'United Kingdom',
    sector: 'Digital Classifieds',
    active: true,
  });
  assert.equal(parsed.currency, 'GBP');
  assert.equal(parsed.quote_unit, 'MINOR');
  assert.equal(parsed.price_decimals, 0);
  assert.equal(parsed.market_data_multiplier, 100);
});

test('company schema rejects unsafe slug and non-positive market multiplier', () => {
  const base = {
    ticker: 'ABC',
    name: 'Example Company',
    exchange: 'Xetra',
    currency: 'EUR',
    quote_unit: 'MAJOR',
    price_decimals: 2,
    market_data_symbol: null,
    country: null,
    sector: null,
    active: true,
  } as const;
  assert.equal(companyInputSchema.safeParse({ ...base, slug: 'Bad Slug', market_data_multiplier: 1 }).success, false);
  assert.equal(companyInputSchema.safeParse({ ...base, slug: 'good-slug', market_data_multiplier: 0 }).success, false);
});

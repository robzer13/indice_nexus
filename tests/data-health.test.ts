import assert from 'node:assert/strict';
import test from 'node:test';
import { getCompanyDataHealth } from '../lib/domain/data-health';
import type { CompanyState } from '../lib/domain/types';

function company(overrides: Partial<CompanyState> = {}): CompanyState {
  return {
    id: '1', slug: 'test', ticker: 'TST', name: 'Test', exchange: 'X', currency: 'EUR', quote_unit: 'MAJOR', price_decimals: 2,
    country: 'France', sector: 'Software', market_data_symbol: 'TST', market_data_multiplier: 1,
    price: 100, price_as_of: '2026-09-04T12:00:00Z', price_source: 'TEST',
    analysis_date: '2026-09-01', model_version: 'V1', status: 'FINALIST', quality_orotitan: true,
    business_quality_score: 90, investment_score: 85, valuation_score: 80, orotitan_score: 85, confidence_score: 8,
    fair_value_low: 90, fair_value_base: 120, fair_value_high: 150,
    price_o85: 110, price_o90: 95, price_o92: 90, price_o95: 80,
    thesis: 'x', main_risk: 'x', invalidation: 'x', source_title: 'source', notes: null, score_components: {},
    ...overrides,
  };
}

test('data health is clean for a complete and fresh company', () => {
  assert.deepEqual(getCompanyDataHealth(company(), new Date('2026-09-05T12:00:00Z')), []);
});

test('data health detects missing O90, market symbol and stale price', () => {
  const issues = getCompanyDataHealth(company({
    price_o90: null,
    market_data_symbol: null,
    price_as_of: '2026-08-20T12:00:00Z',
  }), new Date('2026-09-05T12:00:00Z'));
  const codes = issues.map((issue) => issue.code);
  assert.ok(codes.includes('MISSING_O90'));
  assert.ok(codes.includes('MISSING_MARKET_SYMBOL'));
  assert.ok(codes.includes('STALE_PRICE'));
});

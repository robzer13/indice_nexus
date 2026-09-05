import assert from 'node:assert/strict';
import test from 'node:test';
import { prioritizeCompanies } from '../lib/domain/prioritization';
import type { CompanyState } from '../lib/domain/types';

function company(slug: string, price: number | null, o90: number | null): CompanyState {
  return {
    id: slug, slug, ticker: slug.toUpperCase(), name: slug, exchange: 'X', currency: 'EUR', quote_unit: 'MAJOR', price_decimals: 2,
    country: null, sector: null, market_data_symbol: null, market_data_multiplier: 1, price, price_as_of: null, price_source: null,
    analysis_date: null, model_version: null, status: null, quality_orotitan: null, business_quality_score: null, investment_score: null,
    valuation_score: null, orotitan_score: null, confidence_score: null, fair_value_low: null, fair_value_base: null, fair_value_high: null,
    price_o85: null, price_o90: o90, price_o92: null, price_o95: null, thesis: null, main_risk: null, invalidation: null, source_title: null,
    notes: null, score_components: {},
  };
}

test('prioritization excludes non-calibrated companies and puts reached O90 first', () => {
  const result = prioritizeCompanies([
    company('near', 100, 95),
    company('reached', 90, 100),
    company('far', 100, 60),
    company('uncalibrated', 100, null),
  ]);
  assert.deepEqual(result.map((row) => row.slug), ['reached', 'near', 'far']);
});

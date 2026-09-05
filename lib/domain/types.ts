export type CompanyStatus =
  | 'OROTITAN'
  | 'FINALIST'
  | 'PRICE_WAIT'
  | 'TIER_1'
  | 'WATCHLIST'
  | 'REJECTED';

export type QuoteUnit = 'MAJOR' | 'MINOR';

export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[];

export interface CompanyRecord {
  id: string;
  slug: string;
  ticker: string;
  name: string;
  exchange: string;
  currency: string;
  quote_unit: QuoteUnit;
  price_decimals: number;
  market_data_symbol: string | null;
  market_data_multiplier: number;
  country: string | null;
  sector: string | null;
  active: boolean;
  created_at: string;
  updated_at: string;
}

export interface CompanyState {
  id: string;
  slug: string;
  ticker: string;
  name: string;
  exchange: string;
  currency: string;
  quote_unit: QuoteUnit;
  price_decimals: number;
  country: string | null;
  sector: string | null;
  market_data_symbol: string | null;
  market_data_multiplier: number;
  price: number | null;
  price_as_of: string | null;
  price_source: string | null;
  analysis_date: string | null;
  model_version: string | null;
  status: CompanyStatus | null;
  quality_orotitan: boolean | null;
  business_quality_score: number | null;
  investment_score: number | null;
  valuation_score: number | null;
  orotitan_score: number | null;
  confidence_score: number | null;
  fair_value_low: number | null;
  fair_value_base: number | null;
  fair_value_high: number | null;
  price_o85: number | null;
  price_o90: number | null;
  price_o92: number | null;
  price_o95: number | null;
  thesis: string | null;
  main_risk: string | null;
  invalidation: string | null;
  source_title: string | null;
  notes: string | null;
  score_components: Json;
}

export interface SnapshotHistoryRow {
  id: number;
  company_id: string;
  analysis_date: string;
  model_version: string;
  status: CompanyStatus;
  quality_orotitan: boolean | null;
  business_quality_score: number | null;
  investment_score: number | null;
  valuation_score: number | null;
  orotitan_score: number | null;
  confidence_score: number | null;
  fair_value_low: number | null;
  fair_value_base: number | null;
  fair_value_high: number | null;
  price_o85: number | null;
  price_o90: number | null;
  price_o92: number | null;
  price_o95: number | null;
  thesis: string | null;
  main_risk: string | null;
  invalidation: string | null;
  source_title: string | null;
  notes: string | null;
  score_components: Json;
  created_at: string;
}

export interface MarketPriceRow {
  id: number;
  company_id: string;
  price: number;
  as_of: string;
  source: string;
  raw: Json | null;
  created_at: string;
}

export interface MarketSyncRun {
  id: number;
  started_at: string;
  finished_at: string;
  trigger_source: 'CRON' | 'ADMIN';
  companies: number;
  inserted: number;
  failed: number;
  results: Json;
  created_at: string;
}

export interface ActiveCompanyOption {
  id: string;
  slug: string;
  ticker: string;
  name: string;
  exchange: string;
}

\set ON_ERROR_STOP on

do $$
begin
  if not exists (select 1 from pg_roles where rolname = 'anon') then create role anon nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'authenticated') then create role authenticated nologin; end if;
  if not exists (select 1 from pg_roles where rolname = 'service_role') then create role service_role nologin bypassrls; end if;
end;
$$;

create extension if not exists pgcrypto;

create table public.companies (
  id uuid primary key,
  slug text not null unique,
  ticker text not null,
  name text not null,
  exchange text not null,
  currency text not null,
  quote_unit text not null check (quote_unit in ('MAJOR', 'MINOR')),
  price_decimals integer not null,
  market_data_symbol text,
  market_data_multiplier numeric not null,
  country text
);

create table public.snapshots (
  id bigint generated always as identity primary key,
  company_id uuid not null references public.companies(id) on delete restrict,
  business_quality_score numeric,
  investment_score numeric,
  valuation_score numeric,
  orotitan_score numeric,
  quality_orotitan boolean,
  confidence_score numeric,
  price_o85 numeric,
  price_o90 numeric,
  price_o92 numeric,
  price_o95 numeric,
  score_components jsonb not null
);

create table public.market_prices (
  id bigint generated always as identity primary key,
  company_id uuid not null references public.companies(id) on delete restrict,
  price numeric not null
);

create table public.market_sync_runs (
  id bigint generated always as identity primary key,
  companies integer not null,
  results jsonb not null
);

insert into public.companies values
  ('00000000-0000-4000-8000-000000000001', 'alpha', 'ALP', 'Alpha Display', 'LSE', 'GBP', 'MINOR', 2, 'ALP:LSE', 1, 'GB'),
  ('00000000-0000-4000-8000-000000000002', 'beta', 'BET', 'Beta Display', 'XPAR', 'EUR', 'MAJOR', 2, 'BET:XPAR', 1, 'FR');

insert into public.snapshots (
  company_id, business_quality_score, investment_score, valuation_score, orotitan_score,
  quality_orotitan, confidence_score, price_o85, price_o90, price_o92, price_o95, score_components
) values (
  '00000000-0000-4000-8000-000000000001', 81, 72, 63, 75, true, 8,
  85, 90, 92, 95, '{"legacy":true}'
);
insert into public.market_prices (company_id, price)
values ('00000000-0000-4000-8000-000000000001', 101);
insert into public.market_sync_runs (companies, results) values (2, '[]');

create table public.fixture_legacy_before as
select
  (select count(*) from public.companies) companies,
  (select count(*) from public.snapshots) snapshots,
  (select count(*) from public.market_prices) market_prices,
  (select count(*) from public.market_sync_runs) market_sync_runs,
  (select row_to_json(s)::text from public.snapshots s where id = 1) snapshot_value;

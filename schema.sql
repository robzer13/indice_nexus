create extension if not exists pgcrypto;

create table if not exists public.companies (
  id uuid primary key default gen_random_uuid(),
  slug text not null unique,
  ticker text not null,
  name text not null,
  exchange text not null,
  currency text not null,
  quote_unit text not null default 'MAJOR' check (quote_unit in ('MAJOR','MINOR')),
  price_decimals integer not null default 2 check (price_decimals between 0 and 6),
  market_data_symbol text,
  market_data_multiplier numeric not null default 1 check (market_data_multiplier > 0),
  country text,
  sector text,
  active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.snapshots (
  id bigint generated always as identity primary key,
  company_id uuid not null references public.companies(id) on delete restrict,
  analysis_date date not null,
  model_version text not null,
  status text not null check (status in ('OROTITAN','FINALIST','PRICE_WAIT','TIER_1','WATCHLIST','REJECTED')),
  quality_orotitan boolean,
  business_quality_score numeric check (business_quality_score between 0 and 100),
  investment_score numeric check (investment_score between 0 and 100),
  valuation_score numeric check (valuation_score between 0 and 100),
  orotitan_score numeric check (orotitan_score between 0 and 100),
  confidence_score numeric check (confidence_score between 0 and 10),
  fair_value_low numeric,
  fair_value_base numeric,
  fair_value_high numeric,
  price_o85 numeric,
  price_o90 numeric,
  price_o92 numeric,
  price_o95 numeric,
  thesis text,
  main_risk text,
  invalidation text,
  source_title text,
  notes text,
  score_components jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (company_id, analysis_date, model_version)
);

create table if not exists public.market_prices (
  id bigint generated always as identity primary key,
  company_id uuid not null references public.companies(id) on delete restrict,
  price numeric not null check (price > 0),
  as_of timestamptz not null,
  source text not null,
  raw jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_snapshots_company_date on public.snapshots(company_id, analysis_date desc, created_at desc);
create index if not exists idx_prices_company_asof on public.market_prices(company_id, as_of desc);

-- Data governance: analytical snapshots are immutable, including for service-role writes.
create or replace function public.reject_snapshot_mutation()
returns trigger
language plpgsql
as $$
begin
  raise exception 'OroTitan snapshots are immutable; insert a new analysis_date or model_version instead';
end;
$$;

drop trigger if exists snapshots_immutable on public.snapshots;
create trigger snapshots_immutable
before update or delete on public.snapshots
for each row execute function public.reject_snapshot_mutation();

alter table public.companies enable row level security;
alter table public.snapshots enable row level security;
alter table public.market_prices enable row level security;

-- No public policies. Remove direct API privileges from browser roles as defense in depth.
revoke all on public.companies from anon, authenticated;
revoke all on public.snapshots from anon, authenticated;
revoke all on public.market_prices from anon, authenticated;

grant select on public.companies to service_role;
grant select, insert on public.snapshots to service_role;
grant select, insert on public.market_prices to service_role;
grant usage, select on sequence public.snapshots_id_seq to service_role;
grant usage, select on sequence public.market_prices_id_seq to service_role;

create or replace view public.latest_company_state
with (security_invoker = true)
as
select
  c.id, c.slug, c.ticker, c.name, c.exchange, c.currency, c.quote_unit, c.price_decimals,
  c.country, c.sector, c.market_data_symbol, c.market_data_multiplier,
  p.price, p.as_of as price_as_of, p.source as price_source,
  s.analysis_date, s.model_version, s.status, s.quality_orotitan,
  s.business_quality_score, s.investment_score, s.valuation_score, s.orotitan_score, s.confidence_score,
  s.fair_value_low, s.fair_value_base, s.fair_value_high,
  s.price_o85, s.price_o90, s.price_o92, s.price_o95,
  s.thesis, s.main_risk, s.invalidation, s.source_title, s.notes, s.score_components
from public.companies c
left join lateral (
  select * from public.market_prices mp
  where mp.company_id = c.id
  order by mp.as_of desc, mp.id desc
  limit 1
) p on true
left join lateral (
  select * from public.snapshots ss
  where ss.company_id = c.id
  order by ss.analysis_date desc, ss.created_at desc, ss.id desc
  limit 1
) s on true
where c.active = true;

revoke all on public.latest_company_state from anon, authenticated;
grant select on public.latest_company_state to service_role;

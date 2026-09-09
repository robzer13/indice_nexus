-- OroTitan Equity Research V1 — I1 identity foundation.
-- ADDITIVE ONLY. This migration creates the canonical physical identity layer beside the legacy Screener.
-- It does not modify, delete, rename, or reinterpret legacy analytical snapshots, legacy scores, or market-price rows.
--
-- Canonical boundary:
--   ISSUER != SECURITY
--   ONE ACTIVE CANDIDATE DOSSIER PER ISSUER
--   legacy company rows are retained and linked through an explicit crosswalk.
--
-- Backfill discipline:
--   only fields supported by the legacy row are copied.
--   reporting_currency, security country, primary_listing, listing_status, and authoritative legal_name
--   are intentionally left unresolved when the legacy source cannot establish them safely.
--
-- I3 will add the foreign key from research_dossiers.current_snapshot_id to canonical research_snapshots.
-- Until then current_snapshot_id remains nullable and MUST NOT be treated as a valid canonical current snapshot.

create table if not exists public.issuers (
  issuer_id uuid primary key,
  display_name text not null,
  legal_name text,
  country text,
  reporting_currency text check (
    reporting_currency is null
    or reporting_currency ~ '^[A-Z]{3}$'
  ),
  listing_status text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.securities (
  security_id uuid primary key,
  issuer_id uuid not null references public.issuers(issuer_id) on delete restrict,
  ticker text not null,
  isin text not null default 'UNKNOWN' check (
    isin = 'UNKNOWN'
    or isin ~ '^[A-Z]{2}[A-Z0-9]{9}[0-9]$'
  ),
  exchange text not null,
  country text,
  trading_currency text not null check (trading_currency ~ '^[A-Z]{3}$'),
  primary_listing boolean,
  listing_status text,

  -- Existing market-provider implementation metadata. These are not analytical truth.
  quote_unit text not null default 'MAJOR' check (quote_unit in ('MAJOR','MINOR')),
  price_decimals integer not null default 2 check (price_decimals between 0 and 6),
  market_data_symbol text,
  market_data_multiplier numeric not null default 1 check (market_data_multiplier > 0),

  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create unique index if not exists securities_exchange_ticker_key
  on public.securities (lower(exchange), lower(ticker));

create unique index if not exists one_primary_security_per_issuer
  on public.securities (issuer_id)
  where primary_listing is true;

create table if not exists public.research_dossiers (
  dossier_id uuid primary key,
  issuer_id uuid not null references public.issuers(issuer_id) on delete restrict,
  candidate_episode uuid not null,
  current_snapshot_id uuid,
  active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (issuer_id, candidate_episode)
);

create unique index if not exists one_active_research_dossier_per_issuer
  on public.research_dossiers (issuer_id)
  where active is true;

create table if not exists public.legacy_company_identity_map (
  legacy_company_id uuid primary key references public.companies(id) on delete restrict,
  issuer_id uuid not null references public.issuers(issuer_id) on delete restrict,
  security_id uuid not null unique references public.securities(security_id) on delete restrict,
  dossier_id uuid not null unique references public.research_dossiers(dossier_id) on delete restrict,
  mapped_at timestamptz not null default now()
);

-- Hardened updated_at helper for new canonical identity objects.
-- Existing legacy trigger functions are deliberately not modified in I1.
create or replace function public.set_orotitan_identity_updated_at()
returns trigger
language plpgsql
set search_path = pg_catalog, public
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists issuers_set_updated_at on public.issuers;
create trigger issuers_set_updated_at
before update on public.issuers
for each row execute function public.set_orotitan_identity_updated_at();

drop trigger if exists securities_set_updated_at on public.securities;
create trigger securities_set_updated_at
before update on public.securities
for each row execute function public.set_orotitan_identity_updated_at();

drop trigger if exists research_dossiers_set_updated_at on public.research_dossiers;
create trigger research_dossiers_set_updated_at
before update on public.research_dossiers
for each row execute function public.set_orotitan_identity_updated_at();

-- Deterministic bridge IDs:
-- issuer_id reuses the stable legacy company UUID for the initial one-to-one migration;
-- security/dossier/episode UUIDs are deterministic namespaced hashes of that UUID.
-- The objects remain physically distinct even though the issuer ID is anchored to the legacy row.
with legacy_identity as (
  select
    c.id as legacy_company_id,
    c.id as issuer_id,
    (
      substr(md5('orotitan:security:' || c.id::text),1,8) || '-' ||
      substr(md5('orotitan:security:' || c.id::text),9,4) || '-' ||
      substr(md5('orotitan:security:' || c.id::text),13,4) || '-' ||
      substr(md5('orotitan:security:' || c.id::text),17,4) || '-' ||
      substr(md5('orotitan:security:' || c.id::text),21,12)
    )::uuid as security_id,
    (
      substr(md5('orotitan:dossier:' || c.id::text),1,8) || '-' ||
      substr(md5('orotitan:dossier:' || c.id::text),9,4) || '-' ||
      substr(md5('orotitan:dossier:' || c.id::text),13,4) || '-' ||
      substr(md5('orotitan:dossier:' || c.id::text),17,4) || '-' ||
      substr(md5('orotitan:dossier:' || c.id::text),21,12)
    )::uuid as dossier_id,
    (
      substr(md5('orotitan:episode:' || c.id::text),1,8) || '-' ||
      substr(md5('orotitan:episode:' || c.id::text),9,4) || '-' ||
      substr(md5('orotitan:episode:' || c.id::text),13,4) || '-' ||
      substr(md5('orotitan:episode:' || c.id::text),17,4) || '-' ||
      substr(md5('orotitan:episode:' || c.id::text),21,12)
    )::uuid as candidate_episode,
    c.name,
    c.country,
    c.ticker,
    c.exchange,
    c.currency,
    c.quote_unit,
    c.price_decimals,
    c.market_data_symbol,
    c.market_data_multiplier
  from public.companies c
)
insert into public.issuers (
  issuer_id,
  display_name,
  legal_name,
  country,
  reporting_currency,
  listing_status
)
select
  issuer_id,
  name,
  null,
  country,
  null,
  null
from legacy_identity
on conflict (issuer_id) do nothing;

with legacy_identity as (
  select
    c.id as issuer_id,
    (
      substr(md5('orotitan:security:' || c.id::text),1,8) || '-' ||
      substr(md5('orotitan:security:' || c.id::text),9,4) || '-' ||
      substr(md5('orotitan:security:' || c.id::text),13,4) || '-' ||
      substr(md5('orotitan:security:' || c.id::text),17,4) || '-' ||
      substr(md5('orotitan:security:' || c.id::text),21,12)
    )::uuid as security_id,
    c.ticker,
    c.exchange,
    c.currency,
    c.quote_unit,
    c.price_decimals,
    c.market_data_symbol,
    c.market_data_multiplier
  from public.companies c
)
insert into public.securities (
  security_id,
  issuer_id,
  ticker,
  isin,
  exchange,
  country,
  trading_currency,
  primary_listing,
  listing_status,
  quote_unit,
  price_decimals,
  market_data_symbol,
  market_data_multiplier
)
select
  security_id,
  issuer_id,
  ticker,
  'UNKNOWN',
  exchange,
  null,
  currency,
  null,
  null,
  quote_unit,
  price_decimals,
  market_data_symbol,
  market_data_multiplier
from legacy_identity
on conflict (security_id) do nothing;

with legacy_identity as (
  select
    c.id as issuer_id,
    (
      substr(md5('orotitan:dossier:' || c.id::text),1,8) || '-' ||
      substr(md5('orotitan:dossier:' || c.id::text),9,4) || '-' ||
      substr(md5('orotitan:dossier:' || c.id::text),13,4) || '-' ||
      substr(md5('orotitan:dossier:' || c.id::text),17,4) || '-' ||
      substr(md5('orotitan:dossier:' || c.id::text),21,12)
    )::uuid as dossier_id,
    (
      substr(md5('orotitan:episode:' || c.id::text),1,8) || '-' ||
      substr(md5('orotitan:episode:' || c.id::text),9,4) || '-' ||
      substr(md5('orotitan:episode:' || c.id::text),13,4) || '-' ||
      substr(md5('orotitan:episode:' || c.id::text),17,4) || '-' ||
      substr(md5('orotitan:episode:' || c.id::text),21,12)
    )::uuid as candidate_episode
  from public.companies c
)
insert into public.research_dossiers (
  dossier_id,
  issuer_id,
  candidate_episode,
  current_snapshot_id,
  active
)
select
  dossier_id,
  issuer_id,
  candidate_episode,
  null,
  true
from legacy_identity
on conflict (dossier_id) do nothing;

with legacy_identity as (
  select
    c.id as legacy_company_id,
    c.id as issuer_id,
    (
      substr(md5('orotitan:security:' || c.id::text),1,8) || '-' ||
      substr(md5('orotitan:security:' || c.id::text),9,4) || '-' ||
      substr(md5('orotitan:security:' || c.id::text),13,4) || '-' ||
      substr(md5('orotitan:security:' || c.id::text),17,4) || '-' ||
      substr(md5('orotitan:security:' || c.id::text),21,12)
    )::uuid as security_id,
    (
      substr(md5('orotitan:dossier:' || c.id::text),1,8) || '-' ||
      substr(md5('orotitan:dossier:' || c.id::text),9,4) || '-' ||
      substr(md5('orotitan:dossier:' || c.id::text),13,4) || '-' ||
      substr(md5('orotitan:dossier:' || c.id::text),17,4) || '-' ||
      substr(md5('orotitan:dossier:' || c.id::text),21,12)
    )::uuid as dossier_id
  from public.companies c
)
insert into public.legacy_company_identity_map (
  legacy_company_id,
  issuer_id,
  security_id,
  dossier_id
)
select
  legacy_company_id,
  issuer_id,
  security_id,
  dossier_id
from legacy_identity
on conflict (legacy_company_id) do nothing;

alter table public.issuers enable row level security;
alter table public.securities enable row level security;
alter table public.research_dossiers enable row level security;
alter table public.legacy_company_identity_map enable row level security;

revoke all on public.issuers from anon, authenticated;
revoke all on public.securities from anon, authenticated;
revoke all on public.research_dossiers from anon, authenticated;
revoke all on public.legacy_company_identity_map from anon, authenticated;

grant select, insert, update on public.issuers to service_role;
grant select, insert, update on public.securities to service_role;
grant select, insert, update on public.research_dossiers to service_role;
grant select, insert on public.legacy_company_identity_map to service_role;

comment on table public.issuers is
  'OroTitan Equity Research V1 issuer identity. Distinct from securities. Transitional nullable fields must be resolved before canonical snapshot cutover.';

comment on table public.securities is
  'OroTitan Equity Research V1 security/listing identity plus market-provider implementation metadata.';

comment on table public.research_dossiers is
  'One active candidate dossier per issuer. current_snapshot_id is intentionally nullable until I3 canonical snapshot persistence exists.';

comment on table public.legacy_company_identity_map is
  'Non-destructive crosswalk from the legacy combined companies row to canonical issuer/security/dossier identities.';

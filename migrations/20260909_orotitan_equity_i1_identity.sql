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

begin;

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

-- IF NOT EXISTS only checks object names. Assert the material I1 shape so a
-- partial or independently-created schema can never be accepted as equivalent.
do $$
declare
  invalid_columns text;
  invalid_foreign_keys text;
begin
  with expected(table_name, column_name, data_type, is_nullable) as (
    values
      ('issuers', 'issuer_id', 'uuid', false),
      ('issuers', 'display_name', 'text', false),
      ('issuers', 'legal_name', 'text', true),
      ('issuers', 'country', 'text', true),
      ('issuers', 'reporting_currency', 'text', true),
      ('issuers', 'listing_status', 'text', true),
      ('issuers', 'created_at', 'timestamp with time zone', false),
      ('issuers', 'updated_at', 'timestamp with time zone', false),
      ('securities', 'security_id', 'uuid', false),
      ('securities', 'issuer_id', 'uuid', false),
      ('securities', 'ticker', 'text', false),
      ('securities', 'isin', 'text', false),
      ('securities', 'exchange', 'text', false),
      ('securities', 'country', 'text', true),
      ('securities', 'trading_currency', 'text', false),
      ('securities', 'primary_listing', 'boolean', true),
      ('securities', 'listing_status', 'text', true),
      ('securities', 'quote_unit', 'text', false),
      ('securities', 'price_decimals', 'integer', false),
      ('securities', 'market_data_symbol', 'text', true),
      ('securities', 'market_data_multiplier', 'numeric', false),
      ('securities', 'created_at', 'timestamp with time zone', false),
      ('securities', 'updated_at', 'timestamp with time zone', false),
      ('research_dossiers', 'dossier_id', 'uuid', false),
      ('research_dossiers', 'issuer_id', 'uuid', false),
      ('research_dossiers', 'candidate_episode', 'uuid', false),
      ('research_dossiers', 'current_snapshot_id', 'uuid', true),
      ('research_dossiers', 'active', 'boolean', false),
      ('research_dossiers', 'created_at', 'timestamp with time zone', false),
      ('research_dossiers', 'updated_at', 'timestamp with time zone', false),
      ('legacy_company_identity_map', 'legacy_company_id', 'uuid', false),
      ('legacy_company_identity_map', 'issuer_id', 'uuid', false),
      ('legacy_company_identity_map', 'security_id', 'uuid', false),
      ('legacy_company_identity_map', 'dossier_id', 'uuid', false),
      ('legacy_company_identity_map', 'mapped_at', 'timestamp with time zone', false)
  )
  select string_agg(format('%I.%I', e.table_name, e.column_name), ', ' order by e.table_name, e.column_name)
  into invalid_columns
  from expected e
  left join pg_catalog.pg_class t
    on t.relname = e.table_name and t.relnamespace = 'public'::regnamespace and t.relkind = 'r'
  left join pg_catalog.pg_attribute a
    on a.attrelid = t.oid and a.attname = e.column_name and a.attnum > 0 and not a.attisdropped
  where a.attname is null
     or pg_catalog.format_type(a.atttypid, a.atttypmod) <> e.data_type
     or a.attnotnull = e.is_nullable;

  if invalid_columns is not null then
    raise exception 'Incompatible I1 column definition(s): %', invalid_columns;
  end if;

  if not exists (
    select 1 from pg_catalog.pg_constraint
    where conrelid = 'public.issuers'::regclass and contype = 'p'
      and conkey = array[(select attnum from pg_catalog.pg_attribute where attrelid = 'public.issuers'::regclass and attname = 'issuer_id')]::smallint[]
  ) or not exists (
    select 1 from pg_catalog.pg_constraint
    where conrelid = 'public.securities'::regclass and contype = 'p'
      and conkey = array[(select attnum from pg_catalog.pg_attribute where attrelid = 'public.securities'::regclass and attname = 'security_id')]::smallint[]
  ) or not exists (
    select 1 from pg_catalog.pg_constraint
    where conrelid = 'public.research_dossiers'::regclass and contype = 'p'
      and conkey = array[(select attnum from pg_catalog.pg_attribute where attrelid = 'public.research_dossiers'::regclass and attname = 'dossier_id')]::smallint[]
  ) or not exists (
    select 1 from pg_catalog.pg_constraint
    where conrelid = 'public.legacy_company_identity_map'::regclass and contype = 'p'
      and conkey = array[(select attnum from pg_catalog.pg_attribute where attrelid = 'public.legacy_company_identity_map'::regclass and attname = 'legacy_company_id')]::smallint[]
  ) then
    raise exception 'Incompatible I1 primary key definition';
  end if;

  if not exists (
    select 1 from pg_catalog.pg_constraint where conrelid = 'public.securities'::regclass
      and contype = 'f' and confrelid = 'public.issuers'::regclass and confdeltype = 'r'
      and conkey = array[(select attnum from pg_catalog.pg_attribute where attrelid = 'public.securities'::regclass and attname = 'issuer_id')]::smallint[]
  ) or not exists (
    select 1 from pg_catalog.pg_constraint where conrelid = 'public.research_dossiers'::regclass
      and contype = 'f' and confrelid = 'public.issuers'::regclass and confdeltype = 'r'
      and conkey = array[(select attnum from pg_catalog.pg_attribute where attrelid = 'public.research_dossiers'::regclass and attname = 'issuer_id')]::smallint[]
  ) or (select count(*) from pg_catalog.pg_constraint
        where conrelid = 'public.legacy_company_identity_map'::regclass and contype = 'f' and confdeltype = 'r') <> 4 then
    raise exception 'Incompatible I1 foreign key or ON DELETE definition';
  end if;

  with expected(table_name, column_name, target_table, target_column) as (
    values
      ('securities', 'issuer_id', 'issuers', 'issuer_id'),
      ('research_dossiers', 'issuer_id', 'issuers', 'issuer_id'),
      ('legacy_company_identity_map', 'legacy_company_id', 'companies', 'id'),
      ('legacy_company_identity_map', 'issuer_id', 'issuers', 'issuer_id'),
      ('legacy_company_identity_map', 'security_id', 'securities', 'security_id'),
      ('legacy_company_identity_map', 'dossier_id', 'research_dossiers', 'dossier_id')
  )
  select string_agg(format('%I.%I', e.table_name, e.column_name), ', ' order by e.table_name, e.column_name)
  into invalid_foreign_keys
  from expected e
  where not exists (
    select 1
    from pg_catalog.pg_constraint c
    join pg_catalog.pg_class source_table on source_table.oid = c.conrelid
    join pg_catalog.pg_class target_table on target_table.oid = c.confrelid
    join pg_catalog.pg_attribute source_column
      on source_column.attrelid = source_table.oid and source_column.attnum = c.conkey[1]
    join pg_catalog.pg_attribute target_column
      on target_column.attrelid = target_table.oid and target_column.attnum = c.confkey[1]
    where c.contype = 'f' and c.confdeltype = 'r'
      and source_table.relnamespace = 'public'::regnamespace and source_table.relname = e.table_name
      and source_column.attname = e.column_name
      and target_table.relnamespace = 'public'::regnamespace and target_table.relname = e.target_table
      and target_column.attname = e.target_column
      and cardinality(c.conkey) = 1 and cardinality(c.confkey) = 1
  );

  if invalid_foreign_keys is not null then
    raise exception 'Incompatible I1 foreign key(s): %', invalid_foreign_keys;
  end if;

  if not exists (
    select 1 from pg_catalog.pg_constraint where conrelid = 'public.research_dossiers'::regclass and contype = 'u'
      and conkey = array[
        (select attnum from pg_catalog.pg_attribute where attrelid = 'public.research_dossiers'::regclass and attname = 'issuer_id'),
        (select attnum from pg_catalog.pg_attribute where attrelid = 'public.research_dossiers'::regclass and attname = 'candidate_episode')
      ]::smallint[]
  ) or not exists (
    select 1 from pg_catalog.pg_constraint where conrelid = 'public.legacy_company_identity_map'::regclass and contype = 'u'
      and conkey = array[(select attnum from pg_catalog.pg_attribute where attrelid = 'public.legacy_company_identity_map'::regclass and attname = 'security_id')]::smallint[]
  ) or not exists (
    select 1 from pg_catalog.pg_constraint where conrelid = 'public.legacy_company_identity_map'::regclass and contype = 'u'
      and conkey = array[(select attnum from pg_catalog.pg_attribute where attrelid = 'public.legacy_company_identity_map'::regclass and attname = 'dossier_id')]::smallint[]
  ) then
    raise exception 'Incompatible I1 unique constraint definition';
  end if;

  if not exists (
    select 1 from pg_catalog.pg_index i join pg_catalog.pg_class x on x.oid = i.indexrelid
    where x.relnamespace = 'public'::regnamespace and x.relname = 'securities_exchange_ticker_key'
      and i.indrelid = 'public.securities'::regclass and i.indisunique and i.indpred is null
      and i.indnkeyatts = 2
      and regexp_replace(pg_catalog.pg_get_expr(i.indexprs, i.indrelid), '[[:space:]]', '', 'g') = 'lower(exchange),lower(ticker)'
  ) or not exists (
    select 1 from pg_catalog.pg_index i join pg_catalog.pg_class x on x.oid = i.indexrelid
    where x.relnamespace = 'public'::regnamespace and x.relname = 'one_primary_security_per_issuer'
      and i.indrelid = 'public.securities'::regclass and i.indisunique
      and i.indnkeyatts = 1
      and i.indkey[0] = (select attnum from pg_catalog.pg_attribute where attrelid = 'public.securities'::regclass and attname = 'issuer_id')
      and lower(regexp_replace(pg_catalog.pg_get_expr(i.indpred, i.indrelid), '[[:space:]()]', '', 'g')) = 'primary_listingistrue'
  ) or not exists (
    select 1 from pg_catalog.pg_index i join pg_catalog.pg_class x on x.oid = i.indexrelid
    where x.relnamespace = 'public'::regnamespace and x.relname = 'one_active_research_dossier_per_issuer'
      and i.indrelid = 'public.research_dossiers'::regclass and i.indisunique
      and i.indnkeyatts = 1
      and i.indkey[0] = (select attnum from pg_catalog.pg_attribute where attrelid = 'public.research_dossiers'::regclass and attname = 'issuer_id')
      and lower(regexp_replace(pg_catalog.pg_get_expr(i.indpred, i.indrelid), '[[:space:]()]', '', 'g')) = 'activeistrue'
  ) then
    raise exception 'Incompatible I1 index definition';
  end if;
end;
$$;

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

revoke all on function public.set_orotitan_identity_updated_at() from public, anon, authenticated, service_role;

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
create temporary table orotitan_expected_identity on commit drop as
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
  c.name, c.country, c.ticker, c.exchange, c.currency, c.quote_unit,
  c.price_decimals, c.market_data_symbol, c.market_data_multiplier
from public.companies c;

do $$
begin
  if exists (
    select 1 from orotitan_expected_identity e join public.issuers i on i.issuer_id = e.issuer_id
    where (i.display_name, i.country) is distinct from (e.name, e.country)
  ) then
    raise exception 'Conflicting deterministic OroTitan issuer identity';
  end if;

  if exists (
    select 1 from orotitan_expected_identity e join public.securities s on s.security_id = e.security_id
    where (s.issuer_id, s.ticker, s.exchange, s.trading_currency, s.quote_unit, s.price_decimals,
           s.market_data_symbol, s.market_data_multiplier)
      is distinct from
          (e.issuer_id, e.ticker, e.exchange, e.currency, e.quote_unit, e.price_decimals,
           e.market_data_symbol, e.market_data_multiplier)
  ) then
    raise exception 'Conflicting deterministic OroTitan security identity';
  end if;

  if exists (
    select 1 from orotitan_expected_identity e join public.research_dossiers d on d.dossier_id = e.dossier_id
    where (d.issuer_id, d.candidate_episode) is distinct from (e.issuer_id, e.candidate_episode)
  ) then
    raise exception 'Conflicting deterministic OroTitan dossier identity';
  end if;

  if exists (
    select 1 from orotitan_expected_identity e
    join public.legacy_company_identity_map m on m.legacy_company_id = e.legacy_company_id
    where (m.issuer_id, m.security_id, m.dossier_id)
      is distinct from (e.issuer_id, e.security_id, e.dossier_id)
  ) then
    raise exception 'Conflicting deterministic OroTitan legacy crosswalk';
  end if;
end;
$$;

with legacy_identity as (
  select
    legacy_company_id, issuer_id, security_id, dossier_id, candidate_episode,
    name, country, ticker, exchange, currency, quote_unit, price_decimals,
    market_data_symbol, market_data_multiplier
  from orotitan_expected_identity
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
  select issuer_id, security_id, ticker, exchange, currency, quote_unit,
    price_decimals, market_data_symbol, market_data_multiplier
  from orotitan_expected_identity
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
  select issuer_id, dossier_id, candidate_episode
  from orotitan_expected_identity
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
  select legacy_company_id, issuer_id, security_id, dossier_id
  from orotitan_expected_identity
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

do $$
begin
  if exists (
    select 1
    from public.legacy_company_identity_map m
    join public.securities s on s.security_id = m.security_id
    join public.research_dossiers d on d.dossier_id = m.dossier_id
    where s.issuer_id is distinct from m.issuer_id
       or d.issuer_id is distinct from m.issuer_id
  ) then
    raise exception 'OroTitan crosswalk references identities belonging to another issuer';
  end if;
end;
$$;

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

commit;

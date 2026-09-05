-- OroTitan Screener V1.1 migration.
-- Safe to run on a V1 database. Does not mutate analytical snapshots or market prices.

create table if not exists public.market_sync_runs (
  id bigint generated always as identity primary key,
  started_at timestamptz not null,
  finished_at timestamptz not null,
  trigger_source text not null check (trigger_source in ('CRON','ADMIN')),
  companies integer not null check (companies >= 0),
  inserted integer not null check (inserted >= 0),
  failed integer not null check (failed >= 0),
  results jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_market_sync_runs_created on public.market_sync_runs(created_at desc);

create or replace function public.set_companies_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists companies_set_updated_at on public.companies;
create trigger companies_set_updated_at
before update on public.companies
for each row execute function public.set_companies_updated_at();

create or replace function public.reject_market_sync_run_mutation()
returns trigger
language plpgsql
as $$
begin
  raise exception 'OroTitan market sync runs are append-only';
end;
$$;

drop trigger if exists market_sync_runs_immutable on public.market_sync_runs;
create trigger market_sync_runs_immutable
before update or delete on public.market_sync_runs
for each row execute function public.reject_market_sync_run_mutation();

alter table public.market_sync_runs enable row level security;
revoke all on public.market_sync_runs from anon, authenticated;

-- V1.1 allows administrative metadata changes on companies, but never DELETE.
grant select, insert, update on public.companies to service_role;
grant select, insert on public.market_sync_runs to service_role;
grant usage, select on sequence public.market_sync_runs_id_seq to service_role;

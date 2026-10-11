#!/usr/bin/env bash
set -euo pipefail
# Explicitly local disposable PostgreSQL 17: no shared DB connection possible.
export PGHOST=127.0.0.1 PGPORT=5432 PGUSER=postgres PGPASSWORD=postgres
root="$(cd "$(dirname "$0")/../.." && pwd)"
database="orotitan_paper_$$"
cleanup() { dropdb --if-exists --force "$database" >/dev/null 2>&1 || true; }
trap cleanup EXIT
createdb "$database"
version="$(psql -XAt -d "$database" -c 'show server_version_num')"
if (( version < 170000 || version >= 180000 )); then echo "PostgreSQL 17 required" >&2; exit 3; fi
psql -X -v ON_ERROR_STOP=1 -d "$database" <<'SQL' >/dev/null
create extension if not exists pgcrypto;
do $$ begin
  if not exists (select 1 from pg_roles where rolname='anon') then create role anon nologin; end if;
  if not exists (select 1 from pg_roles where rolname='authenticated') then create role authenticated nologin; end if;
  if not exists (select 1 from pg_roles where rolname='service_role') then create role service_role nologin bypassrls; end if;
end $$;
-- Minimal canonical *test doubles* only. No production schema or existing canonical rows are mutated.
create table public.issuers(issuer_id uuid primary key);
create table public.securities(security_id uuid primary key, trading_currency text not null, quote_unit text not null);
create table public.research_snapshots(snapshot_id uuid primary key, issuer_id uuid not null references public.issuers, security_id uuid not null references public.securities);
create table public.research_dossiers(issuer_id uuid not null references public.issuers, current_snapshot_id uuid references public.research_snapshots, active boolean not null);
create table public.legacy_company_identity_map (legacy_company_id uuid primary key, issuer_id uuid not null references public.issuers,security_id uuid references public.securities);
create table public.market_prices (
  id bigint generated always as identity primary key, company_id uuid not null,
  price numeric not null, as_of timestamptz not null, created_at timestamptz not null default now(),
  source text not null default 'YAHOO_FINANCE',raw jsonb
);
insert into public.issuers values ('00000000-0000-4000-8000-000000000111');
insert into public.securities values ('00000000-0000-4000-8000-000000000222','EUR','MAJOR');
insert into public.legacy_company_identity_map values ('00000000-0000-4000-8000-000000000444','00000000-0000-4000-8000-000000000111','00000000-0000-4000-8000-000000000222');
insert into public.research_snapshots values
  ('00000000-0000-4000-8000-000000000333','00000000-0000-4000-8000-000000000111','00000000-0000-4000-8000-000000000222');
insert into public.research_dossiers values
  ('00000000-0000-4000-8000-000000000111','00000000-0000-4000-8000-000000000333',true);
insert into public.market_prices(company_id,price,as_of) values
  ('00000000-0000-4000-8000-000000000444',4.5,now()-interval '30 minutes');
SQL
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$root/migrations/20261011_paper_long_term_portfolio_v1.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$root/migrations/20261011_paper_autopilot_v1_2.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$database" <<'SQL'
insert into public.paper_portfolios(code) values ('default');

-- V1.2 is off by default. Only a controlled owner RPC can activate it.
do $
declare p uuid;
begin
  select id into p from public.paper_portfolios where code='default';
  if exists(select 1 from public.paper_autopilot_settings where portfolio_id=p)
  then raise exception 'Autopilot must begin disabled and unconfigured'; end if;
  perform public.paper_configure_autopilot(p,false,0.05,1);
  if (select enabled from public.paper_autopilot_settings where portfolio_id=p) then
    raise exception 'Initial autopilot policy must be OFF'; end if;
  begin
    perform public.paper_configure_autopilot(p,true,0.50,1);
    raise exception 'Invalid max weight was accepted';
  exception when sqlstate '23514' then null; end;
  if has_function_privilege('anon','public.paper_configure_autopilot(uuid,boolean,numeric,numeric)','EXECUTE')
  then raise exception 'Anonymous caller can activate autopilot'; end if;
  perform public.paper_configure_autopilot(p,true,0.05,1);
  if not (select enabled from public.paper_autopilot_settings where portfolio_id=p) then
    raise exception 'Explicit opt-in was not persisted'; end if;
  perform public.paper_configure_autopilot(p,false,0.05,1);
  begin
    delete from public.paper_autopilot_policy_events where portfolio_id=p;
    raise exception 'Policy audit events unexpectedly mutable';
  exception when sqlstate '23514' then null; end;
end $;

do $
declare portfolio uuid; pol uuid; sig uuid; same_sig uuid;
begin
  select id into portfolio from public.paper_portfolios where code='default';
  pol:=public.paper_create_policy(portfolio,
    '00000000-0000-4000-8000-000000000111','00000000-0000-4000-8000-000000000333',
    'VALUE','ACTIVE','NOT_APPLICABLE',4.5,12,'Fixture VALUE avec cible de sortie et suivi');
  if (select version from public.paper_policy_versions where id=pol)<>1 then raise exception 'Missing policy version'; end if;
  sig:=public.paper_record_signal(pol,1,'PAPER_LONG_TERM_V1','BUY_CANDIDATE','ENTRY_PRICE_REACHED');
  same_sig:=public.paper_record_signal(pol,1,'PAPER_LONG_TERM_V1','BUY_CANDIDATE','ENTRY_PRICE_REACHED');
  if sig is distinct from same_sig then raise exception 'Idempotent replay changed signal identity'; end if;
  if (select count(*) from public.paper_signals)<>1 then raise exception 'Duplicate signal'; end if;
  begin
    update public.paper_signals set reason='TAMPERED' where id=sig;
    raise exception 'Mutable signal unexpectedly accepted';
  exception when sqlstate '23514' then null; end;
  begin
    perform public.paper_execute_fill(sig,1,1,0);
    raise exception 'Old pre-signal quote unexpectedly filled';
  exception when sqlstate '23514' then null; end;
  if has_function_privilege('anon','public.paper_execute_fill(uuid,bigint,numeric,numeric)','EXECUTE')
  then raise exception 'Anonymous caller can execute a fill'; end if;
end $$;
SQL
# New quote is stored *after* signal; forward-only execution rejects the original quote.
sleep 2
psql -X -v ON_ERROR_STOP=1 -d "$database" <<'SQL'
insert into public.market_prices(company_id,price,as_of) values
  ('00000000-0000-4000-8000-000000000444',5,date_trunc('second',now()));
do $$
declare sig uuid; v_fill_id uuid; portfolio uuid;
begin
  select id into sig from public.paper_signals limit 1;
  select id into portfolio from public.paper_portfolios where code='default';
  v_fill_id:=public.paper_execute_fill(sig,2,10,1);
  if v_fill_id is null then raise exception 'No virtual fill ID'; end if;
  if (select cash_eur from public.paper_portfolios where id=portfolio) <> 99949
    then raise exception 'Incorrect cash after buy'; end if;
  if (select quantity from public.paper_positions where portfolio_id=portfolio) <> 10
    then raise exception 'Incorrect bought quantity'; end if;
  begin
    perform public.paper_execute_fill(sig,2,10,1);
    raise exception 'Duplicate fill unexpectedly accepted';
  exception when unique_violation then null; end;
  begin
    delete from public.paper_fills where id=v_fill_id;
    raise exception 'Mutable fill unexpectedly accepted';
  exception when sqlstate '23514' then null; end;
  perform public.paper_create_policy(portfolio,
    '00000000-0000-4000-8000-000000000111','00000000-0000-4000-8000-000000000333',
    'VALUE','ACTIVE','NOT_APPLICABLE',4.5,12,'Version additionnelle auditable et immuable');
  if (select max(version) from public.paper_policy_versions) <> 2
    then raise exception 'Policy successor version not created'; end if;
  begin
    update public.paper_policy_versions set rationale='REWRITTEN' where version=1;
    raise exception 'Historical policy mutation unexpectedly accepted';
  exception when sqlstate '23514' then null; end;
end $$;
SQL
psql -X -v ON_ERROR_STOP=1 -d "$database" <<'SQL'
-- A valid daily quote is only attestable when provider timestamps match.
update public.market_prices set raw=jsonb_build_object(
  'regularMarketTime',to_char(as_of at time zone 'UTC','YYYY-MM-DD"T"HH24:MI:SS"Z"')
);
do $
declare p uuid; id1 bigint; id2 bigint;
begin
  select id into p from public.paper_portfolios where code='default';
  id1:=public.paper_record_daily_nav(p);
  id2:=public.paper_record_daily_nav(p);
  if id1 is null or id1<>id2 then raise exception 'NAV idempotence failure'; end if;
  if (select nav_eur from public.paper_daily_nav where id=id1) <> 99999
  then raise exception 'Incorrect NAV after fee and mark-to-market'; end if;
  begin
    update public.paper_daily_nav set positions_eur=0 where id=id1;
    raise exception 'NAV history unexpectedly mutable';
  exception when sqlstate '23514' then null; end;
  if has_function_privilege('anon','public.paper_record_daily_nav(uuid)','EXECUTE')
  then raise exception 'Anonymous NAV access'; end if;
end $;
SQL
echo "Paper portfolio: isolated PG17 migration, immutable signals and forward-price virtual fill PASS"

-- Investment Lab V1.2: a separate opt-in switch for paper-only scheduled operation.
-- Requires the reviewed V1.1 paper portfolio migration. NEVER execute on production without separate GO.
begin;
do $$ begin
  if to_regclass('public.paper_portfolios') is null
  or to_regclass('public.paper_policy_versions') is null
  or to_regclass('public.paper_signals') is null
  or to_regclass('public.paper_fills') is null
  then raise exception 'PAPER_V1_1_REQUIRED'; end if;
end $$;

create table public.paper_autopilot_settings (
  portfolio_id uuid primary key references public.paper_portfolios(id) on delete restrict,
  enabled boolean not null default false,
  maximum_initial_capital_weight numeric(8,6) not null default 0.05
    check (maximum_initial_capital_weight > 0 and maximum_initial_capital_weight <= 0.10),
  simulated_fee_eur numeric(12,4) not null default 1
    check (simulated_fee_eur >= 0 and simulated_fee_eur <= 100),
  updated_at timestamptz not null default now()
);
create table public.paper_autopilot_policy_events (
  id bigint generated always as identity primary key,
  portfolio_id uuid not null references public.paper_portfolios(id) on delete restrict,
  enabled boolean not null,
  maximum_initial_capital_weight numeric(8,6) not null,
  simulated_fee_eur numeric(12,4) not null,
  created_at timestamptz not null default now()
);
create trigger paper_autopilot_policy_events_immutable
before update or delete on public.paper_autopilot_policy_events
for each row execute function public.paper_reject_history_mutation();

create function public.paper_configure_autopilot(
  p_portfolio_id uuid, p_enabled boolean, p_maximum_weight numeric, p_fee_eur numeric
) returns void
language plpgsql security definer set search_path = pg_catalog,public as $$
begin
  if p_enabled is null or p_maximum_weight is null or p_maximum_weight <= 0 or p_maximum_weight > 0.1
    or p_fee_eur is null or p_fee_eur < 0 or p_fee_eur > 100
  then raise exception 'PAPER_AUTOPILOT_POLICY_INVALID' using errcode='23514'; end if;
  perform 1 from public.paper_portfolios where id=p_portfolio_id for update;
  if not found then raise exception 'PAPER_PORTFOLIO_MISSING' using errcode='23503'; end if;
  insert into public.paper_autopilot_settings
    (portfolio_id,enabled,maximum_initial_capital_weight,simulated_fee_eur)
    values (p_portfolio_id,p_enabled,p_maximum_weight,p_fee_eur)
  on conflict (portfolio_id) do update
    set enabled=excluded.enabled,
        maximum_initial_capital_weight=excluded.maximum_initial_capital_weight,
        simulated_fee_eur=excluded.simulated_fee_eur,
        updated_at=now();
  insert into public.paper_autopilot_policy_events
    (portfolio_id,enabled,maximum_initial_capital_weight,simulated_fee_eur)
  values (p_portfolio_id,p_enabled,p_maximum_weight,p_fee_eur);
end $$;

alter table public.paper_autopilot_settings enable row level security;
alter table public.paper_autopilot_policy_events enable row level security;
revoke all on public.paper_autopilot_settings,public.paper_autopilot_policy_events
  from public,anon,authenticated;
grant select on public.paper_autopilot_settings,public.paper_autopilot_policy_events to service_role;
revoke all on function public.paper_configure_autopilot(uuid,boolean,numeric,numeric)
  from public,anon,authenticated;
grant execute on function public.paper_configure_autopilot(uuid,boolean,numeric,numeric) to service_role;
grant usage,select on sequence public.paper_autopilot_policy_events_id_seq to service_role;

-- Daily NAV is computed server-side using the actual persisted positions and quote bytes.
-- This is a price-only valuation until verified cash dividends/corporate actions are supported.
create table public.paper_daily_nav (
  id bigint generated always as identity primary key,
  portfolio_id uuid not null references public.paper_portfolios(id) on delete restrict,
  valuation_day date not null,
  valued_at timestamptz not null default now(),
  cash_eur numeric(20,4) not null check (cash_eur >= 0),
  positions_eur numeric(20,4) not null check (positions_eur >= 0),
  nav_eur numeric(20,4) generated always as (cash_eur + positions_eur) stored,
  quote_evidence jsonb not null check (jsonb_typeof(quote_evidence) = 'array'),
  unique (portfolio_id,valuation_day)
);
create trigger paper_daily_nav_immutable before update or delete on public.paper_daily_nav
for each row execute function public.paper_reject_history_mutation();
alter table public.paper_daily_nav enable row level security;
revoke all on public.paper_daily_nav from public,anon,authenticated;
grant select on public.paper_daily_nav to service_role;
grant usage,select on sequence public.paper_daily_nav_id_seq to service_role;

create function public.paper_record_daily_nav(p_portfolio_id uuid) returns bigint
language plpgsql security definer set search_path = pg_catalog,public as $$
declare
  port public.paper_portfolios%rowtype;
  pos record; q record;
  total_positions numeric(20,4):=0;
  receipts jsonb:='[]'::jsonb;
  v_id bigint;
begin
  select * into port from public.paper_portfolios where id=p_portfolio_id for update;
  if not found then raise exception 'PAPER_PORTFOLIO_MISSING' using errcode='23503'; end if;

  for pos in select company_id,quantity from public.paper_positions
    where portfolio_id=port.id and quantity>0 order by company_id loop
    select price.id,price.price,price.as_of,price.created_at,price.source,price.raw,
      security.trading_currency,security.quote_unit
      into q
      from public.legacy_company_identity_map link
      join public.securities security on security.security_id=link.security_id
      join lateral (
        select market.* from public.market_prices market
        where market.company_id=link.legacy_company_id
        order by market.as_of desc, market.id desc limit 1
      ) price on true
      where link.issuer_id=pos.company_id;
    if not found or q.trading_currency<>'EUR' or q.quote_unit<>'MAJOR'
      or q.source<>'YAHOO_FINANCE'
      or q.price is null or q.price<=0
      or q.as_of is null or q.as_of<now()-interval '96 hours'
      or q.as_of>now()+interval '5 minutes'
      or q.raw->>'regularMarketTime' is null
    then raise exception 'PAPER_NAV_UNVERIFIED_QUOTE_OR_CURRENCY' using errcode='23514'; end if;
    if (q.raw->>'regularMarketTime')::timestamptz is distinct from q.as_of
    then raise exception 'PAPER_NAV_MISMATCHED_EXCHANGE_TIMESTAMP' using errcode='23514'; end if;
    total_positions:=total_positions+round(pos.quantity*q.price,4);
    receipts:=receipts||jsonb_build_array(jsonb_build_object(
      'issuer_id',pos.company_id,'quote_id',q.id,'quote_as_of',q.as_of,
      'quantity',pos.quantity,'price_eur',q.price));
  end loop;

  insert into public.paper_daily_nav
    (portfolio_id,valuation_day,cash_eur,positions_eur,quote_evidence)
  values(port.id,(now() at time zone 'UTC')::date,port.cash_eur,total_positions,receipts)
  on conflict (portfolio_id,valuation_day) do nothing
  returning id into v_id;
  if v_id is null then
    select id into v_id from public.paper_daily_nav
    where portfolio_id=port.id and valuation_day=(now() at time zone 'UTC')::date;
  end if;
  return v_id;
end $$;
revoke all on function public.paper_record_daily_nav(uuid) from public,anon,authenticated;
grant execute on function public.paper_record_daily_nav(uuid) to service_role;
commit;

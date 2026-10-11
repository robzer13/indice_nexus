-- Paper-only long-term investment laboratory; does not modify analytical Registry or canonical snapshots.
-- Apply only after separately authorized non-production validation and production migration approval.
begin;

create table public.paper_portfolios (
  id uuid primary key default gen_random_uuid(),
  code text not null unique check (code = 'default'),
  initial_cash_eur numeric(20,4) not null default 100000 check (initial_cash_eur > 0),
  cash_eur numeric(20,4) not null default 100000 check (cash_eur >= 0),
  created_at timestamptz not null default now()
);

create table public.paper_policy_versions (
  id uuid primary key default gen_random_uuid(),
  portfolio_id uuid not null references public.paper_portfolios(id) on delete restrict,
  company_id uuid not null references public.issuers(issuer_id) on delete restrict,
  snapshot_id uuid not null references public.research_snapshots(snapshot_id) on delete restrict,
  version integer not null check (version > 0),
  profile text not null check (profile in ('CORE','VALUE','CYCLICAL_VALUE','TURNAROUND')),
  thesis_state text not null check (thesis_state in ('ACTIVE','UNKNOWN','INVALIDATED')),
  milestone_state text not null check (milestone_state in ('CONFIRMED','UNCONFIRMED','NOT_APPLICABLE')),
  entry_price_eur numeric(20,4) check (entry_price_eur is null or entry_price_eur > 0),
  exit_target_eur numeric(20,4) check (exit_target_eur is null or exit_target_eur > 0),
  rationale text not null check (length(btrim(rationale)) >= 12),
  created_at timestamptz not null default now(),
  unique (portfolio_id,company_id,version),
  unique (id,portfolio_id,company_id)
);
create index paper_policy_lookup_idx on public.paper_policy_versions(portfolio_id,company_id,version desc);

create table public.paper_signals (
  id uuid primary key default gen_random_uuid(),
  portfolio_id uuid not null,
  company_id uuid not null,
  policy_id uuid not null,
  snapshot_id uuid not null references public.research_snapshots(snapshot_id) on delete restrict,
  quote_id bigint not null references public.market_prices(id) on delete restrict,
  quote_price_eur numeric(20,4) not null check (quote_price_eur > 0),
  quote_as_of timestamptz not null,
  ruleset text not null check (ruleset = 'PAPER_LONG_TERM_V1'),
  action text not null check (action in ('BUY_CANDIDATE','SELL_CANDIDATE','HOLD','WAIT','REVIEW_REQUIRED','DATA_BLOCKED')),
  reason text not null check (length(reason) > 0),
  created_at timestamptz not null default now(),
  foreign key (policy_id,portfolio_id,company_id) references public.paper_policy_versions(id,portfolio_id,company_id) on delete restrict,
  unique (policy_id, quote_id, action, reason)
);
create index paper_signal_created_idx on public.paper_signals(portfolio_id,created_at desc);

create table public.paper_positions (
  portfolio_id uuid not null references public.paper_portfolios(id) on delete restrict,
  company_id uuid not null references public.issuers(issuer_id) on delete restrict,
  quantity numeric(24,8) not null default 0 check (quantity >= 0),
  avg_cost_eur numeric(20,6) not null default 0 check (avg_cost_eur >= 0),
  realized_pnl_eur numeric(20,4) not null default 0,
  updated_at timestamptz not null default now(),
  primary key (portfolio_id,company_id)
);

create table public.paper_fills (
  id uuid primary key default gen_random_uuid(),
  portfolio_id uuid not null references public.paper_portfolios(id) on delete restrict,
  company_id uuid not null references public.issuers(issuer_id) on delete restrict,
  signal_id uuid not null unique references public.paper_signals(id) on delete restrict,
  quote_id bigint not null references public.market_prices(id) on delete restrict,
  side text not null check (side in ('BUY','SELL')),
  quantity numeric(24,8) not null check (quantity > 0),
  price_eur numeric(20,4) not null check (price_eur > 0),
  fee_eur numeric(20,4) not null check (fee_eur >= 0),
  realized_pnl_eur numeric(20,4) not null default 0,
  filled_at timestamptz not null default now()
);
create index paper_fill_history_idx on public.paper_fills(portfolio_id,filled_at desc);

-- Immutable policy/decision/fill evidence. Portfolio cash/position balances are mutated only by owner-controlled RPC.
create function public.paper_reject_history_mutation()
returns trigger language plpgsql set search_path = pg_catalog,public as $$
begin
  raise exception 'PAPER_HISTORY_IMMUTABLE' using errcode='23514';
end; $$;

create trigger paper_policy_immutable before update or delete on public.paper_policy_versions
for each row execute function public.paper_reject_history_mutation();
create trigger paper_signal_immutable before update or delete on public.paper_signals
for each row execute function public.paper_reject_history_mutation();
create trigger paper_fill_immutable before update or delete on public.paper_fills
for each row execute function public.paper_reject_history_mutation();

-- Serialized policy version creation. Approvals cannot silently rewrite historical policy records.
create function public.paper_create_policy(
  p_portfolio uuid, p_company uuid, p_snapshot uuid, p_profile text,
  p_thesis text, p_milestone text, p_entry numeric, p_exit numeric, p_rationale text
) returns uuid language plpgsql security definer set search_path = pg_catalog,public as $$
declare v_id uuid; v_version int; v_current_snapshot uuid;
begin
  perform 1 from public.paper_portfolios where id=p_portfolio for update;
  if not found then raise exception 'PAPER_PORTFOLIO_MISSING' using errcode='23503'; end if;
  select d.current_snapshot_id into v_current_snapshot
    from public.research_dossiers d
    where d.issuer_id=p_company and d.active and d.current_snapshot_id=p_snapshot
    limit 1;
  if v_current_snapshot is null then raise exception 'PAPER_SNAPSHOT_NOT_CURRENT' using errcode='23514'; end if;
  if not exists(select 1 from public.research_snapshots s where s.snapshot_id=p_snapshot and s.issuer_id=p_company)
  then raise exception 'PAPER_SNAPSHOT_COMPANY_MISMATCH' using errcode='23514'; end if;
  select coalesce(max(version),0)+1 into v_version from public.paper_policy_versions
    where portfolio_id=p_portfolio and company_id=p_company;
  insert into public.paper_policy_versions
    (portfolio_id,company_id,snapshot_id,version,profile,thesis_state,milestone_state,entry_price_eur,exit_target_eur,rationale)
  values(p_portfolio,p_company,p_snapshot,v_version,p_profile,p_thesis,p_milestone,p_entry,p_exit,p_rationale)
  returning id into v_id;
  return v_id;
end; $$;

-- Server records only signals it calculated, based on an exact persistent quote.
-- SQL additionally rechecks quote & snapshot identities and chronology.
create function public.paper_record_signal(
  p_policy uuid, p_quote bigint, p_ruleset text, p_action text, p_reason text
) returns uuid language plpgsql security definer set search_path = pg_catalog,public as $$
declare pol public.paper_policy_versions%rowtype; q public.market_prices%rowtype; v_id uuid;
begin
  select * into pol from public.paper_policy_versions where id=p_policy;
  if not found then raise exception 'PAPER_POLICY_MISSING' using errcode='23503'; end if;
  if exists(select 1 from public.paper_policy_versions newer
    where newer.portfolio_id=pol.portfolio_id and newer.company_id=pol.company_id and newer.version>pol.version)
  then raise exception 'PAPER_POLICY_SUPERSEDED' using errcode='23514'; end if;
  if not exists(select 1 from public.research_dossiers d where d.issuer_id=pol.company_id
    and d.active and d.current_snapshot_id=pol.snapshot_id)
  then raise exception 'PAPER_SNAPSHOT_NOT_CURRENT' using errcode='23514'; end if;
  select quote.* into q from public.market_prices quote
    join public.legacy_company_identity_map link on link.legacy_company_id=quote.company_id
    where quote.id=p_quote and link.issuer_id=pol.company_id;
  if not found then raise exception 'PAPER_QUOTE_COMPANY_MISMATCH' using errcode='23514'; end if;
  if q.as_of > now()+interval '5 minutes' or q.as_of < now()-interval '96 hours'
  then raise exception 'PAPER_QUOTE_OUT_OF_WINDOW' using errcode='23514'; end if;
  if not exists(select 1 from public.securities sec
      join public.research_snapshots s on s.security_id=sec.security_id
      where s.snapshot_id=pol.snapshot_id and sec.trading_currency='EUR' and sec.quote_unit='MAJOR')
  then raise exception 'PAPER_UNSUPPORTED_QUOTE_CURRENCY_OR_UNIT' using errcode='23514'; end if;
  insert into public.paper_signals
    (portfolio_id,company_id,policy_id,snapshot_id,quote_id,quote_price_eur,quote_as_of,ruleset,action,reason)
  values(pol.portfolio_id,pol.company_id,pol.id,pol.snapshot_id,q.id,q.price,q.as_of,p_ruleset,p_action,p_reason)
  on conflict (policy_id, quote_id, action, reason) do nothing
  returning id into v_id;
  if v_id is null then
    select id into v_id from public.paper_signals
    where policy_id=pol.id and quote_id=q.id and action=p_action and reason=p_reason;
  end if;
  if v_id is null then raise exception 'PAPER_CONFLICTED_SIGNAL_RETRY' using errcode='23505'; end if;
  return v_id;
end; $$;

-- A later market quote is required after a signal: no same-quote/look-ahead execution.
-- Portfolio lock serializes competing fills. Unique signal_id ensures exactly-once.
create function public.paper_execute_fill(
  p_signal uuid, p_quote bigint, p_quantity numeric, p_fee_eur numeric default 0
) returns uuid language plpgsql security definer set search_path = pg_catalog,public as $$
declare s public.paper_signals%rowtype; pol public.paper_policy_versions%rowtype;
  port public.paper_portfolios%rowtype; q public.market_prices%rowtype;
  pos public.paper_positions%rowtype; v_trade uuid; v_notional numeric;
  v_new_qty numeric; v_cost numeric; v_realized numeric := 0;
begin
  select * into s from public.paper_signals where id=p_signal;
  if not found then raise exception 'PAPER_SIGNAL_MISSING' using errcode='23503'; end if;
  if s.action not in ('BUY_CANDIDATE','SELL_CANDIDATE')
  then raise exception 'PAPER_SIGNAL_NOT_ACTIONABLE' using errcode='23514'; end if;
  if p_quantity is null or p_quantity<=0 or p_quantity>100000000 or
     p_fee_eur is null or p_fee_eur<0 or p_fee_eur>1000000
  then raise exception 'PAPER_INVALID_FILL_PARAMS' using errcode='23514'; end if;
  select * into port from public.paper_portfolios where id=s.portfolio_id for update;
  select * into pol from public.paper_policy_versions where id=s.policy_id;
  if exists(select 1 from public.paper_policy_versions newer
    where newer.portfolio_id=pol.portfolio_id and newer.company_id=pol.company_id and newer.version>pol.version)
  then raise exception 'PAPER_POLICY_SUPERSEDED' using errcode='23514'; end if;
  if not exists(select 1 from public.research_dossiers d where d.issuer_id=s.company_id
    and d.active and d.current_snapshot_id=s.snapshot_id)
  then raise exception 'PAPER_SNAPSHOT_NOT_CURRENT' using errcode='23514'; end if;
  if exists(select 1 from public.paper_fills where signal_id=p_signal)
  then raise exception 'PAPER_SIGNAL_ALREADY_FILLED' using errcode='23505'; end if;
  select quote.* into q from public.market_prices quote
    join public.legacy_company_identity_map link on link.legacy_company_id=quote.company_id
    where quote.id=p_quote and link.issuer_id=s.company_id;
  if not found or q.as_of<=s.created_at or q.created_at<=s.created_at
    or q.as_of>now()+interval '5 minutes' or q.as_of<now()-interval '96 hours'
  then raise exception 'PAPER_NO_NEW_VALID_EXECUTION_QUOTE' using errcode='23514'; end if;
  insert into public.paper_positions(portfolio_id,company_id)
    values(s.portfolio_id,s.company_id) on conflict do nothing;
  select * into pos from public.paper_positions where portfolio_id=s.portfolio_id
    and company_id=s.company_id for update;
  v_notional:=round(q.price*p_quantity,4);
  if s.action='BUY_CANDIDATE' then
    if port.cash_eur<v_notional+p_fee_eur
    then raise exception 'PAPER_INSUFFICIENT_CASH' using errcode='23514'; end if;
    v_new_qty:=pos.quantity+p_quantity;
    v_cost:=(pos.avg_cost_eur*pos.quantity+v_notional+p_fee_eur)/v_new_qty;
    update public.paper_portfolios set cash_eur=cash_eur-v_notional-p_fee_eur where id=port.id;
    update public.paper_positions set quantity=v_new_qty,avg_cost_eur=v_cost,updated_at=now()
      where portfolio_id=s.portfolio_id and company_id=s.company_id;
  else
    if pos.quantity<p_quantity
    then raise exception 'PAPER_INSUFFICIENT_SHARES' using errcode='23514'; end if;
    v_new_qty:=pos.quantity-p_quantity;
    v_realized:=round(v_notional-p_fee_eur-pos.avg_cost_eur*p_quantity,4);
    update public.paper_portfolios set cash_eur=cash_eur+v_notional-p_fee_eur where id=port.id;
    update public.paper_positions set quantity=v_new_qty,
      avg_cost_eur=case when v_new_qty=0 then 0 else avg_cost_eur end,
      realized_pnl_eur=realized_pnl_eur+v_realized,updated_at=now()
      where portfolio_id=s.portfolio_id and company_id=s.company_id;
  end if;
  insert into public.paper_fills
    (portfolio_id,company_id,signal_id,quote_id,side,quantity,price_eur,fee_eur,realized_pnl_eur)
  values(s.portfolio_id,s.company_id,s.id,q.id,
    case when s.action='BUY_CANDIDATE' then 'BUY' else 'SELL' end,
    p_quantity,q.price,p_fee_eur,v_realized) returning id into v_trade;
  return v_trade;
end; $$;

-- Narrow privilege boundary: only server-held service_role may write through RPCs.
alter table public.paper_portfolios enable row level security;
alter table public.paper_policy_versions enable row level security;
alter table public.paper_signals enable row level security;
alter table public.paper_positions enable row level security;
alter table public.paper_fills enable row level security;
revoke all on public.paper_portfolios,public.paper_policy_versions,public.paper_signals,
  public.paper_positions,public.paper_fills from public,anon,authenticated;
grant select on public.paper_portfolios,public.paper_policy_versions,public.paper_signals,
  public.paper_positions,public.paper_fills to service_role;
grant insert on public.paper_portfolios to service_role;
revoke all on function public.paper_reject_history_mutation() from public,anon,authenticated;
revoke all on function public.paper_create_policy(uuid,uuid,uuid,text,text,text,numeric,numeric,text) from public,anon,authenticated;
revoke all on function public.paper_record_signal(uuid,bigint,text,text,text) from public,anon,authenticated;
revoke all on function public.paper_execute_fill(uuid,bigint,numeric,numeric) from public,anon,authenticated;
grant execute on function public.paper_create_policy(uuid,uuid,uuid,text,text,text,numeric,numeric,text) to service_role;
grant execute on function public.paper_record_signal(uuid,bigint,text,text,text) to service_role;
grant execute on function public.paper_execute_fill(uuid,bigint,numeric,numeric) to service_role;
commit;

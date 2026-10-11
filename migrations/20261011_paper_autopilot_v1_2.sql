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
commit;

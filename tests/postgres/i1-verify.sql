\set ON_ERROR_STOP on

do $$
declare
  before_state public.fixture_legacy_before%rowtype;
  old_updated_at timestamptz;
  first_security uuid;
  first_issuer constant uuid := '00000000-0000-4000-8000-000000000001';
begin
  -- T1/T2: real objects and fail-closed backfill values.
  if (select count(*) from public.issuers) <> 2
     or (select count(*) from public.securities) <> 2
     or (select count(*) from public.research_dossiers where active) <> 2
     or (select count(*) from public.legacy_company_identity_map) <> 2 then
    raise exception 'T1/T2 canonical backfill counts differ from legacy fixture';
  end if;

  if exists (
    select 1 from public.issuers i join public.companies c on c.id = i.issuer_id
    where i.display_name <> c.name or i.legal_name is not null or i.reporting_currency is not null
  ) or exists (
    select 1 from public.securities s join public.companies c on c.id = s.issuer_id
    where s.ticker <> c.ticker or s.exchange <> c.exchange or s.trading_currency <> c.currency
       or s.country is not null or s.primary_listing is not null or s.listing_status is not null
       or s.isin <> 'UNKNOWN'
  ) then
    raise exception 'T2 fail-closed identity values are incorrect';
  end if;

  -- T8/T9: multiple securities are valid, but only one may be primary.
  update public.securities set primary_listing = true where issuer_id = first_issuer;
  select security_id into first_security from public.securities where issuer_id = first_issuer;
  insert into public.securities (
    security_id, issuer_id, ticker, exchange, trading_currency, primary_listing
  ) values ('10000000-0000-4000-8000-000000000001', first_issuer, 'ALP2', 'NYSE', 'USD', null);
  begin
    update public.securities set primary_listing = true
    where security_id = '10000000-0000-4000-8000-000000000001';
    raise exception 'T9 accepted a second primary listing';
  exception when unique_violation then null;
  end;

  -- T10: an inactive history row is valid; a second active row is rejected.
  insert into public.research_dossiers (dossier_id, issuer_id, candidate_episode, active)
  values ('20000000-0000-4000-8000-000000000001', first_issuer,
          '30000000-0000-4000-8000-000000000001', false);
  begin
    insert into public.research_dossiers (dossier_id, issuer_id, candidate_episode, active)
    values ('20000000-0000-4000-8000-000000000002', first_issuer,
            '30000000-0000-4000-8000-000000000002', true);
    raise exception 'T10 accepted a second active dossier';
  exception when unique_violation then null;
  end;

  -- T11: legacy counts and analytical values remain byte-for-byte equivalent as JSON text.
  select * into before_state from public.fixture_legacy_before;
  if before_state.companies <> (select count(*) from public.companies)
     or before_state.snapshots <> (select count(*) from public.snapshots)
     or before_state.market_prices <> (select count(*) from public.market_prices)
     or before_state.market_sync_runs <> (select count(*) from public.market_sync_runs)
     or before_state.snapshot_value <> (select row_to_json(s)::text from public.snapshots s where id = 1) then
    raise exception 'T11 legacy data changed';
  end if;

  -- T12: RLS and effective table/function privileges.
  if exists (
    select 1 from unnest(array['issuers','securities','research_dossiers','legacy_company_identity_map']) n
    join pg_class c on c.relnamespace = 'public'::regnamespace and c.relname = n
    where not c.relrowsecurity
  ) then raise exception 'T12 RLS is not enabled'; end if;

  if exists (
    select 1 from unnest(array['anon','authenticated']) role_name,
      unnest(array['issuers','securities','research_dossiers','legacy_company_identity_map']) table_name,
      unnest(array['SELECT','INSERT','UPDATE','DELETE']) privilege
    where has_table_privilege(role_name, 'public.' || table_name, privilege)
  ) then raise exception 'T12 browser table privilege exposed'; end if;

  if has_table_privilege('service_role', 'public.issuers', 'DELETE')
     or has_table_privilege('service_role', 'public.securities', 'DELETE')
     or has_table_privilege('service_role', 'public.research_dossiers', 'DELETE')
     or has_table_privilege('service_role', 'public.legacy_company_identity_map', 'DELETE')
     or has_table_privilege('service_role', 'public.legacy_company_identity_map', 'UPDATE')
     or not has_table_privilege('service_role', 'public.issuers', 'SELECT')
     or not has_table_privilege('service_role', 'public.issuers', 'INSERT')
     or not has_table_privilege('service_role', 'public.issuers', 'UPDATE') then
    raise exception 'T12 service_role table privileges differ';
  end if;

  if exists (
       select 1 from pg_proc p
       cross join lateral aclexplode(coalesce(p.proacl, acldefault('f', p.proowner))) privilege
       where p.oid = 'public.set_orotitan_identity_updated_at()'::regprocedure
         and privilege.grantee = 0 and privilege.privilege_type = 'EXECUTE'
     ) or has_function_privilege('anon', 'public.set_orotitan_identity_updated_at()', 'EXECUTE')
     or has_function_privilege('authenticated', 'public.set_orotitan_identity_updated_at()', 'EXECUTE')
     or has_function_privilege('service_role', 'public.set_orotitan_identity_updated_at()', 'EXECUTE') then
    raise exception 'T12 trigger function has direct EXECUTE exposure';
  end if;

  -- T13: trigger execution does not require callers to have function EXECUTE.
  select updated_at into old_updated_at from public.issuers where issuer_id = first_issuer;
  perform pg_sleep(0.01);
  set local role service_role;
  update public.issuers set display_name = display_name where issuer_id = first_issuer;
  reset role;
  if (select updated_at from public.issuers where issuer_id = first_issuer) <= old_updated_at then
    raise exception 'T13 updated_at did not advance';
  end if;

  -- T14: transitional snapshot pointer is nullable, empty, and has no FK.
  if exists (select 1 from public.research_dossiers where current_snapshot_id is not null)
     or (select attnotnull from pg_attribute where attrelid = 'public.research_dossiers'::regclass
         and attname = 'current_snapshot_id')
     or exists (
       select 1 from pg_constraint where conrelid = 'public.research_dossiers'::regclass
         and contype = 'f' and (select attnum from pg_attribute
           where attrelid = 'public.research_dossiers'::regclass and attname = 'current_snapshot_id') = any(conkey)
     ) then raise exception 'T14 current_snapshot_id is not transitional'; end if;
end;
$$;

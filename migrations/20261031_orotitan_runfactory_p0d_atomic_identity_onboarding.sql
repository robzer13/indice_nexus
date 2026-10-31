-- OroTitan Run Factory P0-D: governed, atomic, SHADOW-only onboarding.
-- Requires I1 identity schema and existing constrained indexes.
-- This migration does not create an issuer, security, dossier, run, stage or snapshot.
-- Execute ONLY after isolated PostgreSQL test matrix and scope review.
begin;

create table if not exists public.orotitan_identity_onboarding_receipts_v1 (
  request_key text primary key,
  request_sha256 text not null check (request_sha256 ~ '^[a-f0-9]{64}$'),
  issuer_id uuid not null references public.issuers(issuer_id) on delete restrict,
  security_id uuid not null references public.securities(security_id) on delete restrict,
  dossier_id uuid not null references public.research_dossiers(dossier_id) on delete restrict,
  outcome text not null check (outcome in ('CREATED','REUSED','LISTING_ADDED')),
  evidence jsonb not null,
  reviewed_by text not null check (reviewed_by = 'ADMIN_SESSION_MANUAL_ATTESTATION_V1'),
  created_at timestamptz not null default now()
);
alter table public.orotitan_identity_onboarding_receipts_v1 enable row level security;
revoke all on public.orotitan_identity_onboarding_receipts_v1 from public, anon, authenticated;
grant select, insert on public.orotitan_identity_onboarding_receipts_v1 to service_role;

-- SECURITY INVOKER: privileged service_role on a pinned SHADOW-only server
-- is required. Public/authenticated/anon cannot call this procedure.
create or replace function public.orotitan_onboard_identity_shadow_v1(
 p_idempotency_key text,
 p_isin text,
 p_ticker text,
 p_exchange text,
 p_display_name text,
 p_legal_name text,
 p_issuer_country text,
 p_reporting_currency text,
 p_trading_currency text,
 p_primary_listing boolean,
 p_quote_unit text,
 p_price_decimals integer,
 p_legacy_company_id uuid,
 p_existing_issuer_id uuid,
 p_evidence jsonb,
 p_operator_attestation text
) returns jsonb
language plpgsql
security invoker
set search_path = pg_catalog, public, extensions
as $$
declare
 v_isin text := upper(btrim(p_isin));
 v_ticker text := upper(btrim(p_ticker));
 v_exchange text := upper(btrim(p_exchange));
 v_display_name text := btrim(p_display_name);
 v_legal_name text := btrim(p_legal_name);
 v_country text := upper(btrim(p_issuer_country));
 v_reporting text := upper(btrim(p_reporting_currency));
 v_trading text := upper(btrim(p_trading_currency));
 v_quote text := upper(btrim(p_quote_unit));
 v_hash text;
 v_payload jsonb;
 v_old record;
 v_security record;
 v_issuer record;
 v_dossier record;
 v_issuer_id uuid;
 v_security_id uuid;
 v_dossier_id uuid;
 v_legacy record;
 v_legacy_map record;
 v_existing_count integer;
 v_existing_issuer_count integer;
 v_active_count integer;
 v_name_count integer;
 v_current_primary_count integer;
 v_digits text := '';
 v_char text;
 v_n integer;
 v_sum integer := 0;
 v_offset integer := 0;
 v_action text;
begin
 -- This is an operator attestation, NEVER automatic evidence authentication.
 if p_operator_attestation is distinct from 'APPROVE_SHADOW_CANONICAL_IDENTITY_V1' then
  raise exception 'OROTITAN_ONBOARDING_ATTESTATION_REQUIRED';
 end if;
 if p_idempotency_key is null or length(p_idempotency_key) < 16 or
    length(p_idempotency_key) > 160 or p_idempotency_key !~ '^[a-zA-Z0-9:_-]+$' then
  raise exception 'OROTITAN_ONBOARDING_KEY_INVALID';
 end if;
 if v_isin is null or v_isin !~ '^[A-Z]{2}[A-Z0-9]{9}[0-9]$' then
  raise exception 'OROTITAN_ISIN_INVALID';
 end if;
 -- ISO 6166/Luhn independently verified in database; not trusted from JS.
 for v_n in 1..length(v_isin) loop
  v_char := substring(v_isin from v_n for 1);
  if v_char ~ '[A-Z]' then
   v_digits := v_digits || (ascii(v_char) - 55)::text;
  else
   v_digits := v_digits || v_char;
  end if;
 end loop;
 for v_n in reverse length(v_digits)..1 loop
  v_sum := v_sum + case when (v_offset % 2) = 1 then
   case when substring(v_digits from v_n for 1)::integer * 2 > 9
    then substring(v_digits from v_n for 1)::integer * 2 - 9
    else substring(v_digits from v_n for 1)::integer * 2 end
   else substring(v_digits from v_n for 1)::integer end;
  v_offset := v_offset + 1;
 end loop;
 if (v_sum % 10) <> 0 then raise exception 'OROTITAN_ISIN_CHECK_DIGIT_INVALID'; end if;
 if v_ticker is null or v_ticker !~ '^[A-Z0-9._-]{1,30}$' or
    v_exchange is null or v_exchange !~ '^[A-Z0-9._-]{2,30}$' then
  raise exception 'OROTITAN_LISTING_INVALID';
 end if;
 if v_display_name is null or length(v_display_name) not between 2 and 180 or
    v_legal_name is null or length(v_legal_name) not between 2 and 180 or
    v_country is null or v_country !~ '^[A-Z]{2}$' or
    v_reporting is null or v_reporting !~ '^[A-Z]{3}$' or
    v_trading is null or v_trading !~ '^[A-Z]{3}$' or
    v_quote is null or v_quote not in ('MAJOR','MINOR') or
    p_price_decimals is null or p_price_decimals not between 0 and 6 or
    p_primary_listing is null then
  raise exception 'OROTITAN_ONBOARDING_METADATA_INCOMPLETE';
 end if;
 if p_evidence is null or jsonb_typeof(p_evidence) <> 'object' or
    p_evidence->>'method' is distinct from 'OPERATOR_PRIMARY_SOURCE_REVIEW' or
    p_evidence->>'reviewed_by' is distinct from 'ADMIN_SESSION_MANUAL_ATTESTATION_V1' or
    length(coalesce(p_evidence->>'issuer_source_url','')) not between 15 and 2048 or
    length(coalesce(p_evidence->>'listing_source_url','')) not between 15 and 2048 or
    p_evidence->>'issuer_source_url' !~ '^https://[a-zA-Z0-9.-]+/' or
    p_evidence->>'listing_source_url' !~ '^https://[a-zA-Z0-9.-]+/' or
    p_evidence->>'issuer_source_url' = p_evidence->>'listing_source_url' or
    length(coalesce(p_evidence->>'observed_at','')) < 20 then
  raise exception 'OROTITAN_PRIMARY_SOURCE_ATTESTATION_INCOMPLETE';
 end if;
 v_payload := jsonb_build_object(
  'isin',v_isin,'ticker',v_ticker,'exchange',v_exchange,
  'display_name',v_display_name,'legal_name',v_legal_name,
  'issuer_country',v_country,'reporting_currency',v_reporting,
  'trading_currency',v_trading,'primary_listing',p_primary_listing,
  'quote_unit',v_quote,'price_decimals',p_price_decimals,
  'legacy_company_id',p_legacy_company_id,
  'existing_issuer_id',p_existing_issuer_id,'evidence',p_evidence
 );
 v_hash := encode(extensions.digest(v_payload::text,'sha256'),'hex');
 -- Locks are globally ordered: (ISIN namespace) -> (listing namespace).
 -- They serialize duplicate requests even with different idempotency keys.
 perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(
  'orotitan:p0d:isin:'||v_isin,0));
 perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(
  'orotitan:p0d:listing:'||v_exchange||':'||v_ticker,0));

 select * into v_old from public.orotitan_identity_onboarding_receipts_v1
 where request_key = p_idempotency_key;
 if found then
  if v_old.request_sha256 <> v_hash then
   raise exception 'OROTITAN_ONBOARDING_IDEMPOTENCY_CONFLICT';
  end if;
  if not exists (select 1 from public.securities s
   where s.security_id=v_old.security_id and s.issuer_id=v_old.issuer_id
    and upper(s.isin)=v_isin and upper(s.ticker)=v_ticker
    and upper(s.exchange)=v_exchange) or
   not exists (select 1 from public.research_dossiers d
    where d.dossier_id=v_old.dossier_id and d.issuer_id=v_old.issuer_id
     and d.active) then
    raise exception 'OROTITAN_ONBOARDING_REPLAY_STATE_MISMATCH';
  end if;
  return jsonb_build_object('outcome','REPLAYED','replayed',true,
   'issuer_id',v_old.issuer_id,'security_id',v_old.security_id,
   'dossier_id',v_old.dossier_id,'request_sha256',v_hash,
   'source_automatically_verified',false,'run_created',false,
   'registry_stage_advanced',false);
 end if;

 select count(*) into v_existing_count from public.securities s
 where lower(s.exchange)=lower(v_exchange) and lower(s.ticker)=lower(v_ticker);
 if v_existing_count > 1 then
  raise exception 'OROTITAN_ONBOARDING_LISTING_DUPLICATES';
 end if;
 select * into v_security from public.securities s
 where lower(s.exchange)=lower(v_exchange) and lower(s.ticker)=lower(v_ticker);
 if found and upper(v_security.isin) <> v_isin then
  raise exception 'OROTITAN_ONBOARDING_LISTING_ISIN_CONFLICT';
 end if;

 select count(distinct issuer_id) into v_existing_issuer_count
 from public.securities where upper(isin)=v_isin;
 if v_existing_issuer_count > 1 then
  raise exception 'OROTITAN_ONBOARDING_ISIN_ISSUER_COLLISION';
 end if;
 select issuer_id into v_issuer_id from public.securities
 where upper(isin)=v_isin limit 1;
 if v_security.security_id is not null then
  if v_issuer_id is not null and v_issuer_id <> v_security.issuer_id then
   raise exception 'OROTITAN_ONBOARDING_ISIN_ISSUER_COLLISION';
  end if;
  v_issuer_id := v_security.issuer_id;
 end if;
 if p_existing_issuer_id is not null then
  if v_issuer_id is not null and v_issuer_id <> p_existing_issuer_id then
   raise exception 'OROTITAN_ONBOARDING_ISSUER_OVERRIDE_CONFLICT';
  end if;
  v_issuer_id := p_existing_issuer_id;
 end if;
 if v_issuer_id is not null then
  select * into v_issuer from public.issuers where issuer_id=v_issuer_id;
  if not found then raise exception 'OROTITAN_ONBOARDING_ISSUER_NOT_FOUND'; end if;
  if lower(v_issuer.display_name) <> lower(v_display_name) or
     (v_issuer.legal_name is not null and
      lower(v_issuer.legal_name) <> lower(v_legal_name)) or
     (v_issuer.country is not null and upper(v_issuer.country) <> v_country) or
     (v_issuer.reporting_currency is not null and
      upper(v_issuer.reporting_currency) <> v_reporting) then
   raise exception 'OROTITAN_ONBOARDING_EXISTING_ISSUER_METADATA_CONFLICT';
  end if;
 else
  select count(*) into v_name_count from public.issuers i
  where lower(i.display_name)=lower(v_display_name) or
   lower(coalesce(i.legal_name,''))=lower(v_legal_name);
  if v_name_count > 0 then
   raise exception 'OROTITAN_ONBOARDING_ISSUER_NAME_REVIEW_REQUIRED';
  end if;
  v_issuer_id := pg_catalog.gen_random_uuid();
  insert into public.issuers
   (issuer_id,display_name,legal_name,country,reporting_currency)
   values(v_issuer_id,v_display_name,v_legal_name,v_country,v_reporting);
 end if;
 select count(*) into v_current_primary_count from public.securities
 where issuer_id=v_issuer_id and primary_listing is true;
 if p_primary_listing and v_current_primary_count>0 and
    (v_security.security_id is null or not v_security.primary_listing) then
  raise exception 'OROTITAN_ONBOARDING_PRIMARY_LISTING_CONFLICT';
 end if;
 if v_security.security_id is null then
  v_security_id := pg_catalog.gen_random_uuid();
  insert into public.securities
   (security_id,issuer_id,isin,ticker,exchange,country,trading_currency,
    primary_listing,quote_unit,price_decimals)
  values(v_security_id,v_issuer_id,v_isin,v_ticker,v_exchange,v_country,
   v_trading,p_primary_listing,v_quote,p_price_decimals);
  if v_existing_issuer_count>0 then v_action:='LISTING_ADDED';
  else v_action:='CREATED'; end if;
 else
  v_security_id := v_security.security_id;
  if v_security.issuer_id<>v_issuer_id or
     upper(v_security.trading_currency)<>v_trading or
     v_security.quote_unit<>v_quote or
     v_security.price_decimals<>p_price_decimals or
     v_security.primary_listing is distinct from p_primary_listing then
   raise exception 'OROTITAN_ONBOARDING_EXISTING_SECURITY_METADATA_CONFLICT';
  end if;
  v_action:='REUSED';
 end if;
 select count(*) into v_active_count from public.research_dossiers
 where issuer_id=v_issuer_id and active is true;
 if v_active_count>1 then
  raise exception 'OROTITAN_ONBOARDING_MULTIPLE_ACTIVE_DOSSIERS';
 end if;
 select dossier_id into v_dossier_id from public.research_dossiers
 where issuer_id=v_issuer_id and active is true;
 if v_dossier_id is null then
  v_dossier_id:=pg_catalog.gen_random_uuid();
  insert into public.research_dossiers
   (dossier_id,issuer_id,candidate_episode,current_snapshot_id,active)
  values(v_dossier_id,v_issuer_id,pg_catalog.gen_random_uuid(),null,true);
 end if;

 if p_legacy_company_id is not null then
  select * into v_legacy from public.companies where id=p_legacy_company_id;
  if not found or upper(v_legacy.ticker)<>v_ticker or
     upper(v_legacy.exchange)<>v_exchange or
     upper(v_legacy.currency)<>v_trading then
   raise exception 'OROTITAN_ONBOARDING_LEGACY_IDENTITY_MISMATCH';
  end if;
  select * into v_legacy_map from public.legacy_company_identity_map
  where legacy_company_id=p_legacy_company_id;
  if found then
   if v_legacy_map.issuer_id<>v_issuer_id or
      v_legacy_map.security_id<>v_security_id or
      v_legacy_map.dossier_id<>v_dossier_id then
    raise exception 'OROTITAN_ONBOARDING_LEGACY_CROSSWALK_CONFLICT';
   end if;
  else
   if exists (select 1 from public.legacy_company_identity_map
    where security_id=v_security_id or dossier_id=v_dossier_id) then
    raise exception 'OROTITAN_ONBOARDING_LEGACY_CROSSWALK_UNIQUE_CONFLICT';
   end if;
   insert into public.legacy_company_identity_map
    (legacy_company_id,issuer_id,security_id,dossier_id)
   values (p_legacy_company_id,v_issuer_id,v_security_id,v_dossier_id);
  end if;
 end if;
 insert into public.orotitan_identity_onboarding_receipts_v1
  (request_key,request_sha256,issuer_id,security_id,dossier_id,
   outcome,evidence,reviewed_by)
 values(p_idempotency_key,v_hash,v_issuer_id,v_security_id,v_dossier_id,
  v_action,p_evidence,'ADMIN_SESSION_MANUAL_ATTESTATION_V1');
 return jsonb_build_object('outcome',v_action,'replayed',false,
  'issuer_id',v_issuer_id,'security_id',v_security_id,
  'dossier_id',v_dossier_id,'request_sha256',v_hash,
  'source_automatically_verified',false,'run_created',false,
  'registry_stage_advanced',false);
end;
$$;
revoke all on function public.orotitan_onboard_identity_shadow_v1(
 text,text,text,text,text,text,text,text,text,boolean,text,integer,
 uuid,uuid,jsonb,text) from public, anon, authenticated;
grant execute on function public.orotitan_onboard_identity_shadow_v1(
 text,text,text,text,text,text,text,text,text,boolean,text,integer,
 uuid,uuid,jsonb,text) to service_role;
commit;

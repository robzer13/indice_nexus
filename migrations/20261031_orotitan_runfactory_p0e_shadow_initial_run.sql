-- OroTitan Run Factory P0-E — atomic SHADOW V3.1 INITIAL run admission.
-- Does not change historical Registry contracts, their V1/V2 schema generation
-- label, versioned runtime, and never creates stages/snapshots or publishes.
-- Install on SHADOW only. Requires Registry V1.12 RPCs.
-- The service_role has no direct UPDATE privilege on orotitan_runs:
-- the existing controlled SECURITY DEFINER RPCs perform all writes. The
-- P0-E advisory lock serializes same issuer/cutoff admissions.
begin;

create table if not exists public.orotitan_p0e_run_admission_receipts_v1 (
 run_id uuid primary key references public.orotitan_runs(run_id) on delete restrict,
 creation_idempotency_key text not null unique,
 issuer_id uuid not null references public.issuers(issuer_id) on delete restrict,
 security_id uuid not null references public.securities(security_id) on delete restrict,
 dossier_id uuid not null references public.research_dossiers(dossier_id) on delete restrict,
 data_cutoff date not null,
 profile_id text not null check (profile_id = 'OROTITAN_V3_1'),
 contract_set_sha256 text not null
  check (contract_set_sha256 = '3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2'),
 verification_receipt jsonb not null,
 authority text not null check (authority = 'USER_GO_P0E_SHADOW_INITIAL_V3_1'),
 publication_authorized boolean not null default false check (publication_authorized = false),
 created_at timestamptz not null default now()
);
alter table public.orotitan_p0e_run_admission_receipts_v1 enable row level security;
revoke all on public.orotitan_p0e_run_admission_receipts_v1 from public, anon, authenticated;
grant select,insert on public.orotitan_p0e_run_admission_receipts_v1 to service_role;

create or replace function public.orotitan_shadow_create_initial_run_v3_1(
 p_issuer_id uuid,
 p_security_id uuid,
 p_dossier_id uuid,
 p_isin text,
 p_exchange text,
 p_data_cutoff date,
 p_contract_pins jsonb,
 p_contract_set_sha256 text,
 p_creation_idempotency_key text,
 p_request_fingerprint_sha256 text,
 p_verification_receipt jsonb,
 p_operator_confirmation text
) returns jsonb
language plpgsql
security invoker
set search_path=pg_catalog,public,extensions
as $$
declare
 v_isin text := upper(btrim(p_isin));
 v_exchange text := upper(btrim(p_exchange));
 v_issuer_count integer;
 v_security_count integer;
 v_dossier_count integer;
 v_existing_count integer;
 v_existing public.orotitan_runs%rowtype;
 v_result jsonb;
 v_bind_result jsonb;
 v_run public.orotitan_runs%rowtype;
 v_run_id uuid;
 v_binding_descriptor text;
 v_binding_fingerprint text;
 v_current_receipt public.orotitan_p0e_run_admission_receipts_v1%rowtype;
 v_expected_pin_count integer;
 v_checked_pin_count integer := 0;
 v_entry jsonb;
 v_name text;
 v_sha text;
begin
 if p_operator_confirmation is distinct from
  ('GO P0-E SHADOW '||v_isin||' '||p_data_cutoff::text) then
  raise exception 'P0E_EXPLICIT_GO_REQUIRED';
 end if;
 if p_issuer_id is null or p_security_id is null or p_dossier_id is null or
    v_isin is null or v_isin !~ '^[A-Z]{2}[A-Z0-9]{9}[0-9]$' or
    v_exchange is null or v_exchange !~ '^[A-Z0-9._-]{2,30}$' or
    p_data_cutoff is null or p_data_cutoff > current_date then
  raise exception 'P0E_IDENTITY_OR_CUTOFF_INVALID';
 end if;
 if p_contract_set_sha256 is distinct from
   '3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2'
  or p_contract_pins is null or jsonb_typeof(p_contract_pins)<>'object'
  or (select count(*) from jsonb_object_keys(p_contract_pins))<>15
  or p_contract_pins#>>'{process,version}'<>'3.1'
  or p_contract_pins#>>'{pilotage,version}'<>'3.1'
  or p_contract_pins#>>'{deep_dive_stage,version}'<>'3.1'
  or p_contract_pins#>>'{research_stage,version}'<>'2.0'
  or p_contract_pins#>>'{valuation_date_alignment,version}'<>'1.0'
  or p_contract_pins#>>'{dcf_timing,version}'<>'1.0'
  or not public.orotitan_contract_pins_complete(p_contract_pins)
  or public.orotitan_contract_set_sha256(p_contract_pins)<>p_contract_set_sha256
 then raise exception 'P0E_FROZEN_V3_1_CONTRACT_PIN_MISMATCH';end if;
 if p_creation_idempotency_key is null or
    p_request_fingerprint_sha256 is null or
    p_request_fingerprint_sha256 !~ '^[0-9a-f]{64}$' or
    p_creation_idempotency_key <>
      ('orotitan:v3:create:'||p_request_fingerprint_sha256)
 then raise exception 'P0E_CREATE_FINGERPRINT_INVALID';end if;
 if p_verification_receipt is null or
    jsonb_typeof(p_verification_receipt)<>'object' or
    p_verification_receipt->>'profileId'<>'OROTITAN_V3_1' or
    p_verification_receipt->>'contractSetSha256'<>p_contract_set_sha256 or
    p_verification_receipt->>'pinCount'<>'15' or
    p_verification_receipt->>'passedPinCount'<>'15' or
    p_verification_receipt->>'allOriginalBytesVerified'<>'true' or
    p_verification_receipt->>'registryMutated'<>'false' or
    jsonb_typeof(p_verification_receipt->'checks')<>'array' or
    jsonb_array_length(p_verification_receipt->'checks')<>15
 then raise exception 'P0E_ORIGINAL_BYTES_RECEIPT_REQUIRED';end if;
 begin
  if (p_verification_receipt->>'checkedAt')::timestamptz <
     clock_timestamp() - interval '30 minutes'
   or (p_verification_receipt->>'checkedAt')::timestamptz >
     clock_timestamp() + interval '5 minutes' then
   raise exception 'P0E_PIN_VERIFICATION_STALE';
  end if;
 exception when invalid_datetime_format or datetime_field_overflow then
  raise exception 'P0E_PIN_VERIFICATION_TIMESTAMP_INVALID';
 end;
 for v_entry in select value from jsonb_array_elements(p_verification_receipt->'checks') loop
  v_name := v_entry->>'logicalName';
  v_sha := v_entry->>'sha256';
  if v_name is null or v_sha is null or
     v_entry->>'status'<>'ORIGINAL_CONTRACT_BYTES_VERIFIED' or
     coalesce((v_entry->>'byteLength')::integer,0)<=0 or
     not p_contract_pins ? v_name or
     p_contract_pins->v_name->>'content_sha256'<>v_sha then
    raise exception 'P0E_PIN_VERIFICATION_DETAIL_INVALID';
  end if;
  if (select count(*) from jsonb_array_elements(p_verification_receipt->'checks') x
      where x->>'logicalName'=v_name)<>1 then
    raise exception 'P0E_PIN_VERIFICATION_DUPLICATE_NAME';
  end if;
  v_checked_pin_count := v_checked_pin_count+1;
 end loop;
 if v_checked_pin_count<>15 then
  raise exception 'P0E_PIN_VERIFICATION_INCOMPLETE';
 end if;
 -- One lock serializes competing requests for this canonical issuer/cutoff.
 perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(
  'orotitan:p0e:run:'||p_issuer_id::text||':'||p_data_cutoff::text,0));
 select count(*) into v_security_count from public.securities
  where security_id=p_security_id and issuer_id=p_issuer_id and
   upper(isin)=v_isin and upper(exchange)=v_exchange and primary_listing is true;
 select count(*) into v_issuer_count from public.issuers
  where issuer_id=p_issuer_id and btrim(display_name)<>'';
 select count(*) into v_dossier_count from public.research_dossiers
  where dossier_id=p_dossier_id and issuer_id=p_issuer_id and
   active is true and current_snapshot_id is null;
 if v_security_count<>1 or v_issuer_count<>1 or v_dossier_count<>1 or
    (select count(*) from public.securities where upper(isin)=v_isin
      and upper(exchange)=v_exchange)<>1 or
    (select count(*) from public.research_dossiers
      where issuer_id=p_issuer_id and active is true)<>1 then
  raise exception 'P0E_CANONICAL_IDENTITY_LINEAGE_MISMATCH';
 end if;
 select count(*) into v_existing_count from public.orotitan_runs
  where issuer_id=p_issuer_id and data_cutoff=p_data_cutoff
    and run_status<>'CANCELLED';
 if v_existing_count>1 then
  raise exception 'P0E_MULTIPLE_EXISTING_RUNS';
 end if;
 if v_existing_count=1 then
  select * into v_existing from public.orotitan_runs
   where issuer_id=p_issuer_id and data_cutoff=p_data_cutoff
    and run_status<>'CANCELLED';
  if v_existing.creation_idempotency_key<>p_creation_idempotency_key or
     v_existing.contract_set_sha256<>p_contract_set_sha256 or
     v_existing.security_id is distinct from p_security_id or
     v_existing.dossier_id is distinct from p_dossier_id or
     v_existing.run_type<>'INITIAL' then
   raise exception 'P0E_EXISTING_RUN_CONFLICT';
  end if;
 end if;
 v_result:=public.create_orotitan_run(
  p_creation_idempotency_key,p_issuer_id,'IMPOSED_COMPANY','ANALYZE',
  'INITIAL',p_data_cutoff,null,null,'3.1','3.1',
  p_contract_pins,p_contract_set_sha256,p_request_fingerprint_sha256);
 v_run_id:=(v_result->>'run_id')::uuid;
 select * into v_run from public.orotitan_runs where run_id=v_run_id;
 if not found then raise exception 'P0E_RUN_CREATION_READBACK_MISSING';end if;
 if v_run.security_id is null and v_run.dossier_id is null then
  v_binding_descriptor:='{"dossier_id":'||to_json(p_dossier_id)::text||
   ',"run_id":'||to_json(v_run_id)::text||
   ',"security_id":'||to_json(p_security_id)::text||
   ',"upstream_discovery_artifact_id":null,"upstream_discovery_artifact_version":null}';
  v_binding_fingerprint:=encode(extensions.digest(v_binding_descriptor,'sha256'),'hex');
  v_bind_result:=public.bind_orotitan_run_identity(
   v_run_id,v_run.state_version,p_security_id,p_dossier_id,
   null,null,'orotitan:v3:bind:'||v_binding_fingerprint,
   v_binding_fingerprint);
 elsif v_run.security_id is distinct from p_security_id or
   v_run.dossier_id is distinct from p_dossier_id then
  raise exception 'P0E_PARTIAL_OR_WRONG_RUN_BINDING';
 end if;
 select * into v_run from public.orotitan_runs where run_id=v_run_id;
 if v_run.issuer_id<>p_issuer_id or
   v_run.security_id is distinct from p_security_id or
   v_run.dossier_id is distinct from p_dossier_id or
   v_run.run_type<>'INITIAL' or
   v_run.run_status<>'CREATED' or v_run.current_stage is not null or
   v_run.data_cutoff<>p_data_cutoff or
   v_run.entry_path<>'IMPOSED_COMPANY' or
   v_run.canonical_mode<>'ANALYZE' or
   v_run.contract_set_sha256<>p_contract_set_sha256 or
   v_run.contract_pins<>p_contract_pins or
   v_run.process_version<>'3.1' or
   v_run.pilotage_contract_version<>'3.1' or
   v_run.parent_run_id is not null or
   v_run.baseline_snapshot_id is not null or
   v_run.published_at is not null or
   v_run.cancelled_at is not null or
   v_run.state_version<2 or
   exists (select 1 from public.orotitan_run_stages
    where run_id=v_run_id) then
  raise exception 'P0E_RUN_POST_BIND_INVARIANT_MISMATCH';
 end if;
 insert into public.orotitan_p0e_run_admission_receipts_v1(
  run_id,creation_idempotency_key,issuer_id,security_id,dossier_id,
  data_cutoff,profile_id,contract_set_sha256,verification_receipt,authority)
 values(v_run_id,p_creation_idempotency_key,p_issuer_id,p_security_id,
  p_dossier_id,p_data_cutoff,'OROTITAN_V3_1',p_contract_set_sha256,
  p_verification_receipt,'USER_GO_P0E_SHADOW_INITIAL_V3_1')
 on conflict (run_id) do nothing;
 select * into v_current_receipt from public.orotitan_p0e_run_admission_receipts_v1
  where run_id=v_run_id;
 if not found or
   v_current_receipt.creation_idempotency_key<>p_creation_idempotency_key or
   v_current_receipt.issuer_id<>p_issuer_id or
   v_current_receipt.security_id<>p_security_id or
   v_current_receipt.dossier_id<>p_dossier_id or
   v_current_receipt.data_cutoff<>p_data_cutoff or
   v_current_receipt.contract_set_sha256<>p_contract_set_sha256 then
  raise exception 'P0E_ADMISSION_RECEIPT_CONFLICT';
 end if;
 return jsonb_build_object(
  'status',case when v_result->>'idempotent_replay'='true' then 'REPLAYED'
                 else 'CREATED' end,
  'run_id',v_run_id,'issuer_id',p_issuer_id,
  'security_id',p_security_id,'dossier_id',p_dossier_id,
  'run_type','INITIAL','run_status',v_run.run_status,
  'state_version',v_run.state_version,
  'contract_set_sha256',p_contract_set_sha256,
  'data_cutoff',p_data_cutoff,
  'original_pin_count_verified',15,
  'atomic_creation_and_binding',true,
  'run_stage_started',false,
  'snapshot_created',false,
  'publication_authorized',false);
end;
$$;
revoke all on function public.orotitan_shadow_create_initial_run_v3_1(
 uuid,uuid,uuid,text,text,date,jsonb,text,text,text,jsonb,text)
 from public,anon,authenticated;
grant execute on function public.orotitan_shadow_create_initial_run_v3_1(
 uuid,uuid,uuid,text,text,date,jsonb,text,text,text,jsonb,text)
 to service_role;
commit;

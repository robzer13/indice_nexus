-- D29b PRIVILEGED WRITER PROPOSAL — DO NOT APPLY IN LIVE SUPABASE.
-- Depends on the separately reviewed D29B_SIGNED_RECEIPT_RLS_SQL_REVIEW_ONLY.sql
-- All identifiers are fixtures/placeholders in the public repository.
-- Only a private, separately provisioned service may SET ROLE to this NOLOGIN
-- role. OAuth, anon, authenticated and the pilot Vercel runtime never get it.
begin;

do $preflight$
begin
  if current_setting('server_version_num')::int < 170000
    or to_regclass('public.orotitan_mcp_verified_handoffs') is null
    or to_regclass('public.orotitan_runs') is null
    or to_regclass('public.orotitan_run_stages') is null
  then raise exception 'D29b writer requires reviewed receipt schema and PG17'; end if;
  if exists(select 1 from pg_roles where rolname='orotitan_d29b_writer')
  then raise exception 'D29b writer already exists; manual security review required'; end if;
end;
$preflight$;

create role orotitan_d29b_writer nologin noinherit nobypassrls;
create schema orotitan_d29b_internal;
revoke all on schema orotitan_d29b_internal from public,anon,authenticated;
grant usage on schema orotitan_d29b_internal to orotitan_d29b_writer;
-- Registry permissions are SELECT only; NEVER UPDATE stage/run state.
grant select (run_id,run_status,current_stage,state_version,data_cutoff)
  on public.orotitan_runs to orotitan_d29b_writer;
grant select (run_id,stage_code,state_version,lifecycle_status,
  contract_status_code,handoff_gate_state,active_manifest_kind)
  on public.orotitan_run_stages to orotitan_d29b_writer;

create policy d29b_writer_besi_run_read
on public.orotitan_runs for select to orotitan_d29b_writer
using (run_id='__BESI_RUN_UUID__'::uuid);
create policy d29b_writer_besi_stage_read
on public.orotitan_run_stages for select to orotitan_d29b_writer
using (run_id='__BESI_RUN_UUID__'::uuid);

grant select,insert,update on public.orotitan_mcp_verified_handoffs
  to orotitan_d29b_writer;
create policy d29b_writer_besi_receipt_select
on public.orotitan_mcp_verified_handoffs for select to orotitan_d29b_writer
using (run_id='__BESI_RUN_UUID__'::uuid and
  subject_id='__APPROVED_SUPABASE_AUTH_USER_UUID__'::uuid and
  oauth_client_id='__REGISTERED_CHATGPT_OAUTH_CLIENT_ID__' and
  mcp_resource='https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp');
create policy d29b_writer_besi_receipt_insert
on public.orotitan_mcp_verified_handoffs for insert to orotitan_d29b_writer
with check (run_id='__BESI_RUN_UUID__'::uuid and
  subject_id='__APPROVED_SUPABASE_AUTH_USER_UUID__'::uuid and
  oauth_client_id='__REGISTERED_CHATGPT_OAUTH_CLIENT_ID__' and
  mcp_resource='https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp');
create policy d29b_writer_besi_receipt_update
on public.orotitan_mcp_verified_handoffs for update to orotitan_d29b_writer
using (run_id='__BESI_RUN_UUID__'::uuid and
  subject_id='__APPROVED_SUPABASE_AUTH_USER_UUID__'::uuid and
  oauth_client_id='__REGISTERED_CHATGPT_OAUTH_CLIENT_ID__' and
  mcp_resource='https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp')
with check (run_id='__BESI_RUN_UUID__'::uuid and
  subject_id='__APPROVED_SUPABASE_AUTH_USER_UUID__'::uuid and
  oauth_client_id='__REGISTERED_CHATGPT_OAUTH_CLIENT_ID__' and
  mcp_resource='https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp');

-- Append-only audit events have hashes, not prompt bytes or signed payloads.
create table orotitan_d29b_internal.receipt_audit (
  event_id uuid primary key default gen_random_uuid(),
  receipt_id uuid not null,
  run_id uuid not null,
  event_kind text not null check (event_kind in ('SIGNED_STORED','ADMIN_REVOKED')),
  event_at timestamptz not null default clock_timestamp(),
  payload_sha256 text not null check(payload_sha256 ~ '^[0-9a-f]{64}$'),
  key_id text not null,
  run_state_version bigint,
  stage_state_version bigint
);
revoke all on orotitan_d29b_internal.receipt_audit from public,anon,authenticated;
grant insert on orotitan_d29b_internal.receipt_audit to orotitan_d29b_writer;

-- Security INVOKER: never bypass RLS; private schema not exposed through PostgREST.
-- The privileged service MUST use SERIALIZABLE transaction isolation.
create function orotitan_d29b_internal.put_receipt_v1(
  p_run uuid,p_subject uuid,p_client text,p_resource text,
  p_key text,p_payload text,p_sig text,p_issued timestamptz,p_expires timestamptz,
  p_stage text,p_run_version bigint,p_stage_version bigint,p_cutoff date
) returns uuid
language plpgsql security invoker set search_path=pg_catalog,public,orotitan_d29b_internal
as $function$
declare
  v_run_version bigint; v_stage_version bigint; v_current_stage text;
  v_run_status text; v_cutoff date; v_lifecycle text; v_contract text;
  v_id uuid;
begin
  if current_setting('transaction_isolation')<>'serializable'
  then raise exception 'D29b writer requires SERIALIZABLE transaction'; end if;
  if p_run<>'__BESI_RUN_UUID__'::uuid or
    p_subject<>'__APPROVED_SUPABASE_AUTH_USER_UUID__'::uuid or
    p_client<>'__REGISTERED_CHATGPT_OAUTH_CLIENT_ID__' or
    p_resource<>'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'
  then raise exception 'D29b private writer scope denied'; end if;
  if p_issued<now()-interval '30 seconds' or
    p_issued>now()+interval '5 seconds' or
    p_expires<=now() or p_expires>p_issued+interval '5 minutes'
    or length(p_payload) not between 100 and 24000
    or length(p_sig) not between 80 and 100
  then raise exception 'D29b signature metadata or TTL invalid'; end if;

  perform pg_advisory_xact_lock(hashtextextended(p_run::text,4242));
  select r.state_version,r.current_stage,r.run_status,r.data_cutoff,
    s.state_version,s.lifecycle_status,s.contract_status_code
  into v_run_version,v_current_stage,v_run_status,v_cutoff,
    v_stage_version,v_lifecycle,v_contract
  from public.orotitan_runs r
  join public.orotitan_run_stages s
    on s.run_id=r.run_id and s.stage_code=r.current_stage
  where r.run_id=p_run;
  if not found or v_run_status<>'ACTIVE' or
    v_run_version is distinct from p_run_version or
    v_stage_version is distinct from p_stage_version or
    v_current_stage is distinct from p_stage or
    v_cutoff is distinct from p_cutoff or
    v_lifecycle not in ('IN_PROGRESS','PAUSED','COMPLETE') or
    v_contract like 'BLOCKED_%'
  then raise exception 'D29b Registry CAS changed or blocked'; end if;

  insert into public.orotitan_mcp_verified_handoffs(
    run_id,subject_id,oauth_client_id,mcp_resource,key_id,
    payload_base64url,signature_base64url,issued_at,expires_at,revoked_at)
  values(p_run,p_subject,p_client,p_resource,p_key,p_payload,p_sig,
    p_issued,p_expires,null)
  on conflict on constraint d29b_unique_single_receipt
  do update set receipt_id=gen_random_uuid(),key_id=excluded.key_id,
    payload_base64url=excluded.payload_base64url,
    signature_base64url=excluded.signature_base64url,
    issued_at=excluded.issued_at,expires_at=excluded.expires_at,
    revoked_at=null,created_at=now()
  returning receipt_id into v_id;

  insert into orotitan_d29b_internal.receipt_audit(
    receipt_id,run_id,event_kind,payload_sha256,key_id,run_state_version,
    stage_state_version)
  values(v_id,p_run,'SIGNED_STORED',
    encode(sha256(convert_to(p_payload,'UTF8')),'hex'),
    p_key,p_run_version,p_stage_version);
  return v_id;
end;
$function$;

create function orotitan_d29b_internal.revoke_receipt_v1(
  p_run uuid,p_id uuid
) returns boolean
language plpgsql security invoker set search_path=pg_catalog,public,orotitan_d29b_internal
as $function$
declare v_payload text;v_key text;v_run_version bigint;v_stage_version bigint;
begin
  if current_setting('transaction_isolation')<>'serializable'
  then raise exception 'D29b revocation requires SERIALIZABLE transaction'; end if;
  if p_run<>'__BESI_RUN_UUID__'::uuid then
    raise exception 'D29b revocation run denied';end if;
  perform pg_advisory_xact_lock(hashtextextended(p_run::text,4242));
  update public.orotitan_mcp_verified_handoffs
  set revoked_at=clock_timestamp()
  where run_id=p_run and receipt_id=p_id and revoked_at is null
  returning payload_base64url,key_id into v_payload,v_key;
  if not found then return false;end if;
  select state_version into v_run_version from public.orotitan_runs where run_id=p_run;
  select state_version into v_stage_version from public.orotitan_run_stages
  where run_id=p_run and stage_code=
    (select current_stage from public.orotitan_runs where run_id=p_run);
  insert into orotitan_d29b_internal.receipt_audit(
    receipt_id,run_id,event_kind,payload_sha256,key_id,run_state_version,
    stage_state_version)
  values(p_id,p_run,'ADMIN_REVOKED',
    encode(sha256(convert_to(v_payload,'UTF8')),'hex'),v_key,
    v_run_version,v_stage_version);
  return true;
end;
$function$;

revoke all on function orotitan_d29b_internal.put_receipt_v1(
  uuid,uuid,text,text,text,text,text,timestamptz,timestamptz,
  text,bigint,bigint,date) from public,anon,authenticated;
revoke all on function orotitan_d29b_internal.revoke_receipt_v1(
  uuid,uuid) from public,anon,authenticated;
grant execute on function orotitan_d29b_internal.put_receipt_v1(
  uuid,uuid,text,text,text,text,text,timestamptz,timestamptz,
  text,bigint,bigint,date) to orotitan_d29b_writer;
grant execute on function orotitan_d29b_internal.revoke_receipt_v1(
  uuid,uuid) to orotitan_d29b_writer;
commit;

-- D29b synthetic PG17 serializable writer verification.
-- Database prepared by the isolated D29b writer test runner.
do $preflight$
begin
  if not exists(select 1 from pg_roles where rolname='orotitan_d29b_writer'
    and rolcanlogin=false and rolbypassrls=false)
  then raise exception 'D29b least-privilege NOLOGIN writer absent'; end if;
  if has_table_privilege('authenticated','public.orotitan_mcp_verified_handoffs','INSERT') or
    has_table_privilege('authenticated','public.orotitan_mcp_verified_handoffs','UPDATE') or
    has_schema_privilege('authenticated','orotitan_d29b_internal','USAGE') or
    has_schema_privilege('anon','orotitan_d29b_internal','USAGE') or
    has_table_privilege('orotitan_d29b_writer','public.orotitan_runs','UPDATE') or
    has_table_privilege('orotitan_d29b_writer','public.orotitan_run_stages','UPDATE')
  then raise exception 'D29b privilege escalation detected'; end if;
end;
$preflight$;

set role orotitan_d29b_writer;
begin isolation level serializable;
do $writer$
declare v uuid;
begin
  v:=orotitan_d29b_internal.put_receipt_v1(
    '11111111-1111-4111-8111-111111111111'::uuid,
    '33333333-3333-4333-8333-333333333333'::uuid,
    'pilot-client-fixture',
    'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
    'test-ed25519',repeat('a',240),repeat('b',86),
    now(),now()+interval '2 minutes',
    'RESEARCH',6,4,current_date);
  if v is null then raise exception 'D29b writer did not return receipt ID'; end if;
end;
$writer$;
commit;
reset role;

do $audit$
begin
  if (select count(*) from public.orotitan_mcp_verified_handoffs)<>1 or
    (select count(*) from orotitan_d29b_internal.receipt_audit
     where event_kind='SIGNED_STORED')<>1
  then raise exception 'D29b signed receipt or audit missing';end if;
end;
$audit$;

-- Wrong CAS must fail without replacing receipt or creating audit records.
set role orotitan_d29b_writer;
begin isolation level serializable;
do $deny_cas$
declare err text;
begin
  begin
    perform orotitan_d29b_internal.put_receipt_v1(
      '11111111-1111-4111-8111-111111111111'::uuid,
      '33333333-3333-4333-8333-333333333333'::uuid,
      'pilot-client-fixture',
      'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
      'test-ed25519',repeat('x',240),repeat('b',86),
      now(),now()+interval '2 minutes','RESEARCH',5,4,current_date);
    raise exception 'WRITER_WRONG_CAS_ACCEPTED';
  exception when others then
    get stacked diagnostics err=message_text;
    if err not like '%D29b Registry CAS changed%'
    then raise; end if;
  end;
end;
$deny_cas$;
commit;
reset role;
do $no_audit_change$
begin
  if (select count(*) from orotitan_d29b_internal.receipt_audit)<>1
  then raise exception 'D29b stale CAS should not append audit';end if;
end;
$no_audit_change$;

set role orotitan_d29b_writer;
begin isolation level serializable;
do $revoke$
declare v uuid;done boolean;
begin
  select receipt_id into v from public.orotitan_mcp_verified_handoffs
    where run_id='11111111-1111-4111-8111-111111111111'::uuid;
  done:=orotitan_d29b_internal.revoke_receipt_v1(
    '11111111-1111-4111-8111-111111111111'::uuid,v);
  if done is distinct from true then raise exception 'D29b revoke failed';end if;
end;
$revoke$;
commit;
reset role;

do $final$
begin
  if (select count(*) from orotitan_d29b_internal.receipt_audit)<>2
    or (select count(*) from orotitan_d29b_internal.receipt_audit
      where event_kind='ADMIN_REVOKED')<>1
    or (select count(*) from public.orotitan_mcp_verified_handoffs
      where revoked_at is not null)<>1
  then raise exception 'D29b revoke audit or visibility defect';end if;
end;
$final$;

-- Caller JWT allowed by original OAuth identity and client no longer sees it.
set role authenticated;
select set_config('request.jwt.claims',
  jsonb_build_object('sub','33333333-3333-4333-8333-333333333333',
    'client_id','pilot-client-fixture',
    'resource','https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp')::text,
    false);
do $not_visible$
begin
  if (select count(*) from public.orotitan_mcp_verified_handoffs)<>0
  then raise exception 'D29b revoked receipt still visible to OAuth';end if;
end;
$not_visible$;
reset role;

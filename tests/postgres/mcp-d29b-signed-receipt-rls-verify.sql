-- D29b RLS test-only values; no real identifiers or secrets.
-- Should be loaded after the proposed SQL in a disposable local PG17 DB.
insert into public.orotitan_mcp_verified_handoffs
(run_id,subject_id,oauth_client_id,mcp_resource,key_id,payload_base64url,
 signature_base64url,issued_at,expires_at)
values
('11111111-1111-4111-8111-111111111111',
 '33333333-3333-4333-8333-333333333333',
 'pilot-client-fixture',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 'test-key',repeat('a',200),repeat('b',86),
 now(),now()+interval '2 minutes'),
('22222222-2222-4222-8222-222222222222',
 '33333333-3333-4333-8333-333333333333',
 'pilot-client-fixture',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 'test-key',repeat('a',200),repeat('b',86),
 now(),now()+interval '2 minutes');

set role authenticated;
do $d29b_matrix$
declare scenario record; n integer;
begin
  for scenario in select * from (values
    ('allowed',jsonb_build_object(
      'sub','33333333-3333-4333-8333-333333333333',
      'client_id','pilot-client-fixture',
      'resource','https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'
    ),1),
    ('wrong-subject',jsonb_build_object(
      'sub','44444444-4444-4444-8444-444444444444',
      'client_id','pilot-client-fixture',
      'resource','https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'
    ),0),
    ('wrong-client',jsonb_build_object(
      'sub','33333333-3333-4333-8333-333333333333',
      'client_id','evil-client',
      'resource','https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'
    ),0),
    ('wrong-resource',jsonb_build_object(
      'sub','33333333-3333-4333-8333-333333333333',
      'client_id','pilot-client-fixture',
      'resource','https://other.example/mcp'
    ),0),
    ('ordinary-auth',jsonb_build_object(
      'sub','33333333-3333-4333-8333-333333333333'
    ),0)
  ) as v(label,claims,expected)
  loop
    perform set_config('request.jwt.claims',scenario.claims::text,false);
    select count(*) into n from public.orotitan_mcp_verified_handoffs;
    if n<>scenario.expected then
      raise exception 'D29b: % visible receipts % expected %',scenario.label,n,scenario.expected;
    end if;
  end loop;
end;
$d29b_matrix$;
reset role;

do $grants$
begin
  if has_table_privilege('authenticated','public.orotitan_mcp_verified_handoffs','SELECT')
     or has_table_privilege('authenticated','public.orotitan_mcp_verified_handoffs','INSERT')
     or has_table_privilege('authenticated','public.orotitan_mcp_verified_handoffs','UPDATE')
     or has_table_privilege('authenticated','public.orotitan_mcp_verified_handoffs','DELETE')
     or has_table_privilege('anon','public.orotitan_mcp_verified_handoffs','SELECT')
  then raise exception 'D29b broad/read/write permission unexpected'; end if;
  if not has_column_privilege('authenticated',
      'public.orotitan_mcp_verified_handoffs','payload_base64url','SELECT')
  then raise exception 'D29b intended receipt fields unreadable'; end if;
  if has_column_privilege('authenticated',
      'public.orotitan_mcp_verified_handoffs','subject_id','SELECT')
  then raise exception 'D29b identity columns unexpectedly exposed'; end if;
end;
$grants$;

-- Revocation must hide an already issued row with no change to OAuth token.
update public.orotitan_mcp_verified_handoffs
set revoked_at=now()
where run_id='11111111-1111-4111-8111-111111111111';
set role authenticated;
select set_config('request.jwt.claims',
 jsonb_build_object(
  'sub','33333333-3333-4333-8333-333333333333',
  'client_id','pilot-client-fixture',
  'resource','https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'
 )::text,false);
do $revoked$
declare n integer;
begin
  select count(*) into n from public.orotitan_mcp_verified_handoffs;
  if n<>0 then raise exception 'D29b revoked receipt was still visible'; end if;
end;
$revoked$;
reset role;

-- D28 local test only: run as PostgreSQL authenticated role.
-- Never reuse fixture claims as real Supabase credentials.
set role authenticated;
do $d28_matrix$
declare
  scenario record;
  row_count integer;
begin
  for scenario in
    select * from (values
      ('allowed',jsonb_build_object(
        'sub','33333333-3333-4333-8333-333333333333',
        'client_id','pilot-client-fixture',
        'resource','https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'
      ),1),
      ('wrong-user',jsonb_build_object(
        'sub','44444444-4444-4444-8444-444444444444',
        'client_id','pilot-client-fixture',
        'resource','https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'
      ),0),
      ('wrong-client',jsonb_build_object(
        'sub','33333333-3333-4333-8333-333333333333',
        'client_id','unapproved-client',
        'resource','https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp'
      ),0),
      ('wrong-resource',jsonb_build_object(
        'sub','33333333-3333-4333-8333-333333333333',
        'client_id','pilot-client-fixture',
        'resource','https://unrelated.example/api/orotitan/mcp'
      ),0),
      ('normal-supabase-session',jsonb_build_object(
        'sub','33333333-3333-4333-8333-333333333333'
      ),0),
      ('anonymous-claims','{}'::jsonb,0)
    ) as v(label,claims,expected)
  loop
    perform set_config('request.jwt.claims',scenario.claims::text,false);
    select count(*) into row_count from public.orotitan_runs;
    if row_count <> scenario.expected then
      raise exception 'D28 failed: % run count % != %',scenario.label,row_count,scenario.expected;
    end if;
    select count(*) into row_count from public.orotitan_run_stages;
    if row_count <> scenario.expected then
      raise exception 'D28 failed: % stage count % != %',scenario.label,row_count,scenario.expected;
    end if;
    select count(*) into row_count from public.issuers;
    if row_count <> scenario.expected then
      raise exception 'D28 failed: % issuer count % != %',scenario.label,row_count,scenario.expected;
    end if;
    select count(*) into row_count from public.securities;
    if row_count <> scenario.expected then
      raise exception 'D28 failed: % security count % != %',scenario.label,row_count,scenario.expected;
    end if;
  end loop;
end;
$d28_matrix$;
reset role;

do $d28_grants$
declare
  target text;
begin
  foreach target in array array[
    'orotitan_runs','orotitan_run_stages','issuers','securities'
  ] loop
    if has_table_privilege('authenticated',format('public.%I',target),'SELECT')
       or has_table_privilege('authenticated',format('public.%I',target),'INSERT')
       or has_table_privilege('authenticated',format('public.%I',target),'UPDATE')
       or has_table_privilege('authenticated',format('public.%I',target),'DELETE')
       or has_table_privilege('anon',format('public.%I',target),'SELECT')
    then raise exception 'D28 broad grant: %',target; end if;
    if (select count(*) from pg_policies
         where schemaname='public' and tablename=target and
           policyname='orotitan_mcp_besi_read_v1' and cmd='SELECT' and
           roles='{authenticated}'::name[]) <> 1
    then raise exception 'D28 policy missing for %',target; end if;
  end loop;
  if not has_column_privilege('authenticated','public.orotitan_runs','run_id','SELECT')
     or not has_column_privilege('authenticated','public.orotitan_run_stages','blocker_summary','SELECT')
     or not has_column_privilege('authenticated','public.issuers','display_name','SELECT')
     or not has_column_privilege('authenticated','public.securities','ticker','SELECT')
  then raise exception 'D28 intended columns not readable'; end if;
  if has_column_privilege('authenticated','public.orotitan_runs','creation_idempotency_key','SELECT')
     or has_column_privilege('authenticated','public.issuers','internal_secret','SELECT')
     or has_column_privilege('authenticated','public.securities','internal_secret','SELECT')
  then raise exception 'D28 protected columns readable'; end if;
end;
$d28_grants$;

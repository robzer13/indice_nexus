-- Capture physical and logical history before the forward migration.
create table public.method_generation_test_history as
select run_id, ctid::text as physical_row, to_jsonb(r) as original_row
from public.orotitan_runs r;
create table public.method_generation_test_functions as
select oid, proacl from pg_proc where pronamespace = 'public'::regnamespace;
create table public.method_generation_test_security as
select oid, relacl, relrowsecurity, relforcerowsecurity from pg_class
where relnamespace = 'public'::regnamespace
  and (relname like 'orotitan_%' or relname = 'research_snapshots');
create table public.method_generation_test_policies as select * from pg_policy;
create table public.method_generation_test_snapshots as
select snapshot_id, to_jsonb(s) as original_row from public.research_snapshots s;

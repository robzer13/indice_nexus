-- OroTitan Registry V1 — production read-only postflight
-- SAFE ON LIVE: no persistent mutation. Run only after all five Registry migrations.
-- Compare returned legacy/identity hashes with the immediately preceding preflight output.

begin;
set transaction read only;

with registry_tables as (
  select c.relname, c.relrowsecurity
  from pg_class c
  join pg_namespace ns on ns.oid = c.relnamespace
  where ns.nspname = 'public'
    and c.relname in (
      'orotitan_runs','orotitan_run_stages','orotitan_artifacts',
      'orotitan_artifact_edges','orotitan_run_events'
    )
    and c.relkind = 'r'
),
registry_rows as (
  select jsonb_build_object(
    'orotitan_runs', (select count(*) from public.orotitan_runs),
    'orotitan_run_stages', (select count(*) from public.orotitan_run_stages),
    'orotitan_artifacts', (select count(*) from public.orotitan_artifacts),
    'orotitan_artifact_edges', (select count(*) from public.orotitan_artifact_edges),
    'orotitan_run_events', (select count(*) from public.orotitan_run_events)
  ) as counts
),
policy_count as (
  select count(*)::int as n
  from pg_policies
  where schemaname = 'public'
    and tablename in (
      'orotitan_runs','orotitan_run_stages','orotitan_artifacts',
      'orotitan_artifact_edges','orotitan_run_events'
    )
),
write_grants as (
  select count(*)::int as n
  from information_schema.role_table_grants
  where table_schema = 'public'
    and table_name in (
      'orotitan_runs','orotitan_run_stages','orotitan_artifacts',
      'orotitan_artifact_edges','orotitan_run_events'
    )
    and grantee in ('anon','authenticated','service_role')
    and privilege_type in ('INSERT','UPDATE','DELETE','TRUNCATE')
),
rpcs as (
  select p.oid::regprocedure as signature, p.prosecdef, p.proconfig
  from pg_proc p
  join pg_namespace ns on ns.oid = p.pronamespace
  where ns.nspname = 'public'
    and p.proname in (
      'create_orotitan_run','bind_orotitan_run_identity','start_orotitan_stage',
      'checkpoint_orotitan_stage','pause_orotitan_stage','resume_orotitan_stage',
      'finalize_orotitan_stage','reopen_orotitan_stage','resolve_orotitan_artifact',
      'record_orotitan_publish_authorization','record_orotitan_publish_result'
    )
),
rpc_bad_security as (
  select count(*)::int as n
  from rpcs
  where not prosecdef
     or not (proconfig @> array['search_path=pg_catalog, public']::text[])
),
rpc_bad_privileges as (
  select count(*)::int as n
  from rpcs
  where has_function_privilege('public', signature::text, 'EXECUTE')
     or has_function_privilege('anon', signature::text, 'EXECUTE')
     or has_function_privilege('authenticated', signature::text, 'EXECUTE')
     or not has_function_privilege('service_role', signature::text, 'EXECUTE')
),
canonical_hashes as (
  select
    md5(
      (select string_agg(row_to_json(c)::text, ',' order by c.id) from public.companies c) ||
      (select string_agg(row_to_json(s)::text, ',' order by s.id) from public.snapshots s) ||
      (select string_agg(row_to_json(p)::text, ',' order by p.id) from public.market_prices p) ||
      (select string_agg(row_to_json(r)::text, ',' order by r.id) from public.market_sync_runs r)
    ) as legacy_md5,
    md5(
      (select string_agg(row_to_json(i)::text, ',' order by i.issuer_id) from public.issuers i) ||
      (select string_agg(row_to_json(s)::text, ',' order by s.security_id) from public.securities s) ||
      (select string_agg(row_to_json(d)::text, ',' order by d.dossier_id) from public.research_dossiers d) ||
      (select string_agg(row_to_json(m)::text, ',' order by m.legacy_company_id) from public.legacy_company_identity_map m)
    ) as identity_md5
),
constraint_gate as (
  select
    exists(select 1 from pg_constraint where conname = 'orotitan_run_stages_active_manifest_same_stage_fkey' and condeferrable) as active_manifest_fk,
    exists(select 1 from pg_constraint where conname = 'orotitan_artifacts_manifest_same_stage_fkey' and condeferrable) as artifact_manifest_fk,
    exists(select 1 from pg_constraint where conname = 'orotitan_runs_contract_pins_complete_check') as contract_pin_check
)
select jsonb_pretty(jsonb_build_object(
  'registry_tables', (select count(*) from registry_tables),
  'all_registry_tables_rls', (select count(*) = 5 and bool_and(relrowsecurity) from registry_tables),
  'registry_policies', (select n from policy_count),
  'direct_registry_write_grants', (select n from write_grants),
  'controlled_rpcs', (select count(*) from rpcs),
  'rpc_security_failures', (select n from rpc_bad_security),
  'rpc_privilege_failures', (select n from rpc_bad_privileges),
  'status_view_present', to_regclass('public.orotitan_run_status_view') is not null,
  'active_manifest_deferred_fk', (select active_manifest_fk from constraint_gate),
  'artifact_manifest_deferred_fk', (select artifact_manifest_fk from constraint_gate),
  'contract_pin_completeness_check', (select contract_pin_check from constraint_gate),
  'registry_rows', (select counts from registry_rows),
  'legacy_rowset_md5', (select legacy_md5 from canonical_hashes),
  'identity_rowset_md5', (select identity_md5 from canonical_hashes),
  'postflight_structure_pass',
    (select count(*) = 5 and bool_and(relrowsecurity) from registry_tables)
    and (select n = 0 from policy_count)
    and (select n = 0 from write_grants)
    and (select count(*) = 11 from rpcs)
    and (select n = 0 from rpc_bad_security)
    and (select n = 0 from rpc_bad_privileges)
    and to_regclass('public.orotitan_run_status_view') is not null
    and (select active_manifest_fk and artifact_manifest_fk and contract_pin_check from constraint_gate)
    and (select
      (counts->>'orotitan_runs')::bigint = 0
      and (counts->>'orotitan_run_stages')::bigint = 0
      and (counts->>'orotitan_artifacts')::bigint = 0
      and (counts->>'orotitan_artifact_edges')::bigint = 0
      and (counts->>'orotitan_run_events')::bigint = 0
      from registry_rows)
)) as orotitan_registry_live_postflight;

rollback;

-- OroTitan Registry V1 — production read-only preflight
-- SAFE ON LIVE: no persistent mutation. Run immediately before any Registry DDL.

begin;
set transaction read only;

with registry_relations as (
  select count(*)::int as n
  from pg_class c
  join pg_namespace ns on ns.oid = c.relnamespace
  where ns.nspname = 'public'
    and c.relname in (
      'orotitan_runs','orotitan_run_stages','orotitan_artifacts',
      'orotitan_artifact_edges','orotitan_run_events','orotitan_run_status_view'
    )
),
registry_functions as (
  select count(*)::int as n
  from pg_proc p
  join pg_namespace ns on ns.oid = p.pronamespace
  where ns.nspname = 'public'
    and (
      p.proname like 'orotitan_%'
      or p.proname in (
        'enforce_orotitan_run_update','enforce_orotitan_stage_update',
        'enforce_orotitan_artifact_update','reject_orotitan_history_delete',
        'reject_orotitan_event_mutation','supersede_orotitan_manifest_bundle',
        'create_orotitan_run','bind_orotitan_run_identity','start_orotitan_stage',
        'checkpoint_orotitan_stage','pause_orotitan_stage','resume_orotitan_stage',
        'finalize_orotitan_stage','reopen_orotitan_stage','resolve_orotitan_artifact',
        'record_orotitan_publish_authorization','record_orotitan_publish_result'
      )
    )
),
snapshot_pair_index as (
  select exists (
    select 1
    from pg_class idx
    join pg_namespace ns on ns.oid = idx.relnamespace
    join pg_index ix on ix.indexrelid = idx.oid
    where ns.nspname = 'public'
      and idx.relname = 'research_snapshots_snapshot_dossier_key'
      and ix.indisunique
      and ix.indisvalid
      and ix.indisready
      and ix.indpred is null
  ) as ok
),
current_snapshot_fk as (
  select exists (
    select 1
    from pg_constraint con
    join pg_class c on c.oid = con.conrelid
    join pg_namespace ns on ns.oid = c.relnamespace
    where ns.nspname = 'public'
      and c.relname = 'research_dossiers'
      and con.conname = 'research_dossiers_current_snapshot_fkey'
      and con.contype = 'f'
  ) as ok
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
storage_gate as (
  select jsonb_build_object(
    'text_bucket_private', exists (
      select 1 from storage.buckets
      where id = 'orotitan-text-artifacts-v1' and public = false
    ),
    'source_bucket_private', exists (
      select 1 from storage.buckets
      where id = 'orotitan-source-files-v1' and public = false
    ),
    'text_objects', (select count(*) from storage.objects where bucket_id = 'orotitan-text-artifacts-v1'),
    'source_objects', (select count(*) from storage.objects where bucket_id = 'orotitan-source-files-v1'),
    'matching_policies', (
      select count(*) from pg_policies
      where schemaname = 'storage' and tablename = 'objects'
        and (
          coalesce(qual,'') like '%orotitan-text-artifacts-v1%'
          or coalesce(qual,'') like '%orotitan-source-files-v1%'
          or coalesce(with_check,'') like '%orotitan-text-artifacts-v1%'
          or coalesce(with_check,'') like '%orotitan-source-files-v1%'
        )
    )
  ) as state
),
long_transactions as (
  select count(*)::int as n
  from pg_stat_activity
  where pid <> pg_backend_pid()
    and xact_start is not null
    and xact_start < now() - interval '5 minutes'
)
select jsonb_pretty(jsonb_build_object(
  'server_version', current_setting('server_version'),
  'pgcrypto_digest_present', to_regprocedure('extensions.digest(bytea,text)') is not null,
  'canonical_dependencies_present',
    to_regclass('public.issuers') is not null
    and to_regclass('public.securities') is not null
    and to_regclass('public.research_dossiers') is not null
    and to_regclass('public.research_snapshots') is not null,
  'snapshot_dossier_unique_index', (select ok from snapshot_pair_index),
  'current_snapshot_fk_present', (select ok from current_snapshot_fk),
  'registry_relations_absent', (select n = 0 from registry_relations),
  'registry_functions_absent', (select n = 0 from registry_functions),
  'private_storage', (select state from storage_gate),
  'long_transactions_over_5m', (select n from long_transactions),
  'row_counts', jsonb_build_object(
    'companies', (select count(*) from public.companies),
    'snapshots', (select count(*) from public.snapshots),
    'market_prices', (select count(*) from public.market_prices),
    'market_sync_runs', (select count(*) from public.market_sync_runs),
    'issuers', (select count(*) from public.issuers),
    'securities', (select count(*) from public.securities),
    'research_dossiers', (select count(*) from public.research_dossiers),
    'legacy_company_identity_map', (select count(*) from public.legacy_company_identity_map),
    'research_snapshots', (select count(*) from public.research_snapshots)
  ),
  'legacy_rowset_md5', (select legacy_md5 from canonical_hashes),
  'identity_rowset_md5', (select identity_md5 from canonical_hashes),
  'database_gate_pass',
    to_regprocedure('extensions.digest(bytea,text)') is not null
    and to_regclass('public.issuers') is not null
    and to_regclass('public.securities') is not null
    and to_regclass('public.research_dossiers') is not null
    and to_regclass('public.research_snapshots') is not null
    and (select ok from snapshot_pair_index)
    and (select ok from current_snapshot_fk)
    and (select n = 0 from registry_relations)
    and (select n = 0 from registry_functions)
    and (select n = 0 from long_transactions)
    and (select (state->>'text_bucket_private')::boolean from storage_gate)
    and (select (state->>'source_bucket_private')::boolean from storage_gate)
    and (select (state->>'text_objects')::bigint = 0 from storage_gate)
    and (select (state->>'source_objects')::bigint = 0 from storage_gate)
    and (select (state->>'matching_policies')::bigint = 0 from storage_gate)
)) as orotitan_registry_live_preflight;

rollback;

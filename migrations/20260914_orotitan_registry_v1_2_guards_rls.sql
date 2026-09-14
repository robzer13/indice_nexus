-- OroTitan Registry V1.2 — guards, RLS, privilege firewall.
-- CODE-ONLY IMPLEMENTATION. NOT APPLIED TO PRODUCTION.

begin;

do $$
begin
  if to_regclass('public.orotitan_runs') is null
     or to_regclass('public.orotitan_run_stages') is null
     or to_regclass('public.orotitan_artifacts') is null
     or to_regclass('public.orotitan_artifact_edges') is null
     or to_regclass('public.orotitan_run_events') is null then
    raise exception 'OroTitan registry core migration must exist before guards/RLS';
  end if;
end;
$$;

create or replace function public.enforce_orotitan_run_update()
returns trigger
language plpgsql
set search_path = pg_catalog, public
as $$
begin
  if new.run_id is distinct from old.run_id
     or new.creation_idempotency_key is distinct from old.creation_idempotency_key
     or new.run_scope is distinct from old.run_scope
     or new.parent_run_id is distinct from old.parent_run_id
     or new.baseline_snapshot_id is distinct from old.baseline_snapshot_id
     or new.entry_path is distinct from old.entry_path
     or new.canonical_mode is distinct from old.canonical_mode
     or new.run_type is distinct from old.run_type
     or new.data_cutoff is distinct from old.data_cutoff
     or new.process_version is distinct from old.process_version
     or new.pilotage_contract_version is distinct from old.pilotage_contract_version
     or new.contract_pins is distinct from old.contract_pins
     or new.contract_set_sha256 is distinct from old.contract_set_sha256
     or new.created_at is distinct from old.created_at then
    raise exception 'immutable OroTitan run lock field cannot be changed'
      using errcode = '23514';
  end if;

  if old.issuer_id is not null and new.issuer_id is distinct from old.issuer_id then
    raise exception 'issuer_id cannot be rebound within a run' using errcode = '23514';
  end if;
  if old.security_id is not null and new.security_id is distinct from old.security_id then
    raise exception 'security_id cannot be rebound within a run' using errcode = '23514';
  end if;
  if old.dossier_id is not null and new.dossier_id is distinct from old.dossier_id then
    raise exception 'dossier_id cannot be rebound within a run' using errcode = '23514';
  end if;
  if old.upstream_discovery_artifact_id is not null and (
      new.upstream_discovery_artifact_id is distinct from old.upstream_discovery_artifact_id
      or new.upstream_discovery_artifact_version is distinct from old.upstream_discovery_artifact_version
  ) then
    raise exception 'upstream Discovery artifact cannot be rebound within a run'
      using errcode = '23514';
  end if;

  if old.run_status in ('PUBLISHED', 'CANCELLED') and new.run_status is distinct from old.run_status then
    raise exception 'terminal OroTitan run status cannot be changed'
      using errcode = '23514';
  end if;

  if new.run_status = 'PUBLISHED' and new.published_at is null then
    raise exception 'PUBLISHED run requires published_at' using errcode = '23514';
  end if;
  if new.run_status = 'CANCELLED' and new.cancelled_at is null then
    raise exception 'CANCELLED run requires cancelled_at' using errcode = '23514';
  end if;

  if new.state_version < old.state_version then
    raise exception 'run state_version cannot decrease' using errcode = '23514';
  end if;

  new.updated_at := now();
  return new;
end;
$$;

drop trigger if exists orotitan_runs_enforce_update on public.orotitan_runs;
create trigger orotitan_runs_enforce_update
before update on public.orotitan_runs
for each row execute function public.enforce_orotitan_run_update();

create or replace function public.enforce_orotitan_stage_update()
returns trigger
language plpgsql
set search_path = pg_catalog, public
as $$
begin
  if new.run_id is distinct from old.run_id
     or new.stage_code is distinct from old.stage_code
     or new.stage_contract_name is distinct from old.stage_contract_name
     or new.stage_contract_version is distinct from old.stage_contract_version
     or new.stage_contract_sha256 is distinct from old.stage_contract_sha256
     or new.handoff_gate_name is distinct from old.handoff_gate_name
     or new.started_at is distinct from old.started_at and old.started_at is not null then
    raise exception 'immutable OroTitan stage identity/contract field cannot be changed'
      using errcode = '23514';
  end if;

  if new.stage_revision < old.stage_revision or new.stage_revision > old.stage_revision + 1 then
    raise exception 'stage_revision must stay constant or increase by exactly one'
      using errcode = '23514';
  end if;

  if old.lifecycle_status = 'COMPLETE'
     and new.lifecycle_status <> 'COMPLETE'
     and not (
       new.stage_revision = old.stage_revision + 1
       and new.lifecycle_status in ('IN_PROGRESS', 'BLOCKED')
       and new.handoff_gate_state = 'NOT_EVALUATED'
       and new.completed_at is null
     ) then
    raise exception 'completed stage can only reopen through a new revision'
      using errcode = '23514';
  end if;

  if new.lifecycle_status = 'COMPLETE' and (
      new.active_manifest_kind is distinct from 'FINAL'
      or new.active_manifest_artifact_id is null
      or new.active_manifest_version is null
      or new.completed_at is null
  ) then
    raise exception 'COMPLETE stage requires FINAL active manifest and completed_at'
      using errcode = '23514';
  end if;

  if new.active_manifest_kind = 'CHECKPOINT' and new.handoff_gate_state = 'YES' then
    raise exception 'CHECKPOINT manifest cannot admit downstream stage'
      using errcode = '23514';
  end if;

  if new.state_version < old.state_version then
    raise exception 'stage state_version cannot decrease' using errcode = '23514';
  end if;

  new.updated_at := now();
  return new;
end;
$$;

drop trigger if exists orotitan_run_stages_enforce_update on public.orotitan_run_stages;
create trigger orotitan_run_stages_enforce_update
before update on public.orotitan_run_stages
for each row execute function public.enforce_orotitan_stage_update();

create or replace function public.enforce_orotitan_artifact_update()
returns trigger
language plpgsql
set search_path = pg_catalog, public
as $$
begin
  if new.artifact_id is distinct from old.artifact_id
     or new.version is distinct from old.version
     or new.run_id is distinct from old.run_id
     or new.stage_code is distinct from old.stage_code
     or new.artifact_type is distinct from old.artifact_type
     or new.logical_name is distinct from old.logical_name
     or new.authority_class is distinct from old.authority_class
     or new.media_type is distinct from old.media_type
     or new.size_bytes is distinct from old.size_bytes
     or new.hash_algorithm is distinct from old.hash_algorithm
     or new.content_sha256 is distinct from old.content_sha256
     or new.storage_backend is distinct from old.storage_backend
     or new.storage_uri is distinct from old.storage_uri
     or new.github_repository is distinct from old.github_repository
     or new.github_path is distinct from old.github_path
     or new.github_commit_sha is distinct from old.github_commit_sha
     or new.github_blob_sha is distinct from old.github_blob_sha
     or new.supabase_bucket is distinct from old.supabase_bucket
     or new.supabase_object_path is distinct from old.supabase_object_path
     or new.created_at is distinct from old.created_at
     or new.sealed_at is distinct from old.sealed_at then
    raise exception 'immutable OroTitan artifact content/provenance fields cannot be changed'
      using errcode = '23514';
  end if;

  if old.artifact_status = 'INVALIDATED' and new.artifact_status <> 'INVALIDATED' then
    raise exception 'INVALIDATED artifact cannot return to SEALED' using errcode = '23514';
  end if;

  return new;
end;
$$;

drop trigger if exists orotitan_artifacts_enforce_update on public.orotitan_artifacts;
create trigger orotitan_artifacts_enforce_update
before update on public.orotitan_artifacts
for each row execute function public.enforce_orotitan_artifact_update();

create or replace function public.reject_orotitan_history_delete()
returns trigger
language plpgsql
set search_path = pg_catalog, public
as $$
begin
  raise exception 'OroTitan registry audit/history rows are append-preserved'
    using errcode = '55000';
end;
$$;

create or replace function public.reject_orotitan_event_mutation()
returns trigger
language plpgsql
set search_path = pg_catalog, public
as $$
begin
  raise exception 'OroTitan run events are append-only'
    using errcode = '55000';
end;
$$;

drop trigger if exists orotitan_runs_prevent_delete on public.orotitan_runs;
create trigger orotitan_runs_prevent_delete
before delete on public.orotitan_runs
for each row execute function public.reject_orotitan_history_delete();

drop trigger if exists orotitan_run_stages_prevent_delete on public.orotitan_run_stages;
create trigger orotitan_run_stages_prevent_delete
before delete on public.orotitan_run_stages
for each row execute function public.reject_orotitan_history_delete();

drop trigger if exists orotitan_artifacts_prevent_delete on public.orotitan_artifacts;
create trigger orotitan_artifacts_prevent_delete
before delete on public.orotitan_artifacts
for each row execute function public.reject_orotitan_history_delete();

drop trigger if exists orotitan_artifact_edges_prevent_delete on public.orotitan_artifact_edges;
create trigger orotitan_artifact_edges_prevent_delete
before delete on public.orotitan_artifact_edges
for each row execute function public.reject_orotitan_history_delete();

drop trigger if exists orotitan_run_events_prevent_mutation on public.orotitan_run_events;
create trigger orotitan_run_events_prevent_mutation
before update or delete on public.orotitan_run_events
for each row execute function public.reject_orotitan_event_mutation();

alter table public.orotitan_runs enable row level security;
alter table public.orotitan_run_stages enable row level security;
alter table public.orotitan_artifacts enable row level security;
alter table public.orotitan_artifact_edges enable row level security;
alter table public.orotitan_run_events enable row level security;

revoke all on table public.orotitan_runs from public, anon, authenticated, service_role;
revoke all on table public.orotitan_run_stages from public, anon, authenticated, service_role;
revoke all on table public.orotitan_artifacts from public, anon, authenticated, service_role;
revoke all on table public.orotitan_artifact_edges from public, anon, authenticated, service_role;
revoke all on table public.orotitan_run_events from public, anon, authenticated, service_role;

grant select on table public.orotitan_runs to service_role;
grant select on table public.orotitan_run_stages to service_role;
grant select on table public.orotitan_artifacts to service_role;
grant select on table public.orotitan_artifact_edges to service_role;
grant select on table public.orotitan_run_events to service_role;

revoke all on function public.enforce_orotitan_run_update() from public, anon, authenticated, service_role;
revoke all on function public.enforce_orotitan_stage_update() from public, anon, authenticated, service_role;
revoke all on function public.enforce_orotitan_artifact_update() from public, anon, authenticated, service_role;
revoke all on function public.reject_orotitan_history_delete() from public, anon, authenticated, service_role;
revoke all on function public.reject_orotitan_event_mutation() from public, anon, authenticated, service_role;

commit;

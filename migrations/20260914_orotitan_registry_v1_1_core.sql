-- OROTITAN REGISTRY DDL CANDIDATE V1
-- Generated: 2026-09-14
-- Status: IMPLEMENTATION DDL MIGRATION CANDIDATE. NOT APPLIED TO PRODUCTION.
-- Authority: OROTITAN_RUN_ARTIFACT_REGISTRY_MANIFEST_SPEC_V1_FREEZE_V1.0
-- Live baseline fingerprint before design:
-- eabda3bce8cf43ea40076cc53fd45e1a1dcfda563f499a88ae618bf040c30b91
--
-- Additive only. No production snapshot/data mutation.

begin;

do $$
begin
  if to_regclass('public.issuers') is null
     or to_regclass('public.securities') is null
     or to_regclass('public.research_dossiers') is null
     or to_regclass('public.research_snapshots') is null then
    raise exception 'OroTitan canonical identity/snapshot baseline is missing';
  end if;
  if to_regclass('public.orotitan_runs') is not null
     or to_regclass('public.orotitan_run_stages') is not null
     or to_regclass('public.orotitan_artifacts') is not null
     or to_regclass('public.orotitan_artifact_edges') is not null
     or to_regclass('public.orotitan_run_events') is not null then
    raise exception 'OroTitan registry tables already exist; refusing non-reviewed reapplication';
  end if;
end;
$$;

create table public.orotitan_runs (
  run_id uuid primary key default gen_random_uuid(),
  run_display_key text unique,
  creation_idempotency_key text not null unique,
  run_scope text not null default 'COMPANY_ANALYSIS' check (run_scope = 'COMPANY_ANALYSIS'),
  issuer_id uuid references public.issuers(issuer_id) on delete restrict,
  security_id uuid,
  dossier_id uuid,
  parent_run_id uuid references public.orotitan_runs(run_id) on delete restrict,
  baseline_snapshot_id uuid,
  upstream_discovery_artifact_id uuid,
  upstream_discovery_artifact_version integer check (upstream_discovery_artifact_version is null or upstream_discovery_artifact_version >= 1),
  entry_path text not null check (entry_path in ('IMPOSED_COMPANY', 'DISCOVERY_TO_DECISION')),
  canonical_mode text not null check (canonical_mode in ('DISCOVER', 'ANALYZE', 'DISCOVER + ANALYZE', 'REFRESH', 'ACTIVATION CHECK')),
  run_type text check (run_type is null or run_type in ('INITIAL', 'REFRESH')),
  run_status text not null default 'CREATED' check (run_status in ('CREATED', 'ACTIVE', 'PAUSED', 'BLOCKED', 'READY_TO_PUBLISH', 'PUBLISHED', 'CANCELLED')),
  current_stage text check (current_stage is null or current_stage in ('RESEARCH', 'DEEP_DIVE', 'INTEGRATION')),
  data_cutoff date not null,
  process_version text not null check (length(btrim(process_version)) > 0),
  pilotage_contract_version text not null check (length(btrim(pilotage_contract_version)) > 0),
  contract_pins jsonb not null check (jsonb_typeof(contract_pins) = 'object'),
  contract_set_sha256 text not null check (contract_set_sha256 ~ '^[0-9a-f]{64}$'),
  state_version bigint not null default 1 check (state_version >= 1),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  published_at timestamptz,
  cancelled_at timestamptz,
  foreign key (security_id, issuer_id) references public.securities(security_id, issuer_id) on delete restrict,
  foreign key (dossier_id, issuer_id) references public.research_dossiers(dossier_id, issuer_id) on delete restrict,
  foreign key (baseline_snapshot_id, dossier_id) references public.research_snapshots(snapshot_id, dossier_id) on delete restrict,
  check (security_id is null or issuer_id is not null),
  check (dossier_id is null or issuer_id is not null),
  check (baseline_snapshot_id is null or dossier_id is not null),
  check ((upstream_discovery_artifact_id is null) = (upstream_discovery_artifact_version is null)),
  check ((run_status = 'PUBLISHED') = (published_at is not null) or run_status <> 'PUBLISHED'),
  check ((run_status = 'CANCELLED') = (cancelled_at is not null) or run_status <> 'CANCELLED')
);

comment on table public.orotitan_runs is 'OroTitan persistent company-analysis run registry. Operational truth only; not analytical truth.';
create index orotitan_runs_issuer_created_idx on public.orotitan_runs (issuer_id, created_at desc) where issuer_id is not null;
create index orotitan_runs_dossier_created_idx on public.orotitan_runs (dossier_id, created_at desc) where dossier_id is not null;
create index orotitan_runs_parent_idx on public.orotitan_runs (parent_run_id) where parent_run_id is not null;
create index orotitan_runs_status_stage_idx on public.orotitan_runs (run_status, current_stage, updated_at desc);
create index orotitan_runs_baseline_snapshot_idx on public.orotitan_runs (baseline_snapshot_id) where baseline_snapshot_id is not null;

create table public.orotitan_run_stages (
  run_id uuid not null references public.orotitan_runs(run_id) on delete restrict,
  stage_code text not null check (stage_code in ('RESEARCH', 'DEEP_DIVE', 'INTEGRATION')),
  stage_revision integer not null default 1 check (stage_revision >= 1),
  stage_contract_name text not null check (length(btrim(stage_contract_name)) > 0),
  stage_contract_version text not null check (length(btrim(stage_contract_version)) > 0),
  stage_contract_sha256 text not null check (stage_contract_sha256 ~ '^[0-9a-f]{64}$'),
  lifecycle_status text not null default 'NOT_STARTED' check (lifecycle_status in ('NOT_STARTED', 'IN_PROGRESS', 'PAUSED', 'BLOCKED', 'COMPLETE')),
  contract_status_code text,
  handoff_gate_name text not null,
  handoff_gate_state text not null default 'NOT_EVALUATED' check (handoff_gate_state in ('NOT_EVALUATED', 'YES', 'NO')),
  active_manifest_artifact_id uuid,
  active_manifest_version integer check (active_manifest_version is null or active_manifest_version >= 1),
  active_manifest_kind text check (active_manifest_kind is null or active_manifest_kind in ('CHECKPOINT', 'FINAL')),
  blocker_summary jsonb not null default '[]'::jsonb check (jsonb_typeof(blocker_summary) = 'array'),
  state_version bigint not null default 1 check (state_version >= 1),
  started_at timestamptz,
  completed_at timestamptz,
  updated_at timestamptz not null default now(),
  primary key (run_id, stage_code),
  check ((stage_code = 'RESEARCH' and handoff_gate_name = 'READY_FOR_DEEP_DIVE') or (stage_code = 'DEEP_DIVE' and handoff_gate_name = 'READY_FOR_INTEGRATION') or (stage_code = 'INTEGRATION' and handoff_gate_name = 'READY_TO_PUBLISH')),
  check ((active_manifest_artifact_id is null and active_manifest_version is null and active_manifest_kind is null) or (active_manifest_artifact_id is not null and active_manifest_version is not null and active_manifest_kind is not null)),
  check (lifecycle_status <> 'COMPLETE' or completed_at is not null),
  check (lifecycle_status = 'NOT_STARTED' or started_at is not null)
);

comment on table public.orotitan_run_stages is 'Current operational state for the three company-analysis execution stages.';
create index orotitan_run_stages_lifecycle_idx on public.orotitan_run_stages (lifecycle_status, updated_at desc);
create index orotitan_run_stages_gate_idx on public.orotitan_run_stages (handoff_gate_state, stage_code, updated_at desc);

create table public.orotitan_artifacts (
  artifact_id uuid not null default gen_random_uuid(),
  version integer not null check (version >= 1),
  run_id uuid not null,
  stage_code text not null,
  artifact_type text not null check (length(btrim(artifact_type)) > 0),
  logical_name text not null check (length(btrim(logical_name)) > 0),
  authority_class text not null check (authority_class in ('AUTHORITATIVE_STAGE_OUTPUT','CHECKPOINT_STAGE_OUTPUT','ROUTING_ONLY','HUMAN_SUMMARY','SOURCE_ATTACHMENT','UPSTREAM_DISCOVERY_OUTPUT','IMPLEMENTATION_DIAGNOSTIC')),
  artifact_status text not null default 'SEALED' check (artifact_status in ('SEALED', 'INVALIDATED')),
  authority_state text not null check (authority_state in ('AUTHORITATIVE', 'CHECKPOINT', 'SUPERSEDED', 'NON_AUTHORITATIVE')),
  availability_state text not null default 'AVAILABLE' check (availability_state in ('AVAILABLE', 'WITHDRAWN', 'MISSING')),
  media_type text not null check (length(btrim(media_type)) > 0),
  size_bytes bigint not null check (size_bytes >= 0),
  hash_algorithm text not null default 'SHA-256' check (hash_algorithm = 'SHA-256'),
  content_sha256 text not null check (content_sha256 ~ '^[0-9a-f]{64}$'),
  storage_backend text not null check (storage_backend in ('PRIVATE_GITHUB', 'SUPABASE_STORAGE')),
  storage_uri text not null check (length(btrim(storage_uri)) > 0),
  github_repository text,
  github_path text,
  github_commit_sha text,
  github_blob_sha text,
  supabase_bucket text,
  supabase_object_path text,
  manifest_artifact_id uuid,
  manifest_version integer check (manifest_version is null or manifest_version >= 1),
  created_at timestamptz not null default now(),
  sealed_at timestamptz not null default now(),
  primary key (artifact_id, version),
  unique (run_id, artifact_id, version),
  unique (run_id, stage_code, artifact_id, version),
  foreign key (run_id, stage_code) references public.orotitan_run_stages(run_id, stage_code) on delete restrict,
  check ((manifest_artifact_id is null) = (manifest_version is null)),
  check ((storage_backend = 'PRIVATE_GITHUB' and github_repository is not null and github_path is not null and github_commit_sha is not null and github_blob_sha is not null and supabase_bucket is null and supabase_object_path is null) or (storage_backend = 'SUPABASE_STORAGE' and supabase_bucket is not null and supabase_object_path is not null and github_repository is null and github_path is null and github_commit_sha is null and github_blob_sha is null)),
  check (github_commit_sha is null or github_commit_sha ~ '^[0-9a-f]{40}$'),
  check (github_blob_sha is null or github_blob_sha ~ '^[0-9a-f]{40}$')
);

comment on table public.orotitan_artifacts is 'Exact immutable artifact content identity/version and durable private-storage provenance.';
create index orotitan_artifacts_run_stage_type_idx on public.orotitan_artifacts (run_id, stage_code, artifact_type, version desc);
create index orotitan_artifacts_hash_idx on public.orotitan_artifacts (content_sha256);
create index orotitan_artifacts_manifest_idx on public.orotitan_artifacts (manifest_artifact_id, manifest_version) where manifest_artifact_id is not null;
create index orotitan_artifacts_authority_idx on public.orotitan_artifacts (run_id, stage_code, authority_state, availability_state);

alter table public.orotitan_artifacts add constraint orotitan_artifacts_manifest_same_stage_fkey foreign key (run_id, stage_code, manifest_artifact_id, manifest_version) references public.orotitan_artifacts(run_id, stage_code, artifact_id, version) on delete restrict deferrable initially deferred;
alter table public.orotitan_run_stages add constraint orotitan_run_stages_active_manifest_same_stage_fkey foreign key (run_id, stage_code, active_manifest_artifact_id, active_manifest_version) references public.orotitan_artifacts(run_id, stage_code, artifact_id, version) on delete restrict deferrable initially deferred;
alter table public.orotitan_runs add constraint orotitan_runs_upstream_discovery_artifact_fkey foreign key (upstream_discovery_artifact_id, upstream_discovery_artifact_version) references public.orotitan_artifacts(artifact_id, version) on delete restrict deferrable initially deferred;

create table public.orotitan_artifact_edges (
  edge_id uuid primary key default gen_random_uuid(),
  child_run_id uuid not null,
  child_artifact_id uuid not null,
  child_version integer not null check (child_version >= 1),
  parent_run_id uuid not null,
  parent_artifact_id uuid not null,
  parent_version integer not null check (parent_version >= 1),
  relation_type text not null check (relation_type in ('CONSUMES', 'DERIVED_FROM', 'SUPERSEDES', 'BASELINE_OF', 'REVALIDATES')),
  created_at timestamptz not null default now(),
  foreign key (child_run_id, child_artifact_id, child_version) references public.orotitan_artifacts(run_id, artifact_id, version) on delete restrict,
  foreign key (parent_run_id, parent_artifact_id, parent_version) references public.orotitan_artifacts(run_id, artifact_id, version) on delete restrict,
  unique (child_run_id, child_artifact_id, child_version, parent_run_id, parent_artifact_id, parent_version, relation_type),
  check (child_run_id <> parent_run_id or child_artifact_id <> parent_artifact_id or child_version <> parent_version)
);

comment on table public.orotitan_artifact_edges is 'Explicit artifact dependency/lineage graph, including permitted cross-run refresh lineage.';
create index orotitan_artifact_edges_child_idx on public.orotitan_artifact_edges (child_run_id, child_artifact_id, child_version);
create index orotitan_artifact_edges_parent_idx on public.orotitan_artifact_edges (parent_run_id, parent_artifact_id, parent_version);

create table public.orotitan_run_events (
  event_id uuid primary key default gen_random_uuid(),
  run_id uuid not null references public.orotitan_runs(run_id) on delete restrict,
  stage_code text check (stage_code is null or stage_code in ('RESEARCH', 'DEEP_DIVE', 'INTEGRATION')),
  event_type text not null check (event_type in ('RUN_CREATED','RUN_IDENTITY_BOUND','STAGE_STARTED','STAGE_CHECKPOINTED','STAGE_PAUSED','STAGE_RESUMED','BLOCKER_OPENED','BLOCKER_RESOLVED','STAGE_REOPENED','ARTIFACT_SET_SEALED','STAGE_FINALIZED','READY_TO_PUBLISH_DECLARED','PUBLISH_AUTHORIZED','PUBLISH_SUCCEEDED','PUBLISH_FAILED','RUN_CANCELLED')),
  idempotency_key text not null check (length(btrim(idempotency_key)) > 0),
  request_fingerprint_sha256 text not null check (request_fingerprint_sha256 ~ '^[0-9a-f]{64}$'),
  actor_type text not null check (actor_type in ('USER','PILOTAGE','RESEARCH_WORKER','DEEP_DIVE_WORKER','INTEGRATION_WORKER','PUBLISHER','SYSTEM')),
  payload jsonb not null default '{}'::jsonb check (jsonb_typeof(payload) = 'object'),
  created_at timestamptz not null default now(),
  unique (run_id, idempotency_key)
);

comment on table public.orotitan_run_events is 'Append-only operational audit log. Never an alternative analytical conclusion store.';
create index orotitan_run_events_run_created_idx on public.orotitan_run_events (run_id, created_at desc);
create index orotitan_run_events_stage_created_idx on public.orotitan_run_events (run_id, stage_code, created_at desc) where stage_code is not null;

commit;

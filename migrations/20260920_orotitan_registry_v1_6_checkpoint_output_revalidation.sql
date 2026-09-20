-- OroTitan Registry V1.6 — canonical checkpoint existing-output revalidation.
-- Fixes successor-checkpoint reuse semantics without changing analytical content.

begin;

do $$
begin
  if to_regclass('public.orotitan_runs') is null
     or to_regclass('public.orotitan_run_stages') is null
     or to_regclass('public.orotitan_artifacts') is null
     or to_regclass('public.orotitan_artifact_edges') is null
     or to_regclass('public.orotitan_run_events') is null then
    raise exception 'OroTitan registry core must exist before checkpoint revalidation migration';
  end if;
  if to_regprocedure('public.orotitan_existing_event(uuid,text,text)') is null
     or to_regprocedure('public.orotitan_insert_event(uuid,text,text,text,text,text,jsonb)') is null then
    raise exception 'OroTitan Registry RPC helpers must exist before checkpoint revalidation migration';
  end if;
end;
$$;

create or replace function public.orotitan_insert_artifact_registration(
  p_run_id uuid,
  p_stage_code text,
  p_artifact jsonb,
  p_manifest_artifact_id uuid default null,
  p_manifest_version integer default null
)
returns void
language plpgsql
set search_path = pg_catalog, public
as $$
declare
  v_backend text := p_artifact->>'storage_backend';
  v_existing public.orotitan_artifacts%rowtype;
  v_desired_authority_state text := p_artifact->>'authority_state';
  v_expected_artifact_status text := coalesce(p_artifact->>'artifact_status', 'SEALED');
  v_expected_availability_state text := coalesce(p_artifact->>'availability_state', 'AVAILABLE');
  v_expected_github_repository text := case when v_backend = 'PRIVATE_GITHUB' then p_artifact->>'github_repository' end;
  v_expected_github_path text := case when v_backend = 'PRIVATE_GITHUB' then p_artifact->>'github_path' end;
  v_expected_github_commit_sha text := case when v_backend = 'PRIVATE_GITHUB' then p_artifact->>'github_commit_sha' end;
  v_expected_github_blob_sha text := case when v_backend = 'PRIVATE_GITHUB' then p_artifact->>'github_blob_sha' end;
  v_expected_supabase_bucket text := case when v_backend = 'SUPABASE_STORAGE' then p_artifact->>'supabase_bucket' end;
  v_expected_supabase_object_path text := case when v_backend = 'SUPABASE_STORAGE' then p_artifact->>'supabase_object_path' end;
begin
  if jsonb_typeof(p_artifact) <> 'object' then
    raise exception 'artifact registration must be an object' using errcode = '22023';
  end if;

  if (p_artifact->>'artifact_id') is null
     or (p_artifact->>'version') is null
     or (p_artifact->>'artifact_type') is null
     or (p_artifact->>'logical_name') is null
     or (p_artifact->>'authority_class') is null
     or v_desired_authority_state is null
     or (p_artifact->>'media_type') is null
     or (p_artifact->>'size_bytes') is null
     or (p_artifact->>'content_sha256') is null
     or v_backend is null
     or (p_artifact->>'storage_uri') is null then
    raise exception 'artifact registration is missing required fields'
      using errcode = '22023';
  end if;

  if (p_manifest_artifact_id is null) <> (p_manifest_version is null) then
    raise exception 'manifest artifact id/version must be supplied together'
      using errcode = '22023';
  end if;

  begin
    insert into public.orotitan_artifacts (
      artifact_id, version, run_id, stage_code,
      artifact_type, logical_name, authority_class,
      artifact_status, authority_state, availability_state,
      media_type, size_bytes, hash_algorithm, content_sha256,
      storage_backend, storage_uri,
      github_repository, github_path, github_commit_sha, github_blob_sha,
      supabase_bucket, supabase_object_path,
      manifest_artifact_id, manifest_version
    ) values (
      (p_artifact->>'artifact_id')::uuid,
      (p_artifact->>'version')::integer,
      p_run_id,
      p_stage_code,
      p_artifact->>'artifact_type',
      p_artifact->>'logical_name',
      p_artifact->>'authority_class',
      v_expected_artifact_status,
      v_desired_authority_state,
      v_expected_availability_state,
      p_artifact->>'media_type',
      (p_artifact->>'size_bytes')::bigint,
      'SHA-256',
      p_artifact->>'content_sha256',
      v_backend,
      p_artifact->>'storage_uri',
      v_expected_github_repository,
      v_expected_github_path,
      v_expected_github_commit_sha,
      v_expected_github_blob_sha,
      v_expected_supabase_bucket,
      v_expected_supabase_object_path,
      p_manifest_artifact_id,
      p_manifest_version
    );
    return;
  exception
    when unique_violation then
      select * into v_existing
      from public.orotitan_artifacts
      where artifact_id = (p_artifact->>'artifact_id')::uuid
        and version = (p_artifact->>'version')::integer;

      if not found then
        raise;
      end if;

      if v_existing.run_id is distinct from p_run_id
         or v_existing.stage_code is distinct from p_stage_code
         or v_existing.artifact_type is distinct from p_artifact->>'artifact_type'
         or v_existing.logical_name is distinct from p_artifact->>'logical_name'
         or v_existing.authority_class is distinct from p_artifact->>'authority_class'
         or v_existing.artifact_status is distinct from v_expected_artifact_status
         or v_existing.availability_state is distinct from v_expected_availability_state
         or v_existing.media_type is distinct from p_artifact->>'media_type'
         or v_existing.size_bytes is distinct from (p_artifact->>'size_bytes')::bigint
         or v_existing.hash_algorithm is distinct from 'SHA-256'
         or v_existing.content_sha256 is distinct from p_artifact->>'content_sha256'
         or v_existing.storage_backend is distinct from v_backend
         or v_existing.storage_uri is distinct from p_artifact->>'storage_uri'
         or v_existing.github_repository is distinct from v_expected_github_repository
         or v_existing.github_path is distinct from v_expected_github_path
         or v_existing.github_commit_sha is distinct from v_expected_github_commit_sha
         or v_existing.github_blob_sha is distinct from v_expected_github_blob_sha
         or v_existing.supabase_bucket is distinct from v_expected_supabase_bucket
         or v_existing.supabase_object_path is distinct from v_expected_supabase_object_path then
        raise exception 'ARTIFACT_REGISTRATION_CONFLICT: existing immutable artifact registration differs'
          using errcode = '23514';
      end if;

      if p_manifest_artifact_id is not null then
        if v_existing.artifact_status <> 'SEALED'
           or v_existing.availability_state <> 'AVAILABLE'
           or v_existing.authority_state = 'NON_AUTHORITATIVE' then
          raise exception 'ARTIFACT_REUSE_NOT_ADMITTED: artifact is not a valid sealed available authority candidate'
            using errcode = '23514';
        end if;

        if v_desired_authority_state = 'CHECKPOINT' then
          if v_existing.authority_class <> 'CHECKPOINT_STAGE_OUTPUT' then
            raise exception 'ARTIFACT_REUSE_NOT_ADMITTED: CHECKPOINT authority class mismatch'
              using errcode = '23514';
          end if;
        elsif v_desired_authority_state = 'AUTHORITATIVE' then
          if v_existing.authority_class <> 'AUTHORITATIVE_STAGE_OUTPUT' then
            raise exception 'ARTIFACT_REUSE_NOT_ADMITTED: AUTHORITATIVE authority class mismatch'
              using errcode = '23514';
          end if;
        else
          raise exception 'ARTIFACT_REUSE_NOT_ADMITTED: existing manifest output may only be rebound to current authority'
            using errcode = '23514';
        end if;

        update public.orotitan_artifacts
        set authority_state = v_desired_authority_state,
            manifest_artifact_id = p_manifest_artifact_id,
            manifest_version = p_manifest_version
        where artifact_id = v_existing.artifact_id
          and version = v_existing.version;
      end if;
  end;
end;
$$;

create or replace function public.supersede_orotitan_manifest_bundle()
returns trigger
language plpgsql
set search_path = pg_catalog, public
as $$
begin
  if old.active_manifest_artifact_id is null then
    return new;
  end if;

  if old.active_manifest_artifact_id is not distinct from new.active_manifest_artifact_id
     and old.active_manifest_version is not distinct from new.active_manifest_version then
    return new;
  end if;

  update public.orotitan_artifacts a
  set authority_state = 'SUPERSEDED'
  where a.run_id = old.run_id
    and a.stage_code = old.stage_code
    and a.authority_state in ('AUTHORITATIVE', 'CHECKPOINT')
    and (
      (a.artifact_id = old.active_manifest_artifact_id and a.version = old.active_manifest_version)
      or
      (a.manifest_artifact_id = old.active_manifest_artifact_id and a.manifest_version = old.active_manifest_version)
    )
    and not (
      new.active_manifest_artifact_id is not null
      and new.active_manifest_version is not null
      and a.manifest_artifact_id is not distinct from new.active_manifest_artifact_id
      and a.manifest_version is not distinct from new.active_manifest_version
    );

  return new;
end;
$$;

create or replace function public.revalidate_orotitan_checkpoint_outputs(
  p_run_id uuid,
  p_stage_code text,
  p_expected_run_state_version bigint,
  p_expected_stage_state_version bigint,
  p_expected_manifest_artifact_id uuid,
  p_expected_manifest_version integer,
  p_expected_manifest_sha256 text,
  p_manifest_output_artifacts jsonb,
  p_idempotency_key text,
  p_request_fingerprint_sha256 text
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_run public.orotitan_runs%rowtype;
  v_stage public.orotitan_run_stages%rowtype;
  v_manifest public.orotitan_artifacts%rowtype;
  v_target public.orotitan_artifacts%rowtype;
  v_ref jsonb;
  v_retry jsonb;
  v_event_id uuid;
  v_target_count integer;
  v_expected_count integer;
  v_updated integer;
  v_new_run_state bigint;
  v_new_stage_state bigint;
begin
  if p_idempotency_key is null or length(btrim(p_idempotency_key)) = 0 then
    raise exception 'idempotency key is required' using errcode = '22023';
  end if;
  if p_request_fingerprint_sha256 !~ '^[0-9a-f]{64}$' then
    raise exception 'invalid request fingerprint' using errcode = '22023';
  end if;
  if p_expected_manifest_sha256 !~ '^[0-9a-f]{64}$' then
    raise exception 'invalid expected manifest hash' using errcode = '22023';
  end if;
  if coalesce(jsonb_typeof(p_manifest_output_artifacts), 'null') <> 'array'
     or jsonb_array_length(p_manifest_output_artifacts) = 0 then
    raise exception 'manifest output artifact references must be a non-empty array'
      using errcode = '22023';
  end if;

  v_retry := public.orotitan_existing_event(
    p_run_id, p_idempotency_key, p_request_fingerprint_sha256
  );
  if v_retry is not null then
    if v_retry->>'event_type' <> 'BLOCKER_RESOLVED'
       or v_retry->'payload'->>'repair_class' <> 'EXISTING_OUTPUT_AUTHORITY_REVALIDATION'
       or v_retry->'payload'->>'active_manifest_id' <> p_expected_manifest_artifact_id::text
       or (v_retry->'payload'->>'active_manifest_version')::integer <> p_expected_manifest_version then
      raise exception 'IDEMPOTENCY_CONFLICT: key belongs to a different operation'
        using errcode = '23514';
    end if;
    select state_version into v_new_run_state
    from public.orotitan_runs where run_id = p_run_id;
    select state_version into v_new_stage_state
    from public.orotitan_run_stages where run_id = p_run_id and stage_code = p_stage_code;
    return jsonb_build_object(
      'run_id', p_run_id,
      'stage', p_stage_code,
      'manifest_id', p_expected_manifest_artifact_id,
      'manifest_version', p_expected_manifest_version,
      'artifact_count', (v_retry->'payload'->>'artifact_count')::integer,
      'run_state_version', v_new_run_state,
      'stage_state_version', v_new_stage_state,
      'event_id', v_retry->>'event_id',
      'idempotent_replay', true
    );
  end if;

  select * into v_run
  from public.orotitan_runs
  where run_id = p_run_id
  for update;
  if not found then
    raise exception 'RUN_NOT_FOUND' using errcode = 'P0002';
  end if;
  if v_run.run_status in ('PUBLISHED', 'CANCELLED') then
    raise exception 'RUN_TERMINAL' using errcode = '23514';
  end if;
  if v_run.current_stage is distinct from p_stage_code then
    raise exception 'STAGE_NOT_CURRENT' using errcode = '23514';
  end if;
  if v_run.state_version <> p_expected_run_state_version then
    raise exception 'RUN_STATE_VERSION_MISMATCH' using errcode = '40001';
  end if;

  select * into v_stage
  from public.orotitan_run_stages
  where run_id = p_run_id and stage_code = p_stage_code
  for update;
  if not found then
    raise exception 'STAGE_NOT_FOUND' using errcode = 'P0002';
  end if;
  if v_stage.state_version <> p_expected_stage_state_version then
    raise exception 'STAGE_STATE_VERSION_MISMATCH' using errcode = '40001';
  end if;
  if v_stage.lifecycle_status <> 'IN_PROGRESS' then
    raise exception 'STAGE_NOT_ADMITTED: checkpoint revalidation requires IN_PROGRESS stage'
      using errcode = '23514';
  end if;
  if v_stage.active_manifest_kind <> 'CHECKPOINT'
     or v_stage.active_manifest_artifact_id is distinct from p_expected_manifest_artifact_id
     or v_stage.active_manifest_version is distinct from p_expected_manifest_version then
    raise exception 'ACTIVE_MANIFEST_MISMATCH' using errcode = '23514';
  end if;

  select * into v_manifest
  from public.orotitan_artifacts
  where artifact_id = p_expected_manifest_artifact_id
    and version = p_expected_manifest_version
    and run_id = p_run_id
    and stage_code = p_stage_code;
  if not found then
    raise exception 'ACTIVE_MANIFEST_NOT_FOUND' using errcode = 'P0002';
  end if;
  if v_manifest.artifact_status <> 'SEALED'
     or v_manifest.availability_state <> 'AVAILABLE'
     or v_manifest.authority_state <> 'CHECKPOINT'
     or v_manifest.content_sha256 <> p_expected_manifest_sha256 then
    raise exception 'ACTIVE_MANIFEST_NOT_VALID_CHECKPOINT' using errcode = '23514';
  end if;

  if not exists (
    select 1
    from public.orotitan_run_events e
    where e.run_id = p_run_id
      and e.stage_code = p_stage_code
      and e.event_type = 'STAGE_CHECKPOINTED'
      and e.payload->>'manifest_id' = p_expected_manifest_artifact_id::text
      and (e.payload->>'manifest_version')::integer = p_expected_manifest_version
  ) then
    raise exception 'ACTIVE_MANIFEST_CHECKPOINT_EVENT_MISSING' using errcode = '23514';
  end if;

  if exists (
    select 1
    from jsonb_array_elements(p_manifest_output_artifacts) x
    group by x->>'artifact_id', x->>'version'
    having count(*) > 1
  ) then
    raise exception 'duplicate manifest output artifact reference' using errcode = '23514';
  end if;

  select jsonb_array_length(p_manifest_output_artifacts) into v_target_count;

  select count(*) into v_expected_count
  from public.orotitan_artifact_edges e
  join public.orotitan_artifacts a
    on a.run_id = e.parent_run_id
   and a.artifact_id = e.parent_artifact_id
   and a.version = e.parent_version
  where e.child_run_id = p_run_id
    and e.child_artifact_id = p_expected_manifest_artifact_id
    and e.child_version = p_expected_manifest_version
    and e.relation_type = 'CONSUMES'
    and e.parent_run_id = p_run_id
    and a.stage_code = p_stage_code
    and a.authority_class = 'CHECKPOINT_STAGE_OUTPUT'
    and not (a.artifact_id = p_expected_manifest_artifact_id and a.version = p_expected_manifest_version);

  if v_expected_count <> v_target_count then
    raise exception 'ACTIVE_CHECKPOINT_OUTPUT_SET_MISMATCH: expected %, supplied %', v_expected_count, v_target_count
      using errcode = '23514';
  end if;

  if exists (
    select 1
    from public.orotitan_artifact_edges e
    join public.orotitan_artifacts a
      on a.run_id = e.parent_run_id
     and a.artifact_id = e.parent_artifact_id
     and a.version = e.parent_version
    where e.child_run_id = p_run_id
      and e.child_artifact_id = p_expected_manifest_artifact_id
      and e.child_version = p_expected_manifest_version
      and e.relation_type = 'CONSUMES'
      and e.parent_run_id = p_run_id
      and a.stage_code = p_stage_code
      and a.authority_class = 'CHECKPOINT_STAGE_OUTPUT'
      and not (a.artifact_id = p_expected_manifest_artifact_id and a.version = p_expected_manifest_version)
      and not exists (
        select 1
        from jsonb_array_elements(p_manifest_output_artifacts) r
        where (r->>'artifact_id')::uuid = a.artifact_id
          and (r->>'version')::integer = a.version
      )
  ) then
    raise exception 'ACTIVE_CHECKPOINT_OUTPUT_SET_MISMATCH: successor output omitted'
      using errcode = '23514';
  end if;

  for v_ref in select value from jsonb_array_elements(p_manifest_output_artifacts)
  loop
    if (v_ref->>'artifact_id') is null
       or (v_ref->>'version') is null
       or (v_ref->>'artifact_type') is null
       or (v_ref->>'content_sha256') is null
       or (v_ref->>'authority_class') is null
       or (v_ref->>'media_type') is null
       or (v_ref->>'size_bytes') is null then
      raise exception 'manifest output artifact reference is incomplete' using errcode = '22023';
    end if;
    if v_ref->>'content_sha256' !~ '^[0-9a-f]{64}$'
       or (v_ref->>'size_bytes')::bigint < 0
       or v_ref->>'authority_class' <> 'CHECKPOINT_STAGE_OUTPUT' then
      raise exception 'manifest output artifact reference is invalid' using errcode = '23514';
    end if;

    select * into v_target
    from public.orotitan_artifacts a
    where a.artifact_id = (v_ref->>'artifact_id')::uuid
      and a.version = (v_ref->>'version')::integer;
    if not found then
      raise exception 'TARGET_ARTIFACT_NOT_FOUND' using errcode = 'P0002';
    end if;

    if v_target.run_id <> p_run_id
       or v_target.stage_code <> p_stage_code
       or v_target.artifact_type <> v_ref->>'artifact_type'
       or v_target.authority_class <> v_ref->>'authority_class'
       or v_target.media_type <> v_ref->>'media_type'
       or v_target.content_sha256 <> v_ref->>'content_sha256'
       or v_target.size_bytes <> (v_ref->>'size_bytes')::bigint
       or v_target.artifact_status <> 'SEALED'
       or v_target.availability_state <> 'AVAILABLE'
       or v_target.authority_state not in ('SUPERSEDED', 'CHECKPOINT') then
      raise exception 'TARGET_ARTIFACT_VALIDATION_FAILED' using errcode = '23514';
    end if;

    if not exists (
      select 1
      from public.orotitan_artifact_edges e
      where e.child_run_id = p_run_id
        and e.child_artifact_id = p_expected_manifest_artifact_id
        and e.child_version = p_expected_manifest_version
        and e.parent_run_id = p_run_id
        and e.parent_artifact_id = v_target.artifact_id
        and e.parent_version = v_target.version
        and e.relation_type = 'CONSUMES'
    ) then
      raise exception 'TARGET_NOT_CONSUMED_BY_ACTIVE_CHECKPOINT' using errcode = '23514';
    end if;

    if v_target.authority_state = 'SUPERSEDED' then
      if v_target.manifest_artifact_id is null or v_target.manifest_version is null then
        raise exception 'SUPERSEDED_TARGET_HAS_NO_PREDECESSOR_BINDING' using errcode = '23514';
      end if;
      if not exists (
        select 1
        from public.orotitan_artifact_edges e
        where e.child_run_id = p_run_id
          and e.child_artifact_id = p_expected_manifest_artifact_id
          and e.child_version = p_expected_manifest_version
          and e.parent_run_id = p_run_id
          and e.parent_artifact_id = v_target.manifest_artifact_id
          and e.parent_version = v_target.manifest_version
          and e.relation_type = 'SUPERSEDES'
      ) then
        raise exception 'SUPERSEDED_TARGET_NOT_ATTRIBUTABLE_TO_ACTIVE_RECHECKPOINT'
          using errcode = '23514';
      end if;
      if not exists (
        select 1
        from public.orotitan_run_events e
        where e.run_id = p_run_id
          and e.stage_code = p_stage_code
          and e.event_type = 'STAGE_CHECKPOINTED'
          and e.payload->>'manifest_id' = v_target.manifest_artifact_id::text
          and (e.payload->>'manifest_version')::integer = v_target.manifest_version
      ) then
        raise exception 'PREDECESSOR_CHECKPOINT_EVENT_MISSING' using errcode = '23514';
      end if;
    else
      if v_target.manifest_artifact_id is distinct from p_expected_manifest_artifact_id
         or v_target.manifest_version is distinct from p_expected_manifest_version then
        raise exception 'CURRENT_CHECKPOINT_TARGET_BINDING_MISMATCH' using errcode = '23514';
      end if;
    end if;
  end loop;

  update public.orotitan_artifacts a
  set authority_state = 'CHECKPOINT',
      manifest_artifact_id = p_expected_manifest_artifact_id,
      manifest_version = p_expected_manifest_version
  from jsonb_array_elements(p_manifest_output_artifacts) r
  where a.artifact_id = (r->>'artifact_id')::uuid
    and a.version = (r->>'version')::integer
    and a.run_id = p_run_id
    and a.stage_code = p_stage_code;
  get diagnostics v_updated = row_count;
  if v_updated <> v_target_count then
    raise exception 'REVALIDATION_UPDATE_COUNT_MISMATCH' using errcode = '23514';
  end if;

  update public.orotitan_run_stages
  set state_version = state_version + 1
  where run_id = p_run_id
    and stage_code = p_stage_code
    and state_version = p_expected_stage_state_version
  returning state_version into v_new_stage_state;
  if not found then
    raise exception 'STAGE_STATE_VERSION_MISMATCH' using errcode = '40001';
  end if;

  update public.orotitan_runs
  set state_version = state_version + 1
  where run_id = p_run_id
    and state_version = p_expected_run_state_version
  returning state_version into v_new_run_state;
  if not found then
    raise exception 'RUN_STATE_VERSION_MISMATCH' using errcode = '40001';
  end if;

  v_event_id := public.orotitan_insert_event(
    p_run_id,
    p_stage_code,
    'BLOCKER_RESOLVED',
    p_idempotency_key,
    p_request_fingerprint_sha256,
    'SYSTEM',
    jsonb_build_object(
      'blocker', 'CANONICAL_RECHECKPOINT_SUPERSESSION_TRIGGER_SUPERSEDED_REUSED_FUNDAMENTALS_OUTPUTS',
      'repair_class', 'EXISTING_OUTPUT_AUTHORITY_REVALIDATION',
      'active_manifest_id', p_expected_manifest_artifact_id,
      'active_manifest_version', p_expected_manifest_version,
      'active_manifest_sha256', p_expected_manifest_sha256,
      'artifact_count', v_target_count,
      'analytical_state_changed', false
    )
  );

  return jsonb_build_object(
    'run_id', p_run_id,
    'stage', p_stage_code,
    'manifest_id', p_expected_manifest_artifact_id,
    'manifest_version', p_expected_manifest_version,
    'artifact_count', v_target_count,
    'run_state_version', v_new_run_state,
    'stage_state_version', v_new_stage_state,
    'event_id', v_event_id,
    'idempotent_replay', false
  );
end;
$$;

revoke all on function public.revalidate_orotitan_checkpoint_outputs(
  uuid, text, bigint, bigint, uuid, integer, text, jsonb, text, text
) from public, anon, authenticated;
grant execute on function public.revalidate_orotitan_checkpoint_outputs(
  uuid, text, bigint, bigint, uuid, integer, text, jsonb, text, text
) to service_role;

comment on function public.revalidate_orotitan_checkpoint_outputs(
  uuid, text, bigint, bigint, uuid, integer, text, jsonb, text, text
) is 'Atomically revalidates exact immutable outputs of the active CHECKPOINT after a proven predecessor-supersession metadata defect; changes only authority/binding metadata and Registry state versions.';

comment on function public.supersede_orotitan_manifest_bundle() is
  'Supersedes the predecessor Stage Manifest and predecessor-only outputs when active manifest changes; exact successor-reused output ID/version pairs already rebound to the successor are preserved as current authority.';

commit;

-- OroTitan Registry V1.7 — generic successor FINAL/CHECKPOINT exact-output rebinding.
-- Persistence / authority-binding repair only. No analytical methodology change.

begin;

do $$
begin
  if to_regclass('public.orotitan_runs') is null
     or to_regclass('public.orotitan_run_stages') is null
     or to_regclass('public.orotitan_artifacts') is null
     or to_regclass('public.orotitan_artifact_edges') is null
     or to_regclass('public.orotitan_run_events') is null then
    raise exception 'OroTitan registry core must exist before successor output rebinding migration';
  end if;
  if to_regprocedure('public.orotitan_register_manifest_bundle(uuid,text,jsonb,jsonb,jsonb,jsonb)') is null
     or to_regprocedure('public.orotitan_insert_artifact_registration(uuid,text,jsonb,uuid,integer)') is null
     or to_regprocedure('public.supersede_orotitan_manifest_bundle()') is null then
    raise exception 'OroTitan Registry manifest registration primitives must exist before successor output rebinding migration';
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
  v_stage public.orotitan_run_stages%rowtype;
  v_target_manifest public.orotitan_artifacts%rowtype;
  v_predecessor_manifest public.orotitan_artifacts%rowtype;
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

      if p_manifest_artifact_id is null then
        return;
      end if;

      if v_existing.artifact_status <> 'SEALED'
         or v_existing.availability_state <> 'AVAILABLE' then
        raise exception 'ARTIFACT_REUSE_NOT_ADMITTED: artifact must be SEALED and AVAILABLE'
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
        raise exception 'ARTIFACT_REUSE_NOT_ADMITTED: manifest output may only be rebound to current authority'
          using errcode = '23514';
      end if;

      select * into v_stage
      from public.orotitan_run_stages
      where run_id = p_run_id
        and stage_code = p_stage_code;
      if not found then
        raise exception 'ARTIFACT_REUSE_NOT_ADMITTED: stage not found'
          using errcode = 'P0002';
      end if;

      select * into v_target_manifest
      from public.orotitan_artifacts
      where artifact_id = p_manifest_artifact_id
        and version = p_manifest_version
        and run_id = p_run_id
        and stage_code = p_stage_code;
      if not found
         or v_target_manifest.artifact_status <> 'SEALED'
         or v_target_manifest.availability_state <> 'AVAILABLE'
         or v_target_manifest.authority_state <> v_desired_authority_state then
        raise exception 'ARTIFACT_REUSE_NOT_ADMITTED: target manifest is not a valid current-authority manifest candidate'
          using errcode = '23514';
      end if;

      -- True same-manifest replay is a no-op.
      if v_existing.authority_state = v_desired_authority_state
         and v_existing.manifest_artifact_id is not distinct from p_manifest_artifact_id
         and v_existing.manifest_version is not distinct from p_manifest_version then
        return;
      end if;

      -- A currently authoritative/checkpoint output may move to a successor
      -- only when its current binding is exactly the stage's active manifest.
      if v_existing.authority_state in ('AUTHORITATIVE', 'CHECKPOINT') then
        if v_stage.active_manifest_artifact_id is null
           or v_stage.active_manifest_version is null
           or v_existing.manifest_artifact_id is distinct from v_stage.active_manifest_artifact_id
           or v_existing.manifest_version is distinct from v_stage.active_manifest_version
           or (v_stage.active_manifest_kind = 'FINAL' and v_existing.authority_state <> 'AUTHORITATIVE')
           or (v_stage.active_manifest_kind = 'CHECKPOINT' and v_existing.authority_state <> 'CHECKPOINT')
           or v_stage.active_manifest_kind not in ('FINAL', 'CHECKPOINT') then
          raise exception 'ARTIFACT_AUTHORITY_BINDING_CONFLICT: current authority is not attributable to the active stage manifest'
            using errcode = '23514';
        end if;
      elsif v_existing.authority_state = 'SUPERSEDED' then
        if v_existing.manifest_artifact_id is null or v_existing.manifest_version is null then
          raise exception 'ARTIFACT_REUSE_NOT_ADMITTED: superseded artifact has no predecessor manifest binding'
            using errcode = '23514';
        end if;
        select * into v_predecessor_manifest
        from public.orotitan_artifacts
        where artifact_id = v_existing.manifest_artifact_id
          and version = v_existing.manifest_version
          and run_id = p_run_id
          and stage_code = p_stage_code;
        if not found then
          raise exception 'ARTIFACT_REUSE_NOT_ADMITTED: predecessor manifest binding is not reconstructable'
            using errcode = '23514';
        end if;
      else
        raise exception 'ARTIFACT_REUSE_NOT_ADMITTED: non-current authority state cannot be reactivated'
          using errcode = '23514';
      end if;

      update public.orotitan_artifacts
      set authority_state = v_desired_authority_state,
          manifest_artifact_id = p_manifest_artifact_id,
          manifest_version = p_manifest_version
      where artifact_id = v_existing.artifact_id
        and version = v_existing.version;
  end;
end;
$$;

create or replace function public.orotitan_register_manifest_bundle(
  p_run_id uuid,
  p_stage_code text,
  p_manifest jsonb,
  p_manifest_registration jsonb,
  p_output_artifacts jsonb,
  p_edges jsonb
)
returns void
language plpgsql
set search_path = pg_catalog, public
as $$
declare
  v_stage public.orotitan_run_stages%rowtype;
  v_manifest_id uuid := (p_manifest->>'manifest_id')::uuid;
  v_manifest_version integer := (p_manifest_registration->>'version')::integer;
  v_manifest_kind text := p_manifest->>'manifest_kind';
  v_expected_authority_class text;
  v_expected_authority_state text;
  v_output jsonb;
  v_ref jsonb;
  v_match jsonb;
  v_edge jsonb;
  v_count integer;
begin
  select * into v_stage
  from public.orotitan_run_stages
  where run_id = p_run_id
    and stage_code = p_stage_code;
  if not found then
    raise exception 'STAGE_NOT_FOUND' using errcode = 'P0002';
  end if;

  if v_manifest_kind = 'CHECKPOINT' then
    v_expected_authority_class := 'CHECKPOINT_STAGE_OUTPUT';
    v_expected_authority_state := 'CHECKPOINT';
  elsif v_manifest_kind = 'FINAL' then
    v_expected_authority_class := 'AUTHORITATIVE_STAGE_OUTPUT';
    v_expected_authority_state := 'AUTHORITATIVE';
  else
    raise exception 'MANIFEST_CONTRACT_MISMATCH: unsupported manifest kind'
      using errcode = '23514';
  end if;

  if (p_manifest->>'run_id')::uuid is distinct from p_run_id
     or p_manifest->>'stage' is distinct from p_stage_code
     or (p_manifest->>'stage_revision')::integer is distinct from v_stage.stage_revision then
    raise exception 'MANIFEST_CONTRACT_MISMATCH: manifest run/stage/revision mismatch'
      using errcode = '23514';
  end if;

  if p_manifest_registration->>'artifact_id' <> v_manifest_id::text then
    raise exception 'MANIFEST_CONTRACT_MISMATCH: manifest_id != manifest artifact_id'
      using errcode = '23514';
  end if;

  if p_manifest_registration->>'authority_class' is distinct from v_expected_authority_class
     or p_manifest_registration->>'authority_state' is distinct from v_expected_authority_state
     or coalesce(p_manifest_registration->>'artifact_status','SEALED') <> 'SEALED'
     or coalesce(p_manifest_registration->>'availability_state','AVAILABLE') <> 'AVAILABLE' then
    raise exception 'MANIFEST_CONTRACT_MISMATCH: manifest registration authority/status mismatch'
      using errcode = '23514';
  end if;

  if coalesce(jsonb_typeof(p_output_artifacts), 'null') <> 'array'
     or coalesce(jsonb_typeof(p_edges), 'null') <> 'array'
     or coalesce(jsonb_typeof(p_manifest->'output_artifacts'), 'null') <> 'array' then
    raise exception 'manifest bundle arrays are malformed' using errcode = '22023';
  end if;

  if jsonb_array_length(p_output_artifacts) <> jsonb_array_length(p_manifest->'output_artifacts') then
    raise exception 'ARTIFACT_NOT_IN_MANIFEST: output artifact count mismatch'
      using errcode = '23514';
  end if;

  if exists (
    select 1
    from jsonb_array_elements(p_output_artifacts) x
    group by x->>'artifact_id', x->>'version'
    having count(*) > 1
  ) then
    raise exception 'ARTIFACT_NOT_IN_MANIFEST: duplicate output artifact registration'
      using errcode = '23514';
  end if;

  perform public.orotitan_insert_artifact_registration(
    p_run_id, p_stage_code, p_manifest_registration, null, null
  );

  for v_ref in select value from jsonb_array_elements(p_manifest->'output_artifacts')
  loop
    if v_ref->>'authority_class' is distinct from v_expected_authority_class then
      raise exception 'MANIFEST_CONTRACT_MISMATCH: output reference authority class does not match manifest kind'
        using errcode = '23514';
    end if;

    select value
    into v_match
    from jsonb_array_elements(p_output_artifacts)
    where value->>'artifact_id' = v_ref->>'artifact_id'
      and value->>'version' = v_ref->>'version';

    if not found
       or v_match->>'content_sha256' <> v_ref->>'content_sha256'
       or v_match->>'artifact_type' <> v_ref->>'artifact_type'
       or v_match->>'authority_class' <> v_ref->>'authority_class'
       or v_match->>'media_type' <> v_ref->>'media_type'
       or (v_match->>'size_bytes')::bigint <> (v_ref->>'size_bytes')::bigint
       or v_match->>'authority_class' is distinct from v_expected_authority_class
       or v_match->>'authority_state' is distinct from v_expected_authority_state
       or coalesce(v_match->>'artifact_status','SEALED') <> 'SEALED'
       or coalesce(v_match->>'availability_state','AVAILABLE') <> 'AVAILABLE' then
      raise exception 'ARTIFACT_NOT_IN_MANIFEST: output artifact mismatch'
        using errcode = '23514';
    end if;

    perform public.orotitan_insert_artifact_registration(
      p_run_id, p_stage_code, v_match, v_manifest_id, v_manifest_version
    );
  end loop;

  for v_output in select value from jsonb_array_elements(p_output_artifacts)
  loop
    select count(*)
    into v_count
    from jsonb_array_elements(p_manifest->'output_artifacts')
    where value->>'artifact_id' = v_output->>'artifact_id'
      and value->>'version' = v_output->>'version';
    if v_count <> 1 then
      raise exception 'ARTIFACT_NOT_IN_MANIFEST: extra or duplicate output artifact'
        using errcode = '23514';
    end if;
  end loop;

  for v_edge in select value from jsonb_array_elements(p_edges)
  loop
    insert into public.orotitan_artifact_edges (
      child_run_id, child_artifact_id, child_version,
      parent_run_id, parent_artifact_id, parent_version,
      relation_type
    ) values (
      coalesce((v_edge->>'child_run_id')::uuid, p_run_id),
      (v_edge->>'child_artifact_id')::uuid,
      (v_edge->>'child_version')::integer,
      (v_edge->>'parent_run_id')::uuid,
      (v_edge->>'parent_artifact_id')::uuid,
      (v_edge->>'parent_version')::integer,
      v_edge->>'relation_type'
    )
    on conflict do nothing;
  end loop;
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

revoke all on function public.orotitan_insert_artifact_registration(uuid, text, jsonb, uuid, integer)
  from public, anon, authenticated, service_role;
revoke all on function public.orotitan_register_manifest_bundle(uuid, text, jsonb, jsonb, jsonb, jsonb)
  from public, anon, authenticated, service_role;
revoke all on function public.supersede_orotitan_manifest_bundle()
  from public, anon, authenticated, service_role;

comment on function public.orotitan_insert_artifact_registration(uuid, text, jsonb, uuid, integer) is
  'Registers immutable artifacts and, only inside canonical manifest registration, distinguishes same-manifest replay from validated successor rebind/reactivation while preserving immutable content/provenance.';
comment on function public.orotitan_register_manifest_bundle(uuid, text, jsonb, jsonb, jsonb, jsonb) is
  'Registers CHECKPOINT/FINAL manifest bundles with current stage-revision and manifest-kind authority semantics; exact reused immutable outputs may be rebound only through this canonical path.';
comment on function public.supersede_orotitan_manifest_bundle() is
  'Supersedes the predecessor Stage Manifest and predecessor-only outputs when active manifest changes; exact successor-reused output ID/version pairs already rebound to the successor are preserved as current authority.';

commit;

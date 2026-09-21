-- OroTitan Registry V1.11 — persistence attestation authority reconciliation.
-- Forward-only Registry repair. Historical V1.9/V1.10 migrations and events remain immutable.
-- Analytical methodology / Contract Set / I2 / I3-B semantics: UNCHANGED.
--
-- Repair goals:
-- 1. establish expected immutable artifact metadata in Registry as NON_AUTHORITATIVE;
-- 2. reconcile persistence attestations exactly against that Registry metadata;
-- 3. keep the attestation idempotency identity stable independently of request fingerprint;
-- 4. promote exact staged metadata only inside the existing manifest/finalization path.

begin;

do $$
begin
  if to_regclass('public.orotitan_artifacts') is null
     or to_regclass('public.orotitan_runs') is null
     or to_regclass('public.orotitan_run_stages') is null
     or to_regclass('public.orotitan_run_events') is null
     or to_regprocedure('public.orotitan_insert_artifact_registration(uuid,text,jsonb,uuid,integer)') is null
     or to_regprocedure('public.orotitan_existing_event(uuid,text,text)') is null
     or to_regprocedure('public.orotitan_insert_event(uuid,text,text,text,text,text,jsonb)') is null
     or to_regprocedure('public.attest_orotitan_persistence_locator(uuid,text,jsonb)') is null then
    raise exception 'OroTitan Registry V1.10 primitives must exist before V1.11 attestation authority repair';
  end if;
end;
$$;

-- Preserve the existing artifact registration primitive while adding one narrowly
-- scoped transition: an exact NON_AUTHORITATIVE persistence candidate may be
-- promoted by the canonical manifest/finalization path.
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
as $function$
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

      if v_existing.artifact_status <> 'SEALED'
         or v_existing.availability_state <> 'AVAILABLE' then
        raise exception 'ARTIFACT_REUSE_NOT_ADMITTED: artifact must be SEALED and AVAILABLE'
          using errcode = '23514';
      end if;

      if v_desired_authority_state = 'NON_AUTHORITATIVE' then
        if v_existing.authority_state <> 'NON_AUTHORITATIVE'
           or p_manifest_artifact_id is not null then
          raise exception 'ARTIFACT_AUTHORITY_BINDING_CONFLICT: staged metadata cannot downgrade or bind current authority'
            using errcode = '23514';
        end if;
        return;
      elsif v_desired_authority_state = 'CHECKPOINT' then
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
        raise exception 'ARTIFACT_REUSE_NOT_ADMITTED: unsupported authority transition'
          using errcode = '23514';
      end if;

      -- The manifest itself is inserted/promoted before its output rows.
      if p_manifest_artifact_id is null then
        if v_existing.authority_state = v_desired_authority_state then
          return;
        end if;
        if v_existing.authority_state = 'NON_AUTHORITATIVE'
           and p_artifact->>'artifact_type' like '%_STAGE_MANIFEST' then
          update public.orotitan_artifacts
          set authority_state = v_desired_authority_state
          where artifact_id = v_existing.artifact_id
            and version = v_existing.version;
          return;
        end if;
        raise exception 'ARTIFACT_AUTHORITY_BINDING_CONFLICT: unbound artifact cannot change authority'
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

      -- Exact staged metadata can only become current authority through this
      -- manifest-bound path after persistence validation has succeeded.
      if v_existing.authority_state = 'NON_AUTHORITATIVE' then
        update public.orotitan_artifacts
        set authority_state = v_desired_authority_state,
            manifest_artifact_id = p_manifest_artifact_id,
            manifest_version = p_manifest_version
        where artifact_id = v_existing.artifact_id
          and version = v_existing.version;
        return;
      end if;

      -- True same-manifest replay is a no-op.
      if v_existing.authority_state = v_desired_authority_state
         and v_existing.manifest_artifact_id is not distinct from p_manifest_artifact_id
         and v_existing.manifest_version is not distinct from p_manifest_version then
        return;
      end if;

      -- Existing current authority may move only when attributable to the
      -- stage's active predecessor manifest.
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
$function$;

-- Management-plane preparation step. It records the expected immutable
-- metadata only; it does not grant analytical authority and cannot admit a
-- downstream stage.
create or replace function public.stage_orotitan_persistence_candidate(
  p_run_id uuid,
  p_stage_code text,
  p_artifact jsonb
)
returns jsonb
language plpgsql
set search_path = pg_catalog, public
as $function$
declare
  v_run public.orotitan_runs%rowtype;
  v_stage public.orotitan_run_stages%rowtype;
  v_candidate jsonb;
  v_artifact_id uuid;
  v_version integer;
  v_size bigint;
begin
  if coalesce(jsonb_typeof(p_artifact),'null') <> 'object' then
    raise exception 'PERSISTENCE_CANDIDATE_INVALID: artifact registration must be object'
      using errcode = '22023';
  end if;

  begin
    v_artifact_id := (p_artifact->>'artifact_id')::uuid;
    v_version := (p_artifact->>'version')::integer;
    v_size := (p_artifact->>'size_bytes')::bigint;
  exception when others then
    raise exception 'PERSISTENCE_CANDIDATE_INVALID: artifact_id/version/size invalid'
      using errcode = '22023';
  end;

  if v_version < 1
     or v_size < 0
     or p_artifact->>'artifact_type' is null
     or p_artifact->>'logical_name' is null
     or p_artifact->>'media_type' is null
     or p_artifact->>'authority_class' not in ('AUTHORITATIVE_STAGE_OUTPUT','CHECKPOINT_STAGE_OUTPUT')
     or p_artifact->>'content_sha256' !~ '^[0-9a-f]{64}$'
     or p_artifact->>'storage_backend' is distinct from 'PRIVATE_GITHUB'
     or p_artifact->>'github_repository' is distinct from 'robzer13/real-orotitan'
     or coalesce(length(p_artifact->>'github_path'),0) = 0
     or p_artifact->>'github_commit_sha' !~ '^[0-9a-f]{40}$'
     or p_artifact->>'github_blob_sha' !~ '^[0-9a-f]{40}$'
     or p_artifact->>'storage_uri' is distinct from
        'github://' || (p_artifact->>'github_repository') || '@' ||
        (p_artifact->>'github_commit_sha') || '/' || (p_artifact->>'github_path') then
    raise exception 'PERSISTENCE_CANDIDATE_INVALID: immutable artifact metadata invalid'
      using errcode = '23514';
  end if;

  select * into v_run
  from public.orotitan_runs
  where run_id=p_run_id;

  select * into v_stage
  from public.orotitan_run_stages
  where run_id=p_run_id and stage_code=p_stage_code;

  if not found
     or v_run.run_id is null
     or v_run.current_stage is distinct from p_stage_code
     or v_run.run_status in ('PUBLISHED','CANCELLED','READY_TO_PUBLISH')
     or v_stage.lifecycle_status = 'COMPLETE' then
    raise exception 'PERSISTENCE_CANDIDATE_STAGE_NOT_ACTIVE'
      using errcode = '23514';
  end if;

  v_candidate := p_artifact || jsonb_build_object(
    'artifact_status','SEALED',
    'authority_state','NON_AUTHORITATIVE',
    'availability_state','AVAILABLE'
  );

  perform public.orotitan_insert_artifact_registration(
    p_run_id,p_stage_code,v_candidate,null,null
  );

  return jsonb_build_object(
    'artifact_id',v_artifact_id,
    'version',v_version,
    'run_id',p_run_id,
    'stage_code',p_stage_code,
    'authority_state','NON_AUTHORITATIVE'
  );
end;
$function$;

-- Authority reconciliation gate. The external management plane performs the
-- immutable GitHub reread; this RPC independently requires the resulting
-- receipt to equal the Registry's already-staged immutable artifact metadata.
create or replace function public.attest_orotitan_persistence_locator(
  p_run_id uuid,
  p_stage_code text,
  p_attestation_payload jsonb
)
returns uuid
language plpgsql
security definer
set search_path = pg_catalog, public
as $function$
declare
  v_event_id uuid;
  v_fingerprint text;
  v_idempotency_key text;
  v_retry jsonb;
  v_artifact public.orotitan_artifacts%rowtype;
  v_payload_run_id uuid;
  v_artifact_id uuid;
  v_version integer;
  v_size bigint;
  v_verified_at timestamptz;
  v_expected_uri text;
begin
  if coalesce(jsonb_typeof(p_attestation_payload),'null') <> 'object'
     or p_attestation_payload->>'attestation_schema_version' is distinct from '1.0'
     or p_attestation_payload->>'verification_method' is distinct from 'GITHUB_CONNECTOR_PRIVATE_REREAD_V1'
     or p_attestation_payload->>'trust_boundary' is distinct from 'SUPABASE_MANAGEMENT_PLANE'
     or p_attestation_payload->>'storage_backend' is distinct from 'PRIVATE_GITHUB'
     or p_attestation_payload->>'github_repository' is distinct from 'robzer13/real-orotitan'
     or p_attestation_payload->>'github_commit_sha' !~ '^[0-9a-f]{40}$'
     or p_attestation_payload->>'github_blob_sha' !~ '^[0-9a-f]{40}$'
     or p_attestation_payload->>'content_sha256' !~ '^[0-9a-f]{64}$'
     or p_attestation_payload->>'commit_path_resolved' is distinct from 'true'
     or coalesce(length(p_attestation_payload->>'github_path'),0) = 0
     or p_attestation_payload->>'storage_uri' is null
     or p_attestation_payload->>'artifact_type' is null
     or p_attestation_payload->>'stage_code' is distinct from p_stage_code then
    raise exception 'PERSISTENCE_ATTESTATION_INVALID'
      using errcode = '23514';
  end if;

  begin
    v_payload_run_id := (p_attestation_payload->>'run_id')::uuid;
    v_artifact_id := (p_attestation_payload->>'artifact_id')::uuid;
    v_version := (p_attestation_payload->>'version')::integer;
    v_size := (p_attestation_payload->>'size_bytes')::bigint;
    v_verified_at := (p_attestation_payload->>'verified_at')::timestamptz;
  exception when others then
    raise exception 'PERSISTENCE_ATTESTATION_INVALID'
      using errcode = '22023';
  end;

  if v_payload_run_id is distinct from p_run_id
     or v_version < 1
     or v_size < 0 then
    raise exception 'PERSISTENCE_ATTESTATION_INVALID'
      using errcode = '23514';
  end if;

  select * into v_artifact
  from public.orotitan_artifacts
  where artifact_id=v_artifact_id
    and version=v_version
    and run_id=p_run_id
    and stage_code=p_stage_code
  for share;

  if not found then
    raise exception 'PERSISTENCE_ATTESTATION_ARTIFACT_NOT_REGISTERED'
      using errcode = '23514';
  end if;

  v_expected_uri :=
    'github://' || v_artifact.github_repository || '@' ||
    v_artifact.github_commit_sha || '/' || v_artifact.github_path;

  if v_artifact.artifact_status <> 'SEALED'
     or v_artifact.availability_state <> 'AVAILABLE'
     or v_artifact.hash_algorithm <> 'SHA-256'
     or v_artifact.storage_backend <> 'PRIVATE_GITHUB'
     or v_artifact.github_repository is distinct from 'robzer13/real-orotitan'
     or v_artifact.github_commit_sha !~ '^[0-9a-f]{40}$'
     or v_artifact.github_blob_sha !~ '^[0-9a-f]{40}$'
     or v_artifact.content_sha256 !~ '^[0-9a-f]{64}$'
     or v_artifact.size_bytes < 0
     or v_artifact.storage_uri is distinct from v_expected_uri
     or p_attestation_payload->>'artifact_type' is distinct from v_artifact.artifact_type
     or v_size is distinct from v_artifact.size_bytes
     or p_attestation_payload->>'content_sha256' is distinct from v_artifact.content_sha256
     or p_attestation_payload->>'github_repository' is distinct from v_artifact.github_repository
     or p_attestation_payload->>'github_path' is distinct from v_artifact.github_path
     or p_attestation_payload->>'github_commit_sha' is distinct from v_artifact.github_commit_sha
     or p_attestation_payload->>'github_blob_sha' is distinct from v_artifact.github_blob_sha
     or p_attestation_payload->>'storage_uri' is distinct from v_artifact.storage_uri then
    raise exception 'PERSISTENCE_ATTESTATION_AUTHORITY_MISMATCH'
      using errcode = '23514';
  end if;

  v_fingerprint := encode(
    extensions.digest(convert_to(p_attestation_payload::text,'UTF8'),'sha256'),
    'hex'
  );

  -- Stable logical operation identity: deliberately excludes request fingerprint.
  v_idempotency_key := 'persistence-attest:' ||
    v_artifact_id::text || ':' || v_version::text;

  v_retry := public.orotitan_existing_event(
    p_run_id,v_idempotency_key,v_fingerprint
  );
  if v_retry is not null then
    if v_retry->>'event_type' is distinct from 'PERSISTENCE_ATTESTED' then
      raise exception 'IDEMPOTENCY_CONFLICT: persistence key belongs to different event type'
        using errcode = '23514';
    end if;
    return (v_retry->>'event_id')::uuid;
  end if;

  if v_artifact.authority_state <> 'NON_AUTHORITATIVE' then
    raise exception 'PERSISTENCE_ATTESTATION_ARTIFACT_NOT_STAGED'
      using errcode = '23514';
  end if;

  if not exists (
    select 1
    from public.orotitan_runs r
    join public.orotitan_run_stages s on s.run_id=r.run_id
    where r.run_id=p_run_id
      and r.current_stage=p_stage_code
      and r.run_status not in ('PUBLISHED','CANCELLED','READY_TO_PUBLISH')
      and s.stage_code=p_stage_code
      and s.lifecycle_status <> 'COMPLETE'
  ) then
    raise exception 'PERSISTENCE_ATTESTATION_STAGE_NOT_ACTIVE'
      using errcode = '23514';
  end if;

  begin
    v_event_id := public.orotitan_insert_event(
      p_run_id,p_stage_code,'PERSISTENCE_ATTESTED',
      v_idempotency_key,v_fingerprint,'SYSTEM',p_attestation_payload
    );
  exception when unique_violation then
    v_retry := public.orotitan_existing_event(
      p_run_id,v_idempotency_key,v_fingerprint
    );
    if v_retry is null then
      raise;
    end if;
    if v_retry->>'event_type' is distinct from 'PERSISTENCE_ATTESTED' then
      raise exception 'IDEMPOTENCY_CONFLICT: persistence key belongs to different event type'
        using errcode = '23514';
    end if;
    return (v_retry->>'event_id')::uuid;
  end;

  return v_event_id;
end;
$function$;

revoke all on function public.stage_orotitan_persistence_candidate(uuid,text,jsonb)
  from public,anon,authenticated,service_role;
revoke all on function public.attest_orotitan_persistence_locator(uuid,text,jsonb)
  from public,anon,authenticated,service_role;

comment on function public.stage_orotitan_persistence_candidate(uuid,text,jsonb)
is 'V1.11 management-plane preparation: records exact persisted artifact metadata as NON_AUTHORITATIVE only; final authority remains manifest/finalizer controlled.';

comment on function public.attest_orotitan_persistence_locator(uuid,text,jsonb)
is 'V1.11 persistence authority gate: exact Registry-to-private-GitHub receipt reconciliation with stable idempotency identity and reachable fingerprint conflict.';

commit;

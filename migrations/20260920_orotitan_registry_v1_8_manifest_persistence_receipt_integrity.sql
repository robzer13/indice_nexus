-- OroTitan Registry V1.8 — manifest persistence-receipt integrity boundary.
-- Emergency V2.0.x implementation repair classification:
-- PERSISTENCE / REGISTRY DATA-INTEGRITY DEFECT
-- Methodology / analytical change: NONE.
--
-- Affected-run revalidation is admitted from verified active-manifest bytes,
-- never from an assumed-complete historical edge inventory.
-- This migration does not mutate historical Registry rows. It makes every
-- future Stage Manifest registration fail closed unless the caller supplies
-- exact reread bytes from the immutable PRIVATE_GITHUB object and the Registry
-- can independently recompute size, SHA-256 and Git blob SHA from those bytes.

begin;

do $$
begin
  if to_regprocedure(
       'public.revalidate_orotitan_checkpoint_outputs(uuid,text,bigint,bigint,uuid,integer,text,jsonb,text,text)'
     ) is null then
    raise exception 'OroTitan Registry V1.6 checkpoint revalidation must be deployed before V1.8';
  end if;
  if to_regprocedure(
       'public.orotitan_register_manifest_bundle(uuid,text,jsonb,jsonb,jsonb,jsonb)'
     ) is null
     or to_regprocedure(
       'public.orotitan_insert_artifact_registration(uuid,text,jsonb,uuid,integer)'
     ) is null
     or to_regprocedure(
       'public.supersede_orotitan_manifest_bundle()'
     ) is null then
    raise exception 'OroTitan Registry V1.7 successor rebinding primitives must be deployed before V1.8';
  end if;
  if to_regprocedure('extensions.digest(bytea,text)') is null then
    raise exception 'pgcrypto digest(bytea,text) is required';
  end if;
end;
$$;

create or replace function public.orotitan_validate_manifest_persistence_receipt(
  p_manifest jsonb,
  p_manifest_registration jsonb
)
returns void
language plpgsql
set search_path = pg_catalog, public
as $$
declare
  v_receipt jsonb := p_manifest_registration->'persistence_receipt';
  v_bytes bytea;
  v_text text;
  v_persisted_manifest jsonb;
  v_size bigint;
  v_sha256 text;
  v_git_blob_sha text;
  v_repository text := p_manifest_registration->>'github_repository';
  v_path text := p_manifest_registration->>'github_path';
  v_commit text := p_manifest_registration->>'github_commit_sha';
  v_blob text := p_manifest_registration->>'github_blob_sha';
  v_expected_uri text;
begin
  if jsonb_typeof(p_manifest) <> 'object'
     or jsonb_typeof(p_manifest_registration) <> 'object' then
    raise exception 'MANIFEST_PERSISTENCE_RECEIPT_INVALID: manifest and registration must be objects'
      using errcode = '22023';
  end if;

  if p_manifest_registration->>'artifact_type' is null
     or p_manifest_registration->>'artifact_type' not in (
       'RESEARCH_STAGE_MANIFEST',
       'DEEP_DIVE_STAGE_MANIFEST',
       'INTEGRATION_STAGE_MANIFEST'
     ) then
    raise exception 'MANIFEST_PERSISTENCE_RECEIPT_INVALID: unsupported manifest artifact type'
      using errcode = '23514';
  end if;

  if p_manifest_registration->>'artifact_id' is distinct from p_manifest->>'manifest_id' then
    raise exception 'MANIFEST_PERSISTENCE_RECEIPT_INVALID: manifest identity mismatch'
      using errcode = '23514';
  end if;

  if p_manifest_registration->>'storage_backend' is distinct from 'PRIVATE_GITHUB' then
    raise exception 'MANIFEST_PERSISTENCE_RECEIPT_UNSUPPORTED_BACKEND: Stage Manifests require PRIVATE_GITHUB exact-byte verification'
      using errcode = '23514';
  end if;

  if v_repository is null
     or v_path is null
     or v_commit is null
     or v_blob is null
     or p_manifest_registration->>'storage_uri' is null then
    raise exception 'MANIFEST_PERSISTENCE_RECEIPT_INVALID: GitHub provenance is incomplete'
      using errcode = '22023';
  end if;

  if v_commit !~ '^[0-9a-f]{40}$' or v_blob !~ '^[0-9a-f]{40}$' then
    raise exception 'MANIFEST_PERSISTENCE_RECEIPT_INVALID: GitHub commit/blob format invalid'
      using errcode = '22023';
  end if;

  if coalesce(jsonb_typeof(v_receipt), 'null') <> 'object'
     or v_receipt->>'receipt_schema_version' is distinct from '1.0'
     or v_receipt->>'verification_method' is distinct from 'PRIVATE_GITHUB_REREAD_EXACT_BYTES_V1'
     or v_receipt->>'storage_backend' is distinct from 'PRIVATE_GITHUB'
     or v_receipt->>'github_repository' is null
     or v_receipt->>'github_path' is null
     or v_receipt->>'github_commit_sha' is null
     or v_receipt->>'github_blob_sha' is null
     or v_receipt->>'verified_content_base64' is null
     or v_receipt->>'verified_at' is null
     or coalesce((v_receipt->>'commit_path_resolved')::boolean, false) is not true then
    raise exception 'MANIFEST_PERSISTENCE_RECEIPT_MISSING_OR_INCOMPLETE'
      using errcode = '23514';
  end if;

  if v_receipt->>'github_repository' is distinct from v_repository
     or v_receipt->>'github_path' is distinct from v_path
     or v_receipt->>'github_commit_sha' is distinct from v_commit
     or v_receipt->>'github_blob_sha' is distinct from v_blob then
    raise exception 'MANIFEST_PERSISTENCE_RECEIPT_PROVENANCE_MISMATCH'
      using errcode = '23514';
  end if;

  begin
    perform (v_receipt->>'verified_at')::timestamptz;
  exception when others then
    raise exception 'MANIFEST_PERSISTENCE_RECEIPT_INVALID: verified_at is not a timestamp'
      using errcode = '22023';
  end;

  begin
    v_bytes := decode(v_receipt->>'verified_content_base64', 'base64');
  exception when others then
    raise exception 'MANIFEST_PERSISTENCE_RECEIPT_INVALID: verified bytes are not valid base64'
      using errcode = '22023';
  end;

  v_size := octet_length(v_bytes);
  v_sha256 := encode(extensions.digest(v_bytes, 'sha256'), 'hex');
  v_git_blob_sha := encode(
    extensions.digest(
      convert_to('blob ' || v_size::text, 'UTF8') || decode('00', 'hex') || v_bytes,
      'sha1'
    ),
    'hex'
  );

  if (p_manifest_registration->>'size_bytes')::bigint is distinct from v_size then
    raise exception 'MANIFEST_VERIFIED_SIZE_MISMATCH: declared %, verified %',
      p_manifest_registration->>'size_bytes', v_size
      using errcode = '23514';
  end if;

  if p_manifest_registration->>'content_sha256' is distinct from v_sha256 then
    raise exception 'MANIFEST_VERIFIED_SHA256_MISMATCH: declared %, verified %',
      p_manifest_registration->>'content_sha256', v_sha256
      using errcode = '23514';
  end if;

  if v_blob is distinct from v_git_blob_sha then
    raise exception 'MANIFEST_VERIFIED_GIT_BLOB_MISMATCH: declared %, verified %',
      v_blob, v_git_blob_sha
      using errcode = '23514';
  end if;

  v_expected_uri := 'github://' || v_repository || '@' || v_commit || '/' || v_path;
  if p_manifest_registration->>'storage_uri' is distinct from v_expected_uri then
    raise exception 'MANIFEST_PERSISTENCE_RECEIPT_PROVENANCE_MISMATCH: storage_uri does not match immutable GitHub locator'
      using errcode = '23514';
  end if;

  begin
    v_text := convert_from(v_bytes, 'UTF8');
    v_persisted_manifest := v_text::jsonb;
  exception when others then
    raise exception 'MANIFEST_PERSISTENCE_RECEIPT_INVALID: persisted bytes are not valid UTF-8 JSON'
      using errcode = '23514';
  end;

  if v_persisted_manifest is distinct from p_manifest then
    raise exception 'MANIFEST_VERIFIED_BYTES_PAYLOAD_MISMATCH: persisted manifest JSON differs from submitted manifest'
      using errcode = '23514';
  end if;
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

  -- V1.8 integrity boundary: validate exact reread immutable manifest bytes
  -- before the first artifact/edge/state mutation can survive.
  perform public.orotitan_validate_manifest_persistence_receipt(
    p_manifest,
    p_manifest_registration
  );

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


create or replace function public.orotitan_verify_existing_manifest_persistence_receipt(
  p_run_id uuid,
  p_stage_code text,
  p_manifest_artifact_id uuid,
  p_manifest_version integer,
  p_expected_registered_sha256 text,
  p_expected_actual_sha256 text,
  p_receipt jsonb
)
returns jsonb
language plpgsql
set search_path = pg_catalog, public
as $$
declare
  v_row public.orotitan_artifacts%rowtype;
  v_bytes bytea;
  v_text text;
  v_manifest jsonb;
  v_size bigint;
  v_sha256 text;
  v_git_blob_sha text;
  v_expected_uri text;
begin
  if p_expected_registered_sha256 !~ '^[0-9a-f]{64}$'
     or p_expected_actual_sha256 !~ '^[0-9a-f]{64}$' then
    raise exception 'EXISTING_MANIFEST_RECEIPT_INVALID: expected SHA format invalid'
      using errcode = '22023';
  end if;

  select * into v_row
  from public.orotitan_artifacts
  where artifact_id = p_manifest_artifact_id
    and version = p_manifest_version
    and run_id = p_run_id
    and stage_code = p_stage_code;

  if not found then
    raise exception 'ACTIVE_MANIFEST_NOT_FOUND' using errcode = 'P0002';
  end if;

  if v_row.artifact_type not in (
       'RESEARCH_STAGE_MANIFEST',
       'DEEP_DIVE_STAGE_MANIFEST',
       'INTEGRATION_STAGE_MANIFEST'
     )
     or v_row.artifact_status <> 'SEALED'
     or v_row.availability_state <> 'AVAILABLE'
     or v_row.authority_state <> 'CHECKPOINT'
     or v_row.storage_backend <> 'PRIVATE_GITHUB'
     or v_row.content_sha256 <> p_expected_registered_sha256 then
    raise exception 'ACTIVE_MANIFEST_REGISTRY_STATE_MISMATCH' using errcode = '23514';
  end if;

  if coalesce(jsonb_typeof(p_receipt), 'null') <> 'object'
     or p_receipt->>'receipt_schema_version' is distinct from '1.0'
     or p_receipt->>'verification_method' is distinct from 'PRIVATE_GITHUB_REREAD_EXACT_BYTES_V1'
     or p_receipt->>'storage_backend' is distinct from 'PRIVATE_GITHUB'
     or coalesce((p_receipt->>'commit_path_resolved')::boolean, false) is not true
     or p_receipt->>'verified_content_base64' is null
     or p_receipt->>'verified_at' is null then
    raise exception 'EXISTING_MANIFEST_RECEIPT_MISSING_OR_INCOMPLETE'
      using errcode = '23514';
  end if;

  if p_receipt->>'github_repository' is distinct from v_row.github_repository
     or p_receipt->>'github_path' is distinct from v_row.github_path
     or p_receipt->>'github_commit_sha' is distinct from v_row.github_commit_sha
     or p_receipt->>'github_blob_sha' is distinct from v_row.github_blob_sha then
    raise exception 'EXISTING_MANIFEST_RECEIPT_PROVENANCE_MISMATCH'
      using errcode = '23514';
  end if;

  begin
    perform (p_receipt->>'verified_at')::timestamptz;
    v_bytes := decode(p_receipt->>'verified_content_base64', 'base64');
  exception when others then
    raise exception 'EXISTING_MANIFEST_RECEIPT_INVALID_ENCODING'
      using errcode = '22023';
  end;

  v_size := octet_length(v_bytes);
  v_sha256 := encode(extensions.digest(v_bytes, 'sha256'), 'hex');
  v_git_blob_sha := encode(
    extensions.digest(
      convert_to('blob ' || v_size::text, 'UTF8') || decode('00', 'hex') || v_bytes,
      'sha1'
    ),
    'hex'
  );

  if v_size is distinct from v_row.size_bytes then
    raise exception 'EXISTING_MANIFEST_VERIFIED_SIZE_MISMATCH' using errcode = '23514';
  end if;
  if v_sha256 is distinct from p_expected_actual_sha256 then
    raise exception 'EXISTING_MANIFEST_VERIFIED_SHA256_MISMATCH' using errcode = '23514';
  end if;
  if v_git_blob_sha is distinct from v_row.github_blob_sha then
    raise exception 'EXISTING_MANIFEST_VERIFIED_GIT_BLOB_MISMATCH' using errcode = '23514';
  end if;

  v_expected_uri := 'github://' || v_row.github_repository || '@' || v_row.github_commit_sha || '/' || v_row.github_path;
  if v_row.storage_uri is distinct from v_expected_uri then
    raise exception 'EXISTING_MANIFEST_STORAGE_URI_MISMATCH' using errcode = '23514';
  end if;

  begin
    v_text := convert_from(v_bytes, 'UTF8');
    v_manifest := v_text::jsonb;
  exception when others then
    raise exception 'EXISTING_MANIFEST_VERIFIED_BYTES_NOT_UTF8_JSON'
      using errcode = '23514';
  end;

  if v_manifest->>'manifest_id' is distinct from p_manifest_artifact_id::text
     or v_manifest->>'run_id' is distinct from p_run_id::text
     or v_manifest->>'stage' is distinct from p_stage_code
     or v_manifest->>'manifest_kind' is distinct from 'CHECKPOINT' then
    raise exception 'EXISTING_MANIFEST_VERIFIED_PAYLOAD_IDENTITY_MISMATCH'
      using errcode = '23514';
  end if;

  return jsonb_build_object(
    'manifest', v_manifest,
    'registered_sha256', v_row.content_sha256,
    'actual_sha256', v_sha256,
    'size_bytes', v_size,
    'git_blob_sha', v_git_blob_sha
  );
end;
$$;

drop function if exists public.revalidate_orotitan_checkpoint_outputs(
  uuid, text, bigint, bigint, uuid, integer, text, jsonb, text, text
);

create or replace function public.revalidate_orotitan_checkpoint_outputs(
  p_run_id uuid,
  p_stage_code text,
  p_expected_run_state_version bigint,
  p_expected_stage_state_version bigint,
  p_expected_manifest_artifact_id uuid,
  p_expected_manifest_version integer,
  p_expected_registered_manifest_sha256 text,
  p_expected_actual_manifest_sha256 text,
  p_verified_manifest_receipt jsonb,
  p_target_artifacts jsonb,
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
  v_target public.orotitan_artifacts%rowtype;
  v_verification jsonb;
  v_manifest jsonb;
  v_ref jsonb;
  v_manifest_ref jsonb;
  v_retry jsonb;
  v_event_id uuid;
  v_target_count integer;
  v_updated integer;
  v_new_run_state bigint;
  v_new_stage_state bigint;
  v_match_count integer;
begin
  if p_idempotency_key is null or length(btrim(p_idempotency_key)) = 0 then
    raise exception 'idempotency key is required' using errcode = '22023';
  end if;
  if p_request_fingerprint_sha256 !~ '^[0-9a-f]{64}$' then
    raise exception 'invalid request fingerprint' using errcode = '22023';
  end if;
  if coalesce(jsonb_typeof(p_target_artifacts), 'null') <> 'array'
     or jsonb_array_length(p_target_artifacts) = 0 then
    raise exception 'target artifacts must be a non-empty array' using errcode = '22023';
  end if;
  if exists (
    select 1
    from jsonb_array_elements(p_target_artifacts) x
    group by x->>'artifact_id', x->>'version'
    having count(*) > 1
  ) then
    raise exception 'duplicate target artifact reference' using errcode = '23514';
  end if;

  v_retry := public.orotitan_existing_event(
    p_run_id, p_idempotency_key, p_request_fingerprint_sha256
  );
  if v_retry is not null then
    if v_retry->>'event_type' <> 'BLOCKER_RESOLVED'
       or v_retry->'payload'->>'repair_class' <> 'VERIFIED_EXISTING_OUTPUT_AUTHORITY_REVALIDATION'
       or v_retry->'payload'->>'active_manifest_id' <> p_expected_manifest_artifact_id::text
       or (v_retry->'payload'->>'active_manifest_version')::integer <> p_expected_manifest_version
       or v_retry->'payload'->>'active_manifest_actual_sha256' <> p_expected_actual_manifest_sha256 then
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
  if not found then raise exception 'RUN_NOT_FOUND' using errcode = 'P0002'; end if;
  if v_run.run_status in ('PUBLISHED','CANCELLED') then
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
  if not found then raise exception 'STAGE_NOT_FOUND' using errcode = 'P0002'; end if;
  if v_stage.state_version <> p_expected_stage_state_version then
    raise exception 'STAGE_STATE_VERSION_MISMATCH' using errcode = '40001';
  end if;
  if v_stage.lifecycle_status <> 'IN_PROGRESS'
     or v_stage.active_manifest_kind <> 'CHECKPOINT'
     or v_stage.active_manifest_artifact_id is distinct from p_expected_manifest_artifact_id
     or v_stage.active_manifest_version is distinct from p_expected_manifest_version then
    raise exception 'ACTIVE_CHECKPOINT_MISMATCH' using errcode = '23514';
  end if;

  v_verification := public.orotitan_verify_existing_manifest_persistence_receipt(
    p_run_id,
    p_stage_code,
    p_expected_manifest_artifact_id,
    p_expected_manifest_version,
    p_expected_registered_manifest_sha256,
    p_expected_actual_manifest_sha256,
    p_verified_manifest_receipt
  );
  v_manifest := v_verification->'manifest';

  perform public.orotitan_validate_manifest_lock(v_run, v_stage, v_manifest, 'CHECKPOINT');

  if coalesce(jsonb_typeof(v_manifest->'output_artifacts'), 'null') <> 'array' then
    raise exception 'ACTIVE_MANIFEST_OUTPUT_SET_MALFORMED' using errcode = '23514';
  end if;

  select jsonb_array_length(p_target_artifacts) into v_target_count;

  for v_ref in select value from jsonb_array_elements(p_target_artifacts)
  loop
    if (v_ref->>'artifact_id') is null
       or (v_ref->>'version') is null
       or (v_ref->>'artifact_type') is null
       or (v_ref->>'content_sha256') is null
       or (v_ref->>'authority_class') is null
       or (v_ref->>'media_type') is null
       or (v_ref->>'size_bytes') is null then
      raise exception 'target artifact reference is incomplete' using errcode = '22023';
    end if;

    select count(*), jsonb_agg(value)->0
      into v_match_count, v_manifest_ref
    from jsonb_array_elements(v_manifest->'output_artifacts')
    where value->>'artifact_id' = v_ref->>'artifact_id'
      and value->>'version' = v_ref->>'version';

    if v_match_count <> 1 then
      raise exception 'TARGET_NOT_EXACTLY_ONCE_IN_ACTIVE_MANIFEST' using errcode = '23514';
    end if;

    if v_manifest_ref->>'artifact_type' <> v_ref->>'artifact_type'
       or v_manifest_ref->>'content_sha256' <> v_ref->>'content_sha256'
       or v_manifest_ref->>'authority_class' <> v_ref->>'authority_class'
       or v_manifest_ref->>'media_type' <> v_ref->>'media_type'
       or (v_manifest_ref->>'size_bytes')::bigint <> (v_ref->>'size_bytes')::bigint then
      raise exception 'TARGET_REFERENCE_DIFFERS_FROM_ACTIVE_MANIFEST' using errcode = '23514';
    end if;

    select * into v_target
    from public.orotitan_artifacts
    where artifact_id = (v_ref->>'artifact_id')::uuid
      and version = (v_ref->>'version')::integer;
    if not found then raise exception 'TARGET_ARTIFACT_NOT_FOUND' using errcode = 'P0002'; end if;

    if v_target.run_id <> p_run_id
       or v_target.stage_code <> p_stage_code
       or v_target.artifact_type <> v_ref->>'artifact_type'
       or v_target.authority_class <> 'CHECKPOINT_STAGE_OUTPUT'
       or v_target.authority_class <> v_ref->>'authority_class'
       or v_target.media_type <> v_ref->>'media_type'
       or v_target.content_sha256 <> v_ref->>'content_sha256'
       or v_target.size_bytes <> (v_ref->>'size_bytes')::bigint
       or v_target.artifact_status <> 'SEALED'
       or v_target.availability_state <> 'AVAILABLE'
       or v_target.authority_state not in ('SUPERSEDED','CHECKPOINT') then
      raise exception 'TARGET_ARTIFACT_VALIDATION_FAILED' using errcode = '23514';
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
        raise exception 'SUPERSEDED_TARGET_NOT_ATTRIBUTABLE_TO_ACTIVE_SUCCESSOR'
          using errcode = '23514';
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
  from jsonb_array_elements(p_target_artifacts) r
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
  if not found then raise exception 'STAGE_STATE_VERSION_MISMATCH' using errcode = '40001'; end if;

  update public.orotitan_runs
     set state_version = state_version + 1
   where run_id = p_run_id
     and state_version = p_expected_run_state_version
  returning state_version into v_new_run_state;
  if not found then raise exception 'RUN_STATE_VERSION_MISMATCH' using errcode = '40001'; end if;

  v_event_id := public.orotitan_insert_event(
    p_run_id,
    p_stage_code,
    'BLOCKER_RESOLVED',
    p_idempotency_key,
    p_request_fingerprint_sha256,
    'SYSTEM',
    jsonb_build_object(
      'blocker','REUSED_CHECKPOINT_OUTPUT_AUTHORITY_BINDING_DEFECT',
      'repair_class','VERIFIED_EXISTING_OUTPUT_AUTHORITY_REVALIDATION',
      'active_manifest_id',p_expected_manifest_artifact_id,
      'active_manifest_version',p_expected_manifest_version,
      'active_manifest_registered_sha256',p_expected_registered_manifest_sha256,
      'active_manifest_actual_sha256',p_expected_actual_manifest_sha256,
      'artifact_count',v_target_count,
      'analytical_state_changed',false
    )
  );

  return jsonb_build_object(
    'run_id',p_run_id,
    'stage',p_stage_code,
    'manifest_id',p_expected_manifest_artifact_id,
    'manifest_version',p_expected_manifest_version,
    'artifact_count',v_target_count,
    'run_state_version',v_new_run_state,
    'stage_state_version',v_new_stage_state,
    'event_id',v_event_id,
    'idempotent_replay',false
  );
end;
$$;

revoke all on function public.orotitan_verify_existing_manifest_persistence_receipt(
  uuid, text, uuid, integer, text, text, jsonb
) from public, anon, authenticated, service_role;

revoke all on function public.revalidate_orotitan_checkpoint_outputs(
  uuid, text, bigint, bigint, uuid, integer, text, text, jsonb, jsonb, text, text
) from public, anon, authenticated;
grant execute on function public.revalidate_orotitan_checkpoint_outputs(
  uuid, text, bigint, bigint, uuid, integer, text, text, jsonb, jsonb, text, text
) to service_role;

comment on function public.revalidate_orotitan_checkpoint_outputs(
  uuid, text, bigint, bigint, uuid, integer, text, text, jsonb, jsonb, text, text
) is 'Canonical V1.8 existing-output authority repair. Admits only explicit targets proven exactly once in the verified immutable active CHECKPOINT manifest bytes; preserves analytical/provenance fields and uses CAS plus idempotency.';

revoke all on function public.orotitan_validate_manifest_persistence_receipt(jsonb,jsonb)
  from public, anon, authenticated, service_role;
revoke all on function public.orotitan_register_manifest_bundle(uuid,text,jsonb,jsonb,jsonb,jsonb)
  from public, anon, authenticated, service_role;

comment on function public.orotitan_validate_manifest_persistence_receipt(jsonb,jsonb) is
  'Registry V1.8 fail-closed Stage Manifest persistence receipt validator. Recomputes exact-byte size, SHA-256 and Git blob SHA, reconciles immutable GitHub provenance, and requires byte payload JSON equality before registration.';

comment on function public.orotitan_register_manifest_bundle(uuid,text,jsonb,jsonb,jsonb,jsonb) is
  'Canonical manifest bundle registration boundary. Since V1.8, Stage Manifest registration first requires exact-byte PRIVATE_GITHUB reread receipt verification; no caller-supplied unverified content SHA can activate a manifest.';

commit;

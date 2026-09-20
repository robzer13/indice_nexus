-- OroTitan Registry forward repair — bundle-wide persistence receipt closure.
-- Generated migration identity: 20260920203621_orotitan_registry_bundle_persistence_integrity.sql
-- Classification: PERSISTENCE / REGISTRY DATA-INTEGRITY DEFECT.
-- Analytical methodology change: NONE.
-- Historical Registry row mutation in migration: NONE.
-- Deployed V1.6 / V1.7 / V1.8 are predecessors and are not rewritten.

begin;

do $$
begin
  if to_regprocedure('public.orotitan_validate_manifest_persistence_receipt(jsonb,jsonb)') is null
     or to_regprocedure('public.orotitan_register_manifest_bundle(uuid,text,jsonb,jsonb,jsonb,jsonb)') is null
     or to_regprocedure('public.orotitan_insert_event(uuid,text,text,text,text,text,jsonb)') is null
     or to_regprocedure('public.orotitan_existing_event(uuid,text,text)') is null
     or to_regclass('public.orotitan_run_events') is null then
    raise exception 'OroTitan Registry V1.8 primitives must exist before bundle persistence-integrity migration';
  end if;
end;
$$;

create or replace function public.orotitan_validate_bundle_object_persistence_receipt(
  p_run_id uuid,
  p_stage_code text,
  p_artifact jsonb,
  p_manifest_artifact_id uuid default null,
  p_manifest_version integer default null,
  p_require_locator_authority boolean default true
)
returns void
language plpgsql
set search_path = pg_catalog, public
as $$
declare
  v_receipt jsonb := p_artifact->'persistence_receipt';
  v_artifact_id uuid;
  v_version integer;
  v_bytes bytea;
  v_text text;
  v_payload jsonb;
  v_size bigint;
  v_sha256 text;
  v_git_blob_sha text;
  v_repository text := p_artifact->>'github_repository';
  v_path text := p_artifact->>'github_path';
  v_commit text := p_artifact->>'github_commit_sha';
  v_blob text := p_artifact->>'github_blob_sha';
  v_stage_path text;
  v_expected_prefix text;
  v_expected_uri text;
  v_expected_suffix text;
  v_attestation_fingerprint text;
begin
  if coalesce(jsonb_typeof(p_artifact), 'null') <> 'object' then
    raise exception 'BUNDLE_PERSISTENCE_RECEIPT_INVALID: artifact registration must be an object'
      using errcode = '22023';
  end if;
  if p_stage_code not in ('RESEARCH','DEEP_DIVE','INTEGRATION') then
    raise exception 'BUNDLE_PERSISTENCE_RECEIPT_INVALID: unsupported stage %', p_stage_code
      using errcode = '22023';
  end if;
  if (p_manifest_artifact_id is null) <> (p_manifest_version is null) then
    raise exception 'BUNDLE_PERSISTENCE_RECEIPT_INVALID: manifest binding id/version must be supplied together'
      using errcode = '22023';
  end if;

  begin
    v_artifact_id := (p_artifact->>'artifact_id')::uuid;
    v_version := (p_artifact->>'version')::integer;
  exception when others then
    raise exception 'BUNDLE_PERSISTENCE_RECEIPT_INVALID: artifact identity/version invalid'
      using errcode = '22023';
  end;
  if v_artifact_id is null or v_version is null or v_version < 1 then
    raise exception 'BUNDLE_PERSISTENCE_RECEIPT_INVALID: artifact identity/version missing or invalid'
      using errcode = '22023';
  end if;

  if p_artifact->>'artifact_type' is null
     or p_artifact->>'media_type' is null
     or p_artifact->>'size_bytes' is null
     or p_artifact->>'content_sha256' is null
     or p_artifact->>'storage_uri' is null then
    raise exception 'BUNDLE_PERSISTENCE_RECEIPT_INVALID: required immutable metadata missing'
      using errcode = '22023';
  end if;
  if p_artifact->>'content_sha256' !~ '^[0-9a-f]{64}$'
     or (p_artifact->>'size_bytes')::bigint < 0 then
    raise exception 'BUNDLE_PERSISTENCE_RECEIPT_INVALID: hash/size format invalid'
      using errcode = '22023';
  end if;
  if p_artifact->>'storage_backend' is distinct from 'PRIVATE_GITHUB' then
    raise exception 'BUNDLE_PERSISTENCE_RECEIPT_UNSUPPORTED_BACKEND: registered bundle objects require PRIVATE_GITHUB'
      using errcode = '23514';
  end if;

  if v_repository is distinct from 'robzer13/real-orotitan' then
    raise exception 'IMMUTABLE_LOCATOR_AUTHORITY_MISMATCH: repository is not the canonical private artifact repository'
      using errcode = '23514';
  end if;
  if v_path is null or v_commit is null or v_blob is null
     or v_commit !~ '^[0-9a-f]{40}$'
     or v_blob !~ '^[0-9a-f]{40}$' then
    raise exception 'BUNDLE_PERSISTENCE_RECEIPT_INVALID: immutable GitHub locator incomplete or malformed'
      using errcode = '22023';
  end if;

  v_stage_path := case p_stage_code
    when 'RESEARCH' then 'research'
    when 'DEEP_DIVE' then 'deep_dive'
    when 'INTEGRATION' then 'integration'
  end;
  v_expected_prefix := 'artifacts/orotitan-equity/runs/' || p_run_id::text || '/' || v_stage_path || '/';
  if left(v_path, length(v_expected_prefix)) is distinct from v_expected_prefix then
    raise exception 'IMMUTABLE_LOCATOR_AUTHORITY_MISMATCH: path is outside canonical run/stage namespace'
      using errcode = '23514';
  end if;
  v_expected_suffix := '__' || v_artifact_id::text || '__v' || lpad(v_version::text, 3, '0');
  if position(v_expected_suffix in regexp_replace(v_path, '^.*/', '')) = 0 then
    raise exception 'IMMUTABLE_LOCATOR_AUTHORITY_MISMATCH: path does not bind artifact_id/version'
      using errcode = '23514';
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
    raise exception 'BUNDLE_PERSISTENCE_RECEIPT_MISSING_OR_INCOMPLETE'
      using errcode = '23514';
  end if;
  if v_receipt->>'github_repository' is distinct from v_repository
     or v_receipt->>'github_path' is distinct from v_path
     or v_receipt->>'github_commit_sha' is distinct from v_commit
     or v_receipt->>'github_blob_sha' is distinct from v_blob then
    raise exception 'BUNDLE_PERSISTENCE_RECEIPT_PROVENANCE_MISMATCH'
      using errcode = '23514';
  end if;

  begin
    perform (v_receipt->>'verified_at')::timestamptz;
    v_bytes := decode(v_receipt->>'verified_content_base64', 'base64');
  exception when others then
    raise exception 'BUNDLE_PERSISTENCE_RECEIPT_INVALID: verified_at/base64 invalid'
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

  if (p_artifact->>'size_bytes')::bigint is distinct from v_size then
    raise exception 'BUNDLE_VERIFIED_SIZE_MISMATCH: declared %, verified %', p_artifact->>'size_bytes', v_size
      using errcode = '23514';
  end if;
  if p_artifact->>'content_sha256' is distinct from v_sha256 then
    raise exception 'BUNDLE_VERIFIED_SHA256_MISMATCH: declared %, verified %', p_artifact->>'content_sha256', v_sha256
      using errcode = '23514';
  end if;
  if v_blob is distinct from v_git_blob_sha then
    raise exception 'BUNDLE_VERIFIED_GIT_BLOB_MISMATCH: declared %, verified %', v_blob, v_git_blob_sha
      using errcode = '23514';
  end if;

  v_expected_uri := 'github://' || v_repository || '@' || v_commit || '/' || v_path;
  if p_artifact->>'storage_uri' is distinct from v_expected_uri then
    raise exception 'BUNDLE_PERSISTENCE_RECEIPT_PROVENANCE_MISMATCH: storage_uri does not match immutable locator'
      using errcode = '23514';
  end if;

  begin
    v_text := convert_from(v_bytes, 'UTF8');
  exception when others then
    raise exception 'BUNDLE_PERSISTENCE_RECEIPT_INVALID: persisted bytes are not valid UTF-8'
      using errcode = '23514';
  end;

  if p_artifact->>'media_type' = 'application/json' then
    begin
      v_payload := v_text::jsonb;
    exception when others then
      raise exception 'BUNDLE_PERSISTENCE_RECEIPT_INVALID: JSON artifact bytes are not valid JSON'
        using errcode = '23514';
    end;
    if jsonb_typeof(v_payload) = 'object' then
      if v_payload ? 'artifact_id' and v_payload->>'artifact_id' is distinct from v_artifact_id::text then
        raise exception 'BUNDLE_VERIFIED_ARTIFACT_ID_MISMATCH' using errcode = '23514';
      end if;
      if v_payload ? 'manifest_id' and p_artifact->>'artifact_type' like '%_STAGE_MANIFEST'
         and v_payload->>'manifest_id' is distinct from v_artifact_id::text then
        raise exception 'BUNDLE_VERIFIED_ARTIFACT_ID_MISMATCH' using errcode = '23514';
      end if;
      if v_payload ? 'version' and (v_payload->>'version')::integer is distinct from v_version then
        raise exception 'BUNDLE_VERIFIED_VERSION_MISMATCH' using errcode = '23514';
      end if;
      if v_payload ? 'run_id' and v_payload->>'run_id' is distinct from p_run_id::text then
        raise exception 'BUNDLE_VERIFIED_RUN_ID_MISMATCH' using errcode = '23514';
      end if;
      if v_payload ? 'stage' and v_payload->>'stage' is distinct from p_stage_code then
        raise exception 'BUNDLE_VERIFIED_STAGE_MISMATCH' using errcode = '23514';
      end if;
      if v_payload ? 'artifact_type' and v_payload->>'artifact_type' is distinct from p_artifact->>'artifact_type' then
        raise exception 'BUNDLE_VERIFIED_ARTIFACT_TYPE_MISMATCH' using errcode = '23514';
      end if;
    end if;
  end if;

  if p_require_locator_authority then
    v_attestation_fingerprint := encode(
      extensions.digest(
        convert_to(
          jsonb_build_object(
            'attestation_version','1.0',
            'run_id',p_run_id,
            'stage_code',p_stage_code,
            'artifact_id',v_artifact_id,
            'version',v_version,
            'manifest_artifact_id',p_manifest_artifact_id,
            'manifest_version',p_manifest_version,
            'content_sha256',v_sha256,
            'size_bytes',v_size,
            'github_repository',v_repository,
            'github_path',v_path,
            'github_commit_sha',v_commit,
            'github_blob_sha',v_blob
          )::text,
          'UTF8'
        ),
        'sha256'
      ),
      'hex'
    );

    if not exists (
      select 1
      from public.orotitan_run_events e
      where e.run_id = p_run_id
        and e.stage_code = p_stage_code
        and e.event_type = 'ARTIFACT_SET_SEALED'
        and e.actor_type = 'SYSTEM'
        and e.request_fingerprint_sha256 = v_attestation_fingerprint
        and e.payload->>'seal_class' = 'PERSISTENCE_ATTESTATION_V2'
        and e.payload->>'artifact_id' = v_artifact_id::text
        and (e.payload->>'version')::integer = v_version
        and e.payload->>'manifest_artifact_id' is not distinct from p_manifest_artifact_id::text
        and nullif(e.payload->>'manifest_version','')::integer is not distinct from p_manifest_version
        and e.payload->>'content_sha256' = v_sha256
        and (e.payload->>'size_bytes')::bigint = v_size
        and e.payload->>'github_repository' = v_repository
        and e.payload->>'github_path' = v_path
        and e.payload->>'github_commit_sha' = v_commit
        and e.payload->>'github_blob_sha' = v_blob
    ) then
      raise exception 'IMMUTABLE_LOCATOR_AUTHORITY_MISSING_OR_MISMATCHED'
        using errcode = '23514';
    end if;
  end if;
end;
$$;

create or replace function public.attest_orotitan_bundle_object_persistence(
  p_run_id uuid,
  p_stage_code text,
  p_artifact jsonb,
  p_manifest_artifact_id uuid default null,
  p_manifest_version integer default null
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_run public.orotitan_runs%rowtype;
  v_stage public.orotitan_run_stages%rowtype;
  v_artifact_id uuid := (p_artifact->>'artifact_id')::uuid;
  v_version integer := (p_artifact->>'version')::integer;
  v_fingerprint text;
  v_idempotency_key text;
  v_retry jsonb;
  v_event_id uuid;
  v_payload jsonb;
begin
  select * into v_run from public.orotitan_runs where run_id = p_run_id;
  if not found then raise exception 'RUN_NOT_FOUND' using errcode = 'P0002'; end if;
  if v_run.run_status in ('PUBLISHED','CANCELLED') then raise exception 'RUN_TERMINAL' using errcode = '23514'; end if;
  if v_run.current_stage is distinct from p_stage_code then raise exception 'STAGE_NOT_CURRENT' using errcode = '23514'; end if;

  select * into v_stage from public.orotitan_run_stages where run_id=p_run_id and stage_code=p_stage_code;
  if not found then raise exception 'STAGE_NOT_FOUND' using errcode = 'P0002'; end if;
  if v_stage.lifecycle_status = 'COMPLETE' then raise exception 'STAGE_ALREADY_COMPLETE' using errcode = '23514'; end if;

  perform public.orotitan_validate_bundle_object_persistence_receipt(
    p_run_id,p_stage_code,p_artifact,p_manifest_artifact_id,p_manifest_version,false
  );

  v_fingerprint := encode(
    extensions.digest(
      convert_to(
        jsonb_build_object(
          'attestation_version','1.0',
          'run_id',p_run_id,
          'stage_code',p_stage_code,
          'artifact_id',v_artifact_id,
          'version',v_version,
          'manifest_artifact_id',p_manifest_artifact_id,
          'manifest_version',p_manifest_version,
          'content_sha256',p_artifact->>'content_sha256',
          'size_bytes',(p_artifact->>'size_bytes')::bigint,
          'github_repository',p_artifact->>'github_repository',
          'github_path',p_artifact->>'github_path',
          'github_commit_sha',p_artifact->>'github_commit_sha',
          'github_blob_sha',p_artifact->>'github_blob_sha'
        )::text,
        'UTF8'
      ),
      'sha256'
    ),
    'hex'
  );

  v_idempotency_key := 'registry:persistence-attestation:v2:' || v_artifact_id::text || ':' || v_version::text || ':' ||
    coalesce(p_manifest_artifact_id::text,'UNBOUND') || ':' || coalesce(p_manifest_version::text,'0');

  v_retry := public.orotitan_existing_event(p_run_id,v_idempotency_key,v_fingerprint);
  if v_retry is not null then
    if v_retry->>'event_type' <> 'ARTIFACT_SET_SEALED'
       or v_retry->>'actor_type' <> 'SYSTEM'
       or v_retry->'payload'->>'seal_class' <> 'PERSISTENCE_ATTESTATION_V2' then
      raise exception 'IDEMPOTENCY_CONFLICT: key belongs to a different operation'
        using errcode = '23514';
    end if;
    return jsonb_build_object('event_id',v_retry->>'event_id','idempotent_replay',true,'request_fingerprint_sha256',v_fingerprint);
  end if;

  v_payload := jsonb_build_object(
    'seal_class','PERSISTENCE_ATTESTATION_V2',
    'attestation_version','1.0',
    'attestation_authority','POSTGRES_MANAGEMENT_PLANE_AFTER_INDEPENDENT_GITHUB_REREAD',
    'run_id',p_run_id,
    'stage_code',p_stage_code,
    'artifact_id',v_artifact_id,
    'version',v_version,
    'manifest_artifact_id',p_manifest_artifact_id,
    'manifest_version',p_manifest_version,
    'content_sha256',p_artifact->>'content_sha256',
    'size_bytes',(p_artifact->>'size_bytes')::bigint,
    'github_repository',p_artifact->>'github_repository',
    'github_path',p_artifact->>'github_path',
    'github_commit_sha',p_artifact->>'github_commit_sha',
    'github_blob_sha',p_artifact->>'github_blob_sha',
    'receipt_verified_at',p_artifact->'persistence_receipt'->>'verified_at'
  );

  v_event_id := public.orotitan_insert_event(
    p_run_id,p_stage_code,'ARTIFACT_SET_SEALED',v_idempotency_key,
    v_fingerprint,'SYSTEM',v_payload
  );

  return jsonb_build_object('event_id',v_event_id,'idempotent_replay',false,'request_fingerprint_sha256',v_fingerprint);
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
  select * into v_stage from public.orotitan_run_stages
  where run_id = p_run_id and stage_code = p_stage_code;
  if not found then raise exception 'STAGE_NOT_FOUND' using errcode = 'P0002'; end if;

  if v_manifest_kind = 'CHECKPOINT' then
    v_expected_authority_class := 'CHECKPOINT_STAGE_OUTPUT';
    v_expected_authority_state := 'CHECKPOINT';
  elsif v_manifest_kind = 'FINAL' then
    v_expected_authority_class := 'AUTHORITATIVE_STAGE_OUTPUT';
    v_expected_authority_state := 'AUTHORITATIVE';
  else
    raise exception 'MANIFEST_CONTRACT_MISMATCH: unsupported manifest kind' using errcode = '23514';
  end if;

  if (p_manifest->>'run_id')::uuid is distinct from p_run_id
     or p_manifest->>'stage' is distinct from p_stage_code
     or (p_manifest->>'stage_revision')::integer is distinct from v_stage.stage_revision then
    raise exception 'MANIFEST_CONTRACT_MISMATCH: manifest run/stage/revision mismatch' using errcode = '23514';
  end if;
  if p_manifest_registration->>'artifact_id' <> v_manifest_id::text then
    raise exception 'MANIFEST_CONTRACT_MISMATCH: manifest_id != manifest artifact_id' using errcode = '23514';
  end if;
  if p_manifest_registration->>'authority_class' is distinct from v_expected_authority_class
     or p_manifest_registration->>'authority_state' is distinct from v_expected_authority_state
     or coalesce(p_manifest_registration->>'artifact_status','SEALED') <> 'SEALED'
     or coalesce(p_manifest_registration->>'availability_state','AVAILABLE') <> 'AVAILABLE' then
    raise exception 'MANIFEST_CONTRACT_MISMATCH: manifest registration authority/status mismatch' using errcode = '23514';
  end if;

  if coalesce(jsonb_typeof(p_output_artifacts),'null') <> 'array'
     or coalesce(jsonb_typeof(p_edges),'null') <> 'array'
     or coalesce(jsonb_typeof(p_manifest->'output_artifacts'),'null') <> 'array' then
    raise exception 'manifest bundle arrays are malformed' using errcode = '22023';
  end if;
  if jsonb_array_length(p_output_artifacts) <> jsonb_array_length(p_manifest->'output_artifacts') then
    raise exception 'ARTIFACT_NOT_IN_MANIFEST: output artifact count mismatch' using errcode = '23514';
  end if;
  if exists (
    select 1 from jsonb_array_elements(p_output_artifacts) x
    group by x->>'artifact_id',x->>'version' having count(*) > 1
  ) then
    raise exception 'ARTIFACT_NOT_IN_MANIFEST: duplicate output artifact registration' using errcode = '23514';
  end if;

  perform public.orotitan_validate_manifest_persistence_receipt(p_manifest,p_manifest_registration);
  perform public.orotitan_validate_bundle_object_persistence_receipt(
    p_run_id,p_stage_code,p_manifest_registration,null,null,true
  );

  for v_ref in select value from jsonb_array_elements(p_manifest->'output_artifacts')
  loop
    if v_ref->>'authority_class' is distinct from v_expected_authority_class then
      raise exception 'MANIFEST_CONTRACT_MISMATCH: output reference authority class does not match manifest kind' using errcode = '23514';
    end if;
    select value into v_match
    from jsonb_array_elements(p_output_artifacts)
    where value->>'artifact_id'=v_ref->>'artifact_id' and value->>'version'=v_ref->>'version';

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
      raise exception 'ARTIFACT_NOT_IN_MANIFEST: output artifact mismatch' using errcode = '23514';
    end if;

    perform public.orotitan_validate_bundle_object_persistence_receipt(
      p_run_id,p_stage_code,v_match,v_manifest_id,v_manifest_version,true
    );
  end loop;

  for v_output in select value from jsonb_array_elements(p_output_artifacts)
  loop
    select count(*) into v_count
    from jsonb_array_elements(p_manifest->'output_artifacts')
    where value->>'artifact_id'=v_output->>'artifact_id' and value->>'version'=v_output->>'version';
    if v_count <> 1 then
      raise exception 'ARTIFACT_NOT_IN_MANIFEST: extra or duplicate output artifact' using errcode = '23514';
    end if;
  end loop;

  perform public.orotitan_insert_artifact_registration(
    p_run_id,p_stage_code,p_manifest_registration,null,null
  );

  for v_ref in select value from jsonb_array_elements(p_manifest->'output_artifacts')
  loop
    select value into v_match
    from jsonb_array_elements(p_output_artifacts)
    where value->>'artifact_id'=v_ref->>'artifact_id' and value->>'version'=v_ref->>'version';
    perform public.orotitan_insert_artifact_registration(
      p_run_id,p_stage_code,v_match,v_manifest_id,v_manifest_version
    );
  end loop;

  for v_edge in select value from jsonb_array_elements(p_edges)
  loop
    insert into public.orotitan_artifact_edges(
      child_run_id,child_artifact_id,child_version,
      parent_run_id,parent_artifact_id,parent_version,relation_type
    ) values (
      coalesce((v_edge->>'child_run_id')::uuid,p_run_id),
      (v_edge->>'child_artifact_id')::uuid,
      (v_edge->>'child_version')::integer,
      (v_edge->>'parent_run_id')::uuid,
      (v_edge->>'parent_artifact_id')::uuid,
      (v_edge->>'parent_version')::integer,
      v_edge->>'relation_type'
    ) on conflict do nothing;
  end loop;
end;
$$;

revoke all on function public.attest_orotitan_bundle_object_persistence(uuid,text,jsonb,uuid,integer)
  from public, anon, authenticated, service_role;
revoke all on function public.orotitan_validate_bundle_object_persistence_receipt(uuid,text,jsonb,uuid,integer,boolean)
  from public, anon, authenticated, service_role;
revoke all on function public.orotitan_register_manifest_bundle(uuid,text,jsonb,jsonb,jsonb,jsonb)
  from public, anon, authenticated, service_role;

commit;

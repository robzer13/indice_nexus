-- OroTitan Registry — bundle-wide persistence integrity remediation.
-- Generated migration identity: 20260920203621_orotitan_registry_bundle_persistence_integrity
-- Scope: global / issuer-independent / append-only / forward-only.
-- No analytical methodology change. No historical Registry row rewrite.

begin;

do $$
begin
  if to_regclass('public.orotitan_run_events') is null
     or to_regclass('public.orotitan_artifacts') is null
     or to_regprocedure('public.orotitan_register_manifest_bundle(uuid,text,jsonb,jsonb,jsonb,jsonb)') is null
     or to_regprocedure('public.orotitan_validate_manifest_persistence_receipt(jsonb,jsonb)') is null
     or to_regprocedure('public.orotitan_verify_existing_manifest_persistence_receipt(uuid,text,uuid,integer,text,text,jsonb)') is null then
    raise exception 'OroTitan Registry V1.8 frontier must exist before bundle persistence-integrity remediation';
  end if;
end;
$$;

-- The immutable run-event ledger is reused as the separately authenticated
-- persistence-attestation ledger. Direct INSERT remains owner/management-plane
-- only; service_role retains SELECT only and cannot mint attestations.
alter table public.orotitan_run_events
  drop constraint if exists orotitan_run_events_event_type_check;

alter table public.orotitan_run_events
  add constraint orotitan_run_events_event_type_check
  check (event_type = any (array[
    'RUN_CREATED'::text,
    'RUN_IDENTITY_BOUND'::text,
    'STAGE_STARTED'::text,
    'STAGE_CHECKPOINTED'::text,
    'STAGE_PAUSED'::text,
    'STAGE_RESUMED'::text,
    'BLOCKER_OPENED'::text,
    'BLOCKER_RESOLVED'::text,
    'STAGE_REOPENED'::text,
    'ARTIFACT_SET_SEALED'::text,
    'STAGE_FINALIZED'::text,
    'READY_TO_PUBLISH_DECLARED'::text,
    'PUBLISH_AUTHORIZED'::text,
    'PUBLISH_SUCCEEDED'::text,
    'PUBLISH_FAILED'::text,
    'RUN_CANCELLED'::text,
    'PERSISTENCE_ATTESTED'::text
  ]));

create or replace function public.orotitan_private_github_repository_is_approved(
  p_repository text
)
returns boolean
language sql
immutable
strict
set search_path = pg_catalog, public
as $$
  select p_repository = 'robzer13/real-orotitan';
$$;

create or replace function public.orotitan_verify_attested_persistence_receipt(
  p_run_id uuid,
  p_stage_code text,
  p_registration jsonb,
  p_manifest_ref jsonb default null,
  p_expected_json jsonb default null
)
returns jsonb
language plpgsql
set search_path = pg_catalog, public
as $$
declare
  v_receipt jsonb := p_registration->'persistence_receipt';
  v_attestation_event_id uuid;
  v_event public.orotitan_run_events%rowtype;
  v_attestation jsonb;
  v_artifact_id uuid;
  v_version integer;
  v_artifact_type text := p_registration->>'artifact_type';
  v_repository text := p_registration->>'github_repository';
  v_path text := p_registration->>'github_path';
  v_commit text := p_registration->>'github_commit_sha';
  v_blob text := p_registration->>'github_blob_sha';
  v_expected_uri text;
  v_bytes bytea;
  v_size bigint;
  v_sha256 text;
  v_git_blob_sha text;
  v_text text;
  v_json jsonb;
  v_event_fingerprint text;
  v_receipt_verified_at timestamptz;
  v_attestation_verified_at timestamptz;
begin
  if jsonb_typeof(p_registration) <> 'object' then
    raise exception 'PERSISTENCE_RECEIPT_INVALID: registration must be an object'
      using errcode = '22023';
  end if;

  begin
    v_artifact_id := (p_registration->>'artifact_id')::uuid;
    v_version := (p_registration->>'version')::integer;
  exception when others then
    raise exception 'PERSISTENCE_RECEIPT_INVALID: artifact identity/version malformed'
      using errcode = '22023';
  end;

  if v_version < 1 or v_artifact_type is null or length(btrim(v_artifact_type)) = 0 then
    raise exception 'PERSISTENCE_RECEIPT_INVALID: artifact identity/version/type incomplete'
      using errcode = '22023';
  end if;

  if p_registration->>'storage_backend' is distinct from 'PRIVATE_GITHUB' then
    raise exception 'PERSISTENCE_RECEIPT_UNSUPPORTED_BACKEND: governed stage bundle objects require PRIVATE_GITHUB'
      using errcode = '23514';
  end if;

  if v_repository is null or not public.orotitan_private_github_repository_is_approved(v_repository) then
    raise exception 'PERSISTENCE_REPOSITORY_AUTHORITY_MISMATCH: repository is not approved'
      using errcode = '23514';
  end if;

  if v_path is null or length(btrim(v_path)) = 0
     or v_commit !~ '^[0-9a-f]{40}$'
     or v_blob !~ '^[0-9a-f]{40}$'
     or p_registration->>'content_sha256' !~ '^[0-9a-f]{64}$'
     or p_registration->>'size_bytes' !~ '^[0-9]+$' then
    raise exception 'PERSISTENCE_RECEIPT_INVALID: immutable registration metadata malformed'
      using errcode = '22023';
  end if;

  v_expected_uri := 'github://' || v_repository || '@' || v_commit || '/' || v_path;
  if p_registration->>'storage_uri' is distinct from v_expected_uri then
    raise exception 'PERSISTENCE_LOCATOR_MISMATCH: storage_uri does not match repository/commit/path'
      using errcode = '23514';
  end if;

  if coalesce(jsonb_typeof(v_receipt), 'null') <> 'object'
     or v_receipt->>'receipt_schema_version' is distinct from '1.1'
     or v_receipt->>'verification_method' is distinct from 'PRIVATE_GITHUB_ATTESTED_REREAD_EXACT_BYTES_V1'
     or v_receipt->>'storage_backend' is distinct from 'PRIVATE_GITHUB'
     or v_receipt->>'run_id' is null
     or v_receipt->>'stage_code' is null
     or v_receipt->>'artifact_id' is null
     or v_receipt->>'version' is null
     or v_receipt->>'artifact_type' is null
     or v_receipt->>'github_repository' is null
     or v_receipt->>'github_path' is null
     or v_receipt->>'github_commit_sha' is null
     or v_receipt->>'github_blob_sha' is null
     or v_receipt->>'attestation_event_id' is null
     or v_receipt->>'verified_content_base64' is null
     or v_receipt->>'verified_at' is null
     or coalesce((v_receipt->>'commit_path_resolved')::boolean, false) is not true then
    raise exception 'PERSISTENCE_RECEIPT_MISSING_OR_INCOMPLETE'
      using errcode = '23514';
  end if;

  if v_receipt->>'run_id' is distinct from p_run_id::text
     or v_receipt->>'stage_code' is distinct from p_stage_code
     or v_receipt->>'artifact_id' is distinct from v_artifact_id::text
     or v_receipt->>'version' is distinct from v_version::text
     or v_receipt->>'artifact_type' is distinct from v_artifact_type then
    raise exception 'PERSISTENCE_RECEIPT_IDENTITY_VERSION_MISMATCH'
      using errcode = '23514';
  end if;

  if v_receipt->>'github_repository' is distinct from v_repository
     or v_receipt->>'github_path' is distinct from v_path
     or v_receipt->>'github_commit_sha' is distinct from v_commit
     or v_receipt->>'github_blob_sha' is distinct from v_blob then
    raise exception 'PERSISTENCE_RECEIPT_PROVENANCE_MISMATCH'
      using errcode = '23514';
  end if;

  begin
    v_attestation_event_id := (v_receipt->>'attestation_event_id')::uuid;
    v_receipt_verified_at := (v_receipt->>'verified_at')::timestamptz;
    v_bytes := decode(v_receipt->>'verified_content_base64', 'base64');
  exception when others then
    raise exception 'PERSISTENCE_RECEIPT_INVALID_ENCODING_OR_TIMESTAMP'
      using errcode = '22023';
  end;

  select * into v_event
  from public.orotitan_run_events
  where event_id = v_attestation_event_id
    and run_id = p_run_id
    and stage_code = p_stage_code
    and event_type = 'PERSISTENCE_ATTESTED'
    and actor_type = 'SYSTEM';

  if not found then
    raise exception 'PERSISTENCE_ATTESTATION_NOT_FOUND_OR_UNTRUSTED'
      using errcode = '23514';
  end if;

  v_attestation := v_event.payload;
  v_event_fingerprint := encode(
    extensions.digest(convert_to(v_attestation::text, 'UTF8'), 'sha256'),
    'hex'
  );

  if v_event.request_fingerprint_sha256 is distinct from v_event_fingerprint then
    raise exception 'PERSISTENCE_ATTESTATION_FINGERPRINT_MISMATCH'
      using errcode = '23514';
  end if;

  if coalesce(jsonb_typeof(v_attestation), 'null') <> 'object'
     or v_attestation->>'attestation_schema_version' is distinct from '1.0'
     or v_attestation->>'verification_method' is distinct from 'GITHUB_CONNECTOR_PRIVATE_REREAD_V1'
     or v_attestation->>'trust_boundary' is distinct from 'SUPABASE_MANAGEMENT_PLANE'
     or v_attestation->>'storage_backend' is distinct from 'PRIVATE_GITHUB'
     or coalesce((v_attestation->>'commit_path_resolved')::boolean, false) is not true then
    raise exception 'PERSISTENCE_ATTESTATION_INVALID'
      using errcode = '23514';
  end if;

  begin
    v_attestation_verified_at := (v_attestation->>'verified_at')::timestamptz;
  exception when others then
    raise exception 'PERSISTENCE_ATTESTATION_INVALID_TIMESTAMP'
      using errcode = '22023';
  end;

  if v_attestation->>'run_id' is distinct from p_run_id::text
     or v_attestation->>'stage_code' is distinct from p_stage_code
     or v_attestation->>'artifact_id' is distinct from v_artifact_id::text
     or v_attestation->>'version' is distinct from v_version::text
     or v_attestation->>'artifact_type' is distinct from v_artifact_type
     or v_attestation->>'github_repository' is distinct from v_repository
     or v_attestation->>'github_path' is distinct from v_path
     or v_attestation->>'github_commit_sha' is distinct from v_commit
     or v_attestation->>'github_blob_sha' is distinct from v_blob
     or v_attestation->>'storage_uri' is distinct from v_expected_uri then
    raise exception 'PERSISTENCE_ATTESTATION_BINDING_MISMATCH'
      using errcode = '23514';
  end if;

  if v_attestation_verified_at is distinct from v_receipt_verified_at then
    raise exception 'PERSISTENCE_ATTESTATION_TIMESTAMP_MISMATCH'
      using errcode = '23514';
  end if;

  v_size := octet_length(v_bytes);
  v_sha256 := encode(extensions.digest(v_bytes, 'sha256'), 'hex');
  v_git_blob_sha := encode(
    extensions.digest(
      convert_to('blob ' || v_size::text, 'UTF8') || decode('00', 'hex') || v_bytes,
      'sha1'
    ),
    'hex'
  );

  if (p_registration->>'size_bytes')::bigint is distinct from v_size
     or p_registration->>'content_sha256' is distinct from v_sha256
     or v_blob is distinct from v_git_blob_sha then
    raise exception 'PERSISTENCE_VERIFIED_BYTE_IDENTITY_MISMATCH'
      using errcode = '23514';
  end if;

  if v_attestation->>'size_bytes' !~ '^[0-9]+$'
     or v_attestation->>'content_sha256' !~ '^[0-9a-f]{64}$'
     or (v_attestation->>'size_bytes')::bigint is distinct from v_size
     or v_attestation->>'content_sha256' is distinct from v_sha256
     or v_attestation->>'github_blob_sha' is distinct from v_git_blob_sha then
    raise exception 'PERSISTENCE_ATTESTATION_VERIFIED_BYTE_MISMATCH'
      using errcode = '23514';
  end if;

  if p_manifest_ref is not null then
    if jsonb_typeof(p_manifest_ref) <> 'object'
       or p_manifest_ref->>'artifact_id' is distinct from v_artifact_id::text
       or p_manifest_ref->>'version' is distinct from v_version::text
       or p_manifest_ref->>'artifact_type' is distinct from v_artifact_type
       or p_manifest_ref->>'content_sha256' is distinct from v_sha256
       or p_manifest_ref->>'size_bytes' is distinct from v_size::text
       or coalesce(jsonb_typeof(p_manifest_ref->'storage_ref'), 'null') <> 'object'
       or p_manifest_ref->'storage_ref'->>'backend' is distinct from 'PRIVATE_GITHUB'
       or p_manifest_ref->'storage_ref'->>'repository' is distinct from v_repository
       or p_manifest_ref->'storage_ref'->>'path' is distinct from v_path
       or p_manifest_ref->'storage_ref'->>'commit_sha' is distinct from v_commit
       or p_manifest_ref->'storage_ref'->>'blob_sha' is distinct from v_blob
       or p_manifest_ref->'storage_ref'->>'storage_uri' is distinct from v_expected_uri then
      raise exception 'PERSISTENCE_MANIFEST_REFERENCE_MISMATCH'
        using errcode = '23514';
    end if;
  end if;

  if p_expected_json is not null or p_registration ? 'canonical_json_content' then
    begin
      v_text := convert_from(v_bytes, 'UTF8');
      v_json := v_text::jsonb;
    exception when others then
      raise exception 'PERSISTENCE_VERIFIED_BYTES_NOT_UTF8_JSON'
        using errcode = '23514';
    end;

    if p_expected_json is not null and v_json is distinct from p_expected_json then
      raise exception 'PERSISTENCE_VERIFIED_JSON_SEMANTIC_MISMATCH'
        using errcode = '23514';
    end if;

    if p_registration ? 'canonical_json_content'
       and v_json is distinct from p_registration->'canonical_json_content' then
      raise exception 'PERSISTENCE_REGISTERED_JSON_SEMANTIC_MISMATCH'
        using errcode = '23514';
    end if;
  end if;

  return jsonb_build_object(
    'artifact_id', v_artifact_id,
    'version', v_version,
    'artifact_type', v_artifact_type,
    'size_bytes', v_size,
    'content_sha256', v_sha256,
    'git_blob_sha', v_git_blob_sha,
    'storage_uri', v_expected_uri,
    'attestation_event_id', v_attestation_event_id,
    'verified_json', v_json
  );
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

  perform public.orotitan_verify_attested_persistence_receipt(
    (p_manifest->>'run_id')::uuid,
    p_manifest->>'stage',
    p_manifest_registration,
    null,
    p_manifest
  );
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
  v_attestation_event_id uuid;
  v_event public.orotitan_run_events%rowtype;
  v_attestation jsonb;
  v_bytes bytea;
  v_text text;
  v_manifest jsonb;
  v_size bigint;
  v_sha256 text;
  v_git_blob_sha text;
  v_expected_uri text;
  v_event_fingerprint text;
  v_receipt_verified_at timestamptz;
  v_attestation_verified_at timestamptz;
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

  if not public.orotitan_private_github_repository_is_approved(v_row.github_repository) then
    raise exception 'EXISTING_MANIFEST_REPOSITORY_AUTHORITY_MISMATCH' using errcode = '23514';
  end if;

  if coalesce(jsonb_typeof(p_receipt), 'null') <> 'object'
     or p_receipt->>'receipt_schema_version' is distinct from '1.1'
     or p_receipt->>'verification_method' is distinct from 'PRIVATE_GITHUB_ATTESTED_REREAD_EXACT_BYTES_V1'
     or p_receipt->>'storage_backend' is distinct from 'PRIVATE_GITHUB'
     or p_receipt->>'run_id' is distinct from p_run_id::text
     or p_receipt->>'stage_code' is distinct from p_stage_code
     or p_receipt->>'artifact_id' is distinct from p_manifest_artifact_id::text
     or p_receipt->>'version' is distinct from p_manifest_version::text
     or p_receipt->>'artifact_type' is distinct from v_row.artifact_type
     or coalesce((p_receipt->>'commit_path_resolved')::boolean, false) is not true
     or p_receipt->>'attestation_event_id' is null
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
    v_attestation_event_id := (p_receipt->>'attestation_event_id')::uuid;
    v_receipt_verified_at := (p_receipt->>'verified_at')::timestamptz;
    v_bytes := decode(p_receipt->>'verified_content_base64', 'base64');
  exception when others then
    raise exception 'EXISTING_MANIFEST_RECEIPT_INVALID_ENCODING'
      using errcode = '22023';
  end;

  select * into v_event
  from public.orotitan_run_events
  where event_id = v_attestation_event_id
    and run_id = p_run_id
    and stage_code = p_stage_code
    and event_type = 'PERSISTENCE_ATTESTED'
    and actor_type = 'SYSTEM';

  if not found then
    raise exception 'EXISTING_MANIFEST_ATTESTATION_NOT_FOUND_OR_UNTRUSTED'
      using errcode = '23514';
  end if;

  v_attestation := v_event.payload;
  v_event_fingerprint := encode(
    extensions.digest(convert_to(v_attestation::text, 'UTF8'), 'sha256'),
    'hex'
  );
  if v_event.request_fingerprint_sha256 is distinct from v_event_fingerprint then
    raise exception 'EXISTING_MANIFEST_ATTESTATION_FINGERPRINT_MISMATCH'
      using errcode = '23514';
  end if;

  if v_attestation->>'attestation_schema_version' is distinct from '1.0'
     or v_attestation->>'verification_method' is distinct from 'GITHUB_CONNECTOR_PRIVATE_REREAD_V1'
     or v_attestation->>'trust_boundary' is distinct from 'SUPABASE_MANAGEMENT_PLANE'
     or v_attestation->>'storage_backend' is distinct from 'PRIVATE_GITHUB'
     or coalesce((v_attestation->>'commit_path_resolved')::boolean, false) is not true
     or v_attestation->>'run_id' is distinct from p_run_id::text
     or v_attestation->>'stage_code' is distinct from p_stage_code
     or v_attestation->>'artifact_id' is distinct from p_manifest_artifact_id::text
     or v_attestation->>'version' is distinct from p_manifest_version::text
     or v_attestation->>'artifact_type' is distinct from v_row.artifact_type
     or v_attestation->>'github_repository' is distinct from v_row.github_repository
     or v_attestation->>'github_path' is distinct from v_row.github_path
     or v_attestation->>'github_commit_sha' is distinct from v_row.github_commit_sha
     or v_attestation->>'github_blob_sha' is distinct from v_row.github_blob_sha then
    raise exception 'EXISTING_MANIFEST_ATTESTATION_BINDING_MISMATCH'
      using errcode = '23514';
  end if;

  begin
    v_attestation_verified_at := (v_attestation->>'verified_at')::timestamptz;
  exception when others then
    raise exception 'EXISTING_MANIFEST_ATTESTATION_INVALID_TIMESTAMP'
      using errcode = '22023';
  end;

  if v_attestation_verified_at is distinct from v_receipt_verified_at then
    raise exception 'EXISTING_MANIFEST_ATTESTATION_TIMESTAMP_MISMATCH'
      using errcode = '23514';
  end if;

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
  if v_attestation->>'size_bytes' is distinct from v_size::text
     or v_attestation->>'content_sha256' is distinct from v_sha256
     or v_attestation->>'github_blob_sha' is distinct from v_git_blob_sha then
    raise exception 'EXISTING_MANIFEST_ATTESTATION_VERIFIED_BYTE_MISMATCH'
      using errcode = '23514';
  end if;

  v_expected_uri := 'github://' || v_row.github_repository || '@' || v_row.github_commit_sha || '/' || v_row.github_path;
  if v_row.storage_uri is distinct from v_expected_uri
     or v_attestation->>'storage_uri' is distinct from v_expected_uri then
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
    'git_blob_sha', v_git_blob_sha,
    'attestation_event_id', v_attestation_event_id
  );
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
  v_child_run_id uuid;
  v_child_artifact_id uuid;
  v_child_version integer;
  v_parent_run_id uuid;
  v_parent_artifact_id uuid;
  v_parent_version integer;
  v_relation_type text;
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

  -- Phase 1: close receipt/manifest/locator/edge integrity for every object.
  -- No artifact registration, supersession, active-manifest pointer or CAS state
  -- transition occurs before this complete validation phase succeeds.
  perform public.orotitan_validate_manifest_persistence_receipt(
    p_manifest,
    p_manifest_registration
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

    perform public.orotitan_verify_attested_persistence_receipt(
      p_run_id,
      p_stage_code,
      v_match,
      v_ref,
      null
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
    begin
      v_child_run_id := coalesce((v_edge->>'child_run_id')::uuid, p_run_id);
      v_child_artifact_id := (v_edge->>'child_artifact_id')::uuid;
      v_child_version := (v_edge->>'child_version')::integer;
      v_parent_run_id := (v_edge->>'parent_run_id')::uuid;
      v_parent_artifact_id := (v_edge->>'parent_artifact_id')::uuid;
      v_parent_version := (v_edge->>'parent_version')::integer;
      v_relation_type := v_edge->>'relation_type';
    exception when others then
      raise exception 'LINEAGE_EDGE_INVALID: malformed identity/version'
        using errcode = '22023';
    end;

    if v_child_version < 1 or v_parent_version < 1
       or v_relation_type not in ('CONSUMES','DERIVED_FROM','SUPERSEDES','BASELINE_OF','REVALIDATES')
       or (v_child_run_id = v_parent_run_id
           and v_child_artifact_id = v_parent_artifact_id
           and v_child_version = v_parent_version) then
      raise exception 'LINEAGE_EDGE_INVALID: invalid version/relation/self-edge'
        using errcode = '23514';
    end if;

    if not exists (
      select 1 from public.orotitan_artifacts a
      where a.run_id = v_child_run_id
        and a.artifact_id = v_child_artifact_id
        and a.version = v_child_version
    ) and not (
      v_child_run_id = p_run_id
      and (
        (v_child_artifact_id = v_manifest_id and v_child_version = v_manifest_version)
        or exists (
          select 1 from jsonb_array_elements(p_output_artifacts) x
          where (x->>'artifact_id')::uuid = v_child_artifact_id
            and (x->>'version')::integer = v_child_version
        )
      )
    ) then
      raise exception 'LINEAGE_EDGE_INVALID: child artifact is not registered or in pending bundle'
        using errcode = '23514';
    end if;

    if not exists (
      select 1 from public.orotitan_artifacts a
      where a.run_id = v_parent_run_id
        and a.artifact_id = v_parent_artifact_id
        and a.version = v_parent_version
    ) and not (
      v_parent_run_id = p_run_id
      and (
        (v_parent_artifact_id = v_manifest_id and v_parent_version = v_manifest_version)
        or exists (
          select 1 from jsonb_array_elements(p_output_artifacts) x
          where (x->>'artifact_id')::uuid = v_parent_artifact_id
            and (x->>'version')::integer = v_parent_version
        )
      )
    ) then
      raise exception 'LINEAGE_EDGE_INVALID: parent artifact is not registered or in pending bundle'
        using errcode = '23514';
    end if;
  end loop;

  -- Phase 2: registration/rebinding only after complete bundle validation.
  perform public.orotitan_insert_artifact_registration(
    p_run_id, p_stage_code, p_manifest_registration, null, null
  );

  for v_ref in select value from jsonb_array_elements(p_manifest->'output_artifacts')
  loop
    select value
    into v_match
    from jsonb_array_elements(p_output_artifacts)
    where value->>'artifact_id' = v_ref->>'artifact_id'
      and value->>'version' = v_ref->>'version';

    perform public.orotitan_insert_artifact_registration(
      p_run_id, p_stage_code, v_match, v_manifest_id, v_manifest_version
    );
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

-- Internal-only helpers. Public/server callers reach them only through the
-- existing SECURITY DEFINER checkpoint/finalizer RPCs. The management plane
-- is the sole writer of PERSISTENCE_ATTESTED immutable events.
revoke execute on function public.orotitan_private_github_repository_is_approved(text)
  from public, anon, authenticated, service_role;
revoke execute on function public.orotitan_verify_attested_persistence_receipt(uuid,text,jsonb,jsonb,jsonb)
  from public, anon, authenticated, service_role;
revoke execute on function public.orotitan_validate_manifest_persistence_receipt(jsonb,jsonb)
  from public, anon, authenticated, service_role;
revoke execute on function public.orotitan_verify_existing_manifest_persistence_receipt(uuid,text,uuid,integer,text,text,jsonb)
  from public, anon, authenticated, service_role;
revoke execute on function public.orotitan_register_manifest_bundle(uuid,text,jsonb,jsonb,jsonb,jsonb)
  from public, anon, authenticated, service_role;

comment on function public.orotitan_verify_attested_persistence_receipt(uuid,text,jsonb,jsonb,jsonb) is
  'Validates exact reread bytes plus immutable GitHub locator against an independently minted, immutable PERSISTENCE_ATTESTED run event. Applies to every newly registered CHECKPOINT/FINAL bundle object.';
comment on function public.orotitan_validate_manifest_persistence_receipt(jsonb,jsonb) is
  'Bundle-wide successor of the V1.8 manifest exact-byte gate. Requires receipt schema 1.1 and an independently authenticated persistence attestation bound to run/stage/artifact/version/type/locator/bytes.';
comment on function public.orotitan_register_manifest_bundle(uuid,text,jsonb,jsonb,jsonb,jsonb) is
  'Validates manifest + every output receipt, immutable locator, manifest reference and lineage edge before any bundle registration/rebinding. Existing historical bundles are not rewritten.';
comment on constraint orotitan_run_events_event_type_check on public.orotitan_run_events is
  'Includes PERSISTENCE_ATTESTED events, which are append-only management-plane attestations of independent immutable-storage rereads.';

commit;

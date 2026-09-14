-- OroTitan Registry V1.3 — controlled RPC mutation boundary.
-- CODE-ONLY IMPLEMENTATION. NOT APPLIED TO PRODUCTION.

begin;

do $$
begin
  if to_regprocedure('extensions.digest(bytea,text)') is null then
    raise exception 'pgcrypto digest(bytea,text) must be installed in extensions schema';
  end if;
  if to_regclass('public.orotitan_runs') is null
     or to_regclass('public.orotitan_run_stages') is null
     or to_regclass('public.orotitan_artifacts') is null
     or to_regclass('public.orotitan_artifact_edges') is null
     or to_regclass('public.orotitan_run_events') is null then
    raise exception 'OroTitan registry core migration must exist before RPC layer';
  end if;
end;
$$;

create or replace function public.orotitan_contract_set_sha256(p_contract_pins jsonb)
returns text
language plpgsql
immutable
set search_path = pg_catalog, public
as $$
declare
  v_lines text;
begin
  if p_contract_pins is null or jsonb_typeof(p_contract_pins) <> 'object' then
    raise exception 'contract_pins must be a JSON object' using errcode = '22023';
  end if;

  select string_agg(
    format('%s|%s|%s', e.key, e.value->>'version', e.value->>'content_sha256'),
    E'\n' order by e.key
  )
  into v_lines
  from jsonb_each(p_contract_pins) e
  where jsonb_typeof(e.value) = 'object'
    and jsonb_typeof(e.value->'version') = 'string'
    and jsonb_typeof(e.value->'content_sha256') = 'string'
    and (e.value->>'content_sha256') ~ '^[0-9a-f]{64}$';

  if v_lines is null
     or (select count(*) from jsonb_each(p_contract_pins))
        <> (select count(*) from jsonb_each(p_contract_pins) e
            where jsonb_typeof(e.value) = 'object'
              and jsonb_typeof(e.value->'version') = 'string'
              and jsonb_typeof(e.value->'content_sha256') = 'string'
              and (e.value->>'content_sha256') ~ '^[0-9a-f]{64}$') then
    raise exception 'contract_pins contain malformed version/hash entries'
      using errcode = '22023';
  end if;

  return encode(extensions.digest(convert_to(v_lines || E'\n', 'UTF8'), 'sha256'), 'hex');
end;
$$;

create or replace function public.orotitan_assert_contract_pin(
  p_contract_pins jsonb,
  p_name text,
  p_version text,
  p_sha256 text
)
returns void
language plpgsql
immutable
set search_path = pg_catalog, public
as $$
begin
  if not exists (
    select 1
    from jsonb_each(p_contract_pins) e
    where e.value->>'name' = p_name
      and e.value->>'version' = p_version
      and e.value->>'content_sha256' = p_sha256
  ) then
    raise exception 'RUN_CONTRACT_SET_MISMATCH: stage contract pin not found'
      using errcode = '23514';
  end if;
end;
$$;

create or replace function public.orotitan_existing_event(
  p_run_id uuid,
  p_idempotency_key text,
  p_request_fingerprint_sha256 text
)
returns jsonb
language plpgsql
stable
set search_path = pg_catalog, public
as $$
declare
  v_event public.orotitan_run_events%rowtype;
begin
  select *
  into v_event
  from public.orotitan_run_events
  where run_id = p_run_id
    and idempotency_key = p_idempotency_key;

  if not found then
    return null;
  end if;

  if v_event.request_fingerprint_sha256 <> p_request_fingerprint_sha256 then
    raise exception 'IDEMPOTENCY_CONFLICT: same key with different request fingerprint'
      using errcode = '23514';
  end if;

  return jsonb_build_object(
    'event_id', v_event.event_id,
    'event_type', v_event.event_type,
    'payload', v_event.payload,
    'created_at', v_event.created_at
  );
end;
$$;

create or replace function public.orotitan_insert_event(
  p_run_id uuid,
  p_stage_code text,
  p_event_type text,
  p_idempotency_key text,
  p_request_fingerprint_sha256 text,
  p_actor_type text,
  p_payload jsonb
)
returns uuid
language plpgsql
set search_path = pg_catalog, public
as $$
declare
  v_event_id uuid := gen_random_uuid();
begin
  insert into public.orotitan_run_events (
    event_id, run_id, stage_code, event_type, idempotency_key,
    request_fingerprint_sha256, actor_type, payload
  ) values (
    v_event_id, p_run_id, p_stage_code, p_event_type, p_idempotency_key,
    p_request_fingerprint_sha256, p_actor_type, coalesce(p_payload, '{}'::jsonb)
  );
  return v_event_id;
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
begin
  if jsonb_typeof(p_artifact) <> 'object' then
    raise exception 'artifact registration must be an object' using errcode = '22023';
  end if;

  if (p_artifact->>'artifact_id') is null
     or (p_artifact->>'version') is null
     or (p_artifact->>'artifact_type') is null
     or (p_artifact->>'logical_name') is null
     or (p_artifact->>'authority_class') is null
     or (p_artifact->>'authority_state') is null
     or (p_artifact->>'media_type') is null
     or (p_artifact->>'size_bytes') is null
     or (p_artifact->>'content_sha256') is null
     or (p_artifact->>'storage_backend') is null
     or (p_artifact->>'storage_uri') is null then
    raise exception 'artifact registration is missing required fields'
      using errcode = '22023';
  end if;

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
    coalesce(p_artifact->>'artifact_status', 'SEALED'),
    p_artifact->>'authority_state',
    coalesce(p_artifact->>'availability_state', 'AVAILABLE'),
    p_artifact->>'media_type',
    (p_artifact->>'size_bytes')::bigint,
    'SHA-256',
    p_artifact->>'content_sha256',
    v_backend,
    p_artifact->>'storage_uri',
    case when v_backend = 'PRIVATE_GITHUB' then p_artifact->>'github_repository' end,
    case when v_backend = 'PRIVATE_GITHUB' then p_artifact->>'github_path' end,
    case when v_backend = 'PRIVATE_GITHUB' then p_artifact->>'github_commit_sha' end,
    case when v_backend = 'PRIVATE_GITHUB' then p_artifact->>'github_blob_sha' end,
    case when v_backend = 'SUPABASE_STORAGE' then p_artifact->>'supabase_bucket' end,
    case when v_backend = 'SUPABASE_STORAGE' then p_artifact->>'supabase_object_path' end,
    p_manifest_artifact_id,
    p_manifest_version
  );
exception
  when unique_violation then
    if not exists (
      select 1
      from public.orotitan_artifacts a
      where a.artifact_id = (p_artifact->>'artifact_id')::uuid
        and a.version = (p_artifact->>'version')::integer
        and a.run_id = p_run_id
        and a.stage_code = p_stage_code
        and a.content_sha256 = p_artifact->>'content_sha256'
        and a.size_bytes = (p_artifact->>'size_bytes')::bigint
    ) then
      raise;
    end if;
end;
$$;

create or replace function public.orotitan_assert_required_outputs(
  p_stage_code text,
  p_manifest jsonb
)
returns void
language plpgsql
immutable
set search_path = pg_catalog, public
as $$
declare
  v_required text[];
  v_missing text;
begin
  if p_manifest->>'manifest_kind' <> 'FINAL' then
    return;
  end if;

  if p_stage_code = 'RESEARCH' then
    v_required := array[
      'RESEARCH_SOURCE_MANIFEST',
      'EVIDENCE_LEDGER',
      'CONFLICT_LEDGER',
      'MATERIAL_RESEARCH_HYPOTHESIS_REGISTER',
      'RESEARCH_GAP_REGISTER',
      'DD_INPUT_SUFFICIENCY_RECORD',
      'ANALYSIS_INPUT_LOCK'
    ];
  elsif p_stage_code = 'DEEP_DIVE' then
    v_required := array[
      'DEEP_DIVE_REPORT',
      'EVIDENCE_LEDGER',
      'CONFLICT_LEDGER',
      'CALCULATION_LEDGER',
      'MATERIAL_ASSUMPTION_REGISTER',
      'ANALYTICAL_BLOCK_OUTPUTS',
      'CROSS_BLOCK_RECONCILIATION_RECORD',
      'RED_TEAM_PREMORTEM_RECORD',
      'VALUATION_ARTIFACT',
      'CERTIFICATION_ARTIFACT',
      'OROTITAN_TERMINAL_GATE_ARTIFACT',
      'READINESS_NEXT_ACTION_ARTIFACT'
    ];
  elsif p_stage_code = 'INTEGRATION' then
    v_required := array[
      'CANONICAL_SNAPSHOT_CANDIDATE',
      'INTEGRATION_MAPPING_RECORD',
      'SCHEMA_VALIDATION_REPORT',
      'I2_RECONCILIATION_REPORT',
      'HISTORY_TRANSITION_VALIDATION_REPORT',
      'I3B_ADMISSION_REPORT',
      'PRE_PUBLICATION_CONTROL_CARD'
    ];
  else
    raise exception 'unsupported stage code %', p_stage_code using errcode = '22023';
  end if;

  select string_agg(req, ', ' order by req)
  into v_missing
  from unnest(v_required) req
  where not exists (
    select 1
    from jsonb_array_elements(coalesce(p_manifest->'output_artifacts', '[]'::jsonb)) out_ref
    where out_ref->>'artifact_type' = req
      and out_ref->>'authority_class' = 'AUTHORITATIVE_STAGE_OUTPUT'
  );

  if v_missing is not null then
    raise exception 'MANIFEST_CONTRACT_MISMATCH: missing required authoritative outputs: %', v_missing
      using errcode = '23514';
  end if;
end;
$$;

create or replace function public.orotitan_validate_manifest_lock(
  p_run public.orotitan_runs,
  p_stage public.orotitan_run_stages,
  p_manifest jsonb,
  p_expected_kind text
)
returns void
language plpgsql
set search_path = pg_catalog, public
as $$
declare
  v_gate_name text;
  v_gate_state text;
begin
  if jsonb_typeof(p_manifest) <> 'object' then
    raise exception 'MANIFEST_CONTRACT_MISMATCH: manifest must be object'
      using errcode = '22023';
  end if;

  if p_manifest->>'manifest_kind' <> p_expected_kind
     or (p_manifest->>'run_id')::uuid <> p_run.run_id
     or p_manifest->>'stage' <> p_stage.stage_code
     or (p_manifest->>'stage_revision')::integer <> p_stage.stage_revision
     or p_manifest->>'data_cutoff' <> p_run.data_cutoff::text
     or p_manifest->>'canonical_mode' <> p_run.canonical_mode
     or coalesce(p_manifest->>'run_type', '') <> coalesce(p_run.run_type, '')
     or coalesce(p_manifest->>'issuer_id', '') <> coalesce(p_run.issuer_id::text, '')
     or coalesce(p_manifest->>'security_id', '') <> coalesce(p_run.security_id::text, '')
     or coalesce(p_manifest->>'dossier_id', '') <> coalesce(p_run.dossier_id::text, '')
     or coalesce(p_manifest->>'baseline_snapshot_id', '') <> coalesce(p_run.baseline_snapshot_id::text, '')
     or p_manifest->>'process_version' <> p_run.process_version
     or p_manifest->>'pilotage_contract_version' <> p_run.pilotage_contract_version
     or p_manifest->>'contract_set_sha256' <> p_run.contract_set_sha256
     or p_manifest->'contract_pins' is distinct from p_run.contract_pins then
    raise exception 'MANIFEST_CONTRACT_MISMATCH: run lock mismatch'
      using errcode = '23514';
  end if;

  if p_manifest->'stage_contract'->>'name' <> p_stage.stage_contract_name
     or p_manifest->'stage_contract'->>'version' <> p_stage.stage_contract_version
     or p_manifest->'stage_contract'->>'content_sha256' <> p_stage.stage_contract_sha256 then
    raise exception 'MANIFEST_CONTRACT_MISMATCH: stage contract mismatch'
      using errcode = '23514';
  end if;

  v_gate_name := p_manifest->'handoff_gate'->>'name';
  v_gate_state := p_manifest->'handoff_gate'->>'state';

  if v_gate_name <> p_stage.handoff_gate_name then
    raise exception 'MANIFEST_CONTRACT_MISMATCH: handoff gate name mismatch'
      using errcode = '23514';
  end if;

  if p_expected_kind = 'CHECKPOINT' then
    if p_manifest->>'stage_status' = 'COMPLETE'
       or v_gate_state = 'YES'
       or p_manifest->'completed_at' <> 'null'::jsonb then
      raise exception 'MANIFEST_CONTRACT_MISMATCH: CHECKPOINT cannot complete/admit'
        using errcode = '23514';
    end if;
  elsif p_expected_kind = 'FINAL' then
    if p_manifest->>'stage_status' <> 'COMPLETE'
       or p_manifest->'completed_at' is null
       or p_manifest->'completed_at' = 'null'::jsonb then
      raise exception 'MANIFEST_CONTRACT_MISMATCH: FINAL must be COMPLETE with completed_at'
        using errcode = '23514';
    end if;
    perform public.orotitan_assert_required_outputs(p_stage.stage_code, p_manifest);
  else
    raise exception 'unsupported manifest kind %', p_expected_kind using errcode = '22023';
  end if;

  if v_gate_state = 'YES'
     and jsonb_array_length(coalesce(p_manifest->'critical_blockers', '[]'::jsonb)) > 0 then
    raise exception 'MANIFEST_CONTRACT_MISMATCH: gate YES with critical blockers'
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
  v_manifest_id uuid := (p_manifest->>'manifest_id')::uuid;
  v_manifest_version integer := (p_manifest_registration->>'version')::integer;
  v_output jsonb;
  v_ref jsonb;
  v_match jsonb;
  v_edge jsonb;
  v_count integer;
begin
  if p_manifest_registration->>'artifact_id' <> v_manifest_id::text then
    raise exception 'MANIFEST_CONTRACT_MISMATCH: manifest_id != manifest artifact_id'
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
       or (v_match->>'size_bytes')::bigint <> (v_ref->>'size_bytes')::bigint then
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

create or replace function public.create_orotitan_run(
  p_creation_idempotency_key text,
  p_issuer_id uuid,
  p_entry_path text,
  p_canonical_mode text,
  p_run_type text,
  p_data_cutoff date,
  p_parent_run_id uuid,
  p_baseline_snapshot_id uuid,
  p_process_version text,
  p_pilotage_contract_version text,
  p_contract_pins jsonb,
  p_contract_set_sha256 text,
  p_request_fingerprint_sha256 text
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_existing public.orotitan_runs%rowtype;
  v_run_id uuid := gen_random_uuid();
  v_security_id uuid;
  v_dossier_id uuid;
  v_baseline_issuer uuid;
  v_event_id uuid;
begin
  if p_issuer_id is null then
    raise exception 'STAGE_NOT_ADMITTED: issuer_id is required for company-analysis run'
      using errcode = '23514';
  end if;
  if p_request_fingerprint_sha256 !~ '^[0-9a-f]{64}$' then
    raise exception 'invalid request fingerprint' using errcode = '22023';
  end if;
  if public.orotitan_contract_set_sha256(p_contract_pins) <> p_contract_set_sha256 then
    raise exception 'RUN_CONTRACT_SET_MISMATCH: contract_set_sha256 does not reconcile'
      using errcode = '23514';
  end if;

  select *
  into v_existing
  from public.orotitan_runs
  where creation_idempotency_key = p_creation_idempotency_key;

  if found then
    if not exists (
      select 1 from public.orotitan_run_events e
      where e.run_id = v_existing.run_id
        and e.event_type = 'RUN_CREATED'
        and e.idempotency_key = p_creation_idempotency_key
        and e.request_fingerprint_sha256 = p_request_fingerprint_sha256
    ) then
      raise exception 'IDEMPOTENCY_CONFLICT: run creation key reused with different request'
        using errcode = '23514';
    end if;
    return jsonb_build_object(
      'run_id', v_existing.run_id,
      'state_version', v_existing.state_version,
      'run_status', v_existing.run_status,
      'idempotent_replay', true
    );
  end if;

  if p_run_type = 'REFRESH' then
    if p_baseline_snapshot_id is null then
      raise exception 'REFRESH requires baseline_snapshot_id' using errcode = '23514';
    end if;
    select rs.security_id, rs.dossier_id, rs.issuer_id
    into v_security_id, v_dossier_id, v_baseline_issuer
    from public.research_snapshots rs
    where rs.snapshot_id = p_baseline_snapshot_id;
    if not found or v_baseline_issuer <> p_issuer_id then
      raise exception 'REFRESH baseline snapshot does not match issuer'
        using errcode = '23514';
    end if;
  elsif p_baseline_snapshot_id is not null then
    raise exception 'INITIAL/non-REFRESH run must not silently inherit baseline snapshot'
      using errcode = '23514';
  end if;

  insert into public.orotitan_runs (
    run_id, creation_idempotency_key,
    issuer_id, security_id, dossier_id, parent_run_id, baseline_snapshot_id,
    entry_path, canonical_mode, run_type, run_status, data_cutoff,
    process_version, pilotage_contract_version, contract_pins, contract_set_sha256
  ) values (
    v_run_id, p_creation_idempotency_key,
    p_issuer_id, v_security_id, v_dossier_id, p_parent_run_id, p_baseline_snapshot_id,
    p_entry_path, p_canonical_mode, p_run_type, 'CREATED', p_data_cutoff,
    p_process_version, p_pilotage_contract_version, p_contract_pins, p_contract_set_sha256
  );

  v_event_id := public.orotitan_insert_event(
    v_run_id, null, 'RUN_CREATED',
    p_creation_idempotency_key, p_request_fingerprint_sha256, 'PILOTAGE',
    jsonb_build_object('run_id', v_run_id, 'canonical_mode', p_canonical_mode, 'run_type', p_run_type)
  );

  return jsonb_build_object(
    'run_id', v_run_id,
    'state_version', 1,
    'run_status', 'CREATED',
    'event_id', v_event_id,
    'idempotent_replay', false
  );
end;
$$;

create or replace function public.bind_orotitan_run_identity(
  p_run_id uuid,
  p_expected_state_version bigint,
  p_security_id uuid,
  p_dossier_id uuid,
  p_upstream_discovery_artifact_id uuid,
  p_upstream_discovery_artifact_version integer,
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
  v_retry jsonb;
  v_event_id uuid;
begin
  v_retry := public.orotitan_existing_event(p_run_id, p_idempotency_key, p_request_fingerprint_sha256);
  if v_retry is not null then
    select * into v_run from public.orotitan_runs where run_id = p_run_id;
    return jsonb_build_object('run_id', p_run_id, 'state_version', v_run.state_version, 'idempotent_replay', true);
  end if;

  select * into v_run
  from public.orotitan_runs
  where run_id = p_run_id
  for update;

  if not found then raise exception 'RUN_NOT_FOUND' using errcode = 'P0002'; end if;
  if v_run.state_version <> p_expected_state_version then
    raise exception 'RUN_STATE_VERSION_MISMATCH' using errcode = '40001';
  end if;
  if v_run.run_status in ('PUBLISHED', 'CANCELLED') then
    raise exception 'terminal run cannot bind new identity' using errcode = '23514';
  end if;

  update public.orotitan_runs
  set security_id = coalesce(security_id, p_security_id),
      dossier_id = coalesce(dossier_id, p_dossier_id),
      upstream_discovery_artifact_id = coalesce(upstream_discovery_artifact_id, p_upstream_discovery_artifact_id),
      upstream_discovery_artifact_version = coalesce(upstream_discovery_artifact_version, p_upstream_discovery_artifact_version),
      state_version = state_version + 1
  where run_id = p_run_id;

  v_event_id := public.orotitan_insert_event(
    p_run_id, null, 'RUN_IDENTITY_BOUND', p_idempotency_key,
    p_request_fingerprint_sha256, 'PILOTAGE',
    jsonb_build_object(
      'security_id', p_security_id,
      'dossier_id', p_dossier_id,
      'upstream_discovery_artifact_id', p_upstream_discovery_artifact_id,
      'upstream_discovery_artifact_version', p_upstream_discovery_artifact_version
    )
  );

  select * into v_run from public.orotitan_runs where run_id = p_run_id;
  return jsonb_build_object('run_id', p_run_id, 'state_version', v_run.state_version, 'event_id', v_event_id, 'idempotent_replay', false);
end;
$$;

create or replace function public.start_orotitan_stage(
  p_run_id uuid,
  p_stage_code text,
  p_expected_run_state_version bigint,
  p_stage_contract_name text,
  p_stage_contract_version text,
  p_stage_contract_sha256 text,
  p_idempotency_key text,
  p_request_fingerprint_sha256 text,
  p_actor_type text default 'PILOTAGE'
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_run public.orotitan_runs%rowtype;
  v_upstream public.orotitan_run_stages%rowtype;
  v_stage public.orotitan_run_stages%rowtype;
  v_retry jsonb;
  v_gate_name text;
  v_event_id uuid;
begin
  v_retry := public.orotitan_existing_event(p_run_id, p_idempotency_key, p_request_fingerprint_sha256);
  if v_retry is not null then
    select * into v_stage from public.orotitan_run_stages where run_id = p_run_id and stage_code = p_stage_code;
    return jsonb_build_object('run_id', p_run_id, 'stage', p_stage_code, 'stage_state_version', v_stage.state_version, 'idempotent_replay', true);
  end if;

  select * into v_run from public.orotitan_runs where run_id = p_run_id for update;
  if not found then raise exception 'RUN_NOT_FOUND' using errcode = 'P0002'; end if;
  if v_run.state_version <> p_expected_run_state_version then raise exception 'RUN_STATE_VERSION_MISMATCH' using errcode = '40001'; end if;
  if v_run.run_status in ('PUBLISHED', 'CANCELLED', 'READY_TO_PUBLISH') then
    raise exception 'STAGE_NOT_ADMITTED: run state does not allow stage start' using errcode = '23514';
  end if;

  perform public.orotitan_assert_contract_pin(v_run.contract_pins, p_stage_contract_name, p_stage_contract_version, p_stage_contract_sha256);

  if p_stage_code = 'RESEARCH' then
    if v_run.issuer_id is null then raise exception 'STAGE_NOT_ADMITTED: Research requires issuer_id' using errcode = '23514'; end if;
    v_gate_name := 'READY_FOR_DEEP_DIVE';
  elsif p_stage_code = 'DEEP_DIVE' then
    if v_run.security_id is null then raise exception 'STAGE_NOT_ADMITTED: Deep Dive requires security_id' using errcode = '23514'; end if;
    select * into v_upstream from public.orotitan_run_stages where run_id = p_run_id and stage_code = 'RESEARCH';
    if not found or v_upstream.lifecycle_status <> 'COMPLETE' or v_upstream.active_manifest_kind <> 'FINAL' or v_upstream.handoff_gate_state <> 'YES' then
      raise exception 'STAGE_NOT_ADMITTED: Research handoff is not complete/ready' using errcode = '23514';
    end if;
    v_gate_name := 'READY_FOR_INTEGRATION';
  elsif p_stage_code = 'INTEGRATION' then
    if v_run.security_id is null or v_run.dossier_id is null then
      raise exception 'STAGE_NOT_ADMITTED: Integration requires security_id + dossier_id' using errcode = '23514';
    end if;
    select * into v_upstream from public.orotitan_run_stages where run_id = p_run_id and stage_code = 'DEEP_DIVE';
    if not found or v_upstream.lifecycle_status <> 'COMPLETE' or v_upstream.active_manifest_kind <> 'FINAL' or v_upstream.handoff_gate_state <> 'YES' then
      raise exception 'STAGE_NOT_ADMITTED: Deep Dive handoff is not complete/ready' using errcode = '23514';
    end if;
    v_gate_name := 'READY_TO_PUBLISH';
  else
    raise exception 'unsupported stage_code %', p_stage_code using errcode = '22023';
  end if;

  if exists (select 1 from public.orotitan_run_stages where run_id = p_run_id and stage_code = p_stage_code) then
    raise exception 'STAGE_ALREADY_EXISTS: use resume/reopen for existing stage' using errcode = '23514';
  end if;

  insert into public.orotitan_run_stages (
    run_id, stage_code, stage_contract_name, stage_contract_version,
    stage_contract_sha256, lifecycle_status, handoff_gate_name,
    handoff_gate_state, started_at
  ) values (
    p_run_id, p_stage_code, p_stage_contract_name, p_stage_contract_version,
    p_stage_contract_sha256, 'IN_PROGRESS', v_gate_name,
    'NOT_EVALUATED', now()
  );

  update public.orotitan_runs
  set current_stage = p_stage_code,
      run_status = 'ACTIVE',
      state_version = state_version + 1
  where run_id = p_run_id;

  v_event_id := public.orotitan_insert_event(
    p_run_id, p_stage_code, 'STAGE_STARTED', p_idempotency_key,
    p_request_fingerprint_sha256, p_actor_type,
    jsonb_build_object('stage', p_stage_code, 'stage_revision', 1)
  );

  select * into v_stage from public.orotitan_run_stages where run_id = p_run_id and stage_code = p_stage_code;
  return jsonb_build_object(
    'run_id', p_run_id, 'stage', p_stage_code,
    'stage_revision', v_stage.stage_revision,
    'stage_state_version', v_stage.state_version,
    'event_id', v_event_id,
    'idempotent_replay', false
  );
end;
$$;

create or replace function public.checkpoint_orotitan_stage(
  p_run_id uuid,
  p_stage_code text,
  p_expected_run_state_version bigint,
  p_expected_stage_state_version bigint,
  p_manifest jsonb,
  p_manifest_registration jsonb,
  p_output_artifacts jsonb,
  p_edges jsonb,
  p_target_lifecycle text,
  p_idempotency_key text,
  p_request_fingerprint_sha256 text,
  p_actor_type text
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_run public.orotitan_runs%rowtype;
  v_stage public.orotitan_run_stages%rowtype;
  v_retry jsonb;
  v_manifest_id uuid := (p_manifest->>'manifest_id')::uuid;
  v_manifest_version integer := (p_manifest_registration->>'version')::integer;
  v_event_id uuid;
  v_run_status text;
begin
  v_retry := public.orotitan_existing_event(p_run_id, p_idempotency_key, p_request_fingerprint_sha256);
  if v_retry is not null then
    select * into v_stage from public.orotitan_run_stages where run_id = p_run_id and stage_code = p_stage_code;
    return jsonb_build_object('run_id', p_run_id, 'stage', p_stage_code, 'stage_state_version', v_stage.state_version, 'idempotent_replay', true);
  end if;

  if p_target_lifecycle not in ('IN_PROGRESS', 'PAUSED', 'BLOCKED') then
    raise exception 'invalid checkpoint target lifecycle' using errcode = '22023';
  end if;

  select * into v_run from public.orotitan_runs where run_id = p_run_id for update;
  if not found then raise exception 'RUN_NOT_FOUND' using errcode = 'P0002'; end if;
  if v_run.state_version <> p_expected_run_state_version then raise exception 'RUN_STATE_VERSION_MISMATCH' using errcode = '40001'; end if;

  select * into v_stage from public.orotitan_run_stages where run_id = p_run_id and stage_code = p_stage_code for update;
  if not found then raise exception 'STAGE_NOT_FOUND' using errcode = 'P0002'; end if;
  if v_stage.state_version <> p_expected_stage_state_version then raise exception 'STAGE_STATE_VERSION_MISMATCH' using errcode = '40001'; end if;
  if v_stage.lifecycle_status = 'COMPLETE' then raise exception 'STAGE_ALREADY_COMPLETE' using errcode = '23514'; end if;

  perform public.orotitan_validate_manifest_lock(v_run, v_stage, p_manifest, 'CHECKPOINT');
  perform public.orotitan_register_manifest_bundle(
    p_run_id, p_stage_code, p_manifest, p_manifest_registration, p_output_artifacts, p_edges
  );

  v_run_status := case p_target_lifecycle when 'PAUSED' then 'PAUSED' when 'BLOCKED' then 'BLOCKED' else 'ACTIVE' end;

  update public.orotitan_run_stages
  set lifecycle_status = p_target_lifecycle,
      contract_status_code = p_manifest->>'contract_status_code',
      handoff_gate_state = p_manifest->'handoff_gate'->>'state',
      active_manifest_artifact_id = v_manifest_id,
      active_manifest_version = v_manifest_version,
      active_manifest_kind = 'CHECKPOINT',
      blocker_summary = coalesce(p_manifest->'critical_blockers', '[]'::jsonb),
      state_version = state_version + 1
  where run_id = p_run_id and stage_code = p_stage_code;

  update public.orotitan_runs
  set current_stage = p_stage_code,
      run_status = v_run_status,
      state_version = state_version + 1
  where run_id = p_run_id;

  v_event_id := public.orotitan_insert_event(
    p_run_id, p_stage_code, 'STAGE_CHECKPOINTED', p_idempotency_key,
    p_request_fingerprint_sha256, p_actor_type,
    jsonb_build_object('manifest_id', v_manifest_id, 'manifest_version', v_manifest_version, 'lifecycle', p_target_lifecycle)
  );

  select * into v_stage from public.orotitan_run_stages where run_id = p_run_id and stage_code = p_stage_code;
  return jsonb_build_object('run_id', p_run_id, 'stage', p_stage_code, 'stage_state_version', v_stage.state_version, 'manifest_id', v_manifest_id, 'event_id', v_event_id, 'idempotent_replay', false);
end;
$$;

create or replace function public.pause_orotitan_stage(
  p_run_id uuid,
  p_stage_code text,
  p_expected_run_state_version bigint,
  p_expected_stage_state_version bigint,
  p_idempotency_key text,
  p_request_fingerprint_sha256 text,
  p_reason jsonb
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_run public.orotitan_runs%rowtype;
  v_stage public.orotitan_run_stages%rowtype;
  v_retry jsonb;
  v_event_id uuid;
begin
  v_retry := public.orotitan_existing_event(p_run_id, p_idempotency_key, p_request_fingerprint_sha256);
  if v_retry is not null then
    select * into v_stage from public.orotitan_run_stages where run_id = p_run_id and stage_code = p_stage_code;
    return jsonb_build_object('run_id', p_run_id, 'stage', p_stage_code, 'stage_state_version', v_stage.state_version, 'idempotent_replay', true);
  end if;

  select * into v_run from public.orotitan_runs where run_id = p_run_id for update;
  if not found then raise exception 'RUN_NOT_FOUND' using errcode = 'P0002'; end if;
  if v_run.state_version <> p_expected_run_state_version then raise exception 'RUN_STATE_VERSION_MISMATCH' using errcode = '40001'; end if;
  select * into v_stage from public.orotitan_run_stages where run_id = p_run_id and stage_code = p_stage_code for update;
  if not found then raise exception 'STAGE_NOT_FOUND' using errcode = 'P0002'; end if;
  if v_stage.state_version <> p_expected_stage_state_version then raise exception 'STAGE_STATE_VERSION_MISMATCH' using errcode = '40001'; end if;
  if v_stage.lifecycle_status <> 'IN_PROGRESS' then raise exception 'stage is not IN_PROGRESS' using errcode = '23514'; end if;
  if v_stage.active_manifest_kind <> 'CHECKPOINT' then
    raise exception 'pause requires durable CHECKPOINT manifest' using errcode = '23514';
  end if;

  update public.orotitan_run_stages
  set lifecycle_status = 'PAUSED',
      state_version = state_version + 1
  where run_id = p_run_id and stage_code = p_stage_code;

  update public.orotitan_runs
  set run_status = 'PAUSED',
      current_stage = p_stage_code,
      state_version = state_version + 1
  where run_id = p_run_id;

  v_event_id := public.orotitan_insert_event(
    p_run_id, p_stage_code, 'STAGE_PAUSED', p_idempotency_key,
    p_request_fingerprint_sha256, 'PILOTAGE', coalesce(p_reason, '{}'::jsonb)
  );

  select * into v_stage from public.orotitan_run_stages where run_id = p_run_id and stage_code = p_stage_code;
  return jsonb_build_object('run_id', p_run_id, 'stage', p_stage_code, 'stage_state_version', v_stage.state_version, 'event_id', v_event_id, 'idempotent_replay', false);
end;
$$;

create or replace function public.resume_orotitan_stage(
  p_run_id uuid,
  p_stage_code text,
  p_expected_run_state_version bigint,
  p_expected_stage_state_version bigint,
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
  v_retry jsonb;
  v_event_id uuid;
begin
  v_retry := public.orotitan_existing_event(p_run_id, p_idempotency_key, p_request_fingerprint_sha256);
  if v_retry is not null then
    select * into v_stage from public.orotitan_run_stages where run_id = p_run_id and stage_code = p_stage_code;
    return jsonb_build_object('run_id', p_run_id, 'stage', p_stage_code, 'stage_state_version', v_stage.state_version, 'idempotent_replay', true);
  end if;

  select * into v_run from public.orotitan_runs where run_id = p_run_id for update;
  if not found then raise exception 'RUN_NOT_FOUND' using errcode = 'P0002'; end if;
  if v_run.state_version <> p_expected_run_state_version then raise exception 'RUN_STATE_VERSION_MISMATCH' using errcode = '40001'; end if;
  select * into v_stage from public.orotitan_run_stages where run_id = p_run_id and stage_code = p_stage_code for update;
  if not found then raise exception 'STAGE_NOT_FOUND' using errcode = 'P0002'; end if;
  if v_stage.state_version <> p_expected_stage_state_version then raise exception 'STAGE_STATE_VERSION_MISMATCH' using errcode = '40001'; end if;
  if v_stage.lifecycle_status not in ('PAUSED', 'BLOCKED') then raise exception 'stage is not resumable' using errcode = '23514'; end if;
  if v_stage.active_manifest_kind <> 'CHECKPOINT' then raise exception 'resume requires durable CHECKPOINT manifest' using errcode = '23514'; end if;

  update public.orotitan_run_stages
  set lifecycle_status = 'IN_PROGRESS',
      state_version = state_version + 1
  where run_id = p_run_id and stage_code = p_stage_code;

  update public.orotitan_runs
  set run_status = 'ACTIVE',
      current_stage = p_stage_code,
      state_version = state_version + 1
  where run_id = p_run_id;

  v_event_id := public.orotitan_insert_event(
    p_run_id, p_stage_code, 'STAGE_RESUMED', p_idempotency_key,
    p_request_fingerprint_sha256, 'PILOTAGE', '{}'::jsonb
  );

  select * into v_stage from public.orotitan_run_stages where run_id = p_run_id and stage_code = p_stage_code;
  return jsonb_build_object('run_id', p_run_id, 'stage', p_stage_code, 'stage_state_version', v_stage.state_version, 'event_id', v_event_id, 'idempotent_replay', false);
end;
$$;

create or replace function public.finalize_orotitan_stage(
  p_run_id uuid,
  p_stage_code text,
  p_expected_run_state_version bigint,
  p_expected_stage_state_version bigint,
  p_manifest jsonb,
  p_manifest_registration jsonb,
  p_output_artifacts jsonb,
  p_edges jsonb,
  p_idempotency_key text,
  p_request_fingerprint_sha256 text,
  p_actor_type text
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_run public.orotitan_runs%rowtype;
  v_stage public.orotitan_run_stages%rowtype;
  v_retry jsonb;
  v_manifest_id uuid := (p_manifest->>'manifest_id')::uuid;
  v_manifest_version integer := (p_manifest_registration->>'version')::integer;
  v_gate_state text := p_manifest->'handoff_gate'->>'state';
  v_event_id uuid;
  v_ready_event_id uuid;
  v_run_status text;
begin
  v_retry := public.orotitan_existing_event(p_run_id, p_idempotency_key, p_request_fingerprint_sha256);
  if v_retry is not null then
    select * into v_stage from public.orotitan_run_stages where run_id = p_run_id and stage_code = p_stage_code;
    return jsonb_build_object('run_id', p_run_id, 'stage', p_stage_code, 'stage_state_version', v_stage.state_version, 'idempotent_replay', true);
  end if;

  select * into v_run from public.orotitan_runs where run_id = p_run_id for update;
  if not found then raise exception 'RUN_NOT_FOUND' using errcode = 'P0002'; end if;
  if v_run.state_version <> p_expected_run_state_version then raise exception 'RUN_STATE_VERSION_MISMATCH' using errcode = '40001'; end if;
  select * into v_stage from public.orotitan_run_stages where run_id = p_run_id and stage_code = p_stage_code for update;
  if not found then raise exception 'STAGE_NOT_FOUND' using errcode = 'P0002'; end if;
  if v_stage.state_version <> p_expected_stage_state_version then raise exception 'STAGE_STATE_VERSION_MISMATCH' using errcode = '40001'; end if;
  if v_stage.lifecycle_status = 'COMPLETE' then raise exception 'STAGE_ALREADY_COMPLETE' using errcode = '23514'; end if;

  perform public.orotitan_validate_manifest_lock(v_run, v_stage, p_manifest, 'FINAL');
  perform public.orotitan_register_manifest_bundle(
    p_run_id, p_stage_code, p_manifest, p_manifest_registration, p_output_artifacts, p_edges
  );

  v_run_status := case when p_stage_code = 'INTEGRATION' and v_gate_state = 'YES' then 'READY_TO_PUBLISH'
                       when v_gate_state = 'YES' then 'ACTIVE'
                       else 'BLOCKED' end;

  update public.orotitan_run_stages
  set lifecycle_status = 'COMPLETE',
      contract_status_code = p_manifest->>'contract_status_code',
      handoff_gate_state = v_gate_state,
      active_manifest_artifact_id = v_manifest_id,
      active_manifest_version = v_manifest_version,
      active_manifest_kind = 'FINAL',
      blocker_summary = coalesce(p_manifest->'critical_blockers', '[]'::jsonb),
      completed_at = (p_manifest->>'completed_at')::timestamptz,
      state_version = state_version + 1
  where run_id = p_run_id and stage_code = p_stage_code;

  update public.orotitan_runs
  set current_stage = p_stage_code,
      run_status = v_run_status,
      state_version = state_version + 1
  where run_id = p_run_id;

  v_event_id := public.orotitan_insert_event(
    p_run_id, p_stage_code, 'STAGE_FINALIZED', p_idempotency_key,
    p_request_fingerprint_sha256, p_actor_type,
    jsonb_build_object('manifest_id', v_manifest_id, 'manifest_version', v_manifest_version, 'gate_state', v_gate_state)
  );

  if p_stage_code = 'INTEGRATION' and v_gate_state = 'YES' then
    v_ready_event_id := public.orotitan_insert_event(
      p_run_id, p_stage_code, 'READY_TO_PUBLISH_DECLARED',
      p_idempotency_key || ':ready',
      encode(extensions.digest(convert_to(p_request_fingerprint_sha256 || ':ready', 'UTF8'), 'sha256'), 'hex'),
      'INTEGRATION_WORKER',
      jsonb_build_object('manifest_id', v_manifest_id, 'manifest_version', v_manifest_version)
    );
  end if;

  select * into v_stage from public.orotitan_run_stages where run_id = p_run_id and stage_code = p_stage_code;
  return jsonb_build_object(
    'run_id', p_run_id, 'stage', p_stage_code,
    'stage_state_version', v_stage.state_version,
    'manifest_id', v_manifest_id,
    'handoff_gate_state', v_gate_state,
    'run_status', v_run_status,
    'event_id', v_event_id,
    'ready_event_id', v_ready_event_id,
    'idempotent_replay', false
  );
end;
$$;

create or replace function public.reopen_orotitan_stage(
  p_run_id uuid,
  p_stage_code text,
  p_expected_run_state_version bigint,
  p_expected_stage_state_version bigint,
  p_target_lifecycle text,
  p_reason jsonb,
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
  v_retry jsonb;
  v_event_id uuid;
begin
  v_retry := public.orotitan_existing_event(p_run_id, p_idempotency_key, p_request_fingerprint_sha256);
  if v_retry is not null then
    select * into v_stage from public.orotitan_run_stages where run_id = p_run_id and stage_code = p_stage_code;
    return jsonb_build_object('run_id', p_run_id, 'stage', p_stage_code, 'stage_revision', v_stage.stage_revision, 'idempotent_replay', true);
  end if;

  if p_target_lifecycle not in ('IN_PROGRESS', 'BLOCKED') then
    raise exception 'reopen target lifecycle must be IN_PROGRESS or BLOCKED' using errcode = '22023';
  end if;

  select * into v_run from public.orotitan_runs where run_id = p_run_id for update;
  if not found then raise exception 'RUN_NOT_FOUND' using errcode = 'P0002'; end if;
  if v_run.state_version <> p_expected_run_state_version then raise exception 'RUN_STATE_VERSION_MISMATCH' using errcode = '40001'; end if;
  if v_run.run_status in ('PUBLISHED', 'CANCELLED') then
    raise exception 'terminal run requires successor run, not reopen' using errcode = '23514';
  end if;

  select * into v_stage from public.orotitan_run_stages where run_id = p_run_id and stage_code = p_stage_code for update;
  if not found then raise exception 'STAGE_NOT_FOUND' using errcode = 'P0002'; end if;
  if v_stage.state_version <> p_expected_stage_state_version then raise exception 'STAGE_STATE_VERSION_MISMATCH' using errcode = '40001'; end if;
  if v_stage.lifecycle_status <> 'COMPLETE' then
    raise exception 'only COMPLETE stage may be reopened by this RPC' using errcode = '23514';
  end if;

  update public.orotitan_run_stages
  set stage_revision = stage_revision + 1,
      lifecycle_status = p_target_lifecycle,
      contract_status_code = case when p_target_lifecycle = 'BLOCKED' then 'REOPENED_BLOCKED' else null end,
      handoff_gate_state = 'NOT_EVALUATED',
      active_manifest_artifact_id = null,
      active_manifest_version = null,
      active_manifest_kind = null,
      blocker_summary = case when p_target_lifecycle = 'BLOCKED' then jsonb_build_array(coalesce(p_reason, '{}'::jsonb)) else '[]'::jsonb end,
      completed_at = null,
      state_version = state_version + 1
  where run_id = p_run_id and stage_code = p_stage_code;

  if p_stage_code = 'RESEARCH' then
    update public.orotitan_run_stages
    set stage_revision = case when lifecycle_status = 'COMPLETE' then stage_revision + 1 else stage_revision end,
        lifecycle_status = 'BLOCKED',
        contract_status_code = 'UPSTREAM_STAGE_REOPENED',
        handoff_gate_state = 'NOT_EVALUATED',
        active_manifest_artifact_id = null,
        active_manifest_version = null,
        active_manifest_kind = null,
        blocker_summary = jsonb_build_array(jsonb_build_object('code','UPSTREAM_STAGE_REOPENED','summary','Research was reopened')),
        completed_at = null,
        state_version = state_version + 1
    where run_id = p_run_id and stage_code in ('DEEP_DIVE','INTEGRATION');
  elsif p_stage_code = 'DEEP_DIVE' then
    update public.orotitan_run_stages
    set stage_revision = case when lifecycle_status = 'COMPLETE' then stage_revision + 1 else stage_revision end,
        lifecycle_status = 'BLOCKED',
        contract_status_code = 'UPSTREAM_STAGE_REOPENED',
        handoff_gate_state = 'NOT_EVALUATED',
        active_manifest_artifact_id = null,
        active_manifest_version = null,
        active_manifest_kind = null,
        blocker_summary = jsonb_build_array(jsonb_build_object('code','UPSTREAM_STAGE_REOPENED','summary','Deep Dive was reopened')),
        completed_at = null,
        state_version = state_version + 1
    where run_id = p_run_id and stage_code = 'INTEGRATION';
  end if;

  update public.orotitan_runs
  set current_stage = p_stage_code,
      run_status = case when p_target_lifecycle = 'BLOCKED' then 'BLOCKED' else 'ACTIVE' end,
      state_version = state_version + 1
  where run_id = p_run_id;

  v_event_id := public.orotitan_insert_event(
    p_run_id, p_stage_code, 'STAGE_REOPENED', p_idempotency_key,
    p_request_fingerprint_sha256, 'PILOTAGE', coalesce(p_reason, '{}'::jsonb)
  );

  select * into v_stage from public.orotitan_run_stages where run_id = p_run_id and stage_code = p_stage_code;
  return jsonb_build_object('run_id', p_run_id, 'stage', p_stage_code, 'stage_revision', v_stage.stage_revision, 'stage_state_version', v_stage.state_version, 'event_id', v_event_id, 'idempotent_replay', false);
end;
$$;

create or replace function public.resolve_orotitan_artifact(
  p_run_id uuid,
  p_artifact_id uuid,
  p_version integer,
  p_expected_sha256 text default null,
  p_required_authority_class text default null
)
returns jsonb
language plpgsql
security definer
stable
set search_path = pg_catalog, public
as $$
declare
  v_artifact public.orotitan_artifacts%rowtype;
begin
  select * into v_artifact
  from public.orotitan_artifacts
  where run_id = p_run_id
    and artifact_id = p_artifact_id
    and version = p_version;

  if not found then raise exception 'ARTIFACT_NOT_FOUND' using errcode = 'P0002'; end if;
  if v_artifact.artifact_status <> 'SEALED'
     or v_artifact.availability_state <> 'AVAILABLE'
     or v_artifact.authority_state = 'NON_AUTHORITATIVE' then
    raise exception 'ARTIFACT_STATUS_INVALID' using errcode = '23514';
  end if;
  if p_expected_sha256 is not null and v_artifact.content_sha256 <> p_expected_sha256 then
    raise exception 'ARTIFACT_HASH_MISMATCH' using errcode = '23514';
  end if;
  if p_required_authority_class is not null and v_artifact.authority_class <> p_required_authority_class then
    raise exception 'ARTIFACT_AUTHORITY_CLASS_INVALID' using errcode = '23514';
  end if;

  return jsonb_build_object(
    'artifact_id', v_artifact.artifact_id,
    'version', v_artifact.version,
    'run_id', v_artifact.run_id,
    'stage_code', v_artifact.stage_code,
    'artifact_type', v_artifact.artifact_type,
    'authority_class', v_artifact.authority_class,
    'authority_state', v_artifact.authority_state,
    'content_sha256', v_artifact.content_sha256,
    'size_bytes', v_artifact.size_bytes,
    'media_type', v_artifact.media_type,
    'storage_backend', v_artifact.storage_backend,
    'storage_uri', v_artifact.storage_uri,
    'github_repository', v_artifact.github_repository,
    'github_path', v_artifact.github_path,
    'github_commit_sha', v_artifact.github_commit_sha,
    'github_blob_sha', v_artifact.github_blob_sha,
    'supabase_bucket', v_artifact.supabase_bucket,
    'supabase_object_path', v_artifact.supabase_object_path
  );
end;
$$;

create or replace function public.record_orotitan_publish_authorization(
  p_run_id uuid,
  p_expected_run_state_version bigint,
  p_snapshot_candidate_artifact_id uuid,
  p_snapshot_candidate_version integer,
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
  v_artifact public.orotitan_artifacts%rowtype;
  v_retry jsonb;
  v_event_id uuid;
begin
  v_retry := public.orotitan_existing_event(p_run_id, p_idempotency_key, p_request_fingerprint_sha256);
  if v_retry is not null then
    select * into v_run from public.orotitan_runs where run_id = p_run_id;
    return jsonb_build_object('run_id', p_run_id, 'state_version', v_run.state_version, 'idempotent_replay', true);
  end if;

  select * into v_run from public.orotitan_runs where run_id = p_run_id for update;
  if not found then raise exception 'RUN_NOT_FOUND' using errcode = 'P0002'; end if;
  if v_run.state_version <> p_expected_run_state_version then raise exception 'RUN_STATE_VERSION_MISMATCH' using errcode = '40001'; end if;
  if v_run.run_status <> 'READY_TO_PUBLISH' then raise exception 'PUBLICATION_PRECHECK_FAILED: run not READY_TO_PUBLISH' using errcode = '23514'; end if;

  select * into v_stage from public.orotitan_run_stages where run_id = p_run_id and stage_code = 'INTEGRATION';
  if not found or v_stage.lifecycle_status <> 'COMPLETE' or v_stage.active_manifest_kind <> 'FINAL' or v_stage.handoff_gate_state <> 'YES' then
    raise exception 'PUBLICATION_PRECHECK_FAILED: Integration state invalid' using errcode = '23514';
  end if;

  select * into v_artifact
  from public.orotitan_artifacts
  where run_id = p_run_id
    and artifact_id = p_snapshot_candidate_artifact_id
    and version = p_snapshot_candidate_version;
  if not found
     or v_artifact.artifact_type <> 'CANONICAL_SNAPSHOT_CANDIDATE'
     or v_artifact.artifact_status <> 'SEALED'
     or v_artifact.authority_state <> 'AUTHORITATIVE'
     or v_artifact.availability_state <> 'AVAILABLE' then
    raise exception 'PUBLICATION_PRECHECK_FAILED: snapshot candidate artifact invalid'
      using errcode = '23514';
  end if;

  update public.orotitan_runs
  set state_version = state_version + 1
  where run_id = p_run_id;

  v_event_id := public.orotitan_insert_event(
    p_run_id, 'INTEGRATION', 'PUBLISH_AUTHORIZED', p_idempotency_key,
    p_request_fingerprint_sha256, 'PILOTAGE',
    jsonb_build_object(
      'snapshot_candidate_artifact_id', p_snapshot_candidate_artifact_id,
      'snapshot_candidate_version', p_snapshot_candidate_version
    )
  );

  select * into v_run from public.orotitan_runs where run_id = p_run_id;
  return jsonb_build_object('run_id', p_run_id, 'state_version', v_run.state_version, 'event_id', v_event_id, 'idempotent_replay', false);
end;
$$;

create or replace function public.record_orotitan_publish_result(
  p_run_id uuid,
  p_expected_run_state_version bigint,
  p_authorization_idempotency_key text,
  p_snapshot_id uuid,
  p_result text,
  p_recoverable boolean,
  p_idempotency_key text,
  p_request_fingerprint_sha256 text,
  p_error jsonb default null
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_run public.orotitan_runs%rowtype;
  v_retry jsonb;
  v_event_id uuid;
  v_pointer uuid;
begin
  v_retry := public.orotitan_existing_event(p_run_id, p_idempotency_key, p_request_fingerprint_sha256);
  if v_retry is not null then
    select * into v_run from public.orotitan_runs where run_id = p_run_id;
    return jsonb_build_object('run_id', p_run_id, 'run_status', v_run.run_status, 'state_version', v_run.state_version, 'idempotent_replay', true);
  end if;

  select * into v_run from public.orotitan_runs where run_id = p_run_id for update;
  if not found then raise exception 'RUN_NOT_FOUND' using errcode = 'P0002'; end if;
  if v_run.state_version <> p_expected_run_state_version then raise exception 'RUN_STATE_VERSION_MISMATCH' using errcode = '40001'; end if;
  if not exists (
    select 1 from public.orotitan_run_events
    where run_id = p_run_id
      and event_type = 'PUBLISH_AUTHORIZED'
      and idempotency_key = p_authorization_idempotency_key
  ) then
    raise exception 'PUBLICATION_NOT_AUTHORIZED' using errcode = '23514';
  end if;

  if p_result = 'SUCCEEDED' then
    if v_run.dossier_id is null then raise exception 'PUBLICATION_PRECHECK_FAILED: dossier_id missing' using errcode = '23514'; end if;
    select current_snapshot_id into v_pointer
    from public.research_dossiers
    where dossier_id = v_run.dossier_id;
    if v_pointer is distinct from p_snapshot_id then
      raise exception 'PUBLICATION_PRECHECK_FAILED: canonical pointer does not match published snapshot'
        using errcode = '23514';
    end if;

    update public.orotitan_runs
    set run_status = 'PUBLISHED',
        published_at = now(),
        state_version = state_version + 1
    where run_id = p_run_id;

    v_event_id := public.orotitan_insert_event(
      p_run_id, 'INTEGRATION', 'PUBLISH_SUCCEEDED', p_idempotency_key,
      p_request_fingerprint_sha256, 'PUBLISHER',
      jsonb_build_object('snapshot_id', p_snapshot_id)
    );
  elsif p_result = 'FAILED' then
    update public.orotitan_runs
    set run_status = case when p_recoverable then 'READY_TO_PUBLISH' else 'BLOCKED' end,
        state_version = state_version + 1
    where run_id = p_run_id;

    v_event_id := public.orotitan_insert_event(
      p_run_id, 'INTEGRATION', 'PUBLISH_FAILED', p_idempotency_key,
      p_request_fingerprint_sha256, 'PUBLISHER',
      jsonb_build_object('snapshot_id', p_snapshot_id, 'recoverable', p_recoverable, 'error', p_error)
    );
  else
    raise exception 'publish result must be SUCCEEDED or FAILED' using errcode = '22023';
  end if;

  select * into v_run from public.orotitan_runs where run_id = p_run_id;
  return jsonb_build_object('run_id', p_run_id, 'run_status', v_run.run_status, 'state_version', v_run.state_version, 'event_id', v_event_id, 'idempotent_replay', false);
end;
$$;

create or replace view public.orotitan_run_status_view
with (security_invoker = true)
as
select
  r.run_id,
  r.issuer_id,
  r.security_id,
  r.dossier_id,
  r.run_type,
  r.canonical_mode,
  r.run_status,
  r.current_stage,
  s.lifecycle_status as stage_lifecycle_status,
  s.handoff_gate_name,
  s.handoff_gate_state,
  s.blocker_summary,
  s.active_manifest_artifact_id,
  s.active_manifest_version,
  s.active_manifest_kind,
  r.state_version as run_state_version,
  s.state_version as stage_state_version,
  r.updated_at
from public.orotitan_runs r
left join public.orotitan_run_stages s
  on s.run_id = r.run_id and s.stage_code = r.current_stage;

revoke all on function public.orotitan_contract_set_sha256(jsonb) from public, anon, authenticated, service_role;
revoke all on function public.orotitan_assert_contract_pin(jsonb, text, text, text) from public, anon, authenticated, service_role;
revoke all on function public.orotitan_existing_event(uuid, text, text) from public, anon, authenticated, service_role;
revoke all on function public.orotitan_insert_event(uuid, text, text, text, text, text, jsonb) from public, anon, authenticated, service_role;
revoke all on function public.orotitan_insert_artifact_registration(uuid, text, jsonb, uuid, integer) from public, anon, authenticated, service_role;
revoke all on function public.orotitan_assert_required_outputs(text, jsonb) from public, anon, authenticated, service_role;
revoke all on function public.orotitan_validate_manifest_lock(public.orotitan_runs, public.orotitan_run_stages, jsonb, text) from public, anon, authenticated, service_role;
revoke all on function public.orotitan_register_manifest_bundle(uuid, text, jsonb, jsonb, jsonb, jsonb) from public, anon, authenticated, service_role;

revoke all on function public.create_orotitan_run(text, uuid, text, text, text, date, uuid, uuid, text, text, jsonb, text, text) from public, anon, authenticated;
grant execute on function public.create_orotitan_run(text, uuid, text, text, text, date, uuid, uuid, text, text, jsonb, text, text) to service_role;
revoke all on function public.bind_orotitan_run_identity(uuid, bigint, uuid, uuid, uuid, integer, text, text) from public, anon, authenticated;
grant execute on function public.bind_orotitan_run_identity(uuid, bigint, uuid, uuid, uuid, integer, text, text) to service_role;
revoke all on function public.start_orotitan_stage(uuid, text, bigint, text, text, text, text, text, text) from public, anon, authenticated;
grant execute on function public.start_orotitan_stage(uuid, text, bigint, text, text, text, text, text, text) to service_role;
revoke all on function public.checkpoint_orotitan_stage(uuid, text, bigint, bigint, jsonb, jsonb, jsonb, jsonb, text, text, text, text) from public, anon, authenticated;
grant execute on function public.checkpoint_orotitan_stage(uuid, text, bigint, bigint, jsonb, jsonb, jsonb, jsonb, text, text, text, text) to service_role;
revoke all on function public.pause_orotitan_stage(uuid, text, bigint, bigint, text, text, jsonb) from public, anon, authenticated;
grant execute on function public.pause_orotitan_stage(uuid, text, bigint, bigint, text, text, jsonb) to service_role;
revoke all on function public.resume_orotitan_stage(uuid, text, bigint, bigint, text, text) from public, anon, authenticated;
grant execute on function public.resume_orotitan_stage(uuid, text, bigint, bigint, text, text) to service_role;
revoke all on function public.finalize_orotitan_stage(uuid, text, bigint, bigint, jsonb, jsonb, jsonb, jsonb, text, text, text) from public, anon, authenticated;
grant execute on function public.finalize_orotitan_stage(uuid, text, bigint, bigint, jsonb, jsonb, jsonb, jsonb, text, text, text) to service_role;
revoke all on function public.reopen_orotitan_stage(uuid, text, bigint, bigint, text, jsonb, text, text) from public, anon, authenticated;
grant execute on function public.reopen_orotitan_stage(uuid, text, bigint, bigint, text, jsonb, text, text) to service_role;
revoke all on function public.resolve_orotitan_artifact(uuid, uuid, integer, text, text) from public, anon, authenticated;
grant execute on function public.resolve_orotitan_artifact(uuid, uuid, integer, text, text) to service_role;
revoke all on function public.record_orotitan_publish_authorization(uuid, bigint, uuid, integer, text, text) from public, anon, authenticated;
grant execute on function public.record_orotitan_publish_authorization(uuid, bigint, uuid, integer, text, text) to service_role;
revoke all on function public.record_orotitan_publish_result(uuid, bigint, text, uuid, text, boolean, text, text, jsonb) from public, anon, authenticated;
grant execute on function public.record_orotitan_publish_result(uuid, bigint, text, uuid, text, boolean, text, text, jsonb) to service_role;

revoke all on table public.orotitan_run_status_view from public, anon, authenticated;
grant select on table public.orotitan_run_status_view to service_role;

do $$
declare
  v_sig text;
begin
  foreach v_sig in array array[
    'public.create_orotitan_run(text, uuid, text, text, text, date, uuid, uuid, text, text, jsonb, text, text)',
    'public.bind_orotitan_run_identity(uuid, bigint, uuid, uuid, uuid, integer, text, text)',
    'public.start_orotitan_stage(uuid, text, bigint, text, text, text, text, text, text)',
    'public.checkpoint_orotitan_stage(uuid, text, bigint, bigint, jsonb, jsonb, jsonb, jsonb, text, text, text, text)',
    'public.pause_orotitan_stage(uuid, text, bigint, bigint, text, text, jsonb)',
    'public.resume_orotitan_stage(uuid, text, bigint, bigint, text, text)',
    'public.finalize_orotitan_stage(uuid, text, bigint, bigint, jsonb, jsonb, jsonb, jsonb, text, text, text)',
    'public.reopen_orotitan_stage(uuid, text, bigint, bigint, text, jsonb, text, text)',
    'public.resolve_orotitan_artifact(uuid, uuid, integer, text, text)',
    'public.record_orotitan_publish_authorization(uuid, bigint, uuid, integer, text, text)',
    'public.record_orotitan_publish_result(uuid, bigint, text, uuid, text, boolean, text, text, jsonb)'
  ]
  loop
    if not (
      select p.prosecdef
        and p.proconfig @> array['search_path=pg_catalog, public']::text[]
      from pg_catalog.pg_proc p
      where p.oid = v_sig::regprocedure
    ) then
      raise exception 'registry RPC security boundary incompatible for %', v_sig;
    end if;
    if has_function_privilege('public', v_sig, 'EXECUTE')
       or has_function_privilege('anon', v_sig, 'EXECUTE')
       or has_function_privilege('authenticated', v_sig, 'EXECUTE')
       or not has_function_privilege('service_role', v_sig, 'EXECUTE') then
      raise exception 'registry RPC execute privileges incompatible for %', v_sig;
    end if;
  end loop;
end;
$$;

commit;

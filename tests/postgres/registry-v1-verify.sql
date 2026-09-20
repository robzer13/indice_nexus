\set ON_ERROR_STOP on

-- ---------------------------------------------------------------------------
-- Helpers used only by the disposable PostgreSQL integration matrix.
-- ---------------------------------------------------------------------------

create or replace function pg_temp.registry_artifact_registration(
  p_artifact_id uuid,
  p_artifact_type text,
  p_authority_class text,
  p_authority_state text,
  p_hash_char text,
  p_path text
)
returns jsonb
language sql
as $$
  select jsonb_build_object(
    'artifact_id', p_artifact_id,
    'version', 1,
    'artifact_type', p_artifact_type,
    'logical_name', lower(p_artifact_type),
    'authority_class', p_authority_class,
    'artifact_status', 'SEALED',
    'authority_state', p_authority_state,
    'availability_state', 'AVAILABLE',
    'media_type', 'application/json',
    'size_bytes', 2,
    'content_sha256', repeat(p_hash_char, 64),
    'storage_backend', 'SUPABASE_STORAGE',
    'storage_uri', 'supabase://orotitan-private/' || p_path,
    'supabase_bucket', 'orotitan-private',
    'supabase_object_path', p_path
  );
$$;

create or replace function pg_temp.registry_artifact_ref(p_registration jsonb)
returns jsonb
language sql
as $$
  select jsonb_build_object(
    'artifact_id', p_registration->>'artifact_id',
    'version', (p_registration->>'version')::integer,
    'artifact_type', p_registration->>'artifact_type',
    'content_sha256', p_registration->>'content_sha256',
    'authority_class', p_registration->>'authority_class',
    'media_type', p_registration->>'media_type',
    'size_bytes', (p_registration->>'size_bytes')::bigint
  );
$$;

create or replace function pg_temp.registry_manifest_registration(
  p_artifact_id uuid,
  p_manifest jsonb,
  p_authority_class text,
  p_authority_state text,
  p_path text
)
returns jsonb
language plpgsql
as $$
declare
  v_text text := p_manifest::text;
  v_bytes bytea := convert_to(v_text, 'UTF8');
  v_size bigint := octet_length(v_bytes);
  v_sha256 text := encode(extensions.digest(v_bytes, 'sha256'), 'hex');
  v_blob text;
  v_commit text := repeat('c', 40);
  v_repo text := 'robzer13/real-orotitan';
begin
  v_blob := encode(
    extensions.digest(
      convert_to('blob ' || v_size::text, 'UTF8') || decode('00','hex') || v_bytes,
      'sha1'
    ),
    'hex'
  );
  return jsonb_build_object(
    'artifact_id', p_artifact_id,
    'version', 1,
    'artifact_type', 'RESEARCH_STAGE_MANIFEST',
    'logical_name', 'research_stage_manifest',
    'authority_class', p_authority_class,
    'artifact_status', 'SEALED',
    'authority_state', p_authority_state,
    'availability_state', 'AVAILABLE',
    'media_type', 'application/json',
    'size_bytes', v_size,
    'content_sha256', v_sha256,
    'storage_backend', 'PRIVATE_GITHUB',
    'storage_uri', 'github://' || v_repo || '@' || v_commit || '/' || p_path,
    'github_repository', v_repo,
    'github_path', p_path,
    'github_commit_sha', v_commit,
    'github_blob_sha', v_blob,
    'persistence_receipt', jsonb_build_object(
      'receipt_schema_version','1.0',
      'verification_method','PRIVATE_GITHUB_REREAD_EXACT_BYTES_V1',
      'storage_backend','PRIVATE_GITHUB',
      'github_repository',v_repo,
      'github_path',p_path,
      'github_commit_sha',v_commit,
      'github_blob_sha',v_blob,
      'commit_path_resolved',true,
      'verified_content_base64',encode(v_bytes,'base64'),
      'verified_at','2026-09-20T12:00:00Z'
    )
  );
end;
$$;

-- ---------------------------------------------------------------------------
-- Structural / privilege guards.
-- ---------------------------------------------------------------------------

do $$
declare
  missing text[];
  bad_rls text[];
  write_grants integer;
  policy_count integer;
  rpc_count integer;
  fn record;
begin
  select array_agg(v.name order by v.name) into missing
  from (values
    ('orotitan_runs'),('orotitan_run_stages'),('orotitan_artifacts'),
    ('orotitan_artifact_edges'),('orotitan_run_events')
  ) v(name)
  where to_regclass('public.'||v.name) is null;
  if missing is not null then raise exception 'missing registry tables: %', missing; end if;

  select array_agg(c.relname order by c.relname) into bad_rls
  from pg_class c join pg_namespace n on n.oid=c.relnamespace
  where n.nspname='public'
    and c.relname in ('orotitan_runs','orotitan_run_stages','orotitan_artifacts','orotitan_artifact_edges','orotitan_run_events')
    and not c.relrowsecurity;
  if bad_rls is not null then raise exception 'RLS disabled: %', bad_rls; end if;

  select count(*) into policy_count
  from pg_policies
  where schemaname='public'
    and tablename in ('orotitan_runs','orotitan_run_stages','orotitan_artifacts','orotitan_artifact_edges','orotitan_run_events');
  if policy_count <> 0 then raise exception 'unexpected client RLS policies exist: %', policy_count; end if;

  select count(*) into write_grants
  from information_schema.role_table_grants
  where table_schema='public'
    and table_name in ('orotitan_runs','orotitan_run_stages','orotitan_artifacts','orotitan_artifact_edges','orotitan_run_events')
    and grantee in ('anon','authenticated','service_role')
    and privilege_type in ('INSERT','UPDATE','DELETE','TRUNCATE');
  if write_grants <> 0 then raise exception 'direct registry write grants exist: %', write_grants; end if;

  select count(*) into rpc_count
  from pg_proc p join pg_namespace n on n.oid=p.pronamespace
  where n.nspname='public' and p.proname in (
    'create_orotitan_run','bind_orotitan_run_identity','start_orotitan_stage',
    'checkpoint_orotitan_stage','pause_orotitan_stage','resume_orotitan_stage',
    'finalize_orotitan_stage','reopen_orotitan_stage','resolve_orotitan_artifact',
    'record_orotitan_publish_authorization','record_orotitan_publish_result'
  );
  if rpc_count <> 11 then raise exception 'expected 11 registry RPCs, found %', rpc_count; end if;

  for fn in
    select p.oid::regprocedure as signature, p.prosecdef, p.proconfig
    from pg_proc p join pg_namespace n on n.oid=p.pronamespace
    where n.nspname='public' and p.proname in (
      'create_orotitan_run','bind_orotitan_run_identity','start_orotitan_stage',
      'checkpoint_orotitan_stage','pause_orotitan_stage','resume_orotitan_stage',
      'finalize_orotitan_stage','reopen_orotitan_stage','resolve_orotitan_artifact',
      'record_orotitan_publish_authorization','record_orotitan_publish_result'
    )
  loop
    if not fn.prosecdef or not (fn.proconfig @> array['search_path=pg_catalog, public']::text[]) then
      raise exception 'RPC security boundary invalid: %', fn.signature;
    end if;
    if has_function_privilege('public', fn.signature::text, 'EXECUTE')
       or has_function_privilege('anon', fn.signature::text, 'EXECUTE')
       or has_function_privilege('authenticated', fn.signature::text, 'EXECUTE')
       or not has_function_privilege('service_role', fn.signature::text, 'EXECUTE') then
      raise exception 'RPC privileges invalid: %', fn.signature;
    end if;
  end loop;

  if not exists (
    select 1 from pg_constraint
    where conname='orotitan_run_stages_active_manifest_same_stage_fkey' and condeferrable
  ) then raise exception 'active-manifest same-stage deferred FK missing'; end if;

  if not exists (
    select 1 from pg_constraint
    where conname='orotitan_artifacts_manifest_same_stage_fkey' and condeferrable
  ) then raise exception 'artifact-manifest same-stage deferred FK missing'; end if;

  if not exists (
    select 1 from pg_constraint
    where conname='orotitan_runs_contract_pins_complete_check'
  ) then raise exception 'contract-pin completeness constraint missing'; end if;

  if to_regprocedure('extensions.digest(bytea,text)') is null then
    raise exception 'pgcrypto digest is not in the production-compatible extensions schema';
  end if;
end $$;

-- ---------------------------------------------------------------------------
-- End-to-end operational matrix through Research finalization + downstream
-- admission/reopen. It deliberately does not invoke I3-B publication.
-- ---------------------------------------------------------------------------

do $$
declare
  v_issuer uuid;
  v_security uuid;
  v_dossier uuid;
  v_run uuid;
  v_json jsonb;
  v_run_state bigint;
  v_stage_state bigint;
  v_pins jsonb;
  v_contract_hash text;
  v_research_hash text;
  v_dd_hash text;
  v_pin_key text;
  v_pin_name text;
  v_pin_version text;
  v_pin_hash text;
  v_checkpoint_manifest uuid := '10000000-0000-4000-8000-000000000001';
  v_checkpoint_output uuid := '10000000-0000-4000-8000-000000000002';
  v_final_manifest uuid := '10000000-0000-4000-8000-000000000003';
  v_checkpoint_reg jsonb;
  v_checkpoint_refs jsonb;
  v_final_regs jsonb;
  v_final_refs jsonb;
  v_manifest jsonb;
  v_manifest_reg jsonb;
  v_conflict_seen boolean := false;
  v_rejected boolean := false;
  v_event_id uuid;
  v_event_payload jsonb;
  v_resolved jsonb;
  v_required text[] := array[
    'RESEARCH_SOURCE_MANIFEST',
    'EVIDENCE_LEDGER',
    'CONFLICT_LEDGER',
    'MATERIAL_RESEARCH_HYPOTHESIS_REGISTER',
    'RESEARCH_GAP_REGISTER',
    'DD_INPUT_SUFFICIENCY_RECORD',
    'ANALYSIS_INPUT_LOCK'
  ];
begin
  select m.issuer_id, m.security_id, m.dossier_id
    into v_issuer, v_security, v_dossier
  from public.legacy_company_identity_map m
  order by m.legacy_company_id
  limit 1;

  if v_issuer is null or v_security is null or v_dossier is null then
    raise exception 'identity fixture is incomplete';
  end if;

  v_pins := '{}'::jsonb;
  foreach v_pin_key in array array[
    'process','pilotage','research_stage','deep_dive_stage','integration_stage',
    'analysis_standard','master_prompt','investment_policy','execution_patch',
    'integration_spec','screener_schema','i2','i3b'
  ]
  loop
    v_pin_name := case v_pin_key
      when 'research_stage' then 'RESEARCH'
      when 'deep_dive_stage' then 'DEEP_DIVE'
      when 'integration_stage' then 'INTEGRATION'
      else upper(v_pin_key)
    end;
    v_pin_version := case v_pin_key
      when 'pilotage' then '1.0.1'
      when 'investment_policy' then '1.0.0'
      when 'execution_patch' then '1.0.1'
      when 'screener_schema' then '1.0.0'
      else '1.0'
    end;
    v_pin_hash := encode(extensions.digest(convert_to(v_pin_key, 'UTF8'), 'sha256'), 'hex');
    v_pins := v_pins || jsonb_build_object(
      v_pin_key,
      jsonb_build_object(
        'name', v_pin_name,
        'version', v_pin_version,
        'content_sha256', v_pin_hash,
        'locator', jsonb_build_object(
          'backend','GITHUB_IMMUTABLE',
          'repository','robzer13/indice_nexus',
          'path','contracts/orotitan-equity/v1/execution/' || v_pin_key || '.md',
          'commit_sha',repeat('a',40),
          'blob_sha',repeat('b',40)
        )
      )
    );
  end loop;

  v_contract_hash := public.orotitan_contract_set_sha256(v_pins);
  v_research_hash := v_pins->'research_stage'->>'content_sha256';
  v_dd_hash := v_pins->'deep_dive_stage'->>'content_sha256';

  v_json := public.create_orotitan_run(
    'run-create:test-registry-v1',
    v_issuer,
    'IMPOSED_COMPANY',
    'ANALYZE',
    'INITIAL',
    date '2026-09-14',
    null,
    null,
    '1.0',
    '1.0.1',
    v_pins,
    v_contract_hash,
    repeat('e',64)
  );
  v_run := (v_json->>'run_id')::uuid;
  if v_json->>'run_status' <> 'CREATED' or (v_json->>'idempotent_replay')::boolean then
    raise exception 'run not created correctly: %', v_json;
  end if;

  -- Same create key + same request is a replay of the same RUN_ID.
  v_json := public.create_orotitan_run(
    'run-create:test-registry-v1', v_issuer, 'IMPOSED_COMPANY','ANALYZE','INITIAL',
    date '2026-09-14', null, null, '1.0','1.0.1',v_pins,v_contract_hash,repeat('e',64)
  );
  if not (v_json->>'idempotent_replay')::boolean or (v_json->>'run_id')::uuid <> v_run then
    raise exception 'run retry was not idempotent: %', v_json;
  end if;

  -- Same key + conflicting request fingerprint is rejected.
  begin
    perform public.create_orotitan_run(
      'run-create:test-registry-v1', v_issuer, 'IMPOSED_COMPANY','ANALYZE','INITIAL',
      date '2026-09-14', null, null, '1.0','1.0.1',v_pins,v_contract_hash,repeat('0',64)
    );
  exception when check_violation then
    v_conflict_seen := true;
  end;
  if not v_conflict_seen then raise exception 'conflicting run idempotency key was accepted'; end if;

  -- Incomplete contract pin sets are physically rejected.
  v_rejected := false;
  begin
    perform public.create_orotitan_run(
      'run-create:incomplete-pins', v_issuer, 'IMPOSED_COMPANY','ANALYZE','INITIAL',
      date '2026-09-14', null, null, '1.0','1.0.1',
      jsonb_build_object('process', v_pins->'process'),
      public.orotitan_contract_set_sha256(jsonb_build_object('process', v_pins->'process')),
      repeat('1',64)
    );
  exception when check_violation then
    v_rejected := true;
  end;
  if not v_rejected then raise exception 'incomplete contract pins were accepted'; end if;

  select state_version into v_run_state from public.orotitan_runs where run_id=v_run;
  perform public.bind_orotitan_run_identity(
    v_run,v_run_state,v_security,v_dossier,null,null,
    'bind:test-registry-v1',repeat('2',64)
  );

  select state_version into v_run_state from public.orotitan_runs where run_id=v_run;
  perform public.start_orotitan_stage(
    v_run,'RESEARCH',v_run_state,'RESEARCH','1.0',v_research_hash,
    'start:research',repeat('3',64),'PILOTAGE'
  );

  -- Durable Research checkpoint.
  v_checkpoint_reg := jsonb_build_array(
    pg_temp.registry_artifact_registration(
      v_checkpoint_output,'EVIDENCE_LEDGER_CHECKPOINT','CHECKPOINT_STAGE_OUTPUT','CHECKPOINT','4','research/evidence-checkpoint.json'
    )
  );
  select jsonb_agg(pg_temp.registry_artifact_ref(value))
    into v_checkpoint_refs
  from jsonb_array_elements(v_checkpoint_reg);

  v_manifest := jsonb_build_object(
    'manifest_schema_version','1.0.0',
    'manifest_id',v_checkpoint_manifest,
    'manifest_kind','CHECKPOINT',
    'run_id',v_run,
    'stage','RESEARCH',
    'stage_revision',1,
    'issuer_id',v_issuer,
    'security_id',v_security,
    'dossier_id',v_dossier,
    'canonical_mode','ANALYZE',
    'run_type','INITIAL',
    'data_cutoff','2026-09-14',
    'baseline_snapshot_id',null,
    'process_version','1.0',
    'pilotage_contract_version','1.0.1',
    'contract_pins',v_pins,
    'stage_contract',v_pins->'research_stage',
    'contract_set_sha256',v_contract_hash,
    'input_artifacts','[]'::jsonb,
    'output_artifacts',v_checkpoint_refs,
    'stage_status','PAUSED',
    'contract_status_code','PAUSED_CRITICAL_INPUT',
    'handoff_gate',jsonb_build_object('name','READY_FOR_DEEP_DIVE','state','NOT_EVALUATED'),
    'critical_blockers',jsonb_build_array(jsonb_build_object('code','CRITICAL_INPUT','summary','test pause')),
    'open_material_limitations','[]'::jsonb,
    'parent_manifests','[]'::jsonb,
    'started_at','2026-09-14T14:00:00Z',
    'completed_at',null
  );
  v_manifest_reg := pg_temp.registry_manifest_registration(
    v_checkpoint_manifest,
    v_manifest,
    'CHECKPOINT_STAGE_OUTPUT',
    'CHECKPOINT',
    'research/checkpoint-manifest.json'
  );

  select state_version into v_run_state from public.orotitan_runs where run_id=v_run;
  select state_version into v_stage_state from public.orotitan_run_stages where run_id=v_run and stage_code='RESEARCH';
  perform public.checkpoint_orotitan_stage(
    v_run,'RESEARCH',v_run_state,v_stage_state,
    v_manifest,v_manifest_reg,v_checkpoint_reg,'[]'::jsonb,
    'PAUSED','checkpoint:research',repeat('6',64),'RESEARCH_WORKER'
  );

  if not exists (
    select 1 from public.orotitan_run_stages
    where run_id=v_run and stage_code='RESEARCH'
      and lifecycle_status='PAUSED'
      and active_manifest_kind='CHECKPOINT'
      and handoff_gate_state='NOT_EVALUATED'
  ) then raise exception 'checkpoint stage state invalid'; end if;

  -- A checkpoint can never admit Deep Dive.
  select state_version into v_run_state from public.orotitan_runs where run_id=v_run;
  v_rejected := false;
  begin
    perform public.start_orotitan_stage(
      v_run,'DEEP_DIVE',v_run_state,'DEEP_DIVE','1.0',v_dd_hash,
      'start:dd:early',repeat('7',64),'PILOTAGE'
    );
  exception when check_violation then
    v_rejected := true;
  end;
  if not v_rejected then raise exception 'Deep Dive admitted from CHECKPOINT'; end if;

  select state_version into v_run_state from public.orotitan_runs where run_id=v_run;
  select state_version into v_stage_state from public.orotitan_run_stages where run_id=v_run and stage_code='RESEARCH';
  perform public.resume_orotitan_stage(
    v_run,'RESEARCH',v_run_state,v_stage_state,
    'resume:research',repeat('8',64)
  );

  -- Build all mandatory Research FINAL outputs.
  with items as (
    select artifact_type, ordinality,
           gen_random_uuid() as artifact_id,
           substr('123456789abcdef', ordinality::integer, 1) as hash_char
    from unnest(v_required) with ordinality as t(artifact_type, ordinality)
  ), regs as (
    select ordinality,
           pg_temp.registry_artifact_registration(
             artifact_id,artifact_type,'AUTHORITATIVE_STAGE_OUTPUT','AUTHORITATIVE',hash_char,
             'research/final-' || ordinality || '.json'
           ) as registration
    from items
  )
  select jsonb_agg(registration order by ordinality),
         jsonb_agg(pg_temp.registry_artifact_ref(registration) order by ordinality)
    into v_final_regs, v_final_refs
  from regs;

  v_manifest := jsonb_build_object(
    'manifest_schema_version','1.0.0',
    'manifest_id',v_final_manifest,
    'manifest_kind','FINAL',
    'run_id',v_run,
    'stage','RESEARCH',
    'stage_revision',1,
    'issuer_id',v_issuer,
    'security_id',v_security,
    'dossier_id',v_dossier,
    'canonical_mode','ANALYZE',
    'run_type','INITIAL',
    'data_cutoff','2026-09-14',
    'baseline_snapshot_id',null,
    'process_version','1.0',
    'pilotage_contract_version','1.0.1',
    'contract_pins',v_pins,
    'stage_contract',v_pins->'research_stage',
    'contract_set_sha256',v_contract_hash,
    'input_artifacts','[]'::jsonb,
    'output_artifacts',v_final_refs,
    'stage_status','COMPLETE',
    'contract_status_code','COMPLETE',
    'handoff_gate',jsonb_build_object('name','READY_FOR_DEEP_DIVE','state','YES'),
    'critical_blockers','[]'::jsonb,
    'open_material_limitations','[]'::jsonb,
    'parent_manifests',jsonb_build_array(jsonb_build_object('manifest_id',v_checkpoint_manifest,'version',1,'content_sha256',repeat('5',64))),
    'started_at','2026-09-14T14:00:00Z',
    'completed_at','2026-09-14T15:00:00Z'
  );
  v_manifest_reg := pg_temp.registry_manifest_registration(
    v_final_manifest,
    v_manifest,
    'AUTHORITATIVE_STAGE_OUTPUT',
    'AUTHORITATIVE',
    'research/final-manifest.json'
  );

  select state_version into v_run_state from public.orotitan_runs where run_id=v_run;
  select state_version into v_stage_state from public.orotitan_run_stages where run_id=v_run and stage_code='RESEARCH';
  perform public.finalize_orotitan_stage(
    v_run,'RESEARCH',v_run_state,v_stage_state,
    v_manifest,v_manifest_reg,v_final_regs,'[]'::jsonb,
    'finalize:research',repeat('9',64),'RESEARCH_WORKER'
  );

  if not exists (
    select 1 from public.orotitan_run_stages
    where run_id=v_run and stage_code='RESEARCH'
      and lifecycle_status='COMPLETE'
      and active_manifest_artifact_id=v_final_manifest
      and active_manifest_kind='FINAL'
      and handoff_gate_state='YES'
  ) then raise exception 'Research FINAL state invalid'; end if;

  -- Switching checkpoint -> final automatically supersedes the old bundle.
  if exists (
    select 1 from public.orotitan_artifacts
    where run_id=v_run
      and (artifact_id in (v_checkpoint_manifest,v_checkpoint_output))
      and authority_state <> 'SUPERSEDED'
  ) then raise exception 'checkpoint bundle was not superseded'; end if;

  -- Exact artifact resolver accepts current authoritative output.
  select value into v_resolved from jsonb_array_elements(v_final_regs) limit 1;
  v_json := public.resolve_orotitan_artifact(
    v_run,(v_resolved->>'artifact_id')::uuid,(v_resolved->>'version')::integer,
    v_resolved->>'content_sha256','AUTHORITATIVE_STAGE_OUTPUT'
  );
  if v_json->>'content_sha256' <> v_resolved->>'content_sha256' then
    raise exception 'artifact resolver returned wrong hash';
  end if;

  -- Immutable run lock fields cannot be changed directly, even by owner.
  v_rejected := false;
  begin
    update public.orotitan_runs set data_cutoff = date '2026-09-15' where run_id=v_run;
  exception when check_violation then
    v_rejected := true;
  end;
  if not v_rejected then raise exception 'DATA_CUTOFF mutation was accepted'; end if;

  -- Events are append-only.
  select event_id,payload into v_event_id,v_event_payload
  from public.orotitan_run_events where run_id=v_run order by created_at limit 1;
  v_rejected := false;
  begin
    update public.orotitan_run_events set payload = jsonb_build_object('tampered',true) where event_id=v_event_id;
  exception when others then
    v_rejected := true;
  end;
  if not v_rejected then raise exception 'run event mutation was accepted'; end if;

  -- A valid Research FINAL now admits Deep Dive.
  select state_version into v_run_state from public.orotitan_runs where run_id=v_run;
  perform public.start_orotitan_stage(
    v_run,'DEEP_DIVE',v_run_state,'DEEP_DIVE','1.0',v_dd_hash,
    'start:dd',repeat('a',64),'PILOTAGE'
  );

  if not exists (
    select 1 from public.orotitan_run_stages
    where run_id=v_run and stage_code='DEEP_DIVE' and lifecycle_status='IN_PROGRESS'
  ) then raise exception 'Deep Dive did not start after valid Research handoff'; end if;

  -- Reopening Research invalidates downstream eligibility and supersedes the
  -- prior Research FINAL bundle without deleting historical bytes/metadata.
  select state_version into v_run_state from public.orotitan_runs where run_id=v_run;
  select state_version into v_stage_state from public.orotitan_run_stages where run_id=v_run and stage_code='RESEARCH';
  perform public.reopen_orotitan_stage(
    v_run,'RESEARCH',v_run_state,v_stage_state,'IN_PROGRESS',
    jsonb_build_object('code','NEW_MATERIAL_EVIDENCE','summary','test reopen'),
    'reopen:research',repeat('b',64)
  );

  if not exists (
    select 1 from public.orotitan_run_stages
    where run_id=v_run and stage_code='RESEARCH'
      and stage_revision=2
      and lifecycle_status='IN_PROGRESS'
      and handoff_gate_state='NOT_EVALUATED'
      and active_manifest_artifact_id is null
  ) then raise exception 'Research reopen state invalid'; end if;

  if not exists (
    select 1 from public.orotitan_run_stages
    where run_id=v_run and stage_code='DEEP_DIVE'
      and lifecycle_status='BLOCKED'
      and contract_status_code='UPSTREAM_STAGE_REOPENED'
  ) then raise exception 'Deep Dive was not blocked by upstream reopen'; end if;

  if exists (
    select 1 from public.orotitan_artifacts
    where run_id=v_run
      and (artifact_id=v_final_manifest or manifest_artifact_id=v_final_manifest)
      and authority_state <> 'SUPERSEDED'
  ) then raise exception 'Research FINAL bundle was not superseded on reopen'; end if;
end $$;

select jsonb_build_object(
  'tables', 5,
  'rpcs', 11,
  'rls_enabled', true,
  'direct_writes_anon_authenticated_service_role', 0,
  'checkpoint_final_handoff', 'PASS',
  'manifest_supersession', 'PASS',
  'reopen_downstream_invalidation', 'PASS',
  'result', 'PASS'
) as orotitan_registry_v1_verification;
